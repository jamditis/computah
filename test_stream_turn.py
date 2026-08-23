#!/usr/bin/env python3
"""Tests for the live streaming primitives: stream_detect_wake, capture_request,
and run_turn.

Self-contained and mic-free, like test_pipeline.py: the energy-endpointing checks
build frames directly with numpy (no model), and the streaming checks use the
bundled hey_jarvis model with Piper-synthesized audio (no custom model, no personal
recordings), so a fresh clone can run this.

Run:  .venv/bin/python test_stream_turn.py
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

import pipeline
from pipeline import FRAME_SIZE

TEST_DIR = Path(__file__).resolve().parent / "test_audio"
TEST_DIR.mkdir(exist_ok=True)

# Synth "hey jarvis" fires well above this (0.43-0.98 observed) and the negatives
# score ~0, so 0.2 keeps positives firing with margin against onnxruntime's
# run-to-run float nondeterminism while keeping the silent cases clearly silent.
# This is a test threshold only; the production default lives in config.json.
DETECT_THR = 0.2

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    results.append((PASS if ok else FAIL, name))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")
    return ok


def loud(n: int) -> list[np.ndarray]:
    return [np.full(FRAME_SIZE, 4000, dtype=np.int16) for _ in range(n)]


def silent(n: int) -> list[np.ndarray]:
    return [np.zeros(FRAME_SIZE, dtype=np.int16) for _ in range(n)]


def nframes(pcm: np.ndarray) -> int:
    return len(pcm) // FRAME_SIZE


def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def build_stream(phrase: str, name: str) -> str:
    """Synthesize a phrase with Piper and wrap it in leading/trailing room tone to
    emulate a continuous mic stream. Leading silence is what lets the streaming
    detector fill its context window naturally, the way a live mic does."""
    speech = str(TEST_DIR / f"stream_src_{name}.wav")
    pipeline.speak(phrase, speech)
    pcm = pipeline._load_pcm16(speech)
    lead = np.zeros(int(1.5 * 16000), dtype=np.int16)
    tail = np.zeros(int(1.0 * 16000), dtype=np.int16)
    stream = np.concatenate([lead, pcm, tail])
    import soundfile as sf

    out = str(TEST_DIR / f"stream_{name}.wav")
    sf.write(out, stream, 16000, subtype="PCM_16")
    return out


def main() -> int:
    # ----- capture_request endpointing (fast, no model) -------------------- #
    print("=== capture_request: energy endpointing (no model) ===")
    cap = pipeline.capture_request(iter(loud(5) + silent(15)))
    check(
        "endpoints after trailing silence",
        nframes(cap) == 15,
        f"5 speech + 10 silence (endpoint) captured, got {nframes(cap)} frames",
    )

    cap = pipeline.capture_request(iter(loud(120)))
    check(
        "respects max-request cap",
        nframes(cap) == pipeline._MAX_REQUEST_FRAMES,
        f"unbroken speech capped at {pipeline._MAX_REQUEST_FRAMES}, got {nframes(cap)}",
    )

    cap = pipeline.capture_request(iter(silent(3) + loud(4) + silent(15)))
    check(
        "captures across leading silence then endpoints",
        nframes(cap) == 17,
        f"3 lead silence + 4 speech + 10 trailing silence, got {nframes(cap)} frames",
    )

    it = iter(silent(60))
    cap = pipeline.capture_request(it)
    consumed = 60 - sum(1 for _ in it)
    check(
        "silence-only abandons fast (empty, before the cap)",
        cap.size == 0 and consumed == pipeline._NO_SPEECH_ONSET_FRAMES,
        f"empty after {consumed} frames, not the {pipeline._MAX_REQUEST_FRAMES} cap",
    )

    # ----- peek_cue_gate: gate the cue on a pause (#55), no model ----------- #
    print("\n=== peek_cue_gate: tell a no-pause command from a pause (no model) ===")

    play, peeked = pipeline.peek_cue_gate(iter(silent(pipeline._CUE_PEEK_FRAMES + 3)))
    check(
        "silence peeks the whole window and plays the cue (a pause)",
        play is True and len(peeked) == pipeline._CUE_PEEK_FRAMES,
        f"play={play} peeked={len(peeked)} frames",
    )

    it = iter(loud(6) + silent(4))
    play, peeked = pipeline.peek_cue_gate(it)
    remaining = sum(1 for _ in it)
    check(
        "speech in the window skips the cue and stops at onset (no-pause command)",
        play is False
        and len(peeked) == pipeline._SPEECH_ONSET_FRAMES
        and remaining == 10 - pipeline._SPEECH_ONSET_FRAMES,
        f"play={play} peeked={len(peeked)} remaining={remaining}",
    )

    # A lone transient (a click or echo, 1-2 frames) is not an onset: the gate still
    # plays the cue, matching capture_request's sustained-run onset test (#54).
    play, peeked = pipeline.peek_cue_gate(
        iter(loud(pipeline._SPEECH_ONSET_FRAMES - 1) + silent(5))
    )
    check(
        "a transient below the onset run still plays the cue",
        play is True and len(peeked) == pipeline._CUE_PEEK_FRAMES,
        f"play={play} peeked={len(peeked)}",
    )

    # Sustained room noise can clear the cheap energy gate. The live cue path must use
    # the same speech-sensitive VAD that later protects capture_request, or a fan makes
    # a real pause look like a no-pause command and suppresses the cue.
    real_confirm = pipeline._confirm_speech
    pipeline._confirm_speech = lambda _pcm, _threshold: False
    try:
        play, peeked = pipeline.peek_cue_gate(
            iter(loud(pipeline._CUE_PEEK_FRAMES + 2)), vad_threshold=0.5
        )
    finally:
        pipeline._confirm_speech = real_confirm
    check(
        "sustained non-speech noise still plays the cue",
        play is True and len(peeked) == pipeline._SPEECH_ONSET_FRAMES,
        f"play={play} peeked={len(peeked)}",
    )

    pipeline._confirm_speech = lambda _pcm, _threshold: True
    try:
        play, peeked = pipeline.peek_cue_gate(
            iter(loud(pipeline._CUE_PEEK_FRAMES + 2)), vad_threshold=0.5
        )
    finally:
        pipeline._confirm_speech = real_confirm
    check(
        "VAD-confirmed speech still skips the cue",
        play is False and len(peeked) == pipeline._SPEECH_ONSET_FRAMES,
        f"play={play} peeked={len(peeked)}",
    )

    # The stream ending inside the window decides on what was seen, and never raises.
    play, peeked = pipeline.peek_cue_gate(iter(silent(1)))
    check(
        "a stream that ends inside the window decides on what was seen",
        play is True and len(peeked) == 1,
        f"play={play} peeked={len(peeked)}",
    )

    # Peeking must not drop audio: the frames returned are exactly the ones consumed,
    # in order, so the caller can feed them back to capture on the no-pause branch.
    src = loud(2) + silent(2)
    it = iter(src)
    play, peeked = pipeline.peek_cue_gate(it)
    roundtrip = peeked + list(it)
    check(
        "peeked frames are exactly the consumed frames (no audio dropped)",
        len(roundtrip) == len(src)
        and all(np.array_equal(a, b) for a, b in zip(roundtrip, src)),
        f"roundtrip={len(roundtrip)}/{len(src)}",
    )

    # The window is _SPEECH_ONSET_FRAMES + 1 on purpose: a single quiet leading frame
    # (a breath before the command) must not defeat detection. This pins the +1 -- with
    # _CUE_PEEK_FRAMES == _SPEECH_ONSET_FRAMES the sustained run can't complete after a
    # quiet frame and the command would be misread as a pause and clipped.
    play, peeked = pipeline.peek_cue_gate(
        iter(silent(1) + loud(pipeline._SPEECH_ONSET_FRAMES) + silent(2))
    )
    check(
        "a single quiet leading frame still reads as a no-pause command (+1 window)",
        play is False and len(peeked) == pipeline._CUE_PEEK_FRAMES,
        f"play={play} peeked={len(peeked)}",
    )

    # ----- streaming detection + run_turn (loads bundled hey_jarvis) ------- #
    print("\n=== stream_detect_wake + run_turn (bundled hey_jarvis, synth audio) ===")
    jarvis = build_stream("hey jarvis, what is two plus two?", "jarvis")
    nowake = build_stream("what time is it in tokyo right now?", "nowake")
    model = pipeline._get_oww_model(pipeline._resolve_wake_path("hey_jarvis"))

    score = pipeline.stream_detect_wake(
        pipeline.iter_wav_frames(jarvis), model, DETECT_THR
    )
    check(
        "streaming detect fires on the wake word (no padding)",
        score is not None and score >= DETECT_THR,
        f"peak score {score}",
    )

    none_score = pipeline.stream_detect_wake(
        pipeline.iter_wav_frames(nowake), model, DETECT_THR
    )
    check(
        "streaming detect stays silent without the wake word",
        none_score is None,
        f"returned {none_score}",
    )

    # run_turn end to end, with a stubbed brain so the check is about the
    # stream -> capture -> transcribe -> speak flow, not a live brain.
    real_brain = pipeline.brain
    pipeline.brain = lambda text, **_: "Two plus two is four."
    try:
        out_wav = str(TEST_DIR / "turn_reply.wav")
        r = pipeline.run_turn(
            pipeline.iter_wav_frames(jarvis),
            model_name="hey_jarvis",
            threshold=DETECT_THR,
            out_wav_path=out_wav,
        )
    finally:
        pipeline.brain = real_brain

    ok_turn = (
        r is not None
        and ("two" in (r["transcript"] or "").lower() or "2" in (r["transcript"] or ""))
        and r["reply"] == "Two plus two is four."
        and Path(out_wav).exists()
        and wav_duration(out_wav) > 0.2
    )
    check(
        "run_turn drives a full live turn from a frame stream",
        ok_turn,
        f"transcript={r['transcript']!r} reply={r['reply']!r}"
        if r
        else "returned None",
    )

    no_turn = pipeline.run_turn(
        pipeline.iter_wav_frames(nowake), model_name="hey_jarvis", threshold=DETECT_THR
    )
    check(
        "run_turn returns None when no wake fires",
        no_turn is None,
        f"returned {no_turn}",
    )

    # A wake that fires but is followed by silence (false/abandoned wake) must be
    # ignored before transcribe/brain run, so whisper never hallucinates on silence.
    called = {"transcribe": False, "brain": False}
    real_cap, real_tx, real_brain = (
        pipeline.capture_request,
        pipeline.transcribe_detailed,
        pipeline.brain,
    )
    pipeline.capture_request = lambda fr, preroll=None, vad_threshold=None, **_: (
        np.zeros(0, dtype=np.int16)
    )
    pipeline.transcribe_detailed = lambda p: (
        called.__setitem__("transcribe", True) or pipeline.Transcript("", 0.0, 0.0)
    )
    pipeline.brain = lambda t, **_: called.__setitem__("brain", True) or ""
    try:
        silent_turn = pipeline.run_turn(
            pipeline.iter_wav_frames(jarvis),
            model_name="hey_jarvis",
            threshold=DETECT_THR,
        )
    finally:
        pipeline.capture_request, pipeline.transcribe_detailed, pipeline.brain = (
            real_cap,
            real_tx,
            real_brain,
        )
    check(
        "run_turn ignores a wake with no speech (skips transcribe/brain)",
        silent_turn is None and not called["transcribe"] and not called["brain"],
        f"returned {silent_turn}, called={called}",
    )

    # Non-speech audio (a loud blip that sets speech_seen) can still transcribe to
    # nothing; that turn must not reach the brain.
    brain_hit = {"called": False}
    real_cap2, real_tx2, real_brain2 = (
        pipeline.capture_request,
        pipeline.transcribe_detailed,
        pipeline.brain,
    )
    pipeline.capture_request = lambda fr, preroll=None, vad_threshold=None, **_: (
        np.full(8 * FRAME_SIZE, 4000, dtype=np.int16)
    )
    pipeline.transcribe_detailed = lambda p: pipeline.Transcript("   ", 0.0, 0.0)
    pipeline.brain = lambda t, **_: brain_hit.__setitem__("called", True) or "x"
    try:
        noise_turn = pipeline.run_turn(
            pipeline.iter_wav_frames(jarvis),
            model_name="hey_jarvis",
            threshold=DETECT_THR,
        )
    finally:
        pipeline.capture_request, pipeline.transcribe_detailed, pipeline.brain = (
            real_cap2,
            real_tx2,
            real_brain2,
        )
    check(
        "run_turn ignores audio that transcribes to nothing (skips brain)",
        noise_turn is None and not brain_hit["called"],
        f"returned {noise_turn}, brain_called={brain_hit['called']}",
    )

    # Mishear guard: a low-confidence transcript must never reach the brain (a
    # garbled command must not trigger an action). Stub the transcription to look
    # garbled (avg_logprob below the floor) and assert the brain is skipped, the
    # spoken reply is the re-prompt, and the turn is marked rejected so the loop
    # still gives spoken feedback. stream_detect_wake and capture_request run for
    # real on the jarvis stream; only the transcription and brain are stubbed.
    cfg = pipeline.load_config()
    guard_brain = {"called": False}
    real_td, real_brain4 = pipeline.transcribe_detailed, pipeline.brain
    pipeline.transcribe_detailed = lambda p: pipeline.Transcript(
        "delete everything", cfg["stt_min_avg_logprob"] - 2.0, 0.1
    )
    pipeline.brain = lambda t, **_: guard_brain.__setitem__("called", True) or "x"
    try:
        rej = pipeline.run_turn(
            pipeline.iter_wav_frames(jarvis),
            model_name="hey_jarvis",
            threshold=DETECT_THR,
            out_wav_path=str(TEST_DIR / "turn_reply.wav"),
        )
    finally:
        pipeline.transcribe_detailed, pipeline.brain = real_td, real_brain4
    check(
        "mishear guard rejects a low-confidence transcript (brain skipped)",
        rej is not None
        and rej.get("rejected") == "low_confidence"
        and not guard_brain["called"]
        and rej["reply"] == pipeline.STT_REPROMPT,
        (
            f"reply={rej['reply']!r} rejected={rej.get('rejected')} "
            f"brain_called={guard_brain['called']}"
        )
        if rej
        else "returned None",
    )

    # A confident transcript passes the guard and reaches the brain unchanged.
    pass_brain = {"called": False}
    real_td2, real_brain5 = pipeline.transcribe_detailed, pipeline.brain
    pipeline.transcribe_detailed = lambda p: pipeline.Transcript(
        "what is two plus two", 0.0, 0.0
    )
    pipeline.brain = lambda t, **_: pass_brain.__setitem__("called", True) or "Four."
    try:
        passed = pipeline.run_turn(
            pipeline.iter_wav_frames(jarvis),
            model_name="hey_jarvis",
            threshold=DETECT_THR,
            out_wav_path=str(TEST_DIR / "turn_reply.wav"),
        )
    finally:
        pipeline.transcribe_detailed, pipeline.brain = real_td2, real_brain5
    check(
        "mishear guard passes a confident transcript through to the brain",
        passed is not None
        and pass_brain["called"]
        and "rejected" not in passed
        and passed["reply"] == "Four.",
        (f"reply={passed['reply']!r} brain_called={pass_brain['called']}")
        if passed
        else "returned None",
    )

    # on_capture marks the moment listening for a turn ends. The half-duplex loop
    # pauses the mic there, so it must fire once a request is captured -- on a
    # successful turn AND on a wake followed by silence -- and never when no wake
    # fired, or the loop's pause/resume bookkeeping desyncs.
    fired = {"n": 0}
    real_brain3 = pipeline.brain
    pipeline.brain = lambda text, **_: "Two plus two is four."
    try:
        cap_turn = pipeline.run_turn(
            pipeline.iter_wav_frames(jarvis),
            model_name="hey_jarvis",
            threshold=DETECT_THR,
            out_wav_path=str(TEST_DIR / "turn_reply.wav"),
            on_capture=lambda: fired.__setitem__("n", fired["n"] + 1),
        )
    finally:
        pipeline.brain = real_brain3
    check(
        "on_capture fires once on a successful turn",
        cap_turn is not None and fired["n"] == 1,
        f"calls={fired['n']}",
    )

    fired_nowake = {"n": 0}
    nc_turn = pipeline.run_turn(
        pipeline.iter_wav_frames(nowake),
        model_name="hey_jarvis",
        threshold=DETECT_THR,
        on_capture=lambda: fired_nowake.__setitem__("n", fired_nowake["n"] + 1),
    )
    check(
        "on_capture does not fire when no wake fires",
        nc_turn is None and fired_nowake["n"] == 0,
        f"calls={fired_nowake['n']}",
    )

    fired_silent = {"n": 0}
    real_cap3 = pipeline.capture_request
    pipeline.capture_request = lambda fr, preroll=None, vad_threshold=None, **_: (
        np.zeros(0, dtype=np.int16)
    )
    try:
        si_turn = pipeline.run_turn(
            pipeline.iter_wav_frames(jarvis),
            model_name="hey_jarvis",
            threshold=DETECT_THR,
            on_capture=lambda: fired_silent.__setitem__("n", fired_silent["n"] + 1),
        )
    finally:
        pipeline.capture_request = real_cap3
    check(
        "on_capture fires on a wake even when no speech follows",
        si_turn is None and fired_silent["n"] == 1,
        f"calls={fired_silent['n']}",
    )

    # ----- run_turn wires the cue gate to both branches (#55) --------------- #
    print("\n=== run_turn: the cue gate skips the cue on a no-pause command ===")
    real_sdw, real_cap4, real_confirm4 = (
        pipeline.stream_detect_wake,
        pipeline.capture_request,
        pipeline._confirm_speech,
    )

    def _fire_without_consuming(fr, model, threshold, preroll=None):
        return 0.9  # fire but consume nothing, so the peek sees the frames below

    def _count_capture(store):
        def _cap(fr, preroll=None, vad_threshold=None, **_):
            store["frames"] = sum(1 for _ in fr)  # count what capture actually receives
            return np.zeros(0, dtype=np.int16)  # empty -> the turn returns None

        return _cap

    # No-pause: speech right after the wake. on_wake (the cue) must NOT fire, and the
    # peeked command frames must reach capture -- nothing clipped. Input is 6 loud + 15
    # silent; the peek stops at onset (3 frames) and prepends them, so capture sees all 21.
    store_np, cue_np = {}, {"n": 0}
    pipeline.stream_detect_wake = _fire_without_consuming
    pipeline.capture_request = _count_capture(store_np)
    pipeline._confirm_speech = lambda _pcm, _threshold: True
    try:
        r_np = pipeline.run_turn(
            iter(loud(6) + silent(15)),
            model_name="hey_jarvis",
            threshold=DETECT_THR,
            on_wake=lambda: cue_np.__setitem__("n", cue_np["n"] + 1) or True,
        )
    finally:
        (
            pipeline.stream_detect_wake,
            pipeline.capture_request,
            pipeline._confirm_speech,
        ) = (
            real_sdw,
            real_cap4,
            real_confirm4,
        )
    check(
        "no-pause command: cue skipped, the peeked frames reach capture",
        r_np is None and cue_np["n"] == 0 and store_np.get("frames") == 21,
        f"cue_calls={cue_np['n']} capture_frames={store_np.get('frames')}",
    )

    # Pause: silence right after the wake. on_wake (the cue) fires, and its peeked
    # pre-cue room tone is dropped from capture. Input is 6 + 4 + 12 = 22 frames; the
    # peek reads the first _CUE_PEEK_FRAMES of silence and drops them, so capture sees 18.
    store_pa, cue_pa = {}, {"n": 0}
    pipeline.stream_detect_wake = _fire_without_consuming
    pipeline.capture_request = _count_capture(store_pa)
    try:
        r_pa = pipeline.run_turn(
            iter(silent(6) + loud(4) + silent(12)),
            model_name="hey_jarvis",
            threshold=DETECT_THR,
            on_wake=lambda: cue_pa.__setitem__("n", cue_pa["n"] + 1) or True,
        )
    finally:
        pipeline.stream_detect_wake, pipeline.capture_request = real_sdw, real_cap4
    check(
        "pause: the cue fires and its peeked room tone is dropped from capture",
        r_pa is None
        and cue_pa["n"] == 1
        and store_pa.get("frames") == 22 - pipeline._CUE_PEEK_FRAMES,
        f"cue_calls={cue_pa['n']} capture_frames={store_pa.get('frames')}",
    )

    # Cue failure on the pause branch: on_wake fired but returned falsy (the cue could
    # not play, nothing was flushed). The buffered audio must be kept -- the peeked
    # frames are fed back to capture, not dropped -- so a command starting inside the
    # peek window is not clipped. Input is 22 frames; capture must see all 22.
    store_fail, cue_fail = {}, {"n": 0}
    pipeline.stream_detect_wake = _fire_without_consuming
    pipeline.capture_request = _count_capture(store_fail)
    try:
        r_fail = pipeline.run_turn(
            iter(silent(6) + loud(4) + silent(12)),
            model_name="hey_jarvis",
            threshold=DETECT_THR,
            on_wake=lambda: cue_fail.__setitem__("n", cue_fail["n"] + 1) or False,
        )
    finally:
        pipeline.stream_detect_wake, pipeline.capture_request = real_sdw, real_cap4
    check(
        "cue failure on a pause keeps the peeked frames (no clip)",
        r_fail is None and cue_fail["n"] == 1 and store_fail.get("frames") == 22,
        f"cue_calls={cue_fail['n']} capture_frames={store_fail.get('frames')}",
    )

    # End-to-end with the cue enabled: the gate, a real capture, and real transcription
    # still complete a full turn -- the cue wiring does not break the live path. The
    # deterministic no-pause / pause branch behavior is pinned by the stubbed checks
    # above; this is the real-audio integration smoke test. The jarvis synth has an
    # internal pause, so acoustically it exercises the cue-plays branch here (a genuine
    # one-breath command cannot be synthesized reliably), which is why this asserts a
    # completed turn rather than an unclipped leading word.
    print("\n=== run_turn: the cue-enabled path completes a real turn ===")
    real_brain6 = pipeline.brain
    pipeline.brain = lambda text, **_: "Two plus two is four."
    cue_real = {"n": 0}
    try:
        r_real = pipeline.run_turn(
            pipeline.iter_wav_frames(jarvis),
            model_name="hey_jarvis",
            threshold=DETECT_THR,
            out_wav_path=str(TEST_DIR / "turn_reply.wav"),
            on_wake=lambda: cue_real.__setitem__("n", cue_real["n"] + 1) or True,
        )
    finally:
        pipeline.brain = real_brain6
    ok_cue = (
        r_real is not None
        and (
            "2" in (r_real["transcript"] or "")
            or "two" in (r_real["transcript"] or "").lower()
        )
        and r_real["reply"] == "Two plus two is four."
    )
    check(
        "a full turn completes with the cue enabled (real audio)",
        ok_cue,
        f"transcript={r_real['transcript']!r} reply={r_real['reply']!r} "
        f"cue_calls={cue_real['n']}"
        if r_real
        else "returned None",
    )

    # A model not built by _get_oww_model (a future mic adapter may build its own)
    # must not crash _reset_oww on a missing _blank_buffers snapshot.
    from openwakeword.model import Model

    raw = Model(wakeword_model_paths=[pipeline._resolve_wake_path("hey_jarvis")])
    had_snapshot_before = hasattr(raw, "_blank_buffers")
    raw_score = pipeline.stream_detect_wake(
        pipeline.iter_wav_frames(jarvis), raw, DETECT_THR
    )
    check(
        "stream_detect_wake handles a model not built by the cache",
        raw_score is not None
        and not had_snapshot_before
        and hasattr(raw, "_blank_buffers"),
        f"external model fired (score {raw_score}); snapshot attached lazily",
    )

    n_pass = sum(1 for r in results if r[0] == PASS)
    n_total = len(results)
    print(f"\n=== {n_pass}/{n_total} checks passed ===")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
