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
    check("endpoints after trailing silence", nframes(cap) == 15,
          f"5 speech + 10 silence (endpoint) captured, got {nframes(cap)} frames")

    cap = pipeline.capture_request(iter(loud(120)))
    check("respects max-request cap", nframes(cap) == pipeline._MAX_REQUEST_FRAMES,
          f"unbroken speech capped at {pipeline._MAX_REQUEST_FRAMES}, got {nframes(cap)}")

    cap = pipeline.capture_request(iter(silent(3) + loud(4) + silent(15)))
    check("captures across leading silence then endpoints", nframes(cap) == 17,
          f"3 lead silence + 4 speech + 10 trailing silence, got {nframes(cap)} frames")

    it = iter(silent(60))
    cap = pipeline.capture_request(it)
    consumed = 60 - sum(1 for _ in it)
    check("silence-only abandons fast (empty, before the cap)",
          cap.size == 0 and consumed == pipeline._NO_SPEECH_ONSET_FRAMES,
          f"empty after {consumed} frames, not the {pipeline._MAX_REQUEST_FRAMES} cap")

    # ----- streaming detection + run_turn (loads bundled hey_jarvis) ------- #
    print("\n=== stream_detect_wake + run_turn (bundled hey_jarvis, synth audio) ===")
    jarvis = build_stream("hey jarvis, what is two plus two?", "jarvis")
    nowake = build_stream("what time is it in tokyo right now?", "nowake")
    model = pipeline._get_oww_model(pipeline._resolve_wake_path("hey_jarvis"))

    score = pipeline.stream_detect_wake(pipeline.iter_wav_frames(jarvis), model, DETECT_THR)
    check("streaming detect fires on the wake word (no padding)",
          score is not None and score >= DETECT_THR,
          f"peak score {score}")

    none_score = pipeline.stream_detect_wake(
        pipeline.iter_wav_frames(nowake), model, DETECT_THR)
    check("streaming detect stays silent without the wake word",
          none_score is None, f"returned {none_score}")

    # run_turn end to end, with a stubbed brain so the check is about the
    # stream -> capture -> transcribe -> speak flow, not a live brain.
    real_brain = pipeline.brain
    pipeline.brain = lambda text, **_: "Two plus two is four."
    try:
        out_wav = str(TEST_DIR / "turn_reply.wav")
        r = pipeline.run_turn(pipeline.iter_wav_frames(jarvis),
                              model_name="hey_jarvis", threshold=DETECT_THR,
                              out_wav_path=out_wav)
    finally:
        pipeline.brain = real_brain

    ok_turn = (r is not None
               and ("two" in (r["transcript"] or "").lower()
                    or "2" in (r["transcript"] or ""))
               and r["reply"] == "Two plus two is four."
               and Path(out_wav).exists() and wav_duration(out_wav) > 0.2)
    check("run_turn drives a full live turn from a frame stream", ok_turn,
          f"transcript={r['transcript']!r} reply={r['reply']!r}" if r else "returned None")

    no_turn = pipeline.run_turn(pipeline.iter_wav_frames(nowake),
                                model_name="hey_jarvis", threshold=DETECT_THR)
    check("run_turn returns None when no wake fires", no_turn is None,
          f"returned {no_turn}")

    # A wake that fires but is followed by silence (false/abandoned wake) must be
    # ignored before transcribe/brain run, so whisper never hallucinates on silence.
    called = {"transcribe": False, "brain": False}
    real_cap, real_tx, real_brain = (
        pipeline.capture_request, pipeline.transcribe, pipeline.brain)
    pipeline.capture_request = lambda fr: np.zeros(0, dtype=np.int16)
    pipeline.transcribe = lambda p: called.__setitem__("transcribe", True) or ""
    pipeline.brain = lambda t, **_: called.__setitem__("brain", True) or ""
    try:
        silent_turn = pipeline.run_turn(pipeline.iter_wav_frames(jarvis),
                                        model_name="hey_jarvis", threshold=DETECT_THR)
    finally:
        pipeline.capture_request, pipeline.transcribe, pipeline.brain = (
            real_cap, real_tx, real_brain)
    check("run_turn ignores a wake with no speech (skips transcribe/brain)",
          silent_turn is None and not called["transcribe"] and not called["brain"],
          f"returned {silent_turn}, called={called}")

    # Non-speech audio (a loud blip that sets speech_seen) can still transcribe to
    # nothing; that turn must not reach the brain.
    brain_hit = {"called": False}
    real_cap2, real_tx2, real_brain2 = (
        pipeline.capture_request, pipeline.transcribe, pipeline.brain)
    pipeline.capture_request = lambda fr: np.full(8 * FRAME_SIZE, 4000, dtype=np.int16)
    pipeline.transcribe = lambda p: "   "
    pipeline.brain = lambda t, **_: brain_hit.__setitem__("called", True) or "x"
    try:
        noise_turn = pipeline.run_turn(pipeline.iter_wav_frames(jarvis),
                                       model_name="hey_jarvis", threshold=DETECT_THR)
    finally:
        pipeline.capture_request, pipeline.transcribe, pipeline.brain = (
            real_cap2, real_tx2, real_brain2)
    check("run_turn ignores audio that transcribes to nothing (skips brain)",
          noise_turn is None and not brain_hit["called"],
          f"returned {noise_turn}, brain_called={brain_hit['called']}")

    # A model not built by _get_oww_model (a future mic adapter may build its own)
    # must not crash _reset_oww on a missing _blank_buffers snapshot.
    from openwakeword.model import Model
    raw = Model(wakeword_model_paths=[pipeline._resolve_wake_path("hey_jarvis")])
    had_snapshot_before = hasattr(raw, "_blank_buffers")
    raw_score = pipeline.stream_detect_wake(
        pipeline.iter_wav_frames(jarvis), raw, DETECT_THR)
    check("stream_detect_wake handles a model not built by the cache",
          raw_score is not None and not had_snapshot_before
          and hasattr(raw, "_blank_buffers"),
          f"external model fired (score {raw_score}); snapshot attached lazily")

    n_pass = sum(1 for r in results if r[0] == PASS)
    n_total = len(results)
    print(f"\n=== {n_pass}/{n_total} checks passed ===")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
