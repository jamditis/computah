#!/usr/bin/env python3
"""Fast, no-model tests for the wake-acknowledgment chime (issue #41).

Two concerns, both mic-free and model-free:
  1. The cue generator (chime.py) -- a 16 kHz mono int16 two-tone rising cue with
     edge fades. Checks the format, that the edges fade to near silence (no click),
     and that the pitch actually rises (the "two-tone rising" spec).
  2. The wiring -- both live loops fire the chime at the wake->capture boundary, the
     instant the wake fires and before the request is captured, and never when no
     wake fires. The detection/capture/transcribe/brain/speak/playback stages are
     stubbed, so this runs with no microphone, model, or audio device.

Run:  .venv/bin/python test_chime.py
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import soundfile as sf

import chime
import live_driver
import pipeline
from pipeline import FRAME_SIZE, Transcript

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    results.append((PASS if ok else FAIL, name))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")
    return ok


def _dom_freq(seg: np.ndarray) -> float:
    """Dominant frequency (Hz) of an int16 segment, windowed to limit edge leakage."""
    x = seg.astype(np.float64)
    if x.size < 16:
        return 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    freqs = np.fft.rfftfreq(x.size, 1.0 / chime.SR)
    return float(freqs[int(np.argmax(spec))])


def test_generator() -> None:
    print("=== chime cue generator ===")
    pcm = chime.wake_cue_pcm()

    check("cue is 1-D int16 PCM",
          isinstance(pcm, np.ndarray) and pcm.dtype == np.int16 and pcm.ndim == 1,
          f"dtype={pcm.dtype} ndim={pcm.ndim}")

    # A short cue: long enough to be heard, short enough not to delay the turn.
    ms = pcm.size / chime.SR * 1000.0
    check("cue length is in a sane 80-600 ms range", 80.0 <= ms <= 600.0,
          f"{ms:.0f} ms ({pcm.size} samples)")

    peak = int(np.max(np.abs(pcm))) if pcm.size else 0
    check("cue is audible but not clipped", 1000 < peak < 32768, f"peak={peak}")

    # Raised-cosine edge fades ramp through zero, so the waveform is continuous at
    # both ends -- no step discontinuity, hence no click.
    edge = max(abs(int(pcm[0])), abs(int(pcm[-1]))) if pcm.size else 0
    check("edges fade to near silence (no click)", edge < 500,
          f"max edge sample={edge}")

    # "Two-tone rising": the back of the cue is higher in pitch than the front.
    n = pcm.size
    lo = _dom_freq(pcm[: int(0.4 * n)])
    hi = _dom_freq(pcm[int(0.6 * n):])
    check("pitch rises (second tone higher than first)", hi > lo + 50,
          f"front~{lo:.0f}Hz back~{hi:.0f}Hz")

    # The WAV helper writes a readable 16 kHz mono file and is idempotent.
    d = tempfile.mkdtemp()
    path = os.path.join(d, "cue.wav")
    p1 = chime.wake_cue_wav(path)
    data, sr = sf.read(p1, dtype="int16", always_2d=False)
    check("wake_cue_wav writes a 16 kHz mono WAV at the given path",
          p1 == path and os.path.exists(path) and sr == chime.SR and data.ndim == 1,
          f"path={p1!r} sr={sr} ndim={data.ndim}")
    mtime = os.path.getmtime(path)
    p2 = chime.wake_cue_wav(path)
    check("wake_cue_wav is idempotent (same path, file intact)",
          p2 == path and os.path.getmtime(path) == mtime, f"p2={p2!r}")


# --- wiring: pipeline.run_turn fires on_wake between detection and capture --------

def _run_turn_order(cfg_wake: float, fire: bool, use_hook: bool = True,
                    hook_boundary: bool = True):
    """Drive pipeline.run_turn with every stage stubbed, recording the call order of
    detection, the on_wake hook, and capture, plus the pre-roll capture_request saw.

    `fire` toggles whether a wake fires; `use_hook` toggles wiring the on_wake chime
    hook, so a caller can compare the pre-roll cleared (chime) vs preserved (no chime).
    `hook_boundary` is what the hook returns: True models a cue that played and flushed
    (a fresh capture boundary), False a cue that failed (no boundary, buffer kept) --
    so a caller can check the pre-roll is dropped only on the boundary case.
    Returns (order, seen_preroll)."""
    order: list[str] = []
    seen_preroll: list[list] = []
    real = (pipeline._resolve_wake_path, pipeline._get_oww_model,
            pipeline.stream_detect_wake, pipeline.capture_request,
            pipeline.transcribe_detailed, pipeline.guard_transcript,
            pipeline.brain, pipeline.speak)

    pipeline._resolve_wake_path = lambda name: "dummy"
    pipeline._get_oww_model = lambda path: object()

    def fake_detect(frames, model, threshold, preroll=None):
        order.append("detect")
        if fire and preroll is not None:  # a real detection leaves wake frames here
            preroll.append(np.full(FRAME_SIZE, 7, dtype=np.int16))
        return 0.9 if fire else None

    def fake_capture(frames, preroll=None, vad_threshold=None):
        order.append("capture")
        seen_preroll.append(list(preroll) if preroll is not None else [])
        return np.full(8 * FRAME_SIZE, 4000, dtype=np.int16)

    pipeline.stream_detect_wake = fake_detect
    pipeline.capture_request = fake_capture
    pipeline.transcribe_detailed = lambda p: Transcript("file an issue", -0.2, 0.05)
    pipeline.guard_transcript = lambda heard, cfg: (True, "")
    pipeline.brain = lambda t, **_: "ok"
    pipeline.speak = lambda text, out, **_: out

    def on_wake() -> bool:
        order.append("wake")
        return hook_boundary  # True = cue played and flushed (a fresh boundary)

    try:
        pipeline.run_turn(iter([]), model_name="x", threshold=cfg_wake,
                          out_wav_path=os.devnull,
                          on_wake=on_wake if use_hook else None)
    finally:
        (pipeline._resolve_wake_path, pipeline._get_oww_model,
         pipeline.stream_detect_wake, pipeline.capture_request,
         pipeline.transcribe_detailed, pipeline.guard_transcript,
         pipeline.brain, pipeline.speak) = real
    return order, seen_preroll


def test_run_turn_hook() -> None:
    print("\n=== pipeline.run_turn on_wake hook ===")
    order, pr = _run_turn_order(0.5, fire=True, use_hook=True, hook_boundary=True)
    check("on_wake fires once, after detection and before capture",
          order == ["detect", "wake", "capture"], f"order={order}")
    check("a cue that played (hook returns True) drops the pre-roll so the wake-word "
          "tail is not prepended",
          pr == [[]], f"preroll lengths seen by capture={[len(p) for p in pr]}")

    order, pr = _run_turn_order(0.5, fire=True, use_hook=True, hook_boundary=False)
    check("a cue that failed (hook returns falsy, no boundary) keeps the pre-roll so "
          "a no-pause command is still recovered (issue #30)",
          order == ["detect", "wake", "capture"]
          and len(pr) == 1 and len(pr[0]) == 1,
          f"order={order} preroll lengths={[len(p) for p in pr]}")

    order, pr = _run_turn_order(0.5, fire=True, use_hook=False)
    check("without the chime hook the pre-roll is preserved (issue #30 recovery)",
          order == ["detect", "capture"] and len(pr) == 1 and len(pr[0]) == 1,
          f"order={order} preroll lengths={[len(p) for p in pr]}")

    order, _ = _run_turn_order(0.5, fire=False, use_hook=True)
    check("on_wake does not fire when no wake fires",
          order == ["detect"] and "wake" not in order, f"order={order}")


# --- wiring: live_driver plays the chime between wake and capture ----------------

def test_live_driver_chime() -> None:
    print("\n=== live_driver fires the chime at the wake->capture boundary ===")
    order: list[str] = []
    real = (live_driver.listen_for_wake, pipeline.capture_request,
            pipeline.transcribe_detailed, pipeline.brain, pipeline.speak,
            live_driver._play_wav, live_driver.drain, chime.wake_cue_wav)

    def fake_listen(fr, m, t, d, preroll=None):
        if preroll is not None:  # detection leaves the wake-word tail in the pre-roll
            preroll.append(np.full(FRAME_SIZE, 7, dtype=np.int16))
        return 0.9
    live_driver.listen_for_wake = fake_listen
    chime.wake_cue_wav = lambda path=None: "CUE.wav"

    seen_preroll: list[list] = []

    def fake_capture(fr, preroll=None, vad_threshold=None):
        order.append("capture")
        seen_preroll.append(list(preroll) if preroll is not None else [])
        return np.full(8 * FRAME_SIZE, 4000, dtype=np.int16)

    def fake_play(path, dev):
        order.append(f"play:{'cue' if path == 'CUE.wav' else 'reply'}")

    pipeline.capture_request = fake_capture
    pipeline.transcribe_detailed = lambda p: Transcript("file an issue", -0.2, 0.05)
    pipeline.brain = lambda t, **_: "Brain answer."
    pipeline.speak = lambda text, out, **_: out
    live_driver._play_wav = fake_play
    live_driver.drain = lambda frames, n: order.append("drain")

    cfg = {"stt_confidence_guard": True, "stt_min_avg_logprob": -1.0,
           "stt_max_no_speech_prob": 0.6, "wake_chime": True,
           "capture_vad_threshold": 0.5}
    try:
        ran = live_driver.run_turn(iter([]), object(), 0.5, os.devnull, None,
                                   cfg, False)
    finally:
        (live_driver.listen_for_wake, pipeline.capture_request,
         pipeline.transcribe_detailed, pipeline.brain, pipeline.speak,
         live_driver._play_wav, live_driver.drain, chime.wake_cue_wav) = real

    check("chime plays before the request is captured",
          "play:cue" in order and "capture" in order
          and order.index("play:cue") < order.index("capture"),
          f"ran={ran} order={order}")
    check("reply still plays after capture",
          "play:reply" in order
          and order.index("play:reply") > order.index("capture"),
          f"order={order}")
    check("the chime drops the pre-roll so the wake-word tail is not prepended",
          seen_preroll == [[]],
          f"preroll lengths seen by capture={[len(p) for p in seen_preroll]}")


# --- wiring: a failed cue keeps the pre-roll (no false boundary) ------------------

def test_live_driver_chime_failure_keeps_preroll() -> None:
    print("\n=== live_driver: a failed cue keeps the pre-roll (issue #30 recovery) ===")
    order: list[str] = []
    real = (live_driver.listen_for_wake, pipeline.capture_request,
            pipeline.transcribe_detailed, pipeline.brain, pipeline.speak,
            live_driver._play_wav, live_driver.drain, chime.wake_cue_wav)

    def fake_listen(fr, m, t, d, preroll=None):
        if preroll is not None:
            preroll.append(np.full(FRAME_SIZE, 7, dtype=np.int16))
        return 0.9
    live_driver.listen_for_wake = fake_listen
    chime.wake_cue_wav = lambda path=None: "CUE.wav"

    seen_preroll: list[list] = []

    def fake_capture(fr, preroll=None, vad_threshold=None):
        order.append("capture")
        seen_preroll.append(list(preroll) if preroll is not None else [])
        return np.full(8 * FRAME_SIZE, 4000, dtype=np.int16)

    def failing_play(path, dev):
        # Both the cue and the reply route through _play_wav; the cue must fail so the
        # turn exercises the chime's failure branch. The reply's own try/except logs
        # and continues, so failing it too does not abort the turn.
        raise RuntimeError("aplay unavailable")

    pipeline.capture_request = fake_capture
    pipeline.transcribe_detailed = lambda p: Transcript("file an issue", -0.2, 0.05)
    pipeline.brain = lambda t, **_: "Brain answer."
    pipeline.speak = lambda text, out, **_: out
    live_driver._play_wav = failing_play
    live_driver.drain = lambda frames, n: order.append("drain")

    cfg = {"stt_confidence_guard": True, "stt_min_avg_logprob": -1.0,
           "stt_max_no_speech_prob": 0.6, "wake_chime": True,
           "capture_vad_threshold": 0.5}
    try:
        ran = live_driver.run_turn(iter([]), object(), 0.5, os.devnull, None,
                                   cfg, False)
    finally:
        (live_driver.listen_for_wake, pipeline.capture_request,
         pipeline.transcribe_detailed, pipeline.brain, pipeline.speak,
         live_driver._play_wav, live_driver.drain, chime.wake_cue_wav) = real

    check("a failed cue does not drain (nothing bled into the pipe)",
          "drain" not in order, f"order={order}")
    check("a failed cue keeps the pre-roll so a no-pause command is still recovered",
          len(seen_preroll) == 1 and len(seen_preroll[0]) == 1,
          f"preroll lengths seen by capture={[len(p) for p in seen_preroll]}")
    check("the turn still completes after a failed cue",
          ran is True and "capture" in order, f"ran={ran} order={order}")


# --- wiring: the chime is opt-in (default off) -----------------------------------

def test_chime_opt_in_default_off() -> None:
    print("\n=== chime is opt-in: no cue when wake_chime is off or unset ===")
    # The cue regresses the no-pause case on a half-duplex device, so it ships off by
    # default (issue #41). Both an absent key and an explicit false must suppress it,
    # while the turn itself still runs normally (capture -> reply, just no cue/drain).
    for label, cfg_extra in (("wake_chime unset", {}),
                             ("wake_chime false", {"wake_chime": False})):
        order: list[str] = []
        real = (live_driver.listen_for_wake, pipeline.capture_request,
                pipeline.transcribe_detailed, pipeline.brain, pipeline.speak,
                live_driver._play_wav, live_driver.drain, chime.wake_cue_wav)

        def fake_listen(fr, m, t, d, preroll=None):
            if preroll is not None:
                preroll.append(np.full(FRAME_SIZE, 7, dtype=np.int16))
            return 0.9
        live_driver.listen_for_wake = fake_listen
        chime.wake_cue_wav = lambda path=None: "CUE.wav"

        seen_preroll: list[list] = []

        def fake_capture(fr, preroll=None, vad_threshold=None):
            order.append("capture")
            seen_preroll.append(list(preroll) if preroll is not None else [])
            return np.full(8 * FRAME_SIZE, 4000, dtype=np.int16)

        def fake_play(path, dev):
            order.append(f"play:{'cue' if path == 'CUE.wav' else 'reply'}")

        pipeline.capture_request = fake_capture
        pipeline.transcribe_detailed = lambda p: Transcript("file an issue", -0.2, 0.05)
        pipeline.brain = lambda t, **_: "Brain answer."
        pipeline.speak = lambda text, out, **_: out
        live_driver._play_wav = fake_play
        live_driver.drain = lambda frames, n: order.append("drain")

        cfg = {"stt_confidence_guard": True, "stt_min_avg_logprob": -1.0,
               "stt_max_no_speech_prob": 0.6, "capture_vad_threshold": 0.5,
               **cfg_extra}
        try:
            ran = live_driver.run_turn(iter([]), object(), 0.5, os.devnull, None,
                                       cfg, False)
        finally:
            (live_driver.listen_for_wake, pipeline.capture_request,
             pipeline.transcribe_detailed, pipeline.brain, pipeline.speak,
             live_driver._play_wav, live_driver.drain, chime.wake_cue_wav) = real

        check(f"no cue and no chime-drain when {label}",
              "play:cue" not in order and "drain" not in order,
              f"ran={ran} order={order}")
        check(f"the turn still runs (capture then reply) when {label}",
              "capture" in order and "play:reply" in order,
              f"order={order}")
        check(f"the pre-roll is preserved (not chime-cleared) when {label}",
              len(seen_preroll) == 1 and len(seen_preroll[0]) == 1,
              f"preroll lengths seen by capture={[len(p) for p in seen_preroll]}")


def main() -> int:
    test_generator()
    test_run_turn_hook()
    test_live_driver_chime()
    test_live_driver_chime_failure_keeps_preroll()
    test_chime_opt_in_default_off()
    n_pass = sum(1 for r in results if r[0] == PASS)
    n_total = len(results)
    print(f"\n=== {n_pass}/{n_total} checks passed ===")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
