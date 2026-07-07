#!/usr/bin/env python3
"""Fast, no-model tests for live_driver's turn loop, focused on the mishear guard.

live_driver.py is the real-hardware always-on path (arecord -> wake -> STT -> brain
-> speak). It must apply the same confidence guard as pipeline.run_turn, so a garbled
command never reaches the action-capable brain on the hardware path. These checks
stub the wake, capture, transcription, brain, and playback, so they run with no mic,
no model, and no audio device -- like test_brain_bridge.py.

Run:  .venv/bin/python test_live_driver.py
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import os
import sys

import numpy as np

import live_driver
import pipeline
from pipeline import FRAME_SIZE, Transcript

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    results.append((PASS if ok else FAIL, name))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")
    return ok


class _FakeMic:
    """A mic stand-in for turns that do not reach the post-cue flush (wake_chime off):
    run_turn takes the StdinMic only to call flush(), so a counter suffices here."""

    def __init__(self) -> None:
        self.flushed = 0

    def flush(self) -> None:
        self.flushed += 1


GUARD_ON = {"stt_confidence_guard": True,
            "stt_min_avg_logprob": -1.0, "stt_max_no_speech_prob": 0.6,
            "capture_vad_threshold": 0.5}
GUARD_OFF = dict(GUARD_ON, stt_confidence_guard=False)


def drive(cfg: dict, transcript: Transcript, output_device=None,
          play_error: Exception | None = None) -> dict:
    """Run one live_driver turn with wake/capture/transcribe/brain/playback stubbed.

    Returns what was spoken, whether the brain was called and with what, the
    (path, device) args each reply playback was called with, and the run_turn
    return value -- with no microphone, model, or audio device. Pass play_error to
    simulate a dead output device (a raising _play_wav) and check the turn degrades
    instead of crashing.
    """
    state: dict = {"spoken": None, "brain_called": False, "brain_arg": None,
                   "played": []}

    real = (live_driver.listen_for_wake, pipeline.capture_request,
            pipeline.transcribe_detailed, pipeline.brain, pipeline.speak,
            live_driver._play_wav)
    live_driver.listen_for_wake = lambda fr, m, t, d, preroll=None: 0.9
    pipeline.capture_request = (
        lambda fr, preroll=None, vad_threshold=None: np.full(8 * FRAME_SIZE, 4000, dtype=np.int16))
    pipeline.transcribe_detailed = lambda p: transcript

    def fake_brain(text, **_):
        state["brain_called"] = True
        state["brain_arg"] = text
        return "Brain answer."

    def fake_speak(text, out, **_):
        state["spoken"] = text
        return out

    def fake_play(path, dev):
        state["played"].append((path, dev))
        if play_error is not None:
            raise play_error

    pipeline.brain = fake_brain
    pipeline.speak = fake_speak
    live_driver._play_wav = fake_play
    try:
        state["ran"] = live_driver.run_turn(
            iter([]), _FakeMic(), object(), 0.5, "/dev/null", output_device, cfg, False)
    finally:
        (live_driver.listen_for_wake, pipeline.capture_request,
         pipeline.transcribe_detailed, pipeline.brain, pipeline.speak,
         live_driver._play_wav) = real
    return state


def _frame_bytes(val: int) -> bytes:
    """One 80 ms frame's worth of raw S16_LE bytes, every sample == val."""
    return np.full(FRAME_SIZE, val, dtype=np.int16).tobytes()


def test_stdin_mic() -> None:
    """StdinMic.flush() is flush-to-now: it drops everything buffered up to the
    instant capture starts (the cue bleed), with no wall-time guess and without
    blocking on or consuming fresh post-cue frames (issue #56)."""
    print("\n=== StdinMic: flush-to-now drains the pipe without a wall-time guess ===")

    # frames() reconstructs full 80 ms frames from the raw pipe, in order, writable.
    r, w = os.pipe()
    try:
        gen = live_driver.StdinMic(r).frames()
        os.write(w, _frame_bytes(111) + _frame_bytes(222))
        f1, f2 = next(gen), next(gen)
        check("frames() yields written frames in order at full width",
              f1.shape == (FRAME_SIZE,) and int(f1[0]) == 111 and int(f2[0]) == 222,
              f"f1[0]={int(f1[0])} f2[0]={int(f2[0])} shape={f1.shape}")
        check("yielded frames are writable (the wake model needs a writable array)",
              f1.flags.writeable, f"writeable={f1.flags.writeable}")
    finally:
        os.close(r)
        os.close(w)

    # The core fix: stale cue-bleed frames buffered before the flush never reach
    # capture. The next frame after flush() is post-flush audio only.
    r, w = os.pipe()
    try:
        mic = live_driver.StdinMic(r)
        gen = mic.frames()
        os.write(w, _frame_bytes(1000))            # one frame consumed during detection
        next(gen)
        os.write(w, _frame_bytes(1000) + _frame_bytes(1000))  # cue bleed now buffered
        mic.flush()                                # flush-to-now drops the backlog
        os.write(w, _frame_bytes(5000))            # the command, spoken after the cue
        nxt = next(gen)
        check("flush-to-now drops the buffered backlog (next frame is post-flush only)",
              int(nxt[0]) == 5000,
              f"want 5000 (fresh); got {int(nxt[0])} (1000 == stale bleed leaked through)")
    finally:
        os.close(r)
        os.close(w)

    # flush() never blocks: on an empty pipe it returns at once (the whole point vs the
    # old wall-time drain). If it blocked, this turn would hang.
    r, w = os.pipe()
    try:
        mic = live_driver.StdinMic(r)
        mic.flush()  # empty pipe
        os.write(w, _frame_bytes(7))
        nxt = next(mic.frames())
        check("flush-to-now returns at once on an empty pipe and keeps later audio",
              int(nxt[0]) == 7, f"got {int(nxt[0])}")
    finally:
        os.close(r)
        os.close(w)

    # flush() also drops the sub-frame remainder and restores blocking mode, so the
    # next frames() read blocks normally instead of busy-spinning on EAGAIN.
    r, w = os.pipe()
    try:
        mic = live_driver.StdinMic(r)
        mic._buf = b"\x01\x02\x03\x04"  # a partial-frame remainder
        mic.flush()
        check("flush() clears the sub-frame remainder and restores blocking mode",
              mic._buf == b"" and os.get_blocking(r) is True,
              f"buf={mic._buf!r} blocking={os.get_blocking(r)}")
    finally:
        os.close(r)
        os.close(w)

    # frames() ends when arecord exits (the write end closes).
    r, w = os.pipe()
    try:
        mic = live_driver.StdinMic(r)
        os.write(w, _frame_bytes(9))
        os.close(w)  # arecord exits
        got = list(mic.frames())
        check("frames() ends at EOF (the writer closed)",
              len(got) == 1 and int(got[0][0]) == 9, f"n={len(got)}")
    finally:
        os.close(r)


def main() -> int:
    test_stdin_mic()
    print("\n=== live_driver.run_turn: the hardware path honors the guard ===")

    # A low-confidence transcript on the hardware path must be re-prompted, not
    # dispatched: the brain is skipped and the re-prompt is spoken.
    s = drive(GUARD_ON, Transcript("delete everything", -3.0, 0.1))
    check("low-confidence transcript skips the brain and speaks the re-prompt",
          s["ran"] is True and not s["brain_called"]
          and s["spoken"] == pipeline.STT_REPROMPT,
          f"ran={s['ran']} brain_called={s['brain_called']} spoken={s['spoken']!r}")

    # A confident transcript reaches the brain unchanged, and its reply is spoken.
    s = drive(GUARD_ON, Transcript("file an issue", -0.3, 0.05))
    check("confident transcript reaches the brain and speaks its reply",
          s["ran"] is True and s["brain_called"]
          and s["brain_arg"] == "file an issue" and s["spoken"] == "Brain answer.",
          f"brain_called={s['brain_called']} brain_arg={s['brain_arg']!r} "
          f"spoken={s['spoken']!r}")

    # With the guard disabled, even a low-confidence transcript dispatches (the knob
    # is honored on the hardware path too).
    s = drive(GUARD_OFF, Transcript("delete everything", -3.0, 0.99))
    check("guard disabled: a low-confidence transcript still dispatches",
          s["ran"] is True and s["brain_called"] and s["spoken"] == "Brain answer.",
          f"brain_called={s['brain_called']} spoken={s['spoken']!r}")

    print("\n=== live_driver.run_turn: the reply is played back (issue #11) ===")

    # #11's core guarantee: a live turn must PLAY the reply that speak() produced, not
    # just leave the WAV on disk, or the user never hears the answer. Pin that run_turn
    # feeds the reply WAV to the player on the configured output device -- so a refactor
    # that drops the playback call, or misroutes the device, fails here instead of
    # silently going mute.
    s = drive(GUARD_ON, Transcript("file an issue", -0.3, 0.05),
              output_device="plughw:CARD=PowerConf,DEV=0")
    check("the spoken reply WAV is played on the configured output device",
          s["played"] == [("/dev/null", "plughw:CARD=PowerConf,DEV=0")],
          f"played={s['played']}")

    # A dead or busy output device must degrade to a logged error, never crash: one
    # failed aplay cannot take down the always-on loop (live_driver.run_turn wraps the
    # playback in try/except). The turn still completes and the reply was still spoken.
    s = drive(GUARD_ON, Transcript("file an issue", -0.3, 0.05),
              play_error=RuntimeError("aplay: no such device"))
    check("a playback failure degrades gracefully (the turn still completes)",
          s["ran"] is True and s["spoken"] == "Brain answer." and len(s["played"]) == 1,
          f"ran={s['ran']} spoken={s['spoken']!r} played={s['played']}")

    n_pass = sum(1 for r in results if r[0] == PASS)
    n_total = len(results)
    print(f"\n=== {n_pass}/{n_total} checks passed ===")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
