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


GUARD_ON = {"stt_confidence_guard": True,
            "stt_min_avg_logprob": -1.0, "stt_max_no_speech_prob": 0.6}
GUARD_OFF = dict(GUARD_ON, stt_confidence_guard=False)


def drive(cfg: dict, transcript: Transcript) -> dict:
    """Run one live_driver turn with wake/capture/transcribe/brain/playback stubbed.

    Returns what was spoken, whether the brain was called and with what, and the
    run_turn return value -- with no microphone, model, or audio device.
    """
    state: dict = {"spoken": None, "brain_called": False, "brain_arg": None}

    real = (live_driver.listen_for_wake, pipeline.capture_request,
            pipeline.transcribe_detailed, pipeline.brain, pipeline.speak,
            live_driver._play_wav)
    live_driver.listen_for_wake = lambda fr, m, t, d, preroll=None: 0.9
    pipeline.capture_request = (
        lambda fr, preroll=None: np.full(8 * FRAME_SIZE, 4000, dtype=np.int16))
    pipeline.transcribe_detailed = lambda p: transcript

    def fake_brain(text, **_):
        state["brain_called"] = True
        state["brain_arg"] = text
        return "Brain answer."

    def fake_speak(text, out, **_):
        state["spoken"] = text
        return out

    pipeline.brain = fake_brain
    pipeline.speak = fake_speak
    live_driver._play_wav = lambda path, dev: None
    try:
        state["ran"] = live_driver.run_turn(
            iter([]), object(), 0.5, "/dev/null", None, cfg, False)
    finally:
        (live_driver.listen_for_wake, pipeline.capture_request,
         pipeline.transcribe_detailed, pipeline.brain, pipeline.speak,
         live_driver._play_wav) = real
    return state


def main() -> int:
    print("=== live_driver.run_turn: the hardware path honors the guard ===")

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

    n_pass = sum(1 for r in results if r[0] == PASS)
    n_total = len(results)
    print(f"\n=== {n_pass}/{n_total} checks passed ===")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
