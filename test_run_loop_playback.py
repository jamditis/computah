#!/usr/bin/env python3
"""Regression: pipeline.run_loop must degrade a reply-playback failure, never crash.

The always-on live loop (pipeline.run_loop) plays each reply through the configured
output device. A dead or busy output device that raises mid-playback must be logged and
survived -- one failed playback cannot take down the loop -- exactly as the wake chime
above it and live_driver.run_turn (issue #11) already guarantee. This pins that the
reply-playback call is wrapped, with no microphone or PortAudio needed: run_loop imports
the `audio` module lazily, so a fake stands in for it.

Run:  .venv/bin/python test_run_loop_playback.py
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import sys
import types

import pipeline

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")
    return ok


class _FakeMic:
    """Minimal audio.Microphone stand-in: a context manager whose stream has already
    ended (active() is False), so run_loop breaks right after the single stubbed turn."""

    device_label = "fake-mic"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def frames(self):
        return iter(())

    def pause(self):
        pass

    def resume(self):
        pass

    def flush(self):
        pass

    def active(self) -> bool:
        return False


def main() -> int:
    state = {"turns": 0, "played": []}
    saved_load = pipeline.load_config
    saved_run_turn = pipeline.run_turn
    saved_audio = sys.modules.get("audio")

    def fake_load_config():
        # wake_chime off so the loop never touches the (also-guarded) cue path.
        return {"wake_word": "computah", "wake_chime": False}

    def fake_run_turn(frames, **kw):
        # One captured turn, then the mic stream has ended (active() is False), so the
        # loop breaks. The turn's reply is what run_loop then tries to play.
        state["turns"] += 1
        if state["turns"] == 1:
            return {"transcript": "file an issue", "reply": "Filed it.", "rejected": None}
        return None

    fake_audio = types.ModuleType("audio")
    fake_audio.Microphone = lambda name=None: _FakeMic()

    def fake_play_wav(path, name=None):
        # Stand in for a dead/busy output device: record the attempt, then raise the kind
        # of error PortAudio surfaces so the guard under test is the thing exercised.
        state["played"].append((path, name))
        raise RuntimeError("PortAudioError: output device unavailable")

    fake_audio.play_wav = fake_play_wav

    pipeline.load_config = fake_load_config
    pipeline.run_turn = fake_run_turn
    sys.modules["audio"] = fake_audio
    try:
        crashed = None
        try:
            pipeline.run_loop()
        except Exception as e:  # noqa: BLE001 - the whole point: this must NOT propagate
            crashed = e
        check(
            "a reply-playback failure does not crash run_loop",
            crashed is None,
            "run_loop returned normally"
            if crashed is None
            else f"run_loop raised {type(crashed).__name__}: {crashed}",
        )
        check(
            "the reply playback was actually attempted (the guard wraps the real call)",
            len(state["played"]) == 1,
            f"play_wav calls={state['played']}",
        )
    finally:
        pipeline.load_config = saved_load
        pipeline.run_turn = saved_run_turn
        if saved_audio is None:
            sys.modules.pop("audio", None)
        else:
            sys.modules["audio"] = saved_audio

    n_pass = sum(1 for r in results if r[0] == PASS)
    print(f"\n=== {n_pass}/{len(results)} checks passed ===")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
