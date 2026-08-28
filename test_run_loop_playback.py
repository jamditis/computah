#!/usr/bin/env python3
"""Playback-failure regressions for pipeline.run_loop.

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

import contextlib
import io
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
    # Part of the real Microphone contract the live loop reads (issue #34). None
    # means "nothing wrong with this device", which is what a fake should claim.
    capture_risk = None

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


def test_reply_failure_does_not_crash() -> None:
    print("=== reply playback failure ===")
    state = {"turns": 0, "played": []}
    saved_load = pipeline.load_config
    saved_warm = pipeline.warm_models
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
            return {
                "transcript": "file an issue",
                "reply": "Filed it.",
                "rejected": None,
            }
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
    # Stub warming so this stays a fast, no-model test: run_loop pre-warms the models
    # before listening (issue #13), which would otherwise load real models here. The
    # warm path itself is covered by test_warm_models.py.
    pipeline.warm_models = lambda cfg=None, wake_word=None: {}
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
        pipeline.warm_models = saved_warm
        pipeline.run_turn = saved_run_turn
        if saved_audio is None:
            sys.modules.pop("audio", None)
        else:
            sys.modules["audio"] = saved_audio


def test_failed_wake_chime_stays_disabled() -> None:
    """One failed optional cue must not put every later turn at the same risk (#58)."""
    print("\n=== wake chime stays disabled after failure ===")
    state = {"turns": 0, "cue_calls": 0, "hook_presence": []}
    saved_load = pipeline.load_config
    saved_warm = pipeline.warm_models
    saved_run_turn = pipeline.run_turn
    saved_audio = sys.modules.get("audio")
    saved_chime = sys.modules.get("chime")

    pipeline.load_config = lambda: {"wake_word": "computah", "wake_chime": True}
    pipeline.warm_models = lambda cfg=None, wake_word=None: {}

    def fake_run_turn(frames, **kw):
        state["turns"] += 1
        on_wake = kw.get("on_wake")
        state["hook_presence"].append(on_wake is not None)
        if on_wake is not None:
            on_wake()
        return None

    pipeline.run_turn = fake_run_turn

    class _TwoTurnMic(_FakeMic):
        def active(self) -> bool:
            return state["turns"] < 2

    fake_audio = types.ModuleType("audio")
    fake_audio.Microphone = lambda name=None: _TwoTurnMic()

    def fail_cue(path, name=None):
        state["cue_calls"] += 1
        raise RuntimeError("PortAudioError: output device unavailable")

    fake_audio.play_wav = fail_cue
    fake_chime = types.ModuleType("chime")
    fake_chime.wake_cue_wav = lambda: "cue.wav"
    sys.modules["audio"] = fake_audio
    sys.modules["chime"] = fake_chime

    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            pipeline.run_loop()
        check(
            "a failed cue is attempted only once across later turns",
            state["cue_calls"] == 1 and state["hook_presence"] == [True, False],
            f"cue_calls={state['cue_calls']} hooks={state['hook_presence']}",
        )
        check(
            "the operator is told the cue stays disabled until restart",
            "disabled until restart" in output.getvalue(),
            output.getvalue().strip(),
        )
    finally:
        pipeline.load_config = saved_load
        pipeline.warm_models = saved_warm
        pipeline.run_turn = saved_run_turn
        if saved_audio is None:
            sys.modules.pop("audio", None)
        else:
            sys.modules["audio"] = saved_audio
        if saved_chime is None:
            sys.modules.pop("chime", None)
        else:
            sys.modules["chime"] = saved_chime


def main() -> int:
    test_reply_failure_does_not_crash()
    test_failed_wake_chime_stays_disabled()
    n_pass = sum(1 for r in results if r[0] == PASS)
    print(f"\n=== {n_pass}/{len(results)} checks passed ===")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
