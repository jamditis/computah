#!/usr/bin/env python3
"""Checks for the capture-device suitability warning (issue #34).

Two halves: the classifier's verdicts, and the fact that a verdict reaches the
operator at `--listen` startup. The second half is what makes the feature real --
a correct classifier nobody prints is worth nothing.

Hardware-free by construction. capture_quality takes the fields a device
advertises rather than reading a device, and the startup check stands a fake in
for the lazily-imported audio module, so everything here runs on a box with no
PortAudio and no microphone.

Run:  .venv/bin/python test_capture_quality.py
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import contextlib
import io
import sys
import types

import capture_quality
import pipeline
from capture_quality import CaptureRisk, assess_input_device

PASS, FAIL = "PASS", "FAIL"
results: list[bool] = []


def check(name: str, ok: bool, detail: str) -> bool:
    results.append(bool(ok))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")
    return bool(ok)


def test_flags_bluetooth_hands_free_by_name() -> None:
    """The reported failure: a BT hands-free endpoint on Windows."""
    # The real device name from issue #34's box, in the Windows form.
    risk = assess_input_device("Headset (Anker PowerConf Hands-Free AG Audio)", 16000)
    check(
        "hands-free endpoint flagged",
        risk is not None and risk.kind == "hands-free-profile",
        f"kind={risk.kind if risk else None}",
    )
    check(
        "hands-free message names the fix",
        risk is not None and "USB" in risk.message,
        "message points at USB Audio Class" if risk else "no message",
    )


def test_hands_free_wins_over_rate() -> None:
    """Both signals present: report the profile, since that is the actionable one."""
    risk = assess_input_device("Hands-Free AG Audio", 8000)
    check(
        "profile reported ahead of rate",
        risk is not None and risk.kind == "hands-free-profile",
        f"kind={risk.kind if risk else None}",
    )


def test_flags_narrowband_rate() -> None:
    """A device whose own rate is below the pipeline's 16 kHz."""
    risk = assess_input_device("Some Telephony Mic", 8000)
    check(
        "8 kHz device flagged",
        risk is not None and risk.kind == "narrowband",
        f"kind={risk.kind if risk else None}",
    )
    check(
        "narrowband message states the measured rate",
        risk is not None and "8000" in risk.message,
        "message includes 8000 Hz" if risk else "no message",
    )


def test_good_devices_are_silent() -> None:
    """No warning for the devices the project runs on.

    A warning here would be worse than no warning at all: the operator learns to
    skip the line, and the real one gets skipped with it.
    """
    cases = [
        # The Pi production path, native 16 kHz mono over USB Audio Class. This is
        # the boundary case -- 16 kHz must not read as narrowband.
        ("Anker PowerConf", 16000),
        ("Shure MV7", 48000),
        ("NVIDIA Broadcast", 48000),
        ("Microphone (USB Audio Device)", 44100),
        # A USB headset is a fine STT mic, so "headset" alone must not trip it.
        ("Headset Microphone (Logitech USB Headset)", 48000),
    ]
    for name, sr in cases:
        risk = assess_input_device(name, sr)
        check(
            f"silent for {name}",
            risk is None,
            "no warning" if risk is None else f"unexpected {risk.kind}",
        )


def test_unknown_rate_is_not_a_fault() -> None:
    """Absent rate information must not manufacture a warning."""
    for sr in (None, 0, 0.0, "", "not-a-number"):
        risk = assess_input_device("Mystery Device", sr)
        check(
            f"silent for native_sr={sr!r}",
            risk is None,
            "no warning" if risk is None else f"unexpected {risk.kind}",
        )


def test_survives_a_missing_name() -> None:
    """A device with no usable name must not raise."""
    try:
        risk = assess_input_device(None, 48000)
        check("no name does not raise", risk is None, "returned None")
    except Exception as e:  # noqa: BLE001 - any raise is the failure being checked
        check("no name does not raise", False, f"raised {type(e).__name__}: {e}")


def test_target_rate_is_honored() -> None:
    """The threshold follows the caller's target, not a baked-in 16 kHz.

    audio.py passes its TARGET_SR, so a pipeline retuned to a different frame rate
    keeps a correct warning instead of one pinned to the old rate.
    """
    risk = assess_input_device("Some Mic", 16000, target_sr=48000)
    check(
        "16 kHz is narrowband against a 48 kHz target",
        risk is not None and risk.kind == "narrowband",
        f"kind={risk.kind if risk else None}",
    )
    check(
        "default target is the pipeline's 16 kHz",
        capture_quality.DEFAULT_TARGET_SR == 16000,
        f"DEFAULT_TARGET_SR={capture_quality.DEFAULT_TARGET_SR}",
    )


class _FakeMic:
    """audio.Microphone stand-in whose stream has already ended, so run_loop
    reaches the startup print and then breaks immediately."""

    device_label = "fake-mic"

    def __init__(self, capture_risk):
        self.capture_risk = capture_risk

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


def _run_loop_output(capture_risk) -> str:
    """Drive pipeline.run_loop with a fake mic carrying `capture_risk`, and return
    what it printed. run_loop imports `audio` lazily, so a fake module stands in
    and no device, PortAudio, or model is touched."""
    saved = {
        "load_config": pipeline.load_config,
        "warm_models": pipeline.warm_models,
        "run_turn": pipeline.run_turn,
    }
    saved_audio = sys.modules.get("audio")

    fake_audio = types.ModuleType("audio")
    fake_audio.Microphone = lambda name=None: _FakeMic(capture_risk)
    fake_audio.play_wav = lambda path, name=None: None

    pipeline.load_config = lambda: {"wake_word": "computah", "wake_chime": False}
    pipeline.warm_models = lambda cfg=None, wake_word=None: {}
    pipeline.run_turn = lambda frames, **kw: None
    sys.modules["audio"] = fake_audio
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            pipeline.run_loop()
    finally:
        pipeline.load_config = saved["load_config"]
        pipeline.warm_models = saved["warm_models"]
        pipeline.run_turn = saved["run_turn"]
        if saved_audio is None:
            sys.modules.pop("audio", None)
        else:
            sys.modules["audio"] = saved_audio
    return buf.getvalue()


def test_startup_warns_about_a_bad_device() -> None:
    """The verdict has to reach the operator, not just exist.

    This is the acceptance criterion from issue #34: warn at `--listen` startup.
    A classifier that is right and silent leaves the original failure intact --
    the loop looks healthy and only the transcript is wrong.
    """
    risk = CaptureRisk("hands-free-profile", "narrowband and frame-dropped, use USB")
    out = _run_loop_output(risk)
    check(
        "run_loop prints the warning at startup",
        risk.message in out,
        "message found" if risk.message in out else f"message absent from {out!r}",
    )
    check(
        "the warning is labelled so it reads as a problem",
        "WARNING" in out,
        "labelled WARNING" if "WARNING" in out else "no WARNING label",
    )
    check(
        "warned once, not per frame",
        out.count(risk.message) == 1,
        f"occurrences={out.count(risk.message)}",
    )
    check(
        "the loop still starts on a flagged device",
        "listening" in out,
        "reached listening" if "listening" in out else f"never listened: {out!r}",
    )


def test_startup_is_quiet_on_a_good_device() -> None:
    """No WARNING line when the device is fine, so the line stays meaningful."""
    out = _run_loop_output(None)
    check(
        "no warning for a clean device",
        "WARNING" not in out,
        "no WARNING printed" if "WARNING" not in out else f"unexpected: {out!r}",
    )


def main() -> int:
    print("capture-device suitability checks (issue #34)")
    test_flags_bluetooth_hands_free_by_name()
    test_hands_free_wins_over_rate()
    test_flags_narrowband_rate()
    test_good_devices_are_silent()
    test_unknown_rate_is_not_a_fault()
    test_survives_a_missing_name()
    test_target_rate_is_honored()
    test_startup_warns_about_a_bad_device()
    test_startup_is_quiet_on_a_good_device()
    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
