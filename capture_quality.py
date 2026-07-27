#!/usr/bin/env python3
"""Judge whether a capture device can carry continuous speech (issue #34).

Wake detection and transcription do not fail together. A Bluetooth hands-free
(HFP) mic on Windows still fires the wake word reliably -- a live "computah ..."
scored 0.998 -- while faster-whisper receives narrowband, recompressed,
frame-dropped audio and returns garbled text. The wake model matches a short
fixed pattern that survives the degradation; a full sentence does not. The loop
therefore looks healthy right up to the transcript, which is why this warns up
front. README's "Choosing a microphone" is the operator-facing version.

Hardware-free by design. audio.py is the one OS-specific seam (it imports
sounddevice, which loads native PortAudio), but judging a device's advertised
properties is plain logic, so it lives here and is testable on a box with no
PortAudio and no microphone.
"""

from __future__ import annotations

from typing import NamedTuple

# The pipeline consumes 16 kHz frames (audio.TARGET_SR). A device whose own
# native rate is below that cannot supply them: something upstream is
# upsampling, and the detail is already gone. Callers pass their own target so
# this module stays free of audio.py and its native dependency.
DEFAULT_TARGET_SR = 16000

# Substrings that identify a Bluetooth hands-free endpoint by name. Windows
# exposes a BT mic as a separate hands-free capture endpoint (the "Hands-Free AG
# Audio" family), so the profile shows up in the device name before any audio is
# read.
#
# Excludes the bare word "headset" on purpose: a USB headset is a good STT
# capture device, and matching it would warn about working hardware.
#
# Platform asymmetry worth knowing, because production runs on the Pi: these are
# Windows names. A bluez/PipeWire HFP source on Linux typically carries no
# hands-free substring, so this check never fires there and the native-rate check
# below is the only guard -- and that one misses wideband mSBC at 16 kHz. On
# Linux, treat a clean result as "nothing detected", not "device is fine".
_HFP_NAME_MARKERS = ("hands-free", "hands free", "handsfree", "hfp")


class CaptureRisk(NamedTuple):
    """A reason a device is unsuitable for continuous-speech capture.

    `kind` is the stable machine-readable tag ("hands-free-profile",
    "narrowband"); `message` is the human line a CLI prints. Callers key logic
    off `kind` so the wording can change without breaking them.
    """

    kind: str
    message: str


def assess_input_device(
    name: str | None,
    native_sr: float | int | None,
    target_sr: int = DEFAULT_TARGET_SR,
) -> CaptureRisk | None:
    """Return a CaptureRisk if this input device will garble continuous speech.

    The name check runs first, and wins when both signals fire, because the
    profile is the thing to act on and the rate is its symptom.

    Returns None when nothing is detectably wrong, which is weaker than "this
    device is good". Wideband HFP negotiates 16 kHz, so it clears the rate check,
    and an OS that reports it without a hands-free marker leaves nothing to match
    on. Catching that needs the audio itself (a spectral check for an empty top
    octave), which costs a capture at startup.

    An unknown or unparseable `native_sr` is treated as no information rather
    than a fault, so a device that advertises no rate is not warned about.

    `native_sr` is what the device advertises, not a measured capability, and how
    much that is worth depends on the host API. Under WASAPI it is the shared-mode
    mix format, which is the rate capture actually happens at, so a low value
    there is the fault itself. Under ALSA or CoreAudio it can be a default the
    device is able to exceed, which makes the rate branch advisory rather than
    conclusive -- hence the hedged wording, and why this returns a risk to report
    rather than a verdict to act on.

    Opening a stream at `target_sr` is not the counter-evidence it looks like:
    WASAPI `auto_convert` grants that request by resampling, which is the whole
    reason this consults the advertised rate instead of `Microphone._open_sr`.
    Distinguishing an advertised default from a hard limit needs the audio itself,
    which costs a capture at startup.
    """
    lowered = (name or "").lower()
    if any(marker in lowered for marker in _HFP_NAME_MARKERS):
        return CaptureRisk(
            "hands-free-profile",
            f"{name!r} is a Bluetooth hands-free (HFP) endpoint: narrowband and "
            "frame-dropped, so transcription comes back garbled even though wake "
            "detection keeps firing. Use this mic over USB, or another wired mic.",
        )

    try:
        rate = float(native_sr) if native_sr is not None else 0.0
    except (TypeError, ValueError):
        rate = 0.0
    if 0.0 < rate < target_sr:
        return CaptureRisk(
            "narrowband",
            f"{name!r} advertises {int(rate)} Hz, below the {target_sr} Hz the "
            "pipeline needs, so transcription is likely to come back garbled. On "
            "Windows that figure is the shared-mode capture format and the limit "
            "is real; on other hosts it can be a default the device exceeds. If "
            "this is a Bluetooth mic, use it over USB; otherwise check the rate "
            "in the OS sound settings.",
        )

    return None
