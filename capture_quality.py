#!/usr/bin/env python3
"""Judge whether a capture device can carry continuous speech (issue #34).

Wake detection and transcription do not fail together. A Bluetooth hands-free
(HFP) mic on Windows still fires the wake word reliably -- a live "computah ..."
scored 0.998 -- while faster-whisper receives narrowband, recompressed,
frame-dropped audio and returns garbled text. The wake model matches a short
fixed pattern that survives the degradation; a full sentence does not. The loop
therefore looks healthy right up to the transcript, which is why this warns up
front. README's "Choosing a microphone" is the operator-facing version.

Two evidence sources can identify a problem:

- `assess_input_device` reads what the device advertises -- its name, host API,
  and default rate. Free, works before any audio is captured, and is what the
  `--listen` startup warning uses. It only sees what the OS chooses to report.
- `assess_capture_spectrum` reads the captured audio itself. It cannot run until
  something has recorded, so `--test-mic` (which already captures a few seconds)
  is its home. It catches the cases the advertised properties hide: a bluez HFP
  source on Linux exposes no hands-free name and PipeWire can report a healthy
  16 kHz while resampling 8 kHz audio underneath. Its 4 kHz cliff test cannot
  rule out wideband HFP codec damage, so a capture without that cliff remains
  inconclusive.

Hardware-free by design. audio.py is the one OS-specific seam (it imports
sounddevice, which loads native PortAudio), but neither judging advertised
properties nor analyzing an array of samples needs a device, so both live here
and are testable on a box with no PortAudio and no microphone.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, Sequence

import numpy as np

# The pipeline consumes 16 kHz frames (audio.TARGET_SR). Callers pass their own
# target so this module stays free of audio.py and its native dependency.
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
# hands-free substring, so this check never fires there, and PipeWire's resampling
# can hide the transport rate from the advertised-property check too. On Linux,
# treat a clean result here as "nothing detected", not "device is fine", and run
# `--test-mic`, where assess_capture_spectrum judges the audio instead of the
# advertisement.
_HFP_NAME_MARKERS = ("hands-free", "hands free", "handsfree", "hfp")

# PortAudio's `default_samplerate` is generally a preference, not a bandwidth
# limit. WASAPI shared mode is the exception used here: its default is the mix
# format through which capture runs. Do not grow this list without verifying the
# host API's field has that meaning.
_CAPTURE_FORMAT_HOST_MARKERS = ("wasapi",)


class CaptureRisk(NamedTuple):
    """A reason a device is unsuitable for continuous-speech capture.

    `kind` is the stable machine-readable tag ("hands-free-profile",
    "narrowband", "upsampled-narrowband"); `message` is the human line a CLI
    prints. Callers key logic off `kind` so the wording can change without
    breaking them.
    """

    kind: str
    message: str


class SpectrumAssessment(NamedTuple):
    """The result of inspecting captured audio for a narrowband cliff.

    `status` is explicit because `None` used to mean both "no cliff" and "there
    was nothing to analyze", and the CLI accidentally promoted both to success.
    A non-flagged spectrum stays inconclusive: this measurement cannot detect
    wideband HFP recompression or frame drops.
    """

    status: Literal["flagged", "inconclusive"]
    risk: CaptureRisk | None
    message: str


def assess_input_device(
    name: str | None,
    default_sr: float | int | None,
    target_sr: int = DEFAULT_TARGET_SR,
    host_api: str | None = None,
) -> CaptureRisk | None:
    """Return a CaptureRisk if this input device will garble continuous speech.

    The name check runs first, and wins when both signals fire, because the
    profile is the thing to act on and the rate is its symptom.

    Returns None when nothing is detectably wrong, which is weaker than "this
    device is good". Wideband HFP negotiates 16 kHz, so it clears the rate check,
    and an OS that reports it without a hands-free marker leaves nothing to match
    on. Catching that needs the audio itself, which is what
    assess_capture_spectrum does once something has recorded.

    An unknown or unparseable `default_sr` is treated as no information rather
    than a fault, so a device that advertises no rate is not warned about.

    `default_sr` is not a measured capability or maximum supported rate. The rate
    branch is therefore limited to host APIs where that field represents the
    active capture format. WASAPI shared mode has that meaning; ALSA and CoreAudio
    defaults do not, so a low value from either is ignored instead of warning
    about a stream that may open at `target_sr`.

    Opening a stream at `target_sr` is not the counter-evidence it looks like:
    WASAPI `auto_convert` grants that request by resampling, which is the whole
    reason the WASAPI check consults the shared-mode format instead of
    `Microphone._open_sr`.
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
        rate = float(default_sr) if default_sr is not None else 0.0
    except (TypeError, ValueError):
        rate = 0.0
    host_has_capture_format = any(
        marker in (host_api or "").lower() for marker in _CAPTURE_FORMAT_HOST_MARKERS
    )
    if host_has_capture_format and 0.0 < rate < target_sr:
        return CaptureRisk(
            "narrowband",
            f"{name!r} uses a {int(rate)} Hz WASAPI shared-mode capture format, "
            f"below the {target_sr} Hz the pipeline needs, so transcription is "
            "likely to come back garbled. If this is a Bluetooth mic, use it over "
            "USB; otherwise raise the rate in the Windows sound settings.",
        )

    return None


# Width of the two comparison bands straddling the half-Nyquist edge, in Hz. Wide
# enough to average over formant structure, narrow enough that both bands sit in
# comparable spectral territory.
_CLIFF_BAND_HZ = 1000

# Below this, the energy just above the edge has effectively vanished relative to
# the energy just below it, which is a resampler's signature rather than a voice.
#
# Measured on synthesized signals (the numbers the test pins): genuine full-band
# material runs from 0.34 (a very dull, heavily tilted voice) through 0.63
# (speech-like) to 1.01 (white noise), while 8 kHz-sourced audio runs from 0.000
# (an ideal resampler) up to 0.025 (a sloppy one with a -40 dB noise floor).
# This sits near the geometric middle of that gap: 3.4x under the genuine floor
# and 3.9x over the narrowband ceiling. Picking it by eye put it at 0.05, which is
# a hair under 2x over that ceiling -- close enough that the margin test caught it.
_CLIFF_RATIO_THRESHOLD = 0.10

# Shorter than this and the spectrum is too coarse to trust.
_MIN_ANALYSIS_SECONDS = 0.5


def _cliff_ratio(
    samples: Sequence[float] | np.ndarray,
    sample_rate: int,
    target_sr: int,
) -> float | None:
    """Energy just above the half-Nyquist edge over energy just below it.

    Separate from the verdict so the measurement can be tested on its own: what
    protects this check is the size of the gap between genuine and narrowband
    audio, and a test can only assert on that if it can read the number.

    None means there was nothing to measure, which is not the same as 0.0 (a
    real, total cliff).
    """
    audio = np.asarray(samples, dtype=np.float64).ravel()
    if audio.size < _MIN_ANALYSIS_SECONDS * sample_rate:
        return None

    audio = audio - audio.mean()
    if not np.any(audio):
        return None

    edge_hz = target_sr / 4
    if sample_rate / 2 <= edge_hz:
        # The capture cannot represent the band we would inspect, so there is
        # nothing to measure. The advertised-rate check already covers this.
        return None

    power = np.abs(np.fft.rfft(audio)) ** 2
    freqs = np.fft.rfftfreq(audio.size, 1.0 / sample_rate)

    def band_mean(low: float, high: float) -> float:
        selected = (freqs >= low) & (freqs < high)
        return float(power[selected].mean()) if selected.any() else 0.0

    below = band_mean(edge_hz - _CLIFF_BAND_HZ, edge_hz)
    if below <= 0:
        return None
    return band_mean(edge_hz, edge_hz + _CLIFF_BAND_HZ) / below


def assess_capture_spectrum(
    samples: Sequence[float] | np.ndarray,
    sample_rate: int,
    target_sr: int = DEFAULT_TARGET_SR,
) -> SpectrumAssessment:
    """Assess whether captured audio looks upsampled from a narrower band.

    This is the check that survives a lying device. `assess_input_device` can only
    read what the OS advertises, and on Linux a bluez/PipeWire HFP source reports
    neither a hands-free name nor a low rate -- PipeWire resamples to 16 kHz and
    reports 16 kHz, so both signals look clean while the audio underneath came
    from an 8 kHz transport. The evidence of that is in the audio: upsampling
    cannot invent content it never received, so the band above the original
    Nyquist is empty.

    It measures the *cliff* at half the target's Nyquist rather than the total
    energy above it, because speech has a steep spectral tilt and a raw high-band
    fraction mostly measures the tilt. On the test's own signals that metric does
    not merely separate the two classes badly, it puts them in the wrong order: a
    dull voice carries 0.00007% of its energy above 4 kHz while a sloppily
    resampled 8 kHz signal carries 0.005%, so the genuine case looks *more*
    narrowband than the narrowband one and no threshold exists to fit. Comparing
    the 1 kHz band just above the edge against the 1 kHz band just below
    normalizes the tilt away and puts the same cases an order of magnitude apart,
    in the right order.

    Returns an explicit inconclusive result when there is nothing to judge (too
    short, or digital silence) rather than guessing. A ratio above the threshold
    is also inconclusive: it rules out an 8 kHz upsampling cliff, but wideband HFP
    genuinely has energy above 4 kHz while its codec and frame drops can still
    garble continuous speech. Note that near-silence is safe from a false
    positive: room tone is broadband, so it produces a ratio above the threshold.

    Known limit: a resampler whose noise floor is high enough to fill the empty
    band (around -30 dB, loud enough to hear as hiss) reads as genuine. Real
    resamplers sit at -60 dB or better.

    Args:
        samples: mono PCM, any amplitude scale -- the measure is a ratio.
        sample_rate: the rate `samples` was captured at.
        target_sr: the rate the pipeline needs, which sets the edge to inspect.
    """
    edge_hz = target_sr / 4
    ratio = _cliff_ratio(samples, sample_rate, target_sr)
    if ratio is None:
        message = (
            "the capture did not contain enough varying audio for the spectral "
            "check, so microphone suitability is inconclusive"
        )
        return SpectrumAssessment("inconclusive", None, message)
    if ratio >= _CLIFF_RATIO_THRESHOLD:
        message = (
            f"no {int(edge_hz)} Hz upsampling cliff was detected, but that check "
            "cannot detect wideband HFP codec damage or frame drops, so microphone "
            "suitability is inconclusive"
        )
        return SpectrumAssessment("inconclusive", None, message)

    risk = CaptureRisk(
        "upsampled-narrowband",
        f"the captured audio carries almost nothing above {int(edge_hz)} Hz "
        f"(band ratio {ratio:.4f}), so it was upsampled from a narrower source "
        f"even though it arrived at {sample_rate} Hz. That is what a Bluetooth "
        "hands-free link does, and transcription comes back garbled. Use this "
        "mic over USB, or another wired mic.",
    )
    return SpectrumAssessment("flagged", risk, risk.message)
