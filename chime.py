#!/usr/bin/env python3
"""The wake-acknowledgment chime: a short cue played the instant the wake word
fires, before the request is captured, to tell the user the mic is now listening
(issue #41).

Pure DSP -- numpy in, int16 PCM out -- with no audio backend, on purpose. The two
live loops play the cue through different outputs (sounddevice on the desktop loop,
ALSA `aplay` on the Pi, which has no sounddevice), so the generator must not import
either; it just produces the samples and a WAV both loops can play. That is also why
this is its own module rather than living in audio.py, which imports sounddevice at
module top and so cannot be imported on the Pi driver.

The cue is two ascending tones with raised-cosine edge fades. Rising pitch is the
near-universal "ready/listening" signal; the fades ramp each tone through zero at its
edges, so there is no step discontinuity and no click. The design is a set of
constants, so retuning the sound is a one-line change (a two-way door) -- swap the
notes, durations, or amplitude below.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import soundfile as sf

SR = 16000  # 16 kHz mono int16, matching the pipeline's audio format.

# Cue design. Each note is (frequency Hz, duration ms, amplitude 0..1). The notes
# rise in pitch. Amplitude is modest so the cue is a soft prompt, not a jolt.
_NOTES: tuple[tuple[float, float, float], ...] = (
    (880.0, 90.0, 0.25),    # A5
    (1318.0, 110.0, 0.25),  # E6 -- a rising perfect fifth, the classic "ready" cue
)
_FADE_MS = 10.0  # raised-cosine fade in/out per tone; long enough to kill clicks

# A stable cache path in the system temp dir, so both live loops (and repeated turns)
# reuse one generated file instead of re-rendering it every wake.
_CACHE_WAV = os.path.join(tempfile.gettempdir(), "computah_wake_cue.wav")


def _tone(freq: float, ms: float, amp: float, fade_ms: float = _FADE_MS) -> np.ndarray:
    """One sine tone with raised-cosine fade in/out, as float in [-1, 1]."""
    n = int(SR * ms / 1000.0)
    t = np.arange(n) / SR
    wave = amp * np.sin(2 * np.pi * freq * t)
    f = int(SR * fade_ms / 1000.0)
    if f > 0 and 2 * f <= n:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, f)))  # 0 -> 1
        wave[:f] *= ramp
        wave[-f:] *= ramp[::-1]
    return wave


def wake_cue_pcm() -> np.ndarray:
    """The cue as 16 kHz mono int16 PCM: the rising tones, concatenated."""
    sig = np.concatenate([_tone(freq, ms, amp) for freq, ms, amp in _NOTES])
    return (np.clip(sig, -1.0, 1.0) * 32767.0).astype(np.int16)


def wake_cue_wav(path: str | None = None) -> str:
    """Path to a WAV of the cue, rendering it once if absent. With no path, uses a
    cached file in the temp dir so both live loops and repeated turns reuse it.

    Returns the path. Idempotent: an existing file is left untouched (so its mtime is
    stable), since the cue is deterministic and never needs rewriting."""
    path = path or _CACHE_WAV
    if not os.path.exists(path):
        sf.write(path, wake_cue_pcm(), SR, subtype="PCM_16")
    return path


if __name__ == "__main__":
    out = wake_cue_wav()
    pcm = wake_cue_pcm()
    print(f"wake cue: {pcm.size} samples ({pcm.size / SR * 1000:.0f} ms) -> {out}")
