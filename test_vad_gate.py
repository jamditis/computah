#!/usr/bin/env python3
"""Confirmed-speech capture gate: the silero VAD rejects non-speech the energy gate
cannot (issue #54).

capture_request's energy endpointing decides *when* a command ends, but energy alone
cannot tell a sustained noise (a long cough, a fan, room rumble) from speech, so a
>=240 ms noise after a bare wake could clear the energy onset and prepend the wake-word
pre-roll as a phantom command. _confirm_speech runs the bundled silero VAD over the
post-fire audio and keeps the capture only if real speech is present.

This test loads the real model (no mock -- the model-free plumbing is proven in
test_preroll.py) and shows the separation at the default threshold: Piper-synthesized
speech is confirmed; white noise, a pure tone, and silence are rejected. It prints each
peak probability so the threshold can be hardware-tuned from real numbers.

Run:  .venv/bin/python test_vad_gate.py   (loads the silero + Piper models)
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import pipeline

TEST_DIR = Path(__file__).resolve().parent / "test_audio"
TEST_DIR.mkdir(exist_ok=True)

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    results.append((PASS if ok else FAIL, name))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")
    return ok


def peak_prob(pcm: np.ndarray) -> float:
    """Peak per-480-chunk speech probability -- the value _confirm_speech thresholds.
    Recomputed here so the test can show the speech-vs-noise margin, not just a bool."""
    vad = pipeline._get_vad()
    vad.reset_states()
    frame = 480
    pad = (-len(pcm)) % frame
    if pad:
        pcm = np.concatenate([pcm, np.zeros(pad, dtype=np.int16)])
    peak = 0.0
    for i in range(0, len(pcm), frame):
        peak = max(peak, float(vad.predict(pcm[i : i + frame], frame_size=frame)))
    return peak


def synth_speech() -> np.ndarray:
    out = str(TEST_DIR / "vad_speech.wav")
    pipeline.speak("file an issue about the broken printer", out)
    return pipeline._load_pcm16(out)


def main() -> int:
    thr = pipeline.DEFAULTS["capture_vad_threshold"]
    print(f"=== _confirm_speech at the default threshold {thr} (real silero VAD) ===")

    speech = synth_speech()
    sp_peak = peak_prob(speech)
    check(
        "Piper speech is confirmed as speech",
        pipeline._confirm_speech(speech, thr),
        f"peak={sp_peak:.3f} >= {thr}",
    )

    # White noise, loud enough (RMS ~3.5k, well above _SILENCE_RMS) to clear the energy
    # gate and reach the VAD in the real pipeline. Seeded so the run is deterministic.
    rng = np.random.default_rng(0)
    noise = rng.integers(-6000, 6000, size=16000).astype(np.int16)
    n_peak = peak_prob(noise)
    check(
        "white noise is rejected (not speech)",
        not pipeline._confirm_speech(noise, thr),
        f"peak={n_peak:.3f} < {thr}",
    )

    # A pure tone is loud and periodic but has none of speech's formant structure.
    t = np.arange(16000)
    tone = (5000 * np.sin(2 * np.pi * 440 * t / 16000)).astype(np.int16)
    t_peak = peak_prob(tone)
    check(
        "a 440 Hz tone is rejected (not speech)",
        not pipeline._confirm_speech(tone, thr),
        f"peak={t_peak:.3f} < {thr}",
    )

    silence = np.zeros(16000, dtype=np.int16)
    s_peak = peak_prob(silence)
    check(
        "silence is rejected (not speech)",
        not pipeline._confirm_speech(silence, thr),
        f"peak={s_peak:.3f} < {thr}",
    )

    # The separation should be wide, not a coin-flip across the threshold: speech well
    # above, every non-speech case well below.
    worst_nonspeech = max(n_peak, t_peak, s_peak)
    check(
        "speech clears the loudest non-speech case by a wide margin",
        sp_peak - worst_nonspeech > 0.2,
        f"speech={sp_peak:.3f} vs worst non-speech={worst_nonspeech:.3f}",
    )

    n_pass = sum(1 for x in results if x[0] == PASS)
    n_total = len(results)
    print(f"\n=== {n_pass}/{n_total} checks passed ===")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
