#!/usr/bin/env python3
"""prep_wake_samples.py segmentation and normalization (no models, no recordings).

Synthesizes tone bursts separated by known silence gaps so segment counts and clip
durations are exact and assertable, then checks the load path resamples to 16 kHz
mono and that background audio is kept whole.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

import prep_wake_samples as prep

results: list[tuple[bool, str]] = []


def check(ok: bool, detail: str) -> None:
    results.append((ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {detail}")


def make_bursts(
    n: int, burst_s: float, gap_s: float, sr: int, freq: float = 440.0
) -> np.ndarray:
    """n tone bursts of burst_s separated by gap_s of near-silence."""
    burst_len = int(burst_s * sr)
    gap_len = int(gap_s * sr)
    t = np.arange(burst_len) / sr
    tone = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    # Faint noise floor so the silence is realistic, not a perfect zero.
    rng = np.random.default_rng(0)
    gap = (rng.standard_normal(gap_len) * 1e-4).astype(np.float32)
    parts = [gap]
    for _ in range(n):
        parts.append(tone)
        parts.append(gap)
    return np.concatenate(parts)


def test_segment_count() -> None:
    audio = make_bursts(n=4, burst_s=0.6, gap_s=1.0, sr=prep.TARGET_SR)
    clips, stats = prep.segment_on_silence(audio)
    check(stats["kept"] == 4, f"4 bursts -> 4 clips (got {stats['kept']})")
    durs = [len(c) / prep.TARGET_SR for c in clips]
    ok = all(0.5 < d < 1.1 for d in durs)
    check(ok, f"each clip is ~0.6s plus padding: {[round(d, 2) for d in durs]}")


def test_process_positive(d: Path) -> None:
    audio = make_bursts(n=3, burst_s=0.5, gap_s=0.8, sr=prep.TARGET_SR)
    src = d / "computah_normal.wav"
    sf.write(src, audio, prep.TARGET_SR, subtype="PCM_16")
    out = d / "positive"
    total = prep.process([src], out, "positive", 0.3, 0.2, 3.0)
    files = sorted(out.glob("*.wav"))
    check(
        total == 3 and len(files) == 3,
        f"process wrote 3 positive clips (total={total}, files={len(files)})",
    )
    if files:
        info = sf.info(files[0])
        check(
            info.samplerate == 16000 and info.channels == 1,
            f"clips are 16 kHz mono ({info.samplerate} Hz, {info.channels} ch)",
        )


def test_resample_to_16k_mono(d: Path) -> None:
    sr = 48000
    audio = make_bursts(n=2, burst_s=0.5, gap_s=0.8, sr=sr)
    stereo = np.stack([audio, audio], axis=1)  # 2-channel at 48 kHz
    src = d / "stereo48k.wav"
    sf.write(src, stereo, sr, subtype="PCM_16")
    loaded = prep.load_mono_16k(src)
    expected = int(round(len(audio) * 16000 / sr))
    check(loaded.ndim == 1, "loaded audio is mono")
    check(
        abs(len(loaded) - expected) <= 2,
        f"resampled length matches 16 kHz (got {len(loaded)}, want ~{expected})",
    )


def test_background_kept_whole(d: Path) -> None:
    audio = make_bursts(n=5, burst_s=0.4, gap_s=0.5, sr=prep.TARGET_SR)
    src = d / "background.wav"
    sf.write(src, audio, prep.TARGET_SR, subtype="PCM_16")
    out = d / "background"
    total = prep.process([src], out, "background", 0.3, 0.2, 3.0)
    files = sorted(out.glob("*.wav"))
    check(
        total == 1 and len(files) == 1,
        f"background is one normalized file, not segmented (files={len(files)})",
    )


def test_explicit_files_only(d: Path) -> None:
    """Listing files processes only those; a folder still globs everything.

    Guards the contract that one --label maps to one class: pointing at the
    wake-word files must not pull in a sibling negatives.wav, while passing the
    folder is still allowed when it holds a single class.
    """
    folder = d / "mixed"
    folder.mkdir()
    pos1 = folder / "computah_a.wav"
    pos2 = folder / "computah_b.wav"
    neg = folder / "negatives.wav"
    sf.write(
        pos1, make_bursts(2, 0.5, 0.8, prep.TARGET_SR), prep.TARGET_SR, subtype="PCM_16"
    )
    sf.write(
        pos2, make_bursts(3, 0.5, 0.8, prep.TARGET_SR), prep.TARGET_SR, subtype="PCM_16"
    )
    sf.write(
        neg, make_bursts(4, 0.5, 0.8, prep.TARGET_SR), prep.TARGET_SR, subtype="PCM_16"
    )

    out = d / "pos_only"
    total = prep.process([pos1, pos2], out, "positive", 0.3, 0.2, 3.0)
    files = sorted(out.glob("*.wav"))
    check(total == 5, f"only the two listed files processed (total={total}, want 5)")
    check(
        not any("negatives" in f.name for f in files),
        "the sibling negatives.wav was not pulled into positives",
    )

    out_all = d / "all"
    total_all = prep.process([folder], out_all, "positive", 0.3, 0.2, 3.0)
    check(total_all == 9, f"a folder still globs all files (total={total_all}, want 9)")


def test_empty_input_no_crash(d: Path) -> None:
    """An input with no audio reports cleanly and returns 0, not raise."""
    empty = d / "nothing"
    empty.mkdir()
    total = prep.process([empty], d / "out_empty", "positive", 0.3, 0.2, 3.0)
    check(total == 0, f"empty input returns 0 without crashing (got {total})")


def main() -> int:
    test_segment_count()
    with tempfile.TemporaryDirectory(prefix="prep-wake-") as tmp:
        d = Path(tmp)
        test_process_positive(d)
        test_resample_to_16k_mono(d)
        test_background_kept_whole(d)
        test_explicit_files_only(d)
        test_empty_input_no_crash(d)
    n_pass = sum(1 for ok, _ in results if ok)
    print(f"=== {n_pass}/{len(results)} checks passed ===")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
