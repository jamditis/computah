#!/usr/bin/env python3
"""Turn raw wake-word recordings into clips ready for verifier training.

Takes one or more recordings (any common format, any sample rate, mono or stereo),
resamples to 16 kHz mono (openWakeWord's rate), and either:

  - splits them on silence into individual utterances (labels positive/negative), or
  - normalizes them whole (label background, for continuous negative audio).

Each output clip is a 16 kHz mono int16 WAV. The script reports per-file segment
counts and a duration summary so a bad take (clipped words, no gaps, silence) is
obvious before training.

Recordings are personal data: write them under samples/ (gitignored), not into the
tree.

Usage:
  python prep_wake_samples.py --input <file-or-dir> --output samples/positive \
      --label positive
  python prep_wake_samples.py --input bg.wav --output samples/background \
      --label background
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000
AUDIO_EXTS = {".wav", ".flac", ".ogg", ".m4a", ".mp3", ".aac", ".mp4", ".wma"}


# --------------------------------------------------------------------------- #
# Loading + normalization
# --------------------------------------------------------------------------- #
def _ffmpeg_decode(path: Path) -> tuple[np.ndarray, int]:
    """Decode a non-libsndfile format to 16 kHz mono via ffmpeg."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"cannot read {path.name}: install ffmpeg, or export it to WAV first"
        )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-ac", "1", "-ar", str(TARGET_SR),
             tmp.name],
            capture_output=True, check=True,
        )
        audio, sr = sf.read(tmp.name, dtype="float32", always_2d=False)
    return audio, sr


def load_mono_16k(path: Path) -> np.ndarray:
    """Load any supported file as float32 mono at 16 kHz, in [-1, 1]."""
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        # libsndfile cannot read it (e.g. m4a/mp3) — fall back to ffmpeg.
        audio, sr = _ffmpeg_decode(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        g = gcd(TARGET_SR, int(sr))
        audio = resample_poly(audio, TARGET_SR // g, sr // g)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def _write_clip(out_dir: Path, stem: str, idx: int, audio: np.ndarray) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    path = out_dir / f"{stem}_{idx:03d}.wav"
    sf.write(path, pcm, TARGET_SR, subtype="PCM_16")
    return path


# --------------------------------------------------------------------------- #
# Silence-based segmentation
# --------------------------------------------------------------------------- #
def _frame_rms(audio: np.ndarray, frame: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """RMS per frame, plus the sample index each frame starts at.

    Uses a prefix sum of squares so the whole file is one vectorized pass.
    """
    if len(audio) < frame:
        return np.zeros(0), np.zeros(0, dtype=int)
    sq = np.concatenate(([0.0], np.cumsum(audio.astype(np.float64) ** 2)))
    starts = np.arange(0, len(audio) - frame + 1, hop)
    energy = (sq[starts + frame] - sq[starts]) / frame
    return np.sqrt(energy), starts


def segment_on_silence(
    audio: np.ndarray,
    min_gap_s: float = 0.3,
    min_dur_s: float = 0.2,
    max_dur_s: float = 3.0,
    pad_s: float = 0.1,
) -> tuple[list[np.ndarray], dict]:
    """Split audio into utterances separated by silence.

    Returns (clips, stats). A frame is "speech" when its RMS rises a set fraction
    of the way from the file's noise floor to its loud peak, so the threshold adapts
    to each recording's level. Regions closer together than min_gap_s are merged;
    regions outside [min_dur_s, max_dur_s] are dropped (reported as too short/long).
    """
    frame = int(0.02 * TARGET_SR)   # 20 ms
    hop = int(0.01 * TARGET_SR)     # 10 ms
    rms, starts = _frame_rms(audio, frame, hop)
    if rms.size == 0:
        return [], {"kept": 0, "too_short": 0, "too_long": 0}

    floor = float(np.percentile(rms, 20))
    peak = float(np.percentile(rms, 99))
    thresh = floor + 0.15 * (peak - floor)
    speech = rms > thresh

    # Group contiguous speech frames into (start_sample, end_sample) regions.
    regions: list[list[int]] = []
    in_speech = False
    for i, s in enumerate(speech):
        if s and not in_speech:
            in_speech = True
            seg_start = int(starts[i])
        elif not s and in_speech:
            in_speech = False
            regions.append([seg_start, int(starts[i]) + frame])
    if in_speech:
        regions.append([seg_start, int(starts[-1]) + frame])

    # Merge regions whose gap is below min_gap.
    min_gap = int(min_gap_s * TARGET_SR)
    merged: list[list[int]] = []
    for r in regions:
        if merged and r[0] - merged[-1][1] < min_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    pad = int(pad_s * TARGET_SR)
    min_len = int(min_dur_s * TARGET_SR)
    max_len = int(max_dur_s * TARGET_SR)
    clips: list[np.ndarray] = []
    too_short = too_long = 0
    for a, b in merged:
        length = b - a
        if length < min_len:
            too_short += 1
            continue
        if length > max_len:
            too_long += 1
            continue
        a = max(0, a - pad)
        b = min(len(audio), b + pad)
        clips.append(audio[a:b])

    return clips, {"kept": len(clips), "too_short": too_short, "too_long": too_long}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _inputs(paths: list[Path]) -> list[Path]:
    """Expand the given paths to a deduped file list.

    A directory contributes its audio files; an explicit file is trusted as given.
    Passing files explicitly (or via a shell glob) is what keeps one label to one
    class — pointing a whole mixed folder at --label positive would mislabel the
    negatives and background, so the wake-word files are listed on their own.
    """
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            out.extend(sorted(p for p in path.iterdir()
                              if p.suffix.lower() in AUDIO_EXTS))
        else:
            out.append(path)
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def process(inputs: list[Path], out_dir: Path, label: str,
            min_gap_s: float, min_dur_s: float, max_dur_s: float) -> int:
    """Process every input file; return the total clip count written."""
    files = _inputs(inputs)
    if not files:
        joined = ", ".join(str(p) for p in inputs)
        print(f"no audio files found at {joined}", file=sys.stderr)
        return 0

    total = 0
    all_durs: list[float] = []
    for f in files:
        audio = load_mono_16k(f)
        dur = len(audio) / TARGET_SR
        if label == "background":
            # Continuous negative audio: keep it whole, just normalized.
            _write_clip(out_dir, f.stem, 0, audio)
            total += 1
            print(f"  {f.name}: {dur:.1f}s background -> 1 file")
            continue

        clips, stats = segment_on_silence(audio, min_gap_s, min_dur_s, max_dur_s)
        for i, clip in enumerate(clips):
            _write_clip(out_dir, f.stem, i, clip)
            all_durs.append(len(clip) / TARGET_SR)
        total += stats["kept"]
        note = ""
        if stats["too_short"] or stats["too_long"]:
            note = f" (dropped {stats['too_short']} short, {stats['too_long']} long)"
        print(f"  {f.name}: {dur:.1f}s -> {stats['kept']} clips{note}")

    print(f"\nwrote {total} {label} clip(s) to {out_dir}")
    if all_durs:
        a = np.array(all_durs)
        print(f"clip duration s: min {a.min():.2f}, mean {a.mean():.2f}, "
              f"max {a.max():.2f}")
        if a.mean() > 1.5:
            print("note: clips run long for a single word — check the silence gaps")
    return total


def main() -> int:
    p = argparse.ArgumentParser(description="prep wake-word recordings for training")
    p.add_argument("--input", required=True, nargs="+",
                   help="audio file(s), a shell glob, or a folder of them")
    p.add_argument("--output", required=True, help="output directory for clips")
    p.add_argument("--label", choices=("positive", "negative", "background"),
                   default="positive", help="how to treat the input")
    p.add_argument("--min-gap", type=float, default=0.3,
                   help="silence (s) that separates utterances")
    p.add_argument("--min-dur", type=float, default=0.2,
                   help="drop segments shorter than this (s)")
    p.add_argument("--max-dur", type=float, default=3.0,
                   help="drop segments longer than this (s)")
    args = p.parse_args()

    total = process([Path(p) for p in args.input], Path(args.output), args.label,
                    args.min_gap, args.min_dur, args.max_dur)
    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
