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
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000
AUDIO_EXTS = {".wav", ".flac", ".ogg", ".m4a", ".mp3", ".aac", ".mp4", ".wma"}
# prep writes each clip as "<stem>_NNN.wav" (see _write_unique; NNN is >=3 digits,
# since :03d is a minimum width). --clean pairs this shape with the current run's
# stems so it only removes a re-recorded take's own leftover clips, never a source
# take, an unrelated take's clips, or hand-added audio a user left in the output dir.
_CLIP_NAME = re.compile(r"(.+)_\d{3,}\.wav$")


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
            [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-ac",
                "1",
                "-ar",
                str(TARGET_SR),
                tmp.name,
            ],
            capture_output=True,
            check=True,
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


def _write_unique(
    out_dir: Path, stem: str, idx: int, audio: np.ndarray, seen: set[int]
) -> Path:
    """Write a clip, refusing to clobber one this same run already wrote.

    Rerunning the script into a populated dir is a supported refresh, so a
    leftover from a *prior* run is fine to overwrite. What must never happen is
    two inputs in *one* run mapping to the same file and the second silently
    replacing the first. `seen` holds the inode of every clip written this run;
    the filesystem resolves a case-insensitive collision (a/Computah.wav and
    b/computah.wav) to that same inode, so this catches it where a path-string
    compare would not, and stays quiet on a case-sensitive FS where the two are
    genuinely different files.
    """
    path = out_dir / f"{stem}_{idx:03d}.wav"
    if path.exists() and path.stat().st_ino in seen:
        raise FileExistsError(
            f"two inputs map to {path.name} in one run "
            f"(same basename, or a case-insensitive filesystem collision)"
        )
    _write_clip(out_dir, stem, idx, audio)
    seen.add(path.stat().st_ino)
    return path


# --------------------------------------------------------------------------- #
# Silence-based segmentation
# --------------------------------------------------------------------------- #
def _frame_rms(
    audio: np.ndarray, frame: int, hop: int
) -> tuple[np.ndarray, np.ndarray]:
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
    frame = int(0.02 * TARGET_SR)  # 20 ms
    hop = int(0.01 * TARGET_SR)  # 10 ms
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
            out.extend(
                sorted(p for p in path.iterdir() if p.suffix.lower() in AUDIO_EXTS)
            )
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


def _unique_stems(files: list[Path]) -> list[str]:
    """A distinct output stem per input file, in input order.

    Clip names are derived from the input stem, so two inputs that share a
    basename (a/computah.wav and b/computah.wav) would write the same clip
    names and the second file would silently overwrite the first. A stem that
    is already unique across the inputs is kept as-is. Within a colliding group
    one file keeps the bare stem and the rest get the lowest numeric suffix not
    otherwise taken. Every input stem is reserved up front so a disambiguated
    "computah-1" can never land on a real "computah-1" input -- including one
    that is itself part of another colliding group.

    One member of the group keeps the bare stem so that refreshing a samples dir
    stays a refresh: adding a second computah.wav renames only the new file and
    leaves the original's clips to be overwritten in place. Suffixing every
    member instead would orphan the previous run's computah_###.wav, and
    eval_wake_threshold reads every clip in the directory, so those orphans would
    quietly train and score as duplicates.

    Which member keeps the bare stem is decided by resolved path, not by input
    order, so the mapping depends only on WHICH files are present. Passing the
    same two files in the other order must not move the bare stem to the other
    file -- that would strand the first run's clips just as surely.

    A clip's final name is "<stem>-N_<idx>.wav" when disambiguated (the "-N"
    from here, the "_<idx>" segment index from _write_clip) or "<stem>_<idx>.wav"
    when the stem was already unique.
    """
    counts = Counter(f.stem for f in files)
    used = {f.stem for f in files}
    assigned: dict[int, str] = {}
    groups: dict[str, list[int]] = {}
    for i, f in enumerate(files):
        if counts[f.stem] == 1:
            assigned[i] = f.stem
        else:
            groups.setdefault(f.stem, []).append(i)

    # Sorted group order, and resolved-path order within a group, so every
    # assignment below is a function of the input set alone.
    for stem, idxs in sorted(groups.items()):
        ranked = sorted(idxs, key=lambda j: str(files[j].resolve()))
        for rank, i in enumerate(ranked):
            if rank == 0:
                assigned[i] = stem
                continue
            n = 1
            while f"{stem}-{n}" in used:
                n += 1
            name = f"{stem}-{n}"
            used.add(name)
            assigned[i] = name
    return [assigned[i] for i in range(len(files))]


def _stale_clips(out_dir: Path, written: set[Path]) -> list[Path]:
    """Audio files in `out_dir` this run did not write, sorted; empty if absent.

    Mirrors `eval_wake_threshold._clips()`, precondition included: a directory
    that does not exist holds nothing. The precondition lives here, with the
    read, because the earlier inline copy dropped it and a run that writes no
    clips never creates out_dir (only `_write_clip` does), so the scan died on
    a missing directory.

    The mirror is close but not exact: this module's `AUDIO_EXTS` is the wider
    set, so the warning can name a file eval would skip. #82 tracks giving both
    modules one definition instead of two copies.
    """
    if not out_dir.is_dir():
        return []
    return sorted(
        p
        for p in out_dir.iterdir()
        if p.suffix.lower() in AUDIO_EXTS and p not in written
    )


def _summarize(paths: list[Path], limit: int = 3) -> str:
    """A short "a, b, c, +N more" list of clip names for a one-line status."""
    shown = ", ".join(p.name for p in paths[:limit])
    if len(paths) > limit:
        shown += f", +{len(paths) - limit} more"
    return shown


def _clip_stem(name: str) -> str | None:
    """The stem of a "<stem>_NNN.wav" clip name, or None if it isn't one."""
    m = _CLIP_NAME.fullmatch(name)
    return m.group(1) if m else None


def process(
    inputs: list[Path],
    out_dir: Path,
    label: str,
    min_gap_s: float,
    min_dur_s: float,
    max_dur_s: float,
    clean: bool = False,
) -> int:
    """Process every input file; return the total clip count written.

    With ``clean`` set, a re-recorded take's leftover clips (a prior run's
    now-unused higher-numbered clips for a stem this run wrote) are removed after
    this run writes, so its own count matches what is on disk. It stays narrow on
    purpose: an unrelated take's clips, a source recording, or hand-added audio is
    named but never deleted, so a stray ``--output`` never silently wipes files.
    """
    # Checked before any audio is decoded, so an occupied --output is named once
    # rather than surfacing as whichever exception the run happens to reach first:
    # mkdir() raises FileExistsError once a clip is ready to write, and a run that
    # writes nothing gets as far as the stale scan below.
    if out_dir.exists() and not out_dir.is_dir():
        print(
            f"error: --output {out_dir} exists and is not a directory; "
            f"point --output at a directory, or move that file",
            file=sys.stderr,
        )
        return 0

    files = _inputs(inputs)
    if not files:
        joined = ", ".join(str(p) for p in inputs)
        print(f"no audio files found at {joined}", file=sys.stderr)
        return 0

    stems = _unique_stems(files)
    written: set[Path] = set()
    seen_inodes: set[int] = set()
    all_durs: list[float] = []
    for f, stem in zip(files, stems):
        audio = load_mono_16k(f)
        dur = len(audio) / TARGET_SR
        if label == "background":
            # Continuous negative audio: keep it whole, just normalized.
            written.add(_write_unique(out_dir, stem, 0, audio, seen_inodes))
            print(f"  {f.name}: {dur:.1f}s background -> 1 file")
            continue

        clips, stats = segment_on_silence(audio, min_gap_s, min_dur_s, max_dur_s)
        for i, clip in enumerate(clips):
            written.add(_write_unique(out_dir, stem, i, clip, seen_inodes))
            all_durs.append(len(clip) / TARGET_SR)
        note = ""
        if stats["too_short"] or stats["too_long"]:
            note = f" (dropped {stats['too_short']} short, {stats['too_long']} long)"
        print(f"  {f.name}: {dur:.1f}s -> {stats['kept']} clips{note}")

    # Count files actually on disk, not segments requested, so the report is
    # honest even if two inputs ever resolve to the same clip name.
    total = len(written)
    print(f"\nwrote {total} {label} clip(s) to {out_dir}")

    # Files this run did not write are left from an earlier run over a different
    # input set. Training and eval read every clip in the directory, so they
    # would score as extra data nobody asked for. --clean removes the leftover
    # clips of a take this run just re-recorded (same stem, now-unused index), so
    # the reported count matches what is on disk. It is deliberately narrow: a
    # clip whose stem this run did not write -- an unrelated take, a source
    # recording, hand-added audio -- is named but never auto-deleted, because it
    # is indistinguishable from data the user curated on purpose.
    run_stems = set(stems)
    stale = _stale_clips(out_dir, written)
    if stale:
        removable = (
            [p for p in stale if _clip_stem(p.name) in run_stems] if clean else []
        )
        for p in removable:
            p.unlink()
        if removable:
            print(
                f"removed {len(removable)} clip(s) left from an earlier run "
                f"({_summarize(removable)})"
            )
        lingering = [p for p in stale if p not in removable]
        if lingering:
            fix = (
                "these are not clips this run re-recorded, so --clean leaves them; "
                "remove them by hand"
                if clean
                else "rerun with --clean to remove them, or clear the directory by hand"
            )
            print(
                f"warning: {len(lingering)} file(s) in {out_dir} are left from an "
                f"earlier run ({_summarize(lingering)}). Training and eval read "
                f"every clip in this directory, so {fix}.",
                file=sys.stderr,
            )
    if all_durs:
        a = np.array(all_durs)
        print(
            f"clip duration s: min {a.min():.2f}, mean {a.mean():.2f}, "
            f"max {a.max():.2f}"
        )
        if a.mean() > 1.5:
            print("note: clips run long for a single word — check the silence gaps")
    return total


def main() -> int:
    p = argparse.ArgumentParser(description="prep wake-word recordings for training")
    p.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="audio file(s), a shell glob, or a folder of them",
    )
    p.add_argument("--output", required=True, help="output directory for clips")
    p.add_argument(
        "--label",
        choices=("positive", "negative", "background"),
        default="positive",
        help="how to treat the input",
    )
    p.add_argument(
        "--min-gap",
        type=float,
        default=0.3,
        help="silence (s) that separates utterances",
    )
    p.add_argument(
        "--min-dur", type=float, default=0.2, help="drop segments shorter than this (s)"
    )
    p.add_argument(
        "--max-dur", type=float, default=3.0, help="drop segments longer than this (s)"
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="remove a re-recorded take's leftover clips in --output (a prior run's "
        "now-unused higher-numbered clips for a take in this run); leaves unrelated "
        "or hand-added audio in place",
    )
    args = p.parse_args()

    total = process(
        [Path(p) for p in args.input],
        Path(args.output),
        args.label,
        args.min_gap,
        args.min_dur,
        args.max_dur,
        clean=args.clean,
    )
    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
