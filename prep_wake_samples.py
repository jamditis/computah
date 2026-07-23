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
import json
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


MANIFEST_NAME = ".prep-manifest.json"


def _read_manifest(out_dir: Path) -> tuple[str | None, set[str]]:
    """The label and clip names a previous prep run recorded here.

    Fails soft on every read problem -- absent, unreadable, not JSON, wrong
    shape -- returning ``(None, set())``. An empty record means "no record of
    what prep made," which narrows `--clean` back to the stem rule below.
    Failing loud would be worse in both directions: a corrupt manifest would
    abort a run that has real work to do, and a manifest trusted despite a
    partial read could name files prep never wrote.
    """
    try:
        raw = json.loads((out_dir / MANIFEST_NAME).read_text())
    except (OSError, ValueError):
        return None, set()
    if not isinstance(raw, dict):
        return None, set()
    clips = raw.get("clips")
    if not isinstance(clips, list):
        return None, set()
    label = raw.get("label")
    return (
        label if isinstance(label, str) else None,
        {c for c in clips if isinstance(c, str)},
    )


def _write_manifest(
    out_dir: Path, label: str, names: set[str], prior_label: str | None
) -> None:
    """Record which files in `out_dir` prep created, for a later --clean run.

    Written after the clips are on disk, and a failure only warns: the clips are
    the output that matters, and losing the manifest costs precision on a future
    --clean, not correctness. A run that wrote nothing into a directory that does
    not exist has nothing to record and does not create one.

    A directory already recorded under a different label is left alone. Rewriting
    it would relabel the other dataset's files as this one's, and a second stray
    run into the same directory would then find a matching label and an
    overlapping stem and read them as its own orphans. The mismatch is worth
    saying out loud either way: it usually means ``--output`` points somewhere
    unintended.
    """
    if not out_dir.is_dir():
        return
    if prior_label is not None and prior_label != label:
        print(
            f"warning: {out_dir} was last written as {prior_label!r} clips, not "
            f"{label!r}; leaving its record alone. If --output is right, clear the "
            f"directory (or remove {MANIFEST_NAME}) to start its record over",
            file=sys.stderr,
        )
        return
    try:
        (out_dir / MANIFEST_NAME).write_text(
            json.dumps({"version": 1, "label": label, "clips": sorted(names)}, indent=2)
            + "\n"
        )
    except OSError as e:
        print(
            f"warning: could not write {out_dir / MANIFEST_NAME} ({e}); "
            f"a later --clean will fall back to matching re-recorded stems only",
            file=sys.stderr,
        )


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

    With ``clean`` set, prep's own leftover clips are removed after this run
    writes, so its count matches what is on disk. Two kinds qualify: a
    re-recorded take's now-unused higher-numbered clips (matched by stem), and
    clips a previous run recorded for a take that is not in this run's inputs at
    all (matched against ``MANIFEST_NAME``, and only when this run refreshes the
    recorded set -- same label, overlapping stems). Everything else is named but
    never deleted -- a source recording, hand-added audio, another dataset a
    stray ``--output`` landed in, and the prior clips of a take this run
    attempted but wrote nothing for.
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
    # Read before anything is written, so it describes the directory as this run
    # found it.
    prior_label, prior = _read_manifest(out_dir)
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
    #
    # Scope by the stems this run actually WROTE a clip for, read back from
    # `written`, not by the input list `stems`: an input that produced zero clips
    # this run (silence, a bad re-recording, thresholds that dropped every
    # segment) has no new clip to make its earlier good clips stale, so --clean
    # must not delete them. Keying off written paths leaves them to linger and
    # warn instead of wiping the only copy.
    run_stems = {
        stem for stem in (_clip_stem(p.name) for p in written) if stem is not None
    }
    # The manifest widens --clean to the orphans the stem rule cannot reach: a
    # clip prep recorded writing on an earlier run, whose stem is not in this
    # run's inputs at all. That covers a dropped input and a basename collision
    # whose disambiguated stem changed -- both leave clips no rerun will ever
    # overwrite, and the manifest is what proves prep made them rather than a
    # user curating the directory.
    #
    # But "prep made this file" is provenance, not permission. It does not say
    # the file belongs to the dataset this run is refreshing, and without that
    # second fact a mistyped --output -- positives re-run into the background
    # directory -- would delete every clip there, since none of those stems are
    # in the inputs. So the manifest rule only arms when this run looks like a
    # refresh of the recorded set: same label, and at least one stem in common.
    # A stray --output shares neither, and falls back to the stem rule, which
    # cannot reach outside what this run just wrote.
    #
    # A stem this run DID attempt but wrote nothing for is excluded on purpose,
    # manifest or not: that is the silent-re-recording case, where the prior
    # clips are the only copy. The manifest says prep created a file; it does
    # not say the file is expendable.
    attempted = set(stems)
    prior_stems = {stem for stem in (_clip_stem(n) for n in prior) if stem is not None}
    refreshes_prior = prior_label == label and bool(run_stems & prior_stems)
    stale = _stale_clips(out_dir, written)
    removable = (
        [
            p
            for p in stale
            if _clip_stem(p.name) in run_stems
            or (
                refreshes_prior
                and p.name in prior
                and _clip_stem(p.name) not in attempted
            )
        ]
        if clean
        else []
    )
    for p in removable:
        p.unlink()
    if removable:
        print(
            f"removed {len(removable)} clip(s) left from an earlier run "
            f"({_summarize(removable)})"
        )
    lingering = [p for p in stale if p not in removable]
    # Carry forward the prior run's record for files still on disk, so a
    # directory built up over several runs keeps one provenance list rather than
    # only remembering the most recent run.
    _write_manifest(
        out_dir,
        label,
        {p.name for p in written} | {p.name for p in lingering if p.name in prior},
        prior_label,
    )
    if lingering:
        # Why a file lingered decides what the user should do about it, and the
        # cases want opposite advice. A take this run re-recorded and got nothing
        # from kept its previous clips because they are the only copy -- telling
        # that user to delete them by hand would undo the guard and lose the
        # recording. Split the message rather than send one that is right for the
        # common case and harmful for the case prep went out of its way to
        # protect.
        empty_takes = [p for p in lingering if _clip_stem(p.name) in attempted]
        if empty_takes and clean:
            fix = (
                "this run re-recorded "
                f"{_summarize(empty_takes)} and got no clips from it, so --clean "
                "kept the earlier ones rather than leave you with none; check the "
                "recording before removing anything"
            )
        elif clean:
            fix = (
                "prep has no record of creating these, so --clean leaves them; "
                "remove them by hand if they are not wanted"
            )
        else:
            fix = "rerun with --clean to remove them, or clear the directory by hand"
        print(
            f"warning: {len(lingering)} file(s) in {out_dir} are left from an "
            f"earlier run ({_summarize(lingering)}). Training and eval read every "
            f"clip in this directory, so they count as data nobody asked for: "
            f"{fix}.",
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
        help="remove prep's own leftover clips in --output: a re-recorded take's "
        "now-unused higher-numbered clips, plus clips a previous run recorded for a "
        "take no longer in the inputs; leaves hand-added audio in place",
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
