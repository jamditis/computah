#!/usr/bin/env python3
"""Turn raw wake-word recordings into clips ready for verifier training.

Takes one or more recordings (any common format, any sample rate, mono or stereo),
resamples to 16 kHz mono (openWakeWord's rate), and either:

  - splits them on silence into individual utterances (labels positive/negative), or
  - normalizes them whole (label background, for continuous negative audio).

Each output clip is a 16 kHz mono int16 WAV. The script reports per-file segment
counts and a duration summary so a bad take (clipped words, no gaps, silence) is
obvious before training.

Successful runs also write `.prep-manifest.json` in `--output`; it records the
dataset label and exact source ownership used to guard later refreshes and cleanup.

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
import os
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
# since :03d is a minimum width). Without source ownership, --clean uses this
# shape plus the current run's stems as its narrow legacy fallback. With a v2
# manifest, exact recorded clip names authorize cleanup instead.
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


def _source_key(path: Path) -> str:
    """A stable local identity for one source recording."""
    return os.path.normcase(str(path.resolve()))


def _read_manifest(
    out_dir: Path,
) -> tuple[str | None, set[str], dict[str, set[str]] | None]:
    """The label, clips, and source ownership a previous prep run recorded here.

    An absent file returns ``(None, set(), None)`` and lets the caller use the
    warned legacy stem rule. Every present-but-unreadable, malformed, or
    unsupported record returns no source map; the caller sees that the file
    exists and refuses to decode or write until ownership is bootstrapped
    explicitly.
    """
    manifest_path = out_dir / MANIFEST_NAME

    def warn_manifest_problem(reason: object) -> None:
        print(
            f"warning: could not use {manifest_path} ({reason}); source ownership "
            "protection is unavailable, so this manifest cannot authorize a write",
            file=sys.stderr,
        )

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, set(), None
    except (OSError, ValueError) as e:
        warn_manifest_problem(e)
        return None, set(), None
    if not isinstance(raw, dict):
        warn_manifest_problem("expected a JSON object")
        return None, set(), None
    clips = raw.get("clips")
    if not isinstance(clips, list):
        warn_manifest_problem("expected a clips list")
        return None, set(), None
    label = raw.get("label")
    if not isinstance(label, str):
        warn_manifest_problem("expected a string label")
        return None, set(), None

    def safe_clip_name(name: object) -> bool:
        return (
            isinstance(name, str)
            and name == Path(name).name
            and "/" not in name
            and "\x00" not in name
            and _CLIP_NAME.fullmatch(name) is not None
        )

    if not all(safe_clip_name(name) for name in clips):
        warn_manifest_problem("clips must contain safe output filenames, not paths")
        return label, set(), None
    clip_names = set(clips)
    version = raw.get("version")
    if version != 2:
        warn_manifest_problem(f"unsupported manifest version {version!r}; expected 2")
        return label, clip_names, None
    sources_raw = raw.get("sources")
    if not isinstance(sources_raw, dict):
        warn_manifest_problem("expected a sources map")
        return label, clip_names, None

    sources: dict[str, set[str]] = {}
    claimed_clips: set[str] = set()
    for source, names in sources_raw.items():
        if not isinstance(source, str) or not isinstance(names, list):
            warn_manifest_problem(
                "expected every sources entry to map a path to a clip list"
            )
            return label, clip_names, None
        if not all(safe_clip_name(name) for name in names):
            warn_manifest_problem("sources must own safe output filenames, not paths")
            return label, clip_names, None
        source_clips = set(names)
        if claimed_clips & source_clips:
            warn_manifest_problem(
                "each recorded clip must have exactly one source owner"
            )
            return label, clip_names, None
        sources[source] = source_clips
        claimed_clips.update(source_clips)
    if claimed_clips != clip_names:
        warn_manifest_problem("sources do not own exactly the recorded clips")
        return label, clip_names, None
    return label, clip_names, sources


def _write_manifest(
    out_dir: Path, label: str, source_clips: dict[str, set[str]]
) -> None:
    """Record each source and the clips it owns, for a later --clean run.

    Written after the clips are on disk, and a failure only warns: the clips are
    the output that matters, and losing the manifest costs precision on a future
    --clean, not correctness. A run that wrote nothing into a directory that does
    not exist has nothing to record and does not create one.

    The caller verifies ownership and assembles the source map; this only writes.
    """
    if not out_dir.is_dir():
        return
    clips = set().union(*source_clips.values()) if source_clips else set()
    payload = (
        json.dumps(
            {
                "version": 2,
                "label": label,
                "clips": sorted(clips),
                "sources": {
                    source: sorted(names)
                    for source, names in sorted(source_clips.items())
                },
            },
            indent=2,
        )
        + "\n"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=out_dir,
            prefix=f".{MANIFEST_NAME}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            temporary = Path(tmp.name)
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(temporary, out_dir / MANIFEST_NAME)
    except OSError as e:
        manifest_path = out_dir / MANIFEST_NAME
        if manifest_path.exists():
            recovery = (
                "the previous manifest remains in force and clips written by this "
                "run are unrecorded. Fix the write problem and rerun the full input "
                f"set, or remove {manifest_path} to bootstrap ownership again"
            )
        else:
            recovery = (
                "no manifest was recorded, so clips written by this run have no "
                "source ownership protection. Fix the write problem and rerun the "
                "full input set"
            )
        print(
            f"warning: could not write {manifest_path} ({e}); {recovery}",
            file=sys.stderr,
        )
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _clip_stem(name: str) -> str | None:
    """The stem of a "<stem>_NNN.wav" clip name, or None if it isn't one."""
    m = _CLIP_NAME.fullmatch(name)
    return m.group(1) if m else None


def _manifest_aware_stems(
    files: list[Path],
    source_keys: list[str],
    prior_sources: dict[str, set[str]] | None,
) -> list[str]:
    """Assign output stems without taking names owned by another source.

    The first run uses the path-stable basename rule. Once a source manifest
    exists, a returning source keeps one of its recorded stems when possible,
    and every prior stem stays reserved from other sources. That reservation is
    needed before any clip is written: cleanup cannot save a silent source's
    only good clips after a newcomer has already overwritten their filenames.
    """
    generated = _unique_stems(files)
    if prior_sources is None:
        return generated

    source_stems: dict[str, set[str]] = {}
    owners_by_stem: dict[str, set[str]] = {}
    for source, names in prior_sources.items():
        stems = {stem for name in names if (stem := _clip_stem(name)) is not None}
        source_stems[source] = stems
        for stem in stems:
            owners_by_stem.setdefault(stem, set()).add(source)

    assigned: dict[int, str] = {}
    used: set[str] = set()

    # Anchor returning sources to a stem they already own. Prefer the stem the
    # current basename rule would choose, then a sole prior stem. A malformed
    # record in which two sources share a stem is never used as an anchor.
    for i in sorted(range(len(files)), key=lambda j: str(files[j].resolve())):
        source = source_keys[i]
        owned = source_stems.get(source, set())
        preferred: str | None = None
        if generated[i] in owned and owners_by_stem.get(generated[i]) == {source}:
            preferred = generated[i]
        elif len(owned) == 1:
            candidate = next(iter(owned))
            if owners_by_stem.get(candidate) == {source}:
                preferred = candidate
        if preferred is not None and preferred not in used:
            assigned[i] = preferred
            used.add(preferred)

    # Keep the ordinary mapping for inputs whose candidate has no recorded
    # owner. Anchors run first so a newcomer cannot claim a returning source's
    # prior stem merely because it appeared earlier in the invocation.
    for i in sorted(range(len(files)), key=lambda j: str(files[j].resolve())):
        if i in assigned:
            continue
        candidate = generated[i]
        if candidate not in owners_by_stem and candidate not in used:
            assigned[i] = candidate
            used.add(candidate)

    # Any remaining candidate collides with recorded ownership. Give it a fresh
    # deterministic suffix, avoiding both prior stems and raw basenames that a
    # different pending input may need.
    pending = [i for i in range(len(files)) if i not in assigned]
    unavailable = set(owners_by_stem) | used | {files[i].stem for i in pending}
    for i in sorted(pending, key=lambda j: (files[j].stem, str(files[j].resolve()))):
        n = 1
        while f"{files[i].stem}-{n}" in unavailable:
            n += 1
        assigned[i] = f"{files[i].stem}-{n}"
        unavailable.add(assigned[i])

    return [assigned[i] for i in range(len(files))]


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
    all (matched against ``MANIFEST_NAME``, and only when this run contains a
    source path recorded for that dataset). Everything else is named but never
    deleted -- a source recording, hand-added audio, another dataset a stray
    ``--output`` landed in, and the prior clips of a take this run attempted but
    wrote nothing for.
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

    # Refused before any audio is decoded, because by the time --clean is
    # evaluated the damage is already done: _write_unique would have overwritten
    # this directory's take_000.wav with the new dataset's. A label mismatch is
    # the signature of a mistyped --output, and there is no reading of it where
    # writing positives into a directory of negatives is what the user meant.
    manifest_present = (out_dir / MANIFEST_NAME).exists()
    prior_label, _prior_clips, prior_sources = _read_manifest(out_dir)
    if prior_label is not None and prior_label != label:
        print(
            f"error: --output {out_dir} holds {prior_label!r} clips, not {label!r}; "
            f"nothing was written. Point --output at this run's directory, or clear "
            f"that one (or remove its {MANIFEST_NAME}) to reuse it",
            file=sys.stderr,
        )
        return 0
    if manifest_present and prior_sources is None:
        print(
            f"error: --output {out_dir} has an unusable {MANIFEST_NAME}, so this "
            "run cannot prove its label or source ownership; nothing was written. "
            f"After reviewing the directory, remove {out_dir / MANIFEST_NAME} to "
            "bootstrap ownership from the full input set, or clear the directory "
            "first",
            file=sys.stderr,
        )
        return 0
    if (
        clean
        and prior_sources is None
        and out_dir.is_dir()
        and any(path.suffix.lower() in AUDIO_EXTS for path in out_dir.iterdir())
    ):
        print(
            f"warning: {out_dir} has no {MANIFEST_NAME}; this --clean will use "
            "legacy filename cleanup without source ownership protection and may "
            "remove hand-added <same-stem>_NNN.wav files",
            file=sys.stderr,
        )

    files = _inputs(inputs)
    if not files:
        joined = ", ".join(str(p) for p in inputs)
        print(f"no audio files found at {joined}", file=sys.stderr)
        return 0

    source_keys = [_source_key(f) for f in files]
    # This guards writes as well as --clean: _write_unique intentionally refreshes
    # prior filenames in place, so a different source with the same basename could
    # overwrite take_000.wav before cleanup runs. Add a new source alongside one
    # recorded source in the same invocation to prove which dataset owns the dir.
    if prior_sources is not None and not (set(source_keys) & prior_sources.keys()):
        recorded = sorted(prior_sources)
        preview = ", ".join(repr(source) for source in recorded[:2])
        if len(recorded) > 2:
            preview += f", +{len(recorded) - 2} more"
        print(
            f"error: --output {out_dir} belongs to a {label!r} dataset with "
            f"different source recordings; nothing was written. Recorded sources "
            f"include {preview}. Point --output at this run's directory, or clear "
            f"that one (or remove its {MANIFEST_NAME}) to reuse it",
            file=sys.stderr,
        )
        return 0
    if clean and prior_sources is not None:
        present_names = {path.name for path in _stale_clips(out_dir, set())}
        for source in sorted(prior_sources.keys() - set(source_keys)):
            count = len(prior_sources[source] & present_names)
            if count:
                print(
                    f"warning: source {source!r} is not in this run; --clean will "
                    f"remove its {count} prep-owned clip(s) from {out_dir}",
                    file=sys.stderr,
                )

    stems = _manifest_aware_stems(files, source_keys, prior_sources)
    written: set[Path] = set()
    written_by_source: dict[str, set[Path]] = {source: set() for source in source_keys}
    seen_inodes: set[int] = set()
    all_durs: list[float] = []
    for f, stem, source in zip(files, stems, source_keys):
        audio = load_mono_16k(f)
        dur = len(audio) / TARGET_SR
        if label == "background":
            # Continuous negative audio: keep it whole, just normalized.
            path = _write_unique(out_dir, stem, 0, audio, seen_inodes)
            written.add(path)
            written_by_source[source].add(path)
            print(f"  {f.name}: {dur:.1f}s background -> 1 file")
            continue

        clips, stats = segment_on_silence(audio, min_gap_s, min_dur_s, max_dur_s)
        for i, clip in enumerate(clips):
            path = _write_unique(out_dir, stem, i, clip, seen_inodes)
            written.add(path)
            written_by_source[source].add(path)
            all_durs.append(len(clip) / TARGET_SR)
        note = ""
        if stats["too_short"] or stats["too_long"]:
            note = f" (dropped {stats['too_short']} short, {stats['too_long']} long)"
        print(f"  {f.name}: {dur:.1f}s -> {stats['kept']} clips{note}")

    # Count files actually on disk, not segments requested, so the report is
    # honest even if two inputs ever resolve to the same clip name.
    total = len(written)
    print(f"\nwrote {total} {label} clip(s) to {out_dir}")

    # Files this run did not write are left from an earlier input set. Once a
    # manifest exists, cleanup is based on source ownership rather than inferred
    # from output stems: prior clips belonging to a source this run successfully
    # re-recorded, or to a source dropped from this run, are removable. Prior clips
    # belonging to a source this run attempted but got no clips from are kept as
    # the only good copy. Files absent from the source map are user-owned.
    #
    # Without a readable manifest, the older stem rule remains the narrow
    # fallback. It can clean a re-recorded take's higher-numbered leftovers but
    # cannot reach a dropped input safely.
    run_stems = {
        stem for stem in (_clip_stem(p.name) for p in written) if stem is not None
    }
    current_sources = set(source_keys)
    written_sources = {source for source, paths in written_by_source.items() if paths}
    stale = _stale_clips(out_dir, written)
    removable: list[Path] = []
    if clean:
        if prior_sources is None:
            removable = [p for p in stale if _clip_stem(p.name) in run_stems]
        else:
            replaceable_sources = {
                source
                for source in prior_sources
                if source not in current_sources or source in written_sources
            }
            removable_names = set().union(
                *(prior_sources[source] for source in replaceable_sources)
            )
            removable = [p for p in stale if p.name in removable_names]
    removed: list[Path] = []
    for path in removable:
        try:
            path.unlink()
        except OSError as e:
            print(
                f"warning: could not remove {path} ({e}); leaving it recorded for "
                "a later --clean",
                file=sys.stderr,
            )
        else:
            removed.append(path)
    if removed:
        print(
            f"removed {len(removed)} clip(s) left from an earlier run "
            f"({_summarize(removed)})"
        )
    lingering = [p for p in stale if p not in removed]
    lingering_names = {p.name for p in lingering}
    ambiguous_legacy = prior_sources is None and any(
        _clip_stem(path.name) in run_stems for path in lingering
    )

    # Carry forward every still-present owned clip under its original source,
    # then replace or add the clips written for current sources. A no-manifest
    # run cannot claim same-stem leftovers: they may be old prep output or audio
    # curated by hand. Leave the record absent until legacy --clean resolves
    # that ambiguity, so v2 never turns an inferred filename match into durable
    # provenance.
    next_sources: dict[str, set[str]] = {}
    if prior_sources is not None:
        for source, names in prior_sources.items():
            kept = names & lingering_names
            if kept:
                next_sources[source] = kept
    # Keep an already-owned current source in the record even when it wrote
    # nothing. If its old clips are already absent, the empty entry still proves
    # the next rerun comes from the same source instead of deadlocking the
    # directory. A brand-new source earns no cleanup authority until it actually
    # contributes a clip.
    for source, paths in written_by_source.items():
        if not paths and (prior_sources is None or source not in prior_sources):
            continue
        next_sources.setdefault(source, set()).update(path.name for path in paths)

    # A no-output run cannot establish ownership in an unrecorded directory.
    # Leaving it unclaimed lets a later successful rerun bootstrap the legacy
    # clips instead of stranding them behind an empty manifest.
    if (prior_sources is not None or written) and not ambiguous_legacy:
        _write_manifest(out_dir, label, next_sources)
    elif ambiguous_legacy:
        print(
            f"warning: {MANIFEST_NAME} was not written because same-stem legacy "
            "files remain and prep cannot prove whether it created them; review "
            "the warning below before rerunning with --clean",
            file=sys.stderr,
        )

    if lingering:
        # Why a file lingered decides what the user should do about it, and the
        # cases want opposite advice. A take this run re-recorded and got nothing
        # from kept its previous clips because they are the only copy -- telling
        # that user to delete them by hand would undo the guard and lose the
        # recording. Split the message rather than send one that is right for the
        # common case and harmful for the case prep went out of its way to
        # protect.
        if prior_sources is None:
            attempted = set(stems)
            empty_takes = [
                p for p in lingering if _clip_stem(p.name) in attempted - run_stems
            ]
        else:
            silent_names = set().union(
                *(
                    prior_sources[source]
                    for source in current_sources - written_sources
                    if source in prior_sources
                )
            )
            empty_takes = [p for p in lingering if p.name in silent_names]
        owned_names = (
            set().union(*prior_sources.values()) if prior_sources is not None else set()
        )
        failed_removals = [p for p in lingering if p in removable]
        if empty_takes:
            other_lingering = [
                p
                for p in lingering
                if p not in empty_takes and p not in failed_removals
            ]
            fix = (
                f"{_summarize(empty_takes)} are prior clips for a source this run "
                "re-recorded but got no clips from, so prep kept them rather "
                "than leave you with none; check the recording before removing "
                "anything"
            )
            if other_lingering:
                if clean:
                    fix += (
                        f". {_summarize(other_lingering)} have no prep ownership "
                        "record, so --clean leaves them; remove them by hand if "
                        "they are not wanted"
                    )
                else:
                    cleanable = [
                        p
                        for p in other_lingering
                        if (
                            p.name in owned_names
                            if prior_sources is not None
                            else _clip_stem(p.name) in run_stems
                        )
                    ]
                    manual = [p for p in other_lingering if p not in cleanable]
                    if cleanable:
                        fix += (
                            f". Rerun with --clean to remove prep-owned "
                            f"{_summarize(cleanable)}"
                        )
                    if manual:
                        fix += (
                            f". {_summarize(manual)} have no prep ownership record; "
                            "remove only those files by hand if they are not wanted"
                        )
            if failed_removals:
                fix += (
                    f". {_summarize(failed_removals)} were selected for cleanup "
                    "but could not be removed; address the earlier errors and rerun "
                    "--clean"
                )
        elif clean:
            unrecorded = [p for p in lingering if p not in failed_removals]
            fixes: list[str] = []
            if unrecorded:
                fixes.append(
                    "prep has no record of creating "
                    f"{_summarize(unrecorded)}, so --clean leaves them; remove "
                    "them by hand if they are not wanted"
                )
            if failed_removals:
                fixes.append(
                    f"{_summarize(failed_removals)} were selected for cleanup "
                    "but could not be removed; address the earlier errors and "
                    "rerun --clean"
                )
            fix = ". ".join(fixes)
        else:
            if prior_sources is None:
                fix = (
                    "review these legacy filenames, then rerun with --clean or "
                    "remove only the unwanted files by hand"
                )
            else:
                cleanable = [p for p in lingering if p.name in owned_names]
                manual = [p for p in lingering if p not in cleanable]
                fixes = []
                if cleanable:
                    fixes.append(
                        f"rerun with --clean to remove prep-owned "
                        f"{_summarize(cleanable)}"
                    )
                if manual:
                    fixes.append(
                        f"{_summarize(manual)} have no prep ownership record; "
                        "remove them by hand if they are not wanted"
                    )
                fix = ". ".join(fixes)
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
    p.add_argument(
        "--output",
        required=True,
        help=f"output directory for clips and the {MANIFEST_NAME} ownership record",
    )
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
        "take no longer in the inputs; when no manifest exists, falls back to "
        "same-stem filenames, while an unusable manifest fails closed",
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
