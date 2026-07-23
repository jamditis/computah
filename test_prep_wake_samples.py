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


def test_zero_clips_no_crash(d: Path) -> None:
    """Audio that yields no clips must report cleanly, not raise.

    Distinct from test_empty_input_no_crash: the input file IS found, so this
    runs past the no-files early return and reaches the stale-clip scan. Only
    _write_clip() creates out_dir, so a run that writes nothing never creates
    it and the scan reads a directory that is not there.
    """
    audio = make_bursts(n=3, burst_s=0.6, gap_s=1.0, sr=prep.TARGET_SR)
    src = d / "all_dropped.wav"
    sf.write(src, audio, prep.TARGET_SR, subtype="PCM_16")
    out = d / "out_dropped"
    # A --min-dur longer than every burst, so all three segments drop as short.
    total = prep.process([src], out, "positive", 0.3, 5.0, 10.0)
    check(total == 0, f"all-dropped input returns 0 without crashing (got {total})")
    check(not out.exists(), "a run that writes no clips leaves no output dir")


def test_output_path_not_a_directory(d: Path) -> None:
    """An --output naming an existing file reports it, rather than raising.

    The mistake used to surface as one of two exceptions depending on whether a
    clip was ready to write (FileExistsError from mkdir) or not
    (NotADirectoryError from the stale scan), both after decoding every input.
    """
    audio = make_bursts(n=2, burst_s=0.6, gap_s=1.0, sr=prep.TARGET_SR)
    src = d / "into_a_file.wav"
    sf.write(src, audio, prep.TARGET_SR, subtype="PCM_16")
    occupied = d / "not_a_dir"
    occupied.write_text("keep me")
    total = prep.process([src], occupied, "positive", 0.3, 0.2, 3.0)
    check(total == 0, f"an occupied --output returns 0, not a traceback (got {total})")
    check(occupied.read_text() == "keep me", "the file at --output is left untouched")


def test_duplicate_basename_no_overwrite(d: Path) -> None:
    """Two inputs sharing a basename must not overwrite each other's clips.

    Clip names come from the input stem, so a/computah.wav and b/computah.wav
    once wrote the same names and the second file silently clobbered the first
    while the reported total still counted both. The two files carry different
    burst counts (2 and 3) so any overwrite shows up as fewer files than clips.
    """
    a = d / "a"
    b = d / "b"
    a.mkdir()
    b.mkdir()
    sf.write(
        a / "computah.wav",
        make_bursts(2, 0.5, 0.8, prep.TARGET_SR),
        prep.TARGET_SR,
        subtype="PCM_16",
    )
    sf.write(
        b / "computah.wav",
        make_bursts(3, 0.5, 0.8, prep.TARGET_SR),
        prep.TARGET_SR,
        subtype="PCM_16",
    )

    out = d / "dup_positive"
    total = prep.process(
        [a / "computah.wav", b / "computah.wav"], out, "positive", 0.3, 0.2, 3.0
    )
    files = sorted(out.glob("*.wav"))
    check(
        total == 5 and len(files) == 5,
        f"5 clips from two same-basename inputs, none overwritten "
        f"(total={total}, files={len(files)})",
    )
    check(
        len({f.name for f in files}) == len(files),
        f"every clip name is distinct ({[f.name for f in files]})",
    )

    # The background path derives names the same way, so it shares the bug.
    out_bg = d / "dup_background"
    total_bg = prep.process(
        [a / "computah.wav", b / "computah.wav"], out_bg, "background", 0.3, 0.2, 3.0
    )
    bg_files = sorted(out_bg.glob("*.wav"))
    check(
        total_bg == 2 and len(bg_files) == 2,
        f"two same-basename background files stay two files "
        f"(total={total_bg}, files={len(bg_files)})",
    )


def test_rerun_refreshes_populated_dir(d: Path) -> None:
    """Rerunning into a dir that already holds a prior run's clips must work.

    The docs tell people to run prep_wake_samples straight into samples/, so a
    refresh overwrites last run's clips rather than aborting. Running process
    twice into the same out_dir must succeed both times and leave the same file
    count -- a prior-run leftover is an intended overwrite, not a collision.
    """
    src = d / "rerun_src"
    src.mkdir()
    sf.write(
        src / "take.wav",
        make_bursts(3, 0.5, 0.8, prep.TARGET_SR),
        prep.TARGET_SR,
        subtype="PCM_16",
    )
    out = d / "rerun_out"
    first = prep.process([src / "take.wav"], out, "positive", 0.3, 0.2, 3.0)
    second = prep.process([src / "take.wav"], out, "positive", 0.3, 0.2, 3.0)
    files = sorted(out.glob("*.wav"))
    check(
        first == second == 3 and len(files) == 3,
        f"rerun into a populated dir refreshes cleanly "
        f"(first={first}, second={second}, files={len(files)})",
    )


def test_rerun_with_fewer_clips_orphans_then_clean(d: Path) -> None:
    """Issue #8: re-recording with fewer utterances must not silently keep orphans.

    The first run writes five clips. Re-recording the same take with two
    utterances and rerunning overwrites take_000/_001 but strands take_002..004.
    Without --clean those orphans linger (the warn path, so training would read
    them); with clean=True they are removed and the returned count equals the
    files on disk -- the issue's acceptance criteria.
    """
    src = d / "fewer_src"
    src.mkdir()
    take = src / "take.wav"
    sf.write(
        take, make_bursts(5, 0.5, 0.8, prep.TARGET_SR), prep.TARGET_SR, subtype="PCM_16"
    )
    out = d / "fewer_out"
    first = prep.process([take], out, "positive", 0.3, 0.2, 3.0)
    check(first == 5, f"first run writes five clips (first={first})")

    # Re-record the same file with fewer utterances, then rerun into the same dir.
    sf.write(
        take, make_bursts(2, 0.5, 0.8, prep.TARGET_SR), prep.TARGET_SR, subtype="PCM_16"
    )
    without = prep.process([take], out, "positive", 0.3, 0.2, 3.0)
    present = sorted(p.name for p in out.glob("*.wav"))
    check(
        without == 2 and len(present) == 5,
        f"without --clean the three orphans linger (wrote={without}, files={present})",
    )

    cleaned = prep.process([take], out, "positive", 0.3, 0.2, 3.0, clean=True)
    present = sorted(p.name for p in out.glob("*.wav"))
    check(
        cleaned == 2 and present == ["take_000.wav", "take_001.wav"],
        f"--clean removes the orphans so the count matches files on disk "
        f"(wrote={cleaned}, files={present})",
    )


def test_clean_spares_files_this_run_did_not_record(d: Path) -> None:
    """--clean removes only the re-recorded take's own orphan clips. A source
    take, and a hand-curated clip that happens to match "<stem>_NNN.wav" but
    whose stem this run did not write, are both left in place.
    """
    src = d / "spare_src"
    src.mkdir()
    take = src / "take.wav"
    sf.write(
        take, make_bursts(3, 0.5, 0.8, prep.TARGET_SR), prep.TARGET_SR, subtype="PCM_16"
    )
    out = d / "spare_out"
    prep.process([take], out, "positive", 0.3, 0.2, 3.0)  # take_000.._002

    # A raw recording (no clip shape) and a hand-curated clip whose stem this run
    # never writes -- both must survive --clean.
    for name in ("my_recording.wav", "custom_001.wav"):
        sf.write(
            out / name,
            make_bursts(1, 0.5, 0.8, prep.TARGET_SR),
            prep.TARGET_SR,
            subtype="PCM_16",
        )

    # Re-record with fewer utterances and rerun with --clean: only the take_*
    # orphan is removed; the source take and the foreign clip are untouched.
    sf.write(
        take, make_bursts(2, 0.5, 0.8, prep.TARGET_SR), prep.TARGET_SR, subtype="PCM_16"
    )
    prep.process([take], out, "positive", 0.3, 0.2, 3.0, clean=True)
    present = sorted(p.name for p in out.glob("*.wav"))
    check(
        "my_recording.wav" in present
        and "custom_001.wav" in present
        and "take_002.wav" not in present,
        f"--clean removes only the re-recorded take's orphans ({present})",
    )


def test_clean_removes_disambiguated_stem_orphans(d: Path) -> None:
    """--clean must clean orphans of a *disambiguated* stem too (#7 x #8).

    Two inputs sharing a basename get stems "computah" and "computah-1". The
    greedy match in _CLIP_NAME is load-bearing here: an orphan
    "computah-1_002.wav" has to pair back to the run stem "computah-1", not
    "computah", or --clean would strand it for training to read.
    """
    a = d / "disambig_a"
    b = d / "disambig_b"
    a.mkdir()
    b.mkdir()
    for dir_ in (a, b):
        sf.write(
            dir_ / "computah.wav",
            make_bursts(3, 0.5, 0.8, prep.TARGET_SR),
            prep.TARGET_SR,
            subtype="PCM_16",
        )
    out = d / "disambig_out"
    inputs = [a / "computah.wav", b / "computah.wav"]
    prep.process(inputs, out, "positive", 0.3, 0.2, 3.0)  # computah_* + computah-1_*, 6

    # Re-record both takes shorter, then rerun with --clean.
    for dir_ in (a, b):
        sf.write(
            dir_ / "computah.wav",
            make_bursts(1, 0.5, 0.8, prep.TARGET_SR),
            prep.TARGET_SR,
            subtype="PCM_16",
        )
    prep.process(inputs, out, "positive", 0.3, 0.2, 3.0, clean=True)
    present = sorted(p.name for p in out.glob("*.wav"))
    check(
        present == ["computah-1_000.wav", "computah_000.wav"],
        f"--clean clears orphans of both the bare and disambiguated stems ({present})",
    )


def test_clean_spares_prior_clips_when_rerun_writes_nothing(d: Path) -> None:
    """A re-recording that yields zero clips must not let --clean wipe the good
    run before it. The take's stem is still in the input list, but it wrote
    nothing this run, so its earlier clips are the only copy -- --clean has to
    leave them. Scoping by written stems (not input stems) is what protects them.
    """
    src = d / "wipe_src"
    src.mkdir()
    take = src / "take.wav"
    sf.write(
        take, make_bursts(3, 0.5, 0.8, prep.TARGET_SR), prep.TARGET_SR, subtype="PCM_16"
    )
    out = d / "wipe_out"
    first = prep.process([take], out, "positive", 0.3, 0.2, 3.0)  # take_000.._002
    check(first == 3, f"first run writes the good clips (got {first})")

    # Re-record badly: a --min-dur longer than every burst drops all segments, so
    # this run writes nothing. Without the written-stem scoping, --clean would read
    # the take's prior clips as stale-for-its-own-stem and delete the only copy.
    sf.write(
        take, make_bursts(3, 0.5, 0.8, prep.TARGET_SR), prep.TARGET_SR, subtype="PCM_16"
    )
    second = prep.process([take], out, "positive", 0.3, 5.0, 10.0, clean=True)
    present = sorted(p.name for p in out.glob("*.wav"))
    check(
        second == 0 and present == ["take_000.wav", "take_001.wav", "take_002.wav"],
        f"--clean keeps the prior good clips when the rerun writes nothing ({present})",
    )


def test_in_run_collision_raises(d: Path) -> None:
    """Two clips in one run mapping to the same file must fail loud.

    _unique_stems keeps in-run basenames distinct, but this is the backstop for
    a case-insensitive filesystem where distinct-cased stems resolve to one
    inode. Writing the same name twice with a shared seen-set (as one run would)
    must raise instead of silently dropping the first clip.
    """
    out = d / "in_run_collision"
    audio = make_bursts(1, 0.5, 0.8, prep.TARGET_SR)
    seen: set[int] = set()
    first = prep._write_unique(out, "dup", 0, audio, seen)
    check(first.exists(), f"first clip of the run lands on disk ({first.name})")
    try:
        prep._write_unique(out, "dup", 0, audio, seen)
    except FileExistsError:
        check(True, "a second clip mapping to the same file in one run raises")
    else:
        check(False, "an in-run collision should have raised FileExistsError")


def test_refresh_after_adding_collider_leaves_no_orphans(d: Path) -> None:
    """Adding a second same-basename input must not orphan the first run's clips.

    Run one input, then rerun with a second file of the same basename into the
    same dir. If every colliding input got suffixed, the first run's
    computah_###.wav would be stranded next to an identical computah-1_###.wav --
    and eval_wake_threshold._clips() reads every clip in the directory, so the
    stale pair would silently double-count. The first input keeps the bare stem,
    so its clips are overwritten in place and only the new file is suffixed.
    """
    a = d / "refresh_a"
    b = d / "refresh_b"
    a.mkdir()
    b.mkdir()
    sf.write(
        a / "computah.wav",
        make_bursts(2, 0.5, 0.8, prep.TARGET_SR),
        prep.TARGET_SR,
        subtype="PCM_16",
    )
    sf.write(
        b / "computah.wav",
        make_bursts(3, 0.5, 0.8, prep.TARGET_SR),
        prep.TARGET_SR,
        subtype="PCM_16",
    )

    out = d / "refresh_out"
    prep.process([a / "computah.wav"], out, "positive", 0.3, 0.2, 3.0)
    prep.process(
        [a / "computah.wav", b / "computah.wav"], out, "positive", 0.3, 0.2, 3.0
    )

    files = sorted(p.name for p in out.glob("*.wav"))
    # 2 clips from a (bare stem, overwritten in place) + 3 from b (suffixed).
    check(
        len(files) == 5,
        f"refresh with an added collider leaves no orphaned clips ({files})",
    )
    check(
        any(n.startswith("computah_") for n in files),
        f"the first colliding input keeps the bare stem ({files})",
    )


def test_stem_assignment_ignores_input_order(d: Path) -> None:
    """The same colliding files must map to the same stems in any order.

    If the bare stem went to whichever file was listed first, rerunning with the
    inputs reordered would hand it to the other file: the previous run's
    computah_###.wav would stop being rewritten and would linger as a duplicate
    that eval_wake_threshold still reads. Ranking by resolved path makes the
    mapping a function of which files are present, not how they were passed.
    """
    a = d / "order_a"
    b = d / "order_b"
    a.mkdir()
    b.mkdir()
    for folder in (a, b):
        sf.write(
            folder / "computah.wav",
            make_bursts(2, 0.5, 0.8, prep.TARGET_SR),
            prep.TARGET_SR,
            subtype="PCM_16",
        )

    forward = prep._unique_stems([a / "computah.wav", b / "computah.wav"])
    reverse = prep._unique_stems([b / "computah.wav", a / "computah.wav"])
    # Same file -> same stem regardless of position, so reverse is forward flipped.
    check(
        forward == list(reversed(reverse)),
        f"stem assignment is order-independent (forward={forward}, reverse={reverse})",
    )
    check(
        sorted(forward) == ["computah", "computah-1"],
        f"one file keeps the bare stem, the other is suffixed ({sorted(forward)})",
    )

    # Two colliding groups where one group's bare stem is the name the other
    # group's suffix search wants. Reserving only singletons hands "computah-1"
    # to both, and the run later dies on a bogus same-basename FileExistsError.
    two_groups = prep._unique_stems(
        [
            a / "computah.wav",
            b / "computah.wav",
            a / "computah-1.wav",
            b / "computah-1.wav",
        ]
    )
    check(
        len(set(two_groups)) == len(two_groups),
        f"overlapping collider groups still get distinct stems ({two_groups})",
    )


def _burst_take(path: Path, n: int) -> None:
    """Write a take of `n` well-separated utterances at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        path, make_bursts(n, 0.5, 0.8, prep.TARGET_SR), prep.TARGET_SR, subtype="PCM_16"
    )


def test_manifest_cleans_dropped_input_orphans(d: Path) -> None:
    """--clean removes the orphans of an input dropped between runs (#84 case 1).

    The stem rule alone cannot reach these: `other` is not in the rerun at all,
    so no clip this run writes ever makes its old ones stale. The manifest is
    what distinguishes them from audio a user put in the directory.
    """
    src = d / "dropped_src"
    _burst_take(src / "a" / "take.wav", 3)
    _burst_take(src / "b" / "other.wav", 2)
    out = d / "dropped_out"
    prep.process(
        [src / "a" / "take.wav", src / "b" / "other.wav"],
        out,
        "positive",
        0.3,
        0.2,
        3.0,
    )
    check(
        (out / prep.MANIFEST_NAME).is_file(),
        "a run records what it wrote in the manifest",
    )

    prep.process([src / "a" / "take.wav"], out, "positive", 0.3, 0.2, 3.0, clean=True)
    present = sorted(p.name for p in out.glob("*.wav"))
    check(
        present == ["take_000.wav", "take_001.wav", "take_002.wav"],
        f"--clean removes a dropped input's orphans ({present})",
    )


def test_manifest_cleans_changed_collider_stem_orphans(d: Path) -> None:
    """--clean removes orphans whose disambiguated stem vanished (#84 case 2).

    Two `take.wav` inputs become stems `take` and `take-1`. Rerun with only one
    and `take-1` is not a stem any more, so its clips are orphaned under a name
    nothing will overwrite.
    """
    src = d / "collider_src"
    _burst_take(src / "a" / "take.wav", 3)
    _burst_take(src / "b" / "take.wav", 2)
    out = d / "collider_out"
    prep.process(
        [src / "a" / "take.wav", src / "b" / "take.wav"], out, "positive", 0.3, 0.2, 3.0
    )
    first = sorted(p.name for p in out.glob("*.wav"))
    check(
        any(n.startswith("take-1_") for n in first),
        f"the collider run writes a disambiguated stem ({first})",
    )

    prep.process([src / "b" / "take.wav"], out, "positive", 0.3, 0.2, 3.0, clean=True)
    present = sorted(p.name for p in out.glob("*.wav"))
    check(
        not any(n.startswith("take-1_") for n in present),
        f"--clean removes orphans of a stem the rerun no longer produces ({present})",
    )


def test_manifest_never_authorizes_deleting_curated_audio(d: Path) -> None:
    """The manifest widens --clean only to files prep recorded writing.

    A hand-added clip whose stem no run ever wrote is exactly the shape the new
    rule reaches for -- an orphan of an unattempted stem -- so this pins that it
    is still spared. Being absent from the manifest is the whole guarantee.
    """
    src = d / "curated_src"
    _burst_take(src / "take.wav", 3)
    out = d / "curated_out"
    prep.process([src / "take.wav"], out, "positive", 0.3, 0.2, 3.0)
    _burst_take(out / "custom_001.wav", 1)
    _burst_take(out / "my_recording.wav", 1)

    prep.process([src / "take.wav"], out, "positive", 0.3, 0.2, 3.0, clean=True)
    present = sorted(p.name for p in out.glob("*.wav"))
    check(
        "custom_001.wav" in present and "my_recording.wav" in present,
        f"--clean spares audio prep did not create ({present})",
    )


def test_manifest_spares_prior_clips_when_rerun_writes_nothing(d: Path) -> None:
    """The manifest must not reopen the silent-re-recording data loss.

    Those clips ARE in the manifest, so a rule that removed every manifested
    clip the run did not rewrite would delete the only copy of a take whose
    re-recording came back silent. The stem has to be excluded because the run
    attempted it, regardless of what the manifest says.
    """
    src = d / "silent_src"
    take = src / "take.wav"
    _burst_take(take, 3)
    out = d / "silent_out"
    prep.process([take], out, "positive", 0.3, 0.2, 3.0)

    _burst_take(take, 3)
    second = prep.process([take], out, "positive", 0.3, 5.0, 10.0, clean=True)
    present = sorted(p.name for p in out.glob("*.wav"))
    check(
        second == 0 and present == ["take_000.wav", "take_001.wav", "take_002.wav"],
        f"a manifested clip is still spared when its take wrote nothing ({present})",
    )


def test_manifest_spares_a_silent_take_alongside_a_good_one(d: Path) -> None:
    """The `attempted` guard, with the manifest rule actually armed.

    When the whole run comes back empty nothing is written, so the manifest rule
    never arms and the guard is never reached. The case that reaches it is a run
    where one take records fine and another comes back silent: the good take
    supplies the stem overlap that arms the rule, and then only `attempted`
    stands between the silent take's earlier clips -- its only copy -- and
    unlink(). Delete that clause and this is the test that goes red.
    """
    src = d / "mixed_src"
    good = src / "good.wav"
    flaky = src / "flaky.wav"
    _burst_take(good, 2)
    _burst_take(flaky, 3)
    out = d / "mixed_out"
    prep.process([good, flaky], out, "positive", 0.3, 0.2, 3.0)
    before = sorted(p.name for p in out.glob("*.wav"))
    check(len(before) == 5, f"both takes record on the first run ({before})")

    # Re-record both. `good` is fine; `flaky` comes back as one long unbroken
    # tone -- a mic left running, say -- which the max-duration cap drops
    # entirely, so it writes nothing while `good` writes and arms the rule.
    _burst_take(good, 2)
    sf.write(
        flaky,
        np.ones(int(prep.TARGET_SR * 4.0), dtype=np.float32) * 0.5,
        prep.TARGET_SR,
        subtype="PCM_16",
    )
    prep.process([good, flaky], out, "positive", 0.3, 0.2, 1.0, clean=True)
    present = sorted(p.name for p in out.glob("*.wav"))
    check(
        sorted(n for n in present if n.startswith("flaky_"))
        == ["flaky_000.wav", "flaky_001.wav", "flaky_002.wav"],
        f"--clean spares a silent take's only copy while cleaning beside it ({present})",
    )


def test_manifest_unreadable_falls_back_to_stem_rule(d: Path) -> None:
    """A corrupt manifest degrades --clean, it does not fail the run.

    The clips are the output that matters. Reading garbage as "no record" costs
    precision on the orphans only the manifest can reach; treating it as fatal
    would block a run that has real work to do.
    """
    src = d / "corrupt_src"
    _burst_take(src / "a" / "take.wav", 3)
    _burst_take(src / "b" / "other.wav", 2)
    out = d / "corrupt_out"
    prep.process(
        [src / "a" / "take.wav", src / "b" / "other.wav"],
        out,
        "positive",
        0.3,
        0.2,
        3.0,
    )
    (out / prep.MANIFEST_NAME).write_text("{not json")

    _burst_take(src / "a" / "take.wav", 2)
    total = prep.process(
        [src / "a" / "take.wav"], out, "positive", 0.3, 0.2, 3.0, clean=True
    )
    present = sorted(p.name for p in out.glob("*.wav"))
    check(
        total == 2 and "take_002.wav" not in present and "other_000.wav" in present,
        f"a corrupt manifest leaves the stem rule working ({present})",
    )


def test_manifest_spares_a_different_dataset_on_stray_output(d: Path) -> None:
    """A mistyped --output must not let the manifest wipe another dataset.

    Provenance says prep created a file, not that the file belongs to the set
    being refreshed, so the manifest rule only arms on a matching label with an
    overlapping stem. Without that, every clip in a background dir is one prep
    wrote and no positives rerun attempts, which would be enough to delete the
    whole directory.
    """
    src = d / "stray_src"
    _burst_take(src / "noise.wav", 2)
    bg = d / "stray_background"
    prep.process([src / "noise.wav"], bg, "background", 0.3, 0.2, 3.0)
    before = sorted(p.name for p in bg.glob("*.wav"))
    check(
        before == ["noise_000.wav"], f"the background dir starts populated ({before})"
    )

    # The typo: a positives run pointed at the background dir.
    _burst_take(src / "computah.wav", 3)
    prep.process([src / "computah.wav"], bg, "positive", 0.3, 0.2, 3.0, clean=True)
    present = sorted(p.name for p in bg.glob("*.wav"))
    check(
        "noise_000.wav" in present,
        f"--clean spares a dataset this run is not refreshing ({present})",
    )

    # And the guard has to survive the typo being repeated. If the stray run had
    # relabeled the directory's record as "positive", this second one would find
    # a matching label plus an overlapping stem and read the background clip as
    # its own orphan.
    _burst_take(src / "computah.wav", 2)
    prep.process([src / "computah.wav"], bg, "positive", 0.3, 0.2, 3.0, clean=True)
    present = sorted(p.name for p in bg.glob("*.wav"))
    check(
        "noise_000.wav" in present,
        f"a repeated stray --output still spares the other dataset ({present})",
    )


def main() -> int:
    test_segment_count()
    with tempfile.TemporaryDirectory(prefix="prep-wake-") as tmp:
        d = Path(tmp)
        test_process_positive(d)
        test_resample_to_16k_mono(d)
        test_background_kept_whole(d)
        test_explicit_files_only(d)
        test_empty_input_no_crash(d)
        test_zero_clips_no_crash(d)
        test_output_path_not_a_directory(d)
        test_duplicate_basename_no_overwrite(d)
        test_rerun_refreshes_populated_dir(d)
        test_rerun_with_fewer_clips_orphans_then_clean(d)
        test_clean_spares_files_this_run_did_not_record(d)
        test_clean_removes_disambiguated_stem_orphans(d)
        test_clean_spares_prior_clips_when_rerun_writes_nothing(d)
        test_in_run_collision_raises(d)
        test_refresh_after_adding_collider_leaves_no_orphans(d)
        test_stem_assignment_ignores_input_order(d)
        test_manifest_cleans_dropped_input_orphans(d)
        test_manifest_cleans_changed_collider_stem_orphans(d)
        test_manifest_never_authorizes_deleting_curated_audio(d)
        test_manifest_spares_prior_clips_when_rerun_writes_nothing(d)
        test_manifest_spares_a_silent_take_alongside_a_good_one(d)
        test_manifest_unreadable_falls_back_to_stem_rule(d)
        test_manifest_spares_a_different_dataset_on_stray_output(d)
    n_pass = sum(1 for ok, _ in results if ok)
    print(f"=== {n_pass}/{len(results)} checks passed ===")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
