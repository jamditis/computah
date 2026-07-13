#!/usr/bin/env python3
"""Sweep wake_threshold against a labeled sample set and recommend a default (#17).

This is the scoring driver for the pure metrics core in wake_eval.py. It reads the
gitignored samples/ directories that prep_wake_samples.py writes, scores each clip
with the same openWakeWord model the live loop uses, and prints false-accepts per
hour and false-rejects per activation across a threshold sweep, plus a recommended
default under a false-accept budget.

Sample layout (all under samples/, which is gitignored: personal voice data):
  - positive/    real wake-word activations           -> false-reject rate
  - negative/    hard near-words (computer/commuter)   -> non-wake audio
  - background/  ambient / no-speech recordings        -> non-wake audio
Record and prepare them with prep_wake_samples.py (see docs/recording-computah.md).
The script is committed; the recordings are not.

Usage:
  .venv/bin/python eval_wake_threshold.py                      # defaults under samples/
  .venv/bin/python eval_wake_threshold.py --model hey_jarvis \
      --min-threshold 0.1 --max-threshold 0.9 --step 0.05 --max-fa-per-hour 1.0

Record the recommended value in config.json (wake_threshold) for the shipped model;
see docs/wake-threshold-tuning.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pipeline
import wake_eval

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".m4a", ".mp3"}
FRAME_HOP_S = pipeline.FRAME_SIZE / 16000  # 0.08 s per scored frame

# After a fire the live loop does not re-arm immediately: stream_detect_wake stops on
# the crossing and capture_request consumes the request before wake detection resumes,
# at least _NO_SPEECH_ONSET_FRAMES (~1.6 s) when no speech follows and longer when it
# does. Using that floor as the default post-fire gap makes one sustained near-word or
# noise burst count as a single wake event, matching production, rather than several,
# which would inflate false-accepts/hour and over-tighten the threshold.
DEFAULT_REFRACTORY_S = pipeline._NO_SPEECH_ONSET_FRAMES * FRAME_HOP_S  # ~1.6 s


def _clips(directory: Path) -> list[Path]:
    """Every audio file directly under `directory`, sorted; empty if it is absent."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir()
                  if p.suffix.lower() in AUDIO_EXTS)


def _positive_peak(path: Path, model_name: str) -> float:
    """The detector's peak score for one activation clip.

    The peak is the max over the same padded frame scores detect_wake decides on, so
    a positive and a non-wake clip are scored through one routine and sit on the same
    footing. The sweep applies every threshold afterward.
    """
    scores = pipeline.wake_frame_scores(str(path), model_name)
    return max(scores) if scores else 0.0


def _frame_scores(path: Path, model_name: str) -> tuple[list[float], float]:
    """Per-frame wake scores across a whole clip, and its real duration in seconds.

    Reuses pipeline.wake_frame_scores, the same padded, freshly-reset scoring
    detect_wake runs, so non-wake clips get the identical filled context as positives
    and their opening ~2 s is not under-scored (which would undercount false accepts).
    Unlike detect_wake it keeps every frame's score, not only the peak, so any
    threshold can be applied afterward. The duration is the real, un-padded length,
    the honest denominator for the per-hour rate.
    """
    scores = pipeline.wake_frame_scores(str(path), model_name)
    seconds = len(pipeline._load_pcm16(str(path))) / 16000
    return scores, seconds


def _print_table(rows: list[wake_eval.ThresholdRow]) -> None:
    print(f"\n{'thresh':>7}  {'FR/act':>7}  {'FA/hr':>8}  "
          f"{'rejects':>8}  {'accepts':>8}")
    print("  " + "-" * 44)
    for r in rows:
        print(f"{r.threshold:>7.3f}  {r.false_rejects_per_activation:>7.3f}  "
              f"{r.false_accepts_per_hour:>8.2f}  "
              f"{r.false_rejects:>3}/{r.activations:<4}  {r.false_accepts:>8}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", default="samples",
                   help="root of the labeled sample directories (default: samples)")
    p.add_argument("--model", default=None,
                   help="wake model name (default: config.json wake_word)")
    p.add_argument("--min-threshold", type=float, default=0.1)
    p.add_argument("--max-threshold", type=float, default=0.9)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--max-fa-per-hour", type=float, default=1.0,
                   help="false-accept budget for the recommendation (default: 1.0)")
    p.add_argument("--refractory-s", type=float, default=DEFAULT_REFRACTORY_S,
                   help="post-fire gap before a new false accept can count, seconds "
                        "(default: the live re-arm floor, ~1.6s)")
    args = p.parse_args()

    model_name = args.model or pipeline.load_config()["wake_word"]
    root = Path(args.samples)
    pos_dir, neg_dir, bg_dir = root / "positive", root / "negative", root / "background"

    positives_paths = _clips(pos_dir)
    nonwake_paths = _clips(neg_dir) + _clips(bg_dir)
    if not positives_paths or not nonwake_paths:
        print(
            "not enough labeled audio to evaluate.\n"
            f"  positive activations: {len(positives_paths)} under {pos_dir}\n"
            f"  non-wake clips:       {len(nonwake_paths)} under "
            f"{neg_dir} + {bg_dir}\n"
            "Record and prep samples first (prep_wake_samples.py; see "
            "docs/recording-computah.md), then re-run.",
            file=sys.stderr)
        return 1

    print(f"model: {model_name}")
    print(f"scoring {len(positives_paths)} activations, "
          f"{len(nonwake_paths)} non-wake clips ...")

    positives = [wake_eval.Activation(peak=_positive_peak(path, model_name))
                 for path in positives_paths]
    nonwake = []
    for path in nonwake_paths:
        scores, seconds = _frame_scores(path, model_name)
        nonwake.append(wake_eval.NonWakeClip(frame_scores=scores, seconds=seconds))

    thresholds = wake_eval.threshold_grid(
        args.min_threshold, args.max_threshold, args.step)
    rows = wake_eval.sweep(positives, nonwake, thresholds,
                           refractory_s=args.refractory_s,
                           frame_hop_s=FRAME_HOP_S)
    _print_table(rows)

    rec = wake_eval.recommend(rows, max_false_accepts_per_hour=args.max_fa_per_hour)
    total_hours = rows[0].nonwake_hours if rows else 0.0
    label = "recommended" if rec.meets_budget else "best available (budget not met)"
    print(f"\nnon-wake audio evaluated: {total_hours * 60:.1f} min")
    print(f"{label} wake_threshold for {model_name}: {rec.threshold:.3f}")
    print(f"  {rec.reason}")
    print(f"  at this threshold: false-rejects "
          f"{rec.row.false_rejects_per_activation:.1%} of activations, "
          f"false-accepts {rec.row.false_accepts_per_hour:.2f}/hour")
    if not rec.meets_budget:
        print("  warning: no threshold met the budget; collect more or cleaner "
              "negatives, or accept a higher false-accept rate.")
    print(f'\nrecord this in config.json:  "wake_threshold": {rec.threshold:.3f},')
    print("see docs/wake-threshold-tuning.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
