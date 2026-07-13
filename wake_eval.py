#!/usr/bin/env python3
"""Threshold-sweep metrics for wake-word tuning (#17).

Pure decision core. Given the wake detector's scores on a labeled set, it reports
false-accepts per hour and false-rejects per activation across a `wake_threshold`
sweep and recommends a default under a false-accept budget. No audio, no model, no
numpy is imported here, so the math is unit-testable on any box; the scoring driver
(eval_wake_threshold.py) loads the recordings and supplies the scores.

Metric definitions (matching the issue's acceptance criteria):

  - false-rejects per activation: the fraction of real activations the detector
    misses at a given threshold. One positive recording is one activation.

  - false-accepts per hour: spurious fires during non-wake audio, per hour of that
    audio. Near-word recordings ("computer"/"commuter"/"computing") and ambient
    clips are both non-wake audio: each is scored frame by frame and threshold
    crossings are counted with a refractory gap, so one sustained loud moment is
    a single false accept, the way a live assistant does not re-trigger on the
    same noise. Dividing the fire count by the non-wake audio duration gives a rate
    that is comparable across sample sets of different sizes.

The recommendation walks the sweep for the lowest false-reject threshold whose
false-accept rate stays within a caller-supplied budget, and never returns a
silently-worse default: if nothing meets the budget it says so and hands back the
lowest achievable false-accept rate instead.
"""

from __future__ import annotations

from math import ceil, inf
from typing import NamedTuple, Sequence

# Audio seconds per capture frame: openWakeWord scores 80 ms (1280-sample) frames
# at 16 kHz, matching pipeline.FRAME_SIZE. The driver passes the real value; this
# default keeps the core usable and the tests readable without importing pipeline.
DEFAULT_FRAME_HOP_S = 1280 / 16000  # 0.08


class Activation(NamedTuple):
    """One real wake-word activation (a positive recording).

    peak is the detector's highest score over the clip (pipeline.detect_wake's
    third return). latency_s, when measured, is the time from clip start to the
    first frame that crosses the evaluated threshold; None means latency was not
    measured, and the sweep then judges the activation on peak alone.
    """
    peak: float
    latency_s: float | None = None


class NonWakeClip(NamedTuple):
    """A continuous stretch of audio that must NOT fire the wake word.

    frame_scores is the detector's per-frame score across the whole clip (scored
    without stopping at the first fire, so any threshold can be applied afterward).
    seconds is the clip's real duration, the denominator for the per-hour rate.
    """
    frame_scores: Sequence[float]
    seconds: float


class ThresholdRow(NamedTuple):
    """The measured behavior at one threshold."""
    threshold: float
    activations: int
    false_rejects: int
    false_rejects_per_activation: float
    false_accepts: int
    nonwake_hours: float
    false_accepts_per_hour: float


class Recommendation(NamedTuple):
    """The chosen default threshold, the row it came from, and why."""
    threshold: float
    row: ThresholdRow
    meets_budget: bool
    reason: str


def threshold_grid(start: float = 0.1, stop: float = 0.9,
                   step: float = 0.05) -> list[float]:
    """Inclusive threshold sweep [start, stop] in `step` increments.

    Rounded to 4 decimals so floating-point accumulation does not leave a value
    like 0.35000000000000003 in the table. The last point never exceeds stop: when
    step does not divide the span evenly the grid stops at the last multiple within
    range (threshold_grid(0.1, 0.83, 0.05) ends at 0.8, not 0.85) so the sweep never
    evaluates a threshold above the one asked for. Raises on a non-positive step or
    an inverted range so a typo fails loud instead of producing an empty or backwards
    sweep.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    if stop < start:
        raise ValueError("stop must be >= start")
    # Floor the step count so the grid stays within [start, stop]; the epsilon keeps
    # an even division (e.g. 16.0 arriving as 15.9999999) from dropping the endpoint.
    n = int((stop - start) / step + 1e-9)
    return [round(start + i * step, 4) for i in range(n + 1)]


def _activation_accepted(a: Activation, threshold: float,
                         latency_tolerance_s: float | None) -> bool:
    """True when this activation fires within the peak and (optional) latency budget.

    Peak below threshold is always a miss. When a latency tolerance is set AND this
    activation's latency was measured, a fire later than the tolerance is a miss too
    (a wake that arrives after the user has given up is a reject in practice). An
    unmeasured latency (latency_s is None) is judged on peak alone rather than
    penalized, so peak-only runs are unaffected by the latency feature.
    """
    if a.peak < threshold:
        return False
    if latency_tolerance_s is not None and a.latency_s is not None:
        return a.latency_s <= latency_tolerance_s
    return True


def count_false_accepts(frame_scores: Sequence[float], threshold: float,
                        refractory_frames: int) -> int:
    """Count refractory-gapped threshold crossings in one non-wake clip.

    A live detector that fires does not immediately re-arm on the same noise, so a
    sustained loud region should count once, not once per frame. After a fire at
    frame i, further fires are suppressed until frame i + refractory_frames.
    """
    fires = 0
    next_allowed = 0
    for i, score in enumerate(frame_scores):
        if i >= next_allowed and score >= threshold:
            fires += 1
            next_allowed = i + refractory_frames
    return fires


def sweep(positives: Sequence[Activation], nonwake: Sequence[NonWakeClip],
          thresholds: Sequence[float], *, refractory_s: float = 1.0,
          frame_hop_s: float = DEFAULT_FRAME_HOP_S,
          latency_tolerance_s: float | None = None) -> list[ThresholdRow]:
    """Measure false-reject and false-accept behavior at each threshold.

    Scores are computed once by the caller; this only compares them against each
    threshold, so the sweep is cheap no matter how many thresholds it covers.
    """
    if frame_hop_s <= 0:
        raise ValueError("frame_hop_s must be positive")
    if refractory_s < 0:
        raise ValueError("refractory_s must be non-negative")
    # 0 s means no suppression (every crossing counts); a fraction of a frame still
    # rounds up to a 1-frame gap so a small positive value is not silently a no-op.
    refractory_frames = ceil(refractory_s / frame_hop_s)
    total_hours = sum(c.seconds for c in nonwake) / 3600.0
    n_act = len(positives)

    rows: list[ThresholdRow] = []
    for t in sorted(thresholds):
        fr = sum(1 for a in positives
                 if not _activation_accepted(a, t, latency_tolerance_s))
        fa = sum(count_false_accepts(c.frame_scores, t, refractory_frames)
                 for c in nonwake)
        # With no non-wake audio the rate is undefined: report inf if anything
        # fired (so it never looks safe) and 0.0 only when nothing did.
        if total_hours > 0:
            fa_per_hour = fa / total_hours
        else:
            fa_per_hour = inf if fa else 0.0
        rows.append(ThresholdRow(
            threshold=t,
            activations=n_act,
            false_rejects=fr,
            false_rejects_per_activation=(fr / n_act if n_act else 0.0),
            false_accepts=fa,
            nonwake_hours=total_hours,
            false_accepts_per_hour=fa_per_hour,
        ))
    return rows


def recommend(rows: Sequence[ThresholdRow], *,
              max_false_accepts_per_hour: float) -> Recommendation:
    """Pick the lowest false-reject threshold within the false-accept budget.

    Among thresholds whose false-accept rate is within budget, choose the smallest
    false-rejects-per-activation; ties break toward the lower threshold, which
    leaves more margin for a quiet or clipped activation. If no threshold meets the
    budget, return the threshold with the lowest false-accept rate and flag that the
    budget was not met, so the recommendation is never silently worse than asked.
    """
    if not rows:
        raise ValueError("no rows to recommend from")
    within = [r for r in rows
              if r.false_accepts_per_hour <= max_false_accepts_per_hour]
    if within:
        best = min(within,
                   key=lambda r: (r.false_rejects_per_activation, r.threshold))
        return Recommendation(
            best.threshold, best, True,
            f"lowest false-reject threshold within "
            f"{max_false_accepts_per_hour:g} false accepts/hour")
    fallback = min(rows, key=lambda r: (r.false_accepts_per_hour,
                                        r.false_rejects_per_activation,
                                        r.threshold))
    return Recommendation(
        fallback.threshold, fallback, False,
        f"no threshold met the {max_false_accepts_per_hour:g}/hour budget; "
        f"this is the lowest achievable false-accept rate "
        f"({fallback.false_accepts_per_hour:g}/hour)")
