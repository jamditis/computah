#!/usr/bin/env python3
"""Fast, no-audio tests for the wake-threshold sweep metrics (#17).

wake_eval is a pure decision core over plain score values, so this exercises every
branch with no microphone, no openWakeWord model, and no audio, like
test_confidence_guard.py. It pins the two metrics the issue asks for
(false-accepts/hour, false-rejects/activation), the refractory fire-counting, the
latency tolerance, and the recommendation's budget and tie-break rules.

Run:  .venv/bin/python test_wake_eval.py
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import math

import wake_eval
from wake_eval import Activation, NonWakeClip, ThresholdRow

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    results.append((PASS if ok else FAIL, name))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")
    return ok


def _row(threshold, fr_per_act, fa_per_hour) -> ThresholdRow:
    """A ThresholdRow with only the fields recommend() reads populated."""
    return ThresholdRow(
        threshold=threshold,
        activations=10,
        false_rejects=0,
        false_rejects_per_activation=fr_per_act,
        false_accepts=0,
        nonwake_hours=1.0,
        false_accepts_per_hour=fa_per_hour,
    )


def test_threshold_grid() -> None:
    print("=== threshold_grid: the sweep points ===")
    g = wake_eval.threshold_grid(0.1, 0.9, 0.05)
    check("inclusive endpoints", g[0] == 0.1 and g[-1] == 0.9, f"{g[0]}..{g[-1]}")
    check("length is span/step + 1", len(g) == 17, f"len={len(g)}")
    check("no floating-point dust", 0.35 in g and 0.85 in g, f"has 0.35/0.85: {g}")

    quarters = wake_eval.threshold_grid(0.0, 1.0, 0.25)
    check("quarter grid", quarters == [0.0, 0.25, 0.5, 0.75, 1.0], f"{quarters}")

    uneven = wake_eval.threshold_grid(0.1, 0.83, 0.05)
    check(
        "uneven step never exceeds the requested max",
        uneven[-1] == 0.8,
        f"last={uneven[-1]}",
    )

    for bad, why in (
        ((0.1, 0.9, 0.0), "zero step"),
        ((0.1, 0.9, -0.1), "negative step"),
        ((0.9, 0.1, 0.05), "inverted range"),
    ):
        try:
            wake_eval.threshold_grid(*bad)
            check(f"{why} raises", False, "no error raised")
        except ValueError:
            check(f"{why} raises", True, "ValueError")


def test_count_false_accepts() -> None:
    print("=== count_false_accepts: refractory fire counting ===")
    check(
        "no crossing is no fire",
        wake_eval.count_false_accepts([0.1, 0.2, 0.1], 0.5, 3) == 0,
        "0",
    )
    check(
        "gap of 1 counts every high frame",
        wake_eval.count_false_accepts([0.9, 0.9, 0.9], 0.5, 1) == 3,
        "3",
    )
    fires = wake_eval.count_false_accepts([0.9, 0.9, 0.9, 0.9, 0.9], 0.5, 3)
    check(
        "a sustained run counts once per refractory window",
        fires == 2,
        f"5 highs, gap 3 -> {fires}",
    )
    check(
        "separated fires each count",
        wake_eval.count_false_accepts([0.1, 0.9, 0.1, 0.9], 0.5, 1) == 2,
        "2",
    )


def test_sweep_false_rejects() -> None:
    print("=== sweep: false-rejects per activation ===")
    positives = [Activation(0.2), Activation(0.4), Activation(0.6), Activation(0.8)]
    rows = wake_eval.sweep(positives, [], [0.5])
    r = rows[0]
    check(
        "half the activations fall below 0.5",
        r.false_rejects == 2 and r.false_rejects_per_activation == 0.5,
        f"rejects={r.false_rejects} rate={r.false_rejects_per_activation}",
    )

    rows_hi = wake_eval.sweep(positives, [], [0.9])
    check(
        "all reject above every peak",
        rows_hi[0].false_rejects_per_activation == 1.0,
        f"rate={rows_hi[0].false_rejects_per_activation}",
    )


def test_sweep_false_accepts_per_hour() -> None:
    print("=== sweep: false-accepts per hour ===")
    # One hour of non-wake audio, refractory 1 frame so both highs count.
    clip = NonWakeClip(frame_scores=[0.9, 0.1, 0.9], seconds=3600.0)
    rows = wake_eval.sweep([], [clip], [0.5], refractory_s=0.01, frame_hop_s=0.08)
    check(
        "2 fires over 1 hour is 2.0/hour",
        rows[0].false_accepts == 2 and rows[0].false_accepts_per_hour == 2.0,
        f"fa={rows[0].false_accepts} rate={rows[0].false_accepts_per_hour}",
    )

    empty = wake_eval.sweep([], [], [0.5])
    check(
        "no non-wake audio -> 0.0/hour, not a crash",
        empty[0].false_accepts_per_hour == 0.0,
        f"rate={empty[0].false_accepts_per_hour}",
    )

    zero_dur = NonWakeClip(frame_scores=[0.9], seconds=0.0)
    inf_rows = wake_eval.sweep([], [zero_dur], [0.5])
    check(
        "a fire with zero measured duration is inf, never a safe 0",
        math.isinf(inf_rows[0].false_accepts_per_hour),
        f"rate={inf_rows[0].false_accepts_per_hour}",
    )

    run = NonWakeClip(frame_scores=[0.9, 0.9, 0.9], seconds=3600.0)
    none = wake_eval.sweep([], [run], [0.5], refractory_s=0.0, frame_hop_s=0.08)
    check(
        "refractory 0 suppresses nothing (every crossing counts)",
        none[0].false_accepts == 3,
        f"fa={none[0].false_accepts}",
    )
    try:
        wake_eval.sweep([], [run], [0.5], refractory_s=-1.0)
        check("negative refractory raises", False, "no error")
    except ValueError:
        check("negative refractory raises", True, "ValueError")


def test_latency_tolerance() -> None:
    print("=== sweep: activation-latency tolerance ===")
    late = [Activation(0.9, latency_s=0.5)]
    strict = wake_eval.sweep(late, [], [0.5], latency_tolerance_s=0.3)
    check(
        "a fire later than tolerance is a reject",
        strict[0].false_rejects_per_activation == 1.0,
        f"rate={strict[0].false_rejects_per_activation}",
    )

    lenient = wake_eval.sweep(late, [], [0.5], latency_tolerance_s=1.0)
    check(
        "a fire within tolerance is accepted",
        lenient[0].false_rejects_per_activation == 0.0,
        f"rate={lenient[0].false_rejects_per_activation}",
    )

    unmeasured = wake_eval.sweep(
        [Activation(0.9, latency_s=None)], [], [0.5], latency_tolerance_s=0.3
    )
    check(
        "unmeasured latency is judged on peak alone",
        unmeasured[0].false_rejects_per_activation == 0.0,
        f"rate={unmeasured[0].false_rejects_per_activation}",
    )


def test_recommend() -> None:
    print("=== recommend: budget and tie-break ===")
    rows = [_row(0.3, 0.0, 10.0), _row(0.5, 0.1, 0.5), _row(0.7, 0.4, 0.0)]
    rec = wake_eval.recommend(rows, max_false_accepts_per_hour=1.0)
    check(
        "within budget, pick lowest false-reject",
        rec.threshold == 0.5 and rec.meets_budget,
        f"{rec.threshold} {rec.reason}",
    )

    tight = wake_eval.recommend(rows, max_false_accepts_per_hour=0.1)
    check(
        "tighter budget forces the higher threshold",
        tight.threshold == 0.7 and tight.meets_budget,
        f"{tight.threshold}",
    )

    ties = [_row(0.6, 0.2, 0.0), _row(0.8, 0.2, 0.0)]
    tie = wake_eval.recommend(ties, max_false_accepts_per_hour=1.0)
    check(
        "equal false-rejects tie-break to the lower threshold",
        tie.threshold == 0.6,
        f"{tie.threshold}",
    )

    over = [_row(0.3, 0.0, 5.0), _row(0.5, 0.1, 3.0), _row(0.7, 0.4, 9.0)]
    miss = wake_eval.recommend(over, max_false_accepts_per_hour=0.0)
    check(
        "nothing in budget -> flag it and give the lowest false-accept rate",
        miss.threshold == 0.5 and not miss.meets_budget,
        f"{miss.threshold} {miss.reason}",
    )

    try:
        wake_eval.recommend([], max_false_accepts_per_hour=1.0)
        check("empty rows raises", False, "no error")
    except ValueError:
        check("empty rows raises", True, "ValueError")


def main() -> int:
    test_threshold_grid()
    test_count_false_accepts()
    test_sweep_false_rejects()
    test_sweep_false_accepts_per_hour()
    test_latency_tolerance()
    test_recommend()

    failed = [name for verdict, name in results if verdict == FAIL]
    total = len(results)
    print(f"\n{total - len(failed)}/{total} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
