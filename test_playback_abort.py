#!/usr/bin/env python3
"""Barge-in abort core: playback either runs to completion or is aborted (issue #16).

wait_or_abort is the pure poll loop that makes playback abortable. These checks
inject the four side effects (is_active, should_stop, stop, sleep) so they run with
no audio device and no numpy, which is the point of splitting the primitive out.
They pin the two outcomes and the two failure classes that matter on this hardware:
a dropped sleep would hot-spin a core, and a missed abort would let a long reply run
past the user.

Run:  python test_playback_abort.py
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import sys

from playback_abort import wait_or_abort

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail):
    results.append((PASS if ok else FAIL, name))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")
    return ok


def bounded(values):
    """An is_active() that yields each value in turn, then stays False.

    Bounded so a mutant that never aborts still terminates instead of hanging the
    test process.
    """
    it = iter(values)

    def is_active():
        return next(it, False)

    return is_active


def capped_sleep(cap):
    """A sleep(ms) that records its calls and raises past a cap.

    The cap turns a runaway loop (for example a `while True` mutant) into a failed
    run instead of a hung process.
    """
    calls = []

    def sleep(ms):
        calls.append(ms)
        if len(calls) > cap:
            raise AssertionError(f"sleep called more than {cap} times (runaway loop)")

    return sleep, calls


def main() -> int:
    # 1. Plays to completion: is_active goes True three times then False, the user
    # never barges in, so the loop returns True, never aborts, and paces with sleep.
    stops = []
    sleep, sleep_calls = capped_sleep(50)
    finished = wait_or_abort(
        is_active=bounded([True, True, True]),
        should_stop=lambda: False,
        stop=lambda: stops.append(1),
        sleep=sleep,
        poll_ms=50,
    )
    check("plays to completion returns True", finished is True, f"returned {finished}")
    check("completion never aborts the stream", stops == [], f"stop calls={len(stops)}")
    check(
        "completion paces with sleep(poll_ms)",
        sleep_calls == [50, 50, 50],
        f"sleep calls={sleep_calls}",
    )

    # 2. Aborts on barge-in: is_active stays True well past the barge-in point
    # (bounded so a no-abort mutant terminates and fails), should_stop fires on the
    # third poll, so the loop stops the stream exactly once and returns False.
    stops = []
    sleep, sleep_calls = capped_sleep(50)
    checks = iter([False, False, True])
    aborted = wait_or_abort(
        is_active=bounded([True] * 10),
        should_stop=lambda: next(checks, True),
        stop=lambda: stops.append(1),
        sleep=sleep,
        poll_ms=50,
    )
    check("barge-in returns False", aborted is False, f"returned {aborted}")
    check(
        "barge-in aborts the stream exactly once",
        stops == [1],
        f"stop calls={len(stops)}",
    )

    # 3. Already finished before the first poll: is_active is False from the start,
    # so the loop returns True and never consults should_stop (a reply that ended on
    # its own is not reported as interrupted).
    consulted = []
    finished = wait_or_abort(
        is_active=bounded([]),
        should_stop=lambda: consulted.append(1) or False,
        stop=lambda: None,
        sleep=lambda ms: None,
        poll_ms=50,
    )
    check("already-finished returns True", finished is True, f"returned {finished}")
    check(
        "already-finished never consults should_stop",
        consulted == [],
        f"should_stop calls={len(consulted)}",
    )

    n_pass = sum(1 for x in results if x[0] == PASS)
    n_total = len(results)
    print(f"\n=== {n_pass}/{n_total} checks passed ===")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
