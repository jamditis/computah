#!/usr/bin/env python3
"""Fast, no-model tests for the mishear confidence guard.

The guard stops a misheard or silence-derived transcript from reaching the brain,
so a garbled command never triggers an action. The decision logic
(transcript_confident) and the segment aggregation (_aggregate_segments) are pure
functions over plain values, so this exercises every branch with no microphone, no
Whisper model, and no audio -- like test_brain_bridge.py.

Run:  .venv/bin/python test_confidence_guard.py
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import sys
from typing import NamedTuple

import pipeline
from pipeline import Transcript, transcript_confident

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    results.append((PASS if ok else FAIL, name))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")
    return ok


class Seg(NamedTuple):
    """The faster-whisper segment fields _aggregate_segments reads."""
    text: str
    start: float
    end: float
    avg_logprob: float
    no_speech_prob: float


# Guard thresholds for the tests, fixed here so a tuned deployment default in
# config.json cannot shift what these checks assert.
FLOOR = -1.0   # min avg_logprob
CEIL = 0.6     # max no_speech_prob


def confident(t: Transcript) -> tuple[bool, str]:
    return transcript_confident(t, min_avg_logprob=FLOOR, max_no_speech_prob=CEIL)


def main() -> int:
    print("=== transcript_confident: the gate decision (no model) ===")

    ok, reason = confident(Transcript("file an issue", -0.3, 0.05))
    check("confident transcript passes", ok, f"ok={ok} reason={reason!r}")

    ok, reason = confident(Transcript("", -0.1, 0.01))
    check("empty transcript is rejected", not ok and reason == "empty transcript",
          f"ok={ok} reason={reason!r}")

    ok, reason = confident(Transcript("   ", -0.1, 0.01))
    check("whitespace-only transcript is rejected",
          not ok and reason == "empty transcript", f"ok={ok} reason={reason!r}")

    ok, reason = confident(Transcript("muffled words", -2.5, 0.05))
    check("low avg_logprob is rejected", not ok and "avg_logprob" in reason,
          f"ok={ok} reason={reason!r}")

    ok, reason = confident(Transcript("room tone", -0.2, 0.95))
    check("high no_speech_prob is rejected", not ok and "no_speech_prob" in reason,
          f"ok={ok} reason={reason!r}")

    # no_speech_prob is checked before avg_logprob, so a transcript that fails both
    # reports the silence signal first.
    ok, reason = confident(Transcript("garbled", -5.0, 0.99))
    check("a transcript failing both signals reports no_speech first",
          not ok and "no_speech_prob" in reason, f"ok={ok} reason={reason!r}")

    # Boundary: exactly at the floor/ceiling passes (>= floor, <= ceiling).
    ok, _ = confident(Transcript("edge", FLOOR, CEIL))
    check("values exactly at the floor and ceiling pass", ok, f"ok={ok}")

    ok, _ = confident(Transcript("just under", FLOOR - 0.001, 0.0))
    check("avg_logprob just under the floor is rejected", not ok, f"ok={ok}")

    ok, _ = confident(Transcript("just over", 0.0, CEIL + 0.001))
    check("no_speech_prob just over the ceiling is rejected", not ok, f"ok={ok}")

    print("\n=== _aggregate_segments: collapsing whisper segments (no model) ===")

    # Single segment: metrics pass through unchanged, text is stripped.
    t = pipeline._aggregate_segments([Seg(" hello ", 0.0, 1.0, -0.4, 0.1)])
    check("single segment passes its metrics through",
          t.text == "hello" and abs(t.avg_logprob + 0.4) < 1e-9
          and abs(t.no_speech_prob - 0.1) < 1e-9, f"{t}")

    # Duration-weighted avg_logprob: a long confident segment outweighs a short
    # hesitant one. Weights 3.0 and 1.0 -> (-0.2*3 + -1.0*1)/4 = -0.4.
    t = pipeline._aggregate_segments([
        Seg("the long confident part", 0.0, 3.0, -0.2, 0.05),
        Seg("uh", 3.0, 4.0, -1.0, 0.2),
    ])
    check("avg_logprob is weighted by segment duration",
          abs(t.avg_logprob - (-0.4)) < 1e-9, f"avg_logprob={t.avg_logprob}")

    # no_speech_prob is the max across segments (most conservative).
    check("no_speech_prob takes the max across segments",
          abs(t.no_speech_prob - 0.2) < 1e-9, f"no_speech_prob={t.no_speech_prob}")

    # No segments: empty text with sentinel metrics that fail the guard.
    t = pipeline._aggregate_segments([])
    ok, _ = confident(t)
    check("no segments yields empty text that fails the guard",
          t.text == "" and not ok, f"{t}")

    # Zero-duration segments must not divide by zero (the 1e-3 weight floor).
    t = pipeline._aggregate_segments([
        Seg("a", 1.0, 1.0, -0.5, 0.1),
        Seg("b", 2.0, 2.0, -0.5, 0.1),
    ])
    check("zero-duration segments do not divide by zero",
          abs(t.avg_logprob - (-0.5)) < 1e-9, f"avg_logprob={t.avg_logprob}")

    print("\n=== guard_transcript: the shared gate both live paths use ===")
    on = {"stt_confidence_guard": True, "stt_min_avg_logprob": FLOOR,
          "stt_max_no_speech_prob": CEIL}
    ok, _ = pipeline.guard_transcript(Transcript("go", -0.2, 0.05), on)
    check("guard_transcript passes a confident transcript when enabled", ok,
          f"ok={ok}")

    ok, reason = pipeline.guard_transcript(Transcript("garble", -3.0, 0.05), on)
    check("guard_transcript rejects a low-confidence transcript when enabled",
          not ok and "avg_logprob" in reason, f"ok={ok} reason={reason!r}")

    off = dict(on, stt_confidence_guard=False)
    ok, reason = pipeline.guard_transcript(Transcript("garble", -3.0, 0.99), off)
    check("guard_transcript passes everything when the guard is disabled",
          ok and reason == "guard disabled", f"ok={ok} reason={reason!r}")

    print("\n=== config: the guard is wired into defaults ===")
    for key, want in (("stt_confidence_guard", True),
                      ("stt_min_avg_logprob", -1.0),
                      ("stt_max_no_speech_prob", 0.6)):
        check(f"DEFAULTS carries {key}", pipeline.DEFAULTS.get(key) == want,
              f"{key}={pipeline.DEFAULTS.get(key)!r}")

    cfg = pipeline.load_config()
    check("load_config surfaces the guard keys",
          all(k in cfg for k in ("stt_confidence_guard", "stt_min_avg_logprob",
                                 "stt_max_no_speech_prob")),
          f"guard={cfg.get('stt_confidence_guard')} "
          f"floor={cfg.get('stt_min_avg_logprob')} "
          f"ceil={cfg.get('stt_max_no_speech_prob')}")

    n_pass = sum(1 for r in results if r[0] == PASS)
    n_total = len(results)
    print(f"\n=== {n_pass}/{n_total} checks passed ===")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
