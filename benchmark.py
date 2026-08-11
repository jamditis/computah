#!/usr/bin/env python3
"""Repeatable per-stage latency and memory benchmark for the voice pipeline (issue #27).

The README latency table was one manual measurement. This reproduces it: it runs the
file pipeline N times on a fixed clip with warm models, reports median and p95 per
stage from the `timings_s` the pipeline already returns, and samples peak RSS. Output
is markdown, so the README latency section is updated from real output instead of
hand-edited.

It also times the ssh hop to the brain host separately. That hop is a cost model warmth
cannot remove (#12, #13, #14), and the `brain` stage alone hides it: the stage time is
transport plus however long the assistant took to answer. Nor is it one hop per turn --
brain_bridge.ssh_reply_reader runs `ssh <host> cat` on every reply poll and nothing
multiplexes the connections, so the transport grows with the answer. Measuring
`ssh <host> true` gives the cost of one hop without sending anything into the assistant
session, and the report turns that into the cadence a waiting turn actually pays.

Run it under the documented memory cap:

    systemd-run --user --scope -p MemoryMax=1500M -p MemorySwapMax=0 \\
      .venv/bin/python benchmark.py --runs 20

Exit code is 0 only if every run produced a full set of stage timings.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Report order and display names. run_pipeline emits these keys; anything it adds
# later is still reported, under its raw key, so a new stage cannot vanish from the
# table by being absent from this map.
STAGE_LABELS = {
    "detect_wake": "Wake detection",
    "transcribe": "Speech-to-text",
    "brain": "Brain reply",
    "speak": "Text-to-speech",
    "total": "End-to-end turn",
}

# Below this many runs, nearest-rank p95 is just the slowest sample, which reads as a
# percentile without being one. The report says so rather than quietly implying rigour.
P95_MIN_RUNS = 20


def median(samples: list[float]) -> float:
    """Median of a non-empty sample list."""
    ordered = sorted(samples)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def p95(samples: list[float]) -> float:
    """Nearest-rank 95th percentile: the smallest sample at or above 95% of the set.

    Nearest-rank rather than an interpolating quantile because the samples are
    timings of real runs, and a latency budget wants a number some run actually
    produced. With fewer than P95_MIN_RUNS samples this is the maximum; report_lines
    labels it as such.
    """
    ordered = sorted(samples)
    rank = math.ceil(0.95 * len(ordered))
    return ordered[max(rank, 1) - 1]


def stage_label(key: str) -> str:
    """Display name for a timing key, falling back to the key itself."""
    return STAGE_LABELS.get(key, key)


def ordered_stages(keys: list[str]) -> list[str]:
    """Timing keys in report order: the known stages first, then anything new."""
    known = [k for k in STAGE_LABELS if k in keys]
    return known + sorted(k for k in keys if k not in STAGE_LABELS)


def markdown_table(header: list[str], rows: list[list[str]]) -> list[str]:
    """A markdown table shaped like the README's latency table."""
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _rss_mb(who: int) -> float:
    """ru_maxrss for one rusage target, in MB.

    ru_maxrss is kilobytes on Linux and bytes on macOS; the pipeline runs on Linux,
    and the units are named here so a macOS reading is not silently 1024x off.
    """
    peak = resource.getrusage(who).ru_maxrss
    if sys.platform == "darwin":
        return peak / (1024 * 1024)
    return peak / 1024


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB."""
    return _rss_mb(resource.RUSAGE_SELF)


def peak_child_rss_mb() -> float:
    """Peak RSS of the largest helper process this run waited on, in MB.

    The memory cap in the run command is a cgroup limit, so it counts the ssh probes
    and the piper CLI fallback too, not just this interpreter. Reporting only
    RUSAGE_SELF would understate what the cap is actually holding.
    """
    return _rss_mb(resource.RUSAGE_CHILDREN)


def ssh_hop_samples(host: str, runs: int, connect_timeout_s: int = 5) -> list[float]:
    """Time `ssh <host> true` `runs` times: the transport floor of a brain turn.

    Deliberately a no-op command. Sending a real prompt would inject a turn into the
    persistent assistant session, and the session is somebody's live conversation.
    Returns the successful samples; a failed ssh contributes nothing rather than a
    fabricated number.
    """
    samples: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        completed = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                f"ConnectTimeout={connect_timeout_s}",
                host,
                "true",
            ],
            capture_output=True,
        )
        elapsed = time.perf_counter() - started
        if completed.returncode == 0:
            samples.append(elapsed)
    return samples


def clip_text(wake_word: str) -> str:
    """The fixed utterance the benchmark clip speaks.

    The wake model name is the phrase with underscores for spaces (hey_jarvis ->
    "hey jarvis"), which is the phrasing test_pipeline.py already fires detect_wake on. The question after it is
    short and constant, so the clip's length -- which whisper's time scales with --
    does not drift between runs or between machines.
    """
    return f"{wake_word.replace('_', ' ')}, what is two plus two?"


def ensure_clip(wav_path: str, wake_word: str, synth) -> bool:
    """Make sure `wav_path` exists, synthesizing it with Piper if it does not.

    test_audio/ is gitignored (the tests synthesize their own audio), so a fresh
    checkout has no clip and the documented command would fail on a missing file.
    Returns True if it had to synthesize one.
    """
    if Path(wav_path).exists():
        return False
    Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
    synth(clip_text(wake_word), wav_path)
    return True


def collect(runs: int, wav_path: str, wake_word: str | None) -> dict:
    """Run the pipeline `runs` times on one clip and gather per-stage timings.

    pipeline is imported here, not at module scope, so the pure reporting layer above
    stays importable (and testable) on a machine with no models and no audio stack.
    """
    import pipeline  # noqa: PLC0415 -- deferred on purpose, see docstring

    cfg = pipeline.load_config()
    wake = wake_word or cfg["wake_word"]
    if ensure_clip(wav_path, wake, pipeline.speak):
        print(f'synthesized {wav_path}: "{clip_text(wake)}"')

    warm_started = time.perf_counter()
    warm = pipeline.warm_models(cfg, wake_word=wake_word)
    warm_elapsed = time.perf_counter() - warm_started

    per_stage: dict[str, list[float]] = {}
    misses = 0
    best_miss_score: float | None = None
    for _ in range(runs):
        result = pipeline.run_pipeline(wav_path, wake_word=wake_word)
        if not result["wake_fired"]:
            # Without a wake hit the pipeline returns after one stage, so averaging
            # this run in would report a fast turn that never happened. Keep the
            # closest score: it is the difference between "re-synthesize the clip"
            # and "lower wake_threshold", and the operator cannot tell them apart
            # from a bare "it never fired".
            misses += 1
            score = result["wake_score"]
            if best_miss_score is None or score > best_miss_score:
                best_miss_score = score
            continue
        for key, value in result["timings_s"].items():
            per_stage.setdefault(key, []).append(value)

    return {
        "config": cfg,
        "warm_per_model_s": warm,
        "warm_total_s": warm_elapsed,
        "per_stage": per_stage,
        "runs_requested": runs,
        "runs_measured": runs - misses,
        "wake_misses": misses,
        "best_miss_score": best_miss_score,
        "wake_threshold": cfg["wake_threshold"],
    }


def _miss_hint(collected: dict) -> str:
    """What to do about a clip that did not wake the pipeline."""
    best = collected.get("best_miss_score")
    threshold = collected.get("wake_threshold")
    if best is None or threshold is None:
        return "Re-synthesize the clip or check the configured wake word."
    return (
        f"The closest miss scored {best:.2f} against a wake_threshold of "
        f"{threshold:.2f}: re-synthesize the clip if that is far off, lower the "
        "threshold if it is close."
    )


def _transport_lines(transport: dict | None) -> list[str]:
    """The brain-transport section, or a line saying why there isn't one."""
    if transport is None:
        return []
    if transport["transport"] == "misconfigured":
        return [
            "",
            "Brain transport: brain_transport is ssh but brain_host is empty, so there "
            "is no host to probe. A live turn does not reach the brain either: "
            "_brain_bridge answers 'the brain host is not configured' and speaks that "
            "instead, so the brain row above is that refusal, not a reply.",
        ]
    if transport["transport"] != "ssh":
        return [
            "",
            "Brain transport: the assistant runs on this host "
            f"(brain_transport {transport['transport']}), so a turn pays no ssh hop.",
        ]

    samples = transport["samples"]
    if not samples:
        return [
            "",
            f"Brain transport: the brain host did not answer in "
            f"{transport['attempts']} attempt(s), so the hop is unmeasured. The brain "
            "row above still includes it.",
        ]

    hop_n = len(samples)
    hop = median(samples)
    poll_s = transport["poll_s"]
    lines = ["", "Brain transport:", ""]
    # The hop is probed separately from the pipeline runs and failed probes are
    # dropped, so it carries its own n: enough pipeline runs to earn a p95 does not
    # mean enough ssh probes answered to earn one.
    lines += markdown_table(
        ["Hop", "Median", "p95", "Probes"],
        [
            [
                "ssh round trip to the brain host",
                f"{hop:.2f} s",
                f"{p95(samples):.2f} s",
                f"{hop_n} of {transport['attempts']}",
            ]
        ],
    )
    lines += [
        "",
        "That is one hop, not a turn's transport, and it excludes whatever the "
        "assistant session spends thinking: the probe is a no-op command. A turn pays "
        "two hops up front (the pre-send read and the send), then one more for every "
        "reply poll, because brain_bridge.ssh_reply_reader runs `ssh <host> cat` per "
        "read and nothing multiplexes the connections. With brain_poll_s at "
        f"{poll_s} s that is a hop every {poll_s + hop:.2f} s for as long as the "
        "assistant takes to answer, so the transport inside the brain row grows with "
        "the answer rather than being fixed per turn.",
    ]
    if hop_n < P95_MIN_RUNS:
        lines.append(
            f"The hop p95 is the slowest of {hop_n} probe(s), not a percentile."
        )
    return lines


def report_lines(collected: dict, transport: dict | None) -> list[str]:
    """The markdown report: stage table, transport, memory, and the caveats.

    Nothing here prints the brain host. The whole point of the report is that its
    table gets pasted into the committed README, and brain_host comes from the
    gitignored config.local.json precisely so private hostnames stay out of commits.
    """
    per_stage = collected["per_stage"]
    measured = collected["runs_measured"]
    lines = [
        f"Measured over {measured} run(s) of "
        f"{collected['runs_requested']} with warm models.",
        "",
    ]

    if not per_stage:
        # An empty table under a p95 caveat reads like a result. There isn't one.
        lines.append(
            f"No stage timings: none of the {collected['runs_requested']} run(s) "
            "fired the wake word, so the pipeline returned before the first "
            "measured stage. " + _miss_hint(collected)
        )
        return lines

    rows = [
        [
            stage_label(key),
            f"{median(per_stage[key]):.2f} s",
            f"{p95(per_stage[key]):.2f} s",
        ]
        for key in ordered_stages(list(per_stage))
    ]
    lines += markdown_table(["Stage", "Median", "p95"], rows)

    if measured < P95_MIN_RUNS:
        lines += [
            "",
            f"p95 here is the slowest of {measured} run(s), not a percentile. "
            f"Use --runs {P95_MIN_RUNS} or more for a real one.",
        ]
    if collected["wake_misses"]:
        lines += [
            "",
            f"{collected['wake_misses']} run(s) did not fire the wake word and are "
            "excluded. A clip that does not wake the pipeline measures one stage, "
            "not a turn. " + _miss_hint(collected),
        ]

    lines += _transport_lines(transport)

    warm = collected["warm_per_model_s"]
    lines += [
        "",
        f"Peak RSS: {peak_rss_mb():.0f} MB, plus {peak_child_rss_mb():.0f} MB for the "
        "largest helper process (the memory cap is a cgroup limit, so it counts both). "
        f"Model warm-up: {collected['warm_total_s']:.2f} s "
        f"({', '.join(f'{k}={v:.2f}s' for k, v in warm.items()) or 'none loaded'}).",
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs", type=int, default=5, help="pipeline runs to time (default 5)"
    )
    parser.add_argument(
        "--wav",
        default="test_audio/benchmark_clip.wav",
        help="fixed input clip to run each time, synthesized with Piper if absent",
    )
    parser.add_argument(
        "--wake-word", default=None, help="override the config wake word"
    )
    parser.add_argument(
        "--no-ssh", action="store_true", help="skip the ssh transport measurement"
    )
    parser.add_argument("--json", action="store_true", help="emit raw samples as JSON")
    args = parser.parse_args(argv)

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    # warm_models and the clip synth both log to stdout, which would corrupt --json
    # for any consumer piping it. Their output belongs on stderr either way.
    with (
        contextlib.redirect_stdout(sys.stderr)
        if args.json
        else contextlib.nullcontext()
    ):
        collected = collect(args.runs, args.wav, args.wake_word)

    transport = None
    cfg = collected["config"]
    host = cfg.get("brain_host") or ""
    configured = cfg.get("brain_transport")
    if args.no_ssh:
        pass
    elif configured != "ssh":
        transport = {"transport": configured or "local"}
    elif not host:
        # brain_transport ssh with no brain_host is a half-configured bridge. There is
        # nothing to probe, and the brain row is measuring a refusal, so say that
        # rather than reporting an unreachable host that was never named.
        transport = {"transport": "misconfigured"}
    elif shutil.which("ssh") is None:
        print("ssh is not on PATH; skipping the transport measurement", file=sys.stderr)
    else:
        # The host goes to stderr, never into the report: the report is written to be
        # pasted into the committed README, and brain_host is a config.local.json value.
        print(f"probing the brain host {host} over ssh", file=sys.stderr)
        transport = {
            "transport": "ssh",
            "poll_s": cfg.get("brain_poll_s"),
            "attempts": args.runs,
            "samples": ssh_hop_samples(host, args.runs),
        }

    if args.json:
        print(
            json.dumps(
                {
                    "runs_requested": collected["runs_requested"],
                    "runs_measured": collected["runs_measured"],
                    "wake_misses": collected["wake_misses"],
                    "best_miss_score": collected["best_miss_score"],
                    "wake_threshold": collected["wake_threshold"],
                    "per_stage_s": collected["per_stage"],
                    "warm_per_model_s": collected["warm_per_model_s"],
                    "ssh_hop_s": (transport or {}).get("samples", []),
                    "ssh_hop_attempts": (transport or {}).get("attempts", 0),
                    "peak_rss_mb": round(peak_rss_mb(), 1),
                    "peak_child_rss_mb": round(peak_child_rss_mb(), 1),
                },
                indent=2,
            )
        )
    else:
        print("\n".join(report_lines(collected, transport)))

    if collected["runs_measured"] == 0:
        print(
            f"\nNo run produced timings: {args.wav} never fired the wake word.",
            file=sys.stderr,
        )
        return 1
    return 0 if collected["wake_misses"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
