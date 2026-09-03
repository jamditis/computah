#!/usr/bin/env python3
"""Fast, no-model tests for benchmark.py (issue #27).

benchmark.py reports the latency numbers the README quotes, so the thing worth
pinning is that it cannot report a number it did not measure: a run where the wake
word never fired must not average in as a fast turn, an ssh hop that failed must not
appear as a timing, a p95 over five samples must not read as a percentile, and a
stage the pipeline adds later must not fall out of the table by being absent from the
label map.

These import only benchmark, whose model-dependent work is behind a deferred import,
so they run with no models and no audio stack.

Run:  .venv/bin/python test_benchmark.py
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import benchmark

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    results.append((PASS if ok else FAIL, name))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")
    return ok


def _collected(per_stage: dict, *, requested: int, misses: int) -> dict:
    """A collect() result with the fields report_lines reads."""
    return {
        "config": {
            "brain_backend": "cli",
            "brain_transport": "local",
            "brain_host": "",
            "brain_poll_s": 0.5,
        },
        "wav_path": "test_audio/benchmark_clip_hey_jarvis.wav",
        "warm_per_model_s": {"whisper": 1.5},
        "warm_total_s": 1.5,
        "per_stage": per_stage,
        "runs_requested": requested,
        "runs_measured": requested - misses,
        "wake_misses": misses,
        "best_miss_score": 0.31 if misses else None,
        "wake_threshold": 0.5,
    }


def test_statistics() -> None:
    print("\nstatistics")
    check(
        "median of an odd sample",
        benchmark.median([3.0, 1.0, 2.0]) == 2.0,
        "median([3,1,2]) == 2",
    )
    check(
        "median of an even sample",
        benchmark.median([4.0, 1.0, 2.0, 3.0]) == 2.5,
        "median([4,1,2,3]) == 2.5",
    )

    # Nearest-rank p95 of 1..20 is rank ceil(0.95*20) == 19, so the 19th value.
    twenty = [float(n) for n in range(1, 21)]
    check(
        "p95 is nearest-rank, not interpolated",
        benchmark.p95(twenty) == 19.0,
        f"p95(1..20) == {benchmark.p95(twenty)}",
    )
    check(
        "p95 returns a sample that was actually observed",
        benchmark.p95([0.4, 0.9, 0.5]) in {0.4, 0.9, 0.5},
        f"p95 of three samples == {benchmark.p95([0.4, 0.9, 0.5])}",
    )
    check(
        "p95 of one sample is that sample",
        benchmark.p95([2.5]) == 2.5,
        "no index error on a single run",
    )


def test_no_stage_is_silently_dropped() -> None:
    print("\nstage reporting")
    keys = ["speak", "detect_wake", "vad_confirm", "total"]
    order = benchmark.ordered_stages(keys)
    check(
        "known stages report in pipeline order",
        order[:2] == ["detect_wake", "speak"],
        f"order == {order}",
    )
    check(
        "a stage with no label still reports",
        "vad_confirm" in order
        and benchmark.stage_label("vad_confirm") == "vad_confirm",
        "an unlabelled key falls back to itself instead of vanishing",
    )
    check(
        "every reported key appears exactly once",
        sorted(order) == sorted(keys),
        f"{len(order)} keys in, {len(set(order))} unique out",
    )


def test_p95_over_a_small_sample_says_so() -> None:
    print("\nsmall-sample honesty")
    lines = benchmark.report_lines(
        _collected({"total": [1.0, 2.0, 3.0]}, requested=3, misses=0), None
    )
    text = "\n".join(lines)
    check(
        "a sub-threshold p95 is labelled as the slowest run",
        "not a percentile" in text,
        f"threshold is {benchmark.P95_MIN_RUNS} runs",
    )

    many = [float(n) for n in range(benchmark.P95_MIN_RUNS)]
    text_many = "\n".join(
        benchmark.report_lines(
            _collected({"total": many}, requested=benchmark.P95_MIN_RUNS, misses=0),
            None,
        )
    )
    check(
        "at the threshold the caveat is gone",
        "not a percentile" not in text_many,
        f"{benchmark.P95_MIN_RUNS} runs reports p95 plainly",
    )


def test_a_missed_wake_is_excluded_and_named() -> None:
    print("\nruns that measured nothing")
    lines = benchmark.report_lines(
        _collected({"total": [1.0, 1.1]}, requested=5, misses=3), None
    )
    text = "\n".join(lines)
    check(
        "the report counts only the runs that fired",
        "Measured over 2 run(s) of 5" in text,
        text.splitlines()[0],
    )
    check(
        "the missed runs are stated, not dropped quietly",
        "3 run(s) did not fire the wake word" in text,
        "a silent exclusion would overstate the sample",
    )


def test_zero_measured_runs_report_no_table() -> None:
    print("\nnothing measured")
    text = "\n".join(
        benchmark.report_lines(_collected({}, requested=4, misses=4), None)
    )
    check(
        "no timings means no table",
        "| Stage |" not in text and "No stage timings" in text,
        "an empty table under a p95 caveat would read as a result",
    )
    check(
        "no p95 caveat when there is no p95",
        "not a percentile" not in text,
        "the caveat only makes sense next to a number",
    )


def test_a_failed_ssh_hop_is_not_a_timing() -> None:
    print("\nssh transport")
    collected = _collected({"total": [1.0]}, requested=1, misses=0)
    text = "\n".join(
        benchmark.report_lines(
            collected, {"transport": "ssh", "poll_s": 0.5, "attempts": 3, "samples": []}
        )
    )
    check(
        "an unreachable host reports as unmeasured",
        "did not answer" in text and "round trip" not in text,
        "no fabricated number for a hop that never completed",
    )

    text_ok = "\n".join(
        benchmark.report_lines(
            collected,
            {
                "transport": "ssh",
                "poll_s": 0.5,
                "attempts": 5,
                "samples": [0.10, 0.12, 0.11],
            },
        )
    )
    check(
        "a measured hop reports as its own row",
        "ssh round trip to the brain host" in text_ok,
        "the row is there without naming the host",
    )
    check(
        "the private brain host never reaches the pasteable report",
        "pi-secret"
        not in "\n".join(
            benchmark.report_lines(
                collected,
                {
                    "transport": "ssh",
                    "host": "pi-secret",
                    "poll_s": 0.5,
                    "attempts": 1,
                    "samples": [0.1],
                },
            )
        ),
        "brain_host lives in gitignored config.local.json; the report gets committed",
    )
    check(
        "the report separates transport from the brain's answer time",
        "one hop, not a turn's transport" in text_ok and "brain_poll_s" in text_ok,
        "the brain row includes both, so the caveat has to travel with it",
    )
    check(
        "the hop is priced as a cadence, not one per turn",
        "a hop every 0.61 s" in text_ok and "ssh_reply_reader" in text_ok,
        "every reply poll opens its own connection, so transport scales with the answer",
    )
    misconfigured = "\n".join(
        benchmark.report_lines(collected, {"transport": "misconfigured"})
    )
    check(
        "ssh with no host reports the half-configured bridge",
        "brain_host is empty" in misconfigured and "not configured" in misconfigured,
        "the brain row is measuring a refusal, and the report has to say so",
    )
    local = "\n".join(benchmark.report_lines(collected, {"transport": "local"}))
    check(
        "a local brain says so instead of going quiet",
        "no ssh hop" in local,
        "silence would read as an unmeasured hop rather than an absent one",
    )
    check(
        "the hop row carries its own probe count and caveat",
        "3 of 5" in text_ok and "slowest of 3 probe(s)" in text_ok,
        "dropped probes make the hop's n smaller than the run count",
    )

    text_many = "\n".join(
        benchmark.report_lines(
            collected,
            {
                "transport": "ssh",
                "poll_s": 0.5,
                "attempts": benchmark.P95_MIN_RUNS,
                "samples": [0.1] * benchmark.P95_MIN_RUNS,
            },
        )
    )
    check(
        "enough probes drops the hop caveat",
        "probe(s), not a percentile" not in text_many,
        f"{benchmark.P95_MIN_RUNS} answered probes reports p95 plainly",
    )


def test_a_dead_clip_says_how_close_it_got() -> None:
    print("\nmiss diagnostics")
    text = "\n".join(
        benchmark.report_lines(_collected({}, requested=4, misses=4), None)
    )
    check(
        "the closest miss is reported against the threshold",
        "scored 0.31 against a wake_threshold of 0.50" in text,
        "re-synthesize or retune are different fixes, and the score picks one",
    )
    clean = "\n".join(
        benchmark.report_lines(
            _collected({"total": [1.0]}, requested=1, misses=0), None
        )
    )
    check(
        "no miss, no hint",
        "wake_threshold" not in clean,
        "the hint only belongs next to a miss",
    )


def test_the_clip_is_fixed_and_self_supplying() -> None:
    print("\nbenchmark clip")
    check(
        "the clip speaks the configured wake phrase",
        benchmark.clip_text("hey_jarvis").startswith("hey jarvis,"),
        benchmark.clip_text("hey_jarvis"),
    )
    check(
        "the utterance does not vary between runs",
        benchmark.clip_text("alexa") == benchmark.clip_text("alexa"),
        "a clip whose length drifts would move the transcribe timing with it",
    )

    with tempfile.TemporaryDirectory() as tmp:
        target = str(Path(tmp) / "nested" / "clip.wav")
        spoken: list[tuple[str, str]] = []
        made = benchmark.ensure_clip(
            target, "hey_jarvis", lambda text, out: spoken.append((text, out))
        )
        check(
            "a missing clip is synthesized, not a crash",
            made and spoken == [(benchmark.clip_text("hey_jarvis"), target)],
            "test_audio/ is gitignored, so a fresh checkout has no clip",
        )
        Path(target).write_bytes(b"")
        spoken.clear()
        again = benchmark.ensure_clip(
            target, "hey_jarvis", lambda text, out: spoken.append((text, out))
        )
        check(
            "an existing clip is reused untouched",
            not again and spoken == [],
            "re-synthesizing every run would defeat the point of a fixed clip",
        )

    check(
        "each wake word caches its own clip",
        benchmark.default_clip_path("hey_jarvis")
        != benchmark.default_clip_path("alexa"),
        "one shared filename would score the previous phrase against the new model, "
        "and the run would report no timings at all",
    )
    check(
        "the cached clip says which phrase it speaks",
        "alexa" in benchmark.default_clip_path("alexa"),
        benchmark.default_clip_path("alexa"),
    )
    with tempfile.TemporaryDirectory() as elsewhere:
        saved_cwd = os.getcwd()
        try:
            os.chdir(elsewhere)
            anchored = Path(benchmark.default_clip_path("hey_jarvis"))
        finally:
            os.chdir(saved_cwd)
    check(
        "the default clip cache is anchored to the repository",
        anchored.is_absolute()
        and anchored.parent == Path(benchmark.__file__).resolve().parent / "test_audio",
        f"running elsewhere must not create a stray cache at {anchored}",
    )


def test_a_cli_backend_pays_no_ssh_hop() -> None:
    print("\nbackend gates the probe")
    collected = _collected({"total": [1.0]}, requested=1, misses=0)
    text = "\n".join(
        benchmark.report_lines(collected, {"transport": "not-bridge", "backend": "cli"})
    )
    check(
        "a cli backend reports no hop even with an ssh transport configured",
        "brain_backend is cli" in text and "no ssh hop" in text,
        "brain() dispatches on the backend, so brain_transport alone measured nothing",
    )
    check(
        "no hop row is invented for a turn that never used the bridge",
        "round trip" not in text,
        "an ssh row here would price a hop the measured turn did not pay",
    )


def test_a_bridge_brain_is_not_written_to_by_accident() -> None:
    print("\nlive brain guard")
    # A stub module, not the real pipeline: pipeline imports numpy at module scope, and
    # these tests run with no models and no audio stack. collect() resolves `pipeline`
    # through sys.modules at call time, so the stub is what it gets. Every model entry
    # point records instead of running, so the check below can say the guard fired
    # BEFORE any of them rather than only that it fired.
    touched: list[str] = []

    stub = types.ModuleType("pipeline")
    stub.load_config = lambda: {
        "brain_backend": "bridge",
        "brain_reply_path": "/tmp/replies",
        "brain_transport": "local",
        "wake_word": "hey_jarvis",
        "wake_threshold": 0.5,
    }
    stub.warm_models = lambda *a, **k: (touched.append("warm_models"), {})[1]
    # A real-shaped result, not None: with the guard removed this has to run to
    # completion so the check below reports a FAIL, rather than crashing the suite on
    # the way and reporting nothing at all.
    stub.run_pipeline = lambda *a, **k: (
        touched.append("run_pipeline"),
        {"wake_fired": True, "wake_score": 0.9, "timings_s": {"total": 1.0}},
    )[1]
    stub.speak = lambda *a, **k: touched.append("speak")

    saved = sys.modules.get("pipeline")
    sys.modules["pipeline"] = stub
    try:
        raised = ""
        try:
            benchmark.collect(20, None, None)
        except SystemExit as e:
            raised = str(e)
        check(
            "a bridge backend refuses to run without --live-brain",
            "live assistant session" in raised and "20 run(s)" in raised,
            "each run sends the transcript into somebody's live conversation",
        )
        check(
            "it refuses before it touches a model, not after",
            touched == [],
            f"reached {touched or 'nothing'}: a refusal after run_pipeline would "
            "already have written to the session it is protecting",
        )
        check(
            "the refusal says what to pass instead",
            "--live-brain" in raised and "brain_backend to cli" in raised,
            "an operator should not have to read the source to get past a refusal",
        )

        # A bridge that cannot send has no session to protect: _brain_bridge answers
        # 'the brain reply path is not configured' locally and writes nothing, and
        # main() has a report line for exactly that state. Refusing here would take
        # away a measurement without protecting anybody.
        check(
            "a working bridge is what needs consent",
            benchmark.bridge_reaches_a_session(
                {
                    "brain_backend": "bridge",
                    "brain_reply_path": "/tmp/replies",
                    "brain_transport": "local",
                }
            ),
            "a fully configured bridge writes to the session",
        )
        for name, half in (
            ("no reply path", {"brain_backend": "bridge", "brain_reply_path": ""}),
            (
                "ssh with no host",
                {
                    "brain_backend": "bridge",
                    "brain_reply_path": "/tmp/replies",
                    "brain_transport": "ssh",
                    "brain_host": "",
                },
            ),
            (
                "simulation transport",
                {
                    "brain_backend": "bridge",
                    "brain_reply_path": "/tmp/replies",
                    "brain_transport": "sim",
                    "brain_inbox_path": "/tmp/inbox",
                },
            ),
            (
                "unsupported transport",
                {
                    "brain_backend": "bridge",
                    "brain_reply_path": "/tmp/replies",
                    "brain_transport": "carrier-pigeon",
                },
            ),
            ("cli backend", {"brain_backend": "cli", "brain_reply_path": "/tmp/r"}),
        ):
            check(
                f"a half-configured bridge stays measurable: {name}",
                not benchmark.bridge_reaches_a_session(half),
                "the turn never leaves this host, so there is nothing to consent to",
            )
    finally:
        if saved is None:
            del sys.modules["pipeline"]
        else:
            sys.modules["pipeline"] = saved


def test_an_explicit_wav_is_measured_or_refused() -> None:
    """A named sample is the question being asked, so it is never filled in.

    ensure_clip synthesizes anything missing, which is right for the default clip and
    wrong for --wav: a typo would otherwise score the canned utterance and print a
    table that answers a question nobody asked.
    """
    print("\nexplicit --wav")
    stub = types.ModuleType("pipeline")
    stub.load_config = lambda: {
        "brain_backend": "cli",
        "brain_transport": "local",
        "brain_host": "",
        "brain_poll_s": 0.5,
        "wake_word": "hey_jarvis",
        "wake_threshold": 0.5,
    }
    touched: list[str] = []
    stub.warm_models = lambda cfg=None, wake_word=None: (
        touched.append("warm"),
        {"whisper": 1.5},
    )[1]
    stub.run_pipeline = lambda wav, wake_word=None: (
        touched.append("run"),
        {"wake_fired": True, "wake_score": 0.9, "timings_s": {"total": 1.0}},
    )[1]

    saved_mod = sys.modules.get("pipeline")
    saved_cwd = os.getcwd()
    saved_project_dir = benchmark.PROJECT_DIR
    scratch = tempfile.TemporaryDirectory()
    os.chdir(scratch.name)
    benchmark.PROJECT_DIR = Path(scratch.name)
    sys.modules["pipeline"] = stub
    try:
        spoken: list[str] = []
        stub.speak = lambda text, out: (touched.append("speak"), spoken.append(out))[0]

        missing = str(Path(scratch.name) / "not-here.wav")
        refused = None
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                benchmark.collect(1, missing, None)
            except SystemExit as e:
                refused = str(e.code)
        check(
            "a missing --wav refuses instead of synthesizing over it",
            refused is not None and missing in refused and spoken == [],
            f"refusal: {refused!r}",
        )
        check(
            "the refusal names the way forward, not just the fault",
            refused is not None and "Drop --wav" in refused,
            "an operator who typo'd a path needs to know the default is the fallback",
        )
        check(
            "it refuses before it loads a model, not after",
            touched == [],
            f"reached {touched}: on the Pi a typo would otherwise cost a whisper "
            "and Piper load before the operator hears about it",
        )

        with contextlib.redirect_stdout(io.StringIO()):
            collected = benchmark.collect(1, None, None)
        check(
            "the default clip is still synthesized on demand",
            spoken == [benchmark.default_clip_path("hey_jarvis")],
            f"spoke into {spoken}: only the clip we own is filled in",
        )
        check(
            "the result keeps the local checkout path private",
            collected["wav_path"] == "test_audio/benchmark_clip_hey_jarvis.wav",
            f"reported {collected['wav_path']}",
        )
    finally:
        benchmark.PROJECT_DIR = saved_project_dir
        os.chdir(saved_cwd)
        scratch.cleanup()
        if saved_mod is None:
            del sys.modules["pipeline"]
        else:
            sys.modules["pipeline"] = saved_mod


def test_main_decides_the_probe_and_the_refusal() -> None:
    """The decisions, not the rendering: report_lines is given a transport dict by the
    tests above, so nothing there pins main() choosing to build one. These drive main()
    from argv against a deployment-shaped config (bridge over ssh, host set), which is
    the shape both new gates were written for.
    """
    print("\nmain gates the probe")
    probes: list[tuple[str, int]] = []
    order: list[str] = []

    stub = types.ModuleType("pipeline")
    stub.load_config = lambda: {
        "brain_backend": "bridge",
        "brain_reply_path": "/tmp/replies",
        "brain_transport": "ssh",
        "brain_host": "pi-secret",
        "brain_poll_s": 0.5,
        "wake_word": "hey_jarvis",
        "wake_threshold": 0.5,
    }
    stub.warm_models = lambda cfg=None, wake_word=None: (
        order.append("warm"),
        {"whisper": 1.5},
    )[1]
    stub.speak = lambda text, out: order.append("speak")
    stub.run_pipeline = lambda wav, wake_word=None: {
        "wake_fired": True,
        "wake_score": 0.9,
        "timings_s": {"total": 1.0},
    }

    saved_mod = sys.modules.get("pipeline")
    saved_probe = benchmark.ssh_hop_samples
    saved_which = benchmark.shutil.which
    saved_cwd = os.getcwd()
    saved_project_dir = benchmark.PROJECT_DIR
    # This test's "was the clip synthesized after warming" check needs a reliably
    # absent cache without writing a generated clip into the real checkout.
    scratch = tempfile.TemporaryDirectory()
    os.chdir(scratch.name)
    benchmark.PROJECT_DIR = Path(scratch.name)
    sys.modules["pipeline"] = stub
    # The transport decision, not whether this test host happens to ship OpenSSH,
    # is under test. Keep the check portable to a minimal Python environment.
    benchmark.shutil.which = lambda name: "/usr/bin/ssh" if name == "ssh" else None
    benchmark.ssh_hop_samples = lambda host, runs, *a, **k: (
        probes.append((host, runs)),
        [0.1] * runs,
    )[1]

    def run_main(argv: list[str], out: io.StringIO | None = None):
        """main() with SystemExit as a return value, so a refusal is a result here.

        The guard refuses by raising, and a raise that escapes an assertion aborts the
        whole suite instead of failing one check. Catching it is what lets the checks
        below say which call refused and which ran.
        """
        with (
            contextlib.redirect_stdout(out or io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            try:
                return benchmark.main(argv)
            except SystemExit as e:
                return e.code

    try:
        code = run_main(["--runs", "3"])
        check(
            "a bridge config refuses before it probes anything",
            isinstance(code, str) and "--live-brain" in code and probes == [],
            "the refusal has to come first: the probe would follow 3 live turns",
        )

        code = run_main(["--runs", "3", "--live-brain"])
        check(
            "--live-brain lets the run through to the hop probe",
            code == 0 and probes == [("pi-secret", 3)],
            f"probed {probes} once consent was explicit",
        )
        check(
            "the clip is synthesized after the models are warm",
            order[:2] == ["warm", "speak"],
            f"order was {order[:2]}: synthesizing first would pre-load Piper and "
            "report a warm-up it did not measure",
        )

        probes.clear()
        stub.load_config = lambda: {
            "brain_backend": "cli",
            "brain_transport": "ssh",
            "brain_host": "pi-secret",
            "brain_poll_s": 0.5,
            "wake_word": "hey_jarvis",
            "wake_threshold": 0.5,
        }
        out = io.StringIO()
        code = run_main(["--runs", "3"], out)
        check(
            "a cli backend runs freely and probes no host",
            code == 0 and probes == [],
            "brain_transport is stale config here, not a hop this turn paid",
        )
        check(
            "and the report says why there is no hop row",
            "brain_backend is cli" in out.getvalue(),
            "a silently missing row reads as a failed probe",
        )

        probes.clear()
        stub.load_config = lambda: {
            "brain_backend": "bridge",
            "brain_reply_path": "",
            "brain_transport": "ssh",
            "brain_host": "pi-secret",
            "brain_poll_s": 0.5,
            "wake_word": "hey_jarvis",
            "wake_threshold": 0.5,
        }
        out = io.StringIO()
        code = run_main(["--runs", "3"], out)
        check(
            "a missing reply path probes no host",
            code == 0
            and probes == []
            and "brain reply path is not configured" in out.getvalue(),
            "the measured brain row is a local refusal and paid no ssh transport",
        )
    finally:
        benchmark.ssh_hop_samples = saved_probe
        benchmark.shutil.which = saved_which
        benchmark.PROJECT_DIR = saved_project_dir
        os.chdir(saved_cwd)
        scratch.cleanup()
        if saved_mod is None:
            del sys.modules["pipeline"]
        else:
            sys.modules["pipeline"] = saved_mod


def test_json_mode_emits_only_json() -> None:
    print("\njson output")
    saved = benchmark.collect

    def noisy_collect(runs, wav, wake, live_brain=False):
        print("  warm whisper: 1.50s")  # what warm_models writes to stdout
        return _collected({"total": [1.0]}, requested=runs, misses=0)

    out, err = io.StringIO(), io.StringIO()
    try:
        benchmark.collect = noisy_collect
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = benchmark.main(["--runs", "1", "--no-ssh", "--json"])
    finally:
        benchmark.collect = saved

    check(
        "stdout parses as JSON",
        code == 0 and json.loads(out.getvalue())["runs_measured"] == 1,
        "a consumer pipes stdout straight into a parser",
    )
    check(
        "the warm-up log goes to stderr, not into the JSON",
        "warm whisper" in err.getvalue() and "warm whisper" not in out.getvalue(),
        "it is still worth seeing, just not on the data channel",
    )


def test_exit_code_reflects_what_was_measured() -> None:
    print("\nexit code")
    saved = benchmark.collect
    try:
        benchmark.collect = lambda runs, wav, wake, live_brain=False: _collected(
            {}, requested=runs, misses=runs
        )
        check(
            "no measured run exits non-zero",
            benchmark.main(["--runs", "3", "--no-ssh"]) == 1,
            "a benchmark that measured nothing must not look like a pass",
        )
        benchmark.collect = lambda runs, wav, wake, live_brain=False: _collected(
            {"total": [1.0, 1.0]}, requested=runs, misses=1
        )
        check(
            "a partial run exits non-zero",
            benchmark.main(["--runs", "3", "--no-ssh"]) == 1,
            "a clip that fires intermittently is a broken benchmark, not a result",
        )
        benchmark.collect = lambda runs, wav, wake, live_brain=False: _collected(
            {"total": [1.0, 1.0, 1.0]}, requested=runs, misses=0
        )
        check(
            "a full run exits zero",
            benchmark.main(["--runs", "3", "--no-ssh"]) == 0,
            "every requested run produced timings",
        )
    finally:
        benchmark.collect = saved


def main() -> int:
    test_statistics()
    test_no_stage_is_silently_dropped()
    test_p95_over_a_small_sample_says_so()
    test_a_missed_wake_is_excluded_and_named()
    test_zero_measured_runs_report_no_table()
    test_a_failed_ssh_hop_is_not_a_timing()
    test_a_dead_clip_says_how_close_it_got()
    test_the_clip_is_fixed_and_self_supplying()
    test_a_cli_backend_pays_no_ssh_hop()
    test_a_bridge_brain_is_not_written_to_by_accident()
    test_an_explicit_wav_is_measured_or_refused()
    test_main_decides_the_probe_and_the_refusal()
    test_json_mode_emits_only_json()
    test_exit_code_reflects_what_was_measured()

    n_pass = sum(1 for r in results if r[0] == PASS)
    n_total = len(results)
    print(f"\n=== {n_pass}/{n_total} checks passed ===")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
