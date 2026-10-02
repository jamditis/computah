#!/usr/bin/env python3
"""Brain failures stay speakable and never become benchmark answer timings.

Run with Python directly. No models, audio device, assistant CLI, or live inbox.
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import benchmark
import brain_bridge
import live_driver
import pipeline


class BrainFailureTests(unittest.TestCase):
    def test_cli_errors_survive_sanitizing(self):
        cases = (
            (FileNotFoundError(), "cli_unavailable"),
            (OSError(), "cli_unavailable"),
            (subprocess.TimeoutExpired("unused", 1), "cli_timeout"),
            (SimpleNamespace(returncode=1, stdout="", stderr="**failed**"), "cli_exit"),
        )
        for outcome, reason in cases:
            with self.subTest(reason=reason):
                kwargs = (
                    {"side_effect": outcome}
                    if isinstance(outcome, Exception)
                    else {"return_value": outcome}
                )
                with (
                    patch.object(
                        pipeline, "load_config", return_value=pipeline.DEFAULTS
                    ),
                    patch.object(pipeline.subprocess, "run", **kwargs),
                ):
                    reply = pipeline.brain("hello")
                self.assertIsInstance(reply, brain_bridge.BrainFailure)
                self.assertEqual(reply.reason, reason)
                self.assertTrue(reply.startswith("Sorry,"))
                self.assertNotIn("**", reply)

    def test_bridge_configuration_errors_have_reasons(self):
        cases = (
            ({}, "bridge_reply_path"),
            ({"brain_reply_path": "/unused", "brain_transport": "ssh"}, "bridge_host"),
            (
                {"brain_reply_path": "/unused", "brain_transport": "sim"},
                "bridge_inbox_path",
            ),
            (
                {
                    "brain_reply_path": "/unused",
                    "brain_transport": "sim",
                    "brain_inbox_path": [],
                },
                "bridge_inbox_path",
            ),
            (
                {"brain_reply_path": "/unused", "brain_transport": "invalid"},
                "bridge_transport",
            ),
        )
        for cfg, reason in cases:
            with self.subTest(reason=reason):
                reply = brain_bridge.build_brain(cfg)("hello")
                self.assertIsInstance(reply, brain_bridge.BrainFailure)
                self.assertEqual(reply.reason, reason)

    def test_bridge_transport_errors_have_reasons(self):
        for send, extra, reason in (
            (lambda *_a, **_k: None, {}, "bridge_timeout"),
            (
                lambda *_a, **_k: None,
                {"confirm_landing": lambda _id: False, "landing_timeout_s": 0},
                "bridge_not_landed",
            ),
        ):
            with self.subTest(reason=reason):
                reply = brain_bridge.brain_via_bridge(
                    "hello",
                    persona="test",
                    send=send,
                    read_reply=lambda: "",
                    timeout_s=0,
                    **extra,
                )
                self.assertIsInstance(reply, brain_bridge.BrainFailure)
                self.assertEqual(reply.reason, reason)
        reply = brain_bridge.brain_via_bridge(
            "hello",
            persona="test",
            send=Mock(side_effect=OSError()),
            read_reply=lambda: "",
            timeout_s=0,
        )
        self.assertIsInstance(reply, brain_bridge.BrainFailure)
        self.assertEqual(reply.reason, "bridge_send")

    def test_pipeline_speaks_error_and_marks_failure(self):
        with (
            patch.object(pipeline, "load_config", return_value=pipeline.DEFAULTS),
            patch.object(pipeline, "CLAUDE_BIN", "/this/assistant/does/not/exist"),
            patch.object(
                pipeline, "detect_wake", return_value=(True, "hey_jarvis", 0.9)
            ),
            patch.object(
                pipeline,
                "transcribe_detailed",
                return_value=pipeline.Transcript("hello", -0.2, 0.01),
            ),
            patch.object(pipeline, "speak") as speak,
        ):
            result = pipeline.run_pipeline("/unused.wav", "/unused-reply.wav")
        self.assertEqual(result["rejected"], "brain_failure")
        self.assertEqual(result["reject_reason"], "cli_unavailable")
        speak.assert_called_once_with(result["reply"], "/unused-reply.wav")
        self.assertEqual(
            str(result["reply"]), "Sorry, the brain is not available right now."
        )

    def test_answer_with_error_words_is_still_an_answer(self):
        text = "Sorry, the brain is not available right now."
        with (
            patch.object(pipeline, "load_config", return_value=pipeline.DEFAULTS),
            patch.object(
                pipeline.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0, stdout=text, stderr=""),
            ),
        ):
            reply = pipeline.brain("quote the error message")
        self.assertEqual(reply, text)
        self.assertNotIsInstance(reply, brain_bridge.BrainFailure)

    def test_live_turn_plays_error_and_keeps_listening(self):
        cfg = dict(pipeline.DEFAULTS, wake_chime=False)
        with (
            patch.object(pipeline, "load_config", return_value=cfg),
            patch.object(pipeline, "CLAUDE_BIN", "/this/assistant/does/not/exist"),
            patch.object(live_driver, "listen_for_wake", return_value=0.9),
            patch.object(
                pipeline, "capture_request", return_value=SimpleNamespace(size=1)
            ),
            patch.object(
                pipeline,
                "transcribe_detailed",
                return_value=pipeline.Transcript("hello", -0.2, 0.01),
            ),
            patch.object(pipeline, "speak") as speak,
            patch.object(live_driver, "_play_wav") as play,
        ):
            keep_listening = live_driver.run_turn(
                iter(()), None, None, 0.5, "/unused-reply.wav", None, cfg, False
            )
        self.assertTrue(keep_listening)
        self.assertIsInstance(speak.call_args.args[0], brain_bridge.BrainFailure)
        self.assertEqual(speak.call_args.args[0].reason, "cli_unavailable")
        play.assert_called_once_with("/unused-reply.wav", None)

    def test_empty_reply_keeps_spoken_fallback_and_marks_failure(self):
        for text in ("", "   ", "\x00"):
            with (
                self.subTest(text=text),
                patch.object(pipeline, "load_config", return_value=pipeline.DEFAULTS),
                patch.object(
                    pipeline.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0, stdout=text, stderr=""),
                ),
            ):
                reply = pipeline.brain("hello")
            self.assertEqual(reply, pipeline.EMPTY_REPLY_FALLBACK)
            self.assertIsInstance(reply, brain_bridge.BrainFailure)
            self.assertEqual(reply.reason, "empty_reply")

    def collect(self, turns):
        with (
            patch.object(pipeline, "load_config", return_value=pipeline.DEFAULTS),
            patch.object(pipeline, "warm_models", return_value={}),
            patch.object(benchmark, "ensure_clip", return_value=False),
            patch.object(pipeline, "run_pipeline", side_effect=turns),
        ):
            return benchmark.collect(len(turns), None, None)

    def test_mixed_runs_keep_only_answer_timings(self):
        collected = self.collect(
            [
                {"wake_fired": False, "wake_score": 0.2, "timings_s": {"total": 0.01}},
                {
                    "wake_fired": True,
                    "rejected": "brain_failure",
                    "reject_reason": "cli_unavailable",
                    "timings_s": {"brain": 0.01, "total": 0.02},
                },
                {"wake_fired": True, "timings_s": {"brain": 3.0, "total": 4.0}},
                {
                    "wake_fired": True,
                    "rejected": "brain_failure",
                    "reject_reason": "bridge_timeout",
                    "timings_s": {"brain": 120.0, "total": 121.0},
                },
            ]
        )
        self.assertEqual(collected["runs_measured"], 1)
        self.assertEqual(collected["wake_misses"], 1)
        self.assertEqual(
            collected["brain_failures"], {"cli_unavailable": 1, "bridge_timeout": 1}
        )
        self.assertEqual(collected["per_stage"], {"brain": [3.0], "total": [4.0]})
        report = "\n".join(benchmark.report_lines(collected, None))
        self.assertIn("2 run(s) failed in the brain", report)
        self.assertIn("cli_unavailable=1", report)
        with (
            patch.object(benchmark, "collect", return_value=collected),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(benchmark.main(["--runs", "4", "--no-ssh"]), 1)

    def test_all_brain_failures_report_no_answers_and_exit_nonzero(self):
        collected = self.collect(
            [
                {
                    "wake_fired": True,
                    "rejected": "brain_failure",
                    "reject_reason": "cli_unavailable",
                    "timings_s": {"total": 0.02},
                }
            ]
            * 2
        )
        self.assertEqual(collected["runs_measured"], 0)
        self.assertEqual(collected["per_stage"], {})
        report = "\n".join(benchmark.report_lines(collected, None))
        self.assertIn("No stage timings: no run produced a brain answer.", report)
        self.assertNotIn("none of the 2 run(s) fired", report)
        self.assertNotIn("| Stage", report)
        for json_mode in (False, True):
            out, err = io.StringIO(), io.StringIO()
            with (
                patch.object(benchmark, "collect", return_value=collected),
                contextlib.redirect_stdout(out),
                contextlib.redirect_stderr(err),
            ):
                code = benchmark.main(
                    ["--runs", "2", "--no-ssh"] + (["--json"] if json_mode else [])
                )
            self.assertEqual(code, 1)
            self.assertIn("the brain failed", err.getvalue())
            self.assertNotIn("never fired", err.getvalue())
            if json_mode:
                data = json.loads(out.getvalue())
                self.assertEqual(data["brain_failures"], {"cli_unavailable": 2})
                self.assertEqual(data["runs_measured"], 0)


if __name__ == "__main__":
    unittest.main()
