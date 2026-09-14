#!/usr/bin/env python3
"""Hardware-free regressions for nested replies and silent completed captures."""

import logging
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

import pipeline
import live_driver


class ReplyCaptureRegressions(unittest.TestCase):
    def test_nested_emphasis_keeps_content_without_stalling(self):
        for reply, expected in (
            ("*" * 40000 + "x" + "*" * 40000 + " After.", "x After."),
            ("*a " * 12000 + "x" + " a*" * 12000, "a " * 12000 + "x" + " a" * 12000),
            ("_" * 40000 + "x" + "_" * 40000, "x"),
            ("~" * 40000 + "x" + "~" * 40000, "x"),
        ):
            start = time.monotonic()
            self.assertEqual(pipeline._strip_emphasis(reply), expected)
            self.assertLess(time.monotonic() - start, 2.0)

    def test_mixed_nested_emphasis_and_literal_identifier_boundaries(self):
        for text in (
            "_**x**_",
            "__*x*__",
            "**_x_**",
            "_~~x~~_",
            "_**_x_**_",
            "__*_*x*_*__",
            "**~~*~~x~~*~~**",
        ):
            self.assertEqual(pipeline._strip_emphasis(text), "x")
        self.assertEqual(pipeline._strip_emphasis("foo_*bar*_baz"), "foo_*bar*_baz")

    def test_completed_silence_has_a_distinct_reason_and_warning(self):
        for amplitude in (0, 100, 249):
            frame = np.full(1280, amplitude, dtype=np.int16)
            with self.assertLogs(pipeline.WAKE_LOGGER, logging.WARNING) as logs:
                captured = pipeline.capture_request(iter([frame] * 5))
            self.assertEqual(captured.size, 0)
            self.assertEqual(captured.empty_reason, pipeline._EMPTY_ALL_SILENT)
            self.assertIn("mute", logs.output[0])
            self.assertIn("device selection", logs.output[0])

    def test_silent_request_cap_is_not_transcribed(self):
        frame = np.zeros(1280, dtype=np.int16)
        with self.assertLogs(pipeline.WAKE_LOGGER, logging.WARNING):
            captured = pipeline.capture_request(iter([frame] * 100), max_request_ms=400)
        self.assertEqual(captured.empty_reason, pipeline._EMPTY_ALL_SILENT)

    def test_abandoned_wake_keeps_its_existing_recovery_path(self):
        frame = np.zeros(1280, dtype=np.int16)
        with self.assertNoLogs(pipeline.WAKE_LOGGER, logging.WARNING):
            captured = pipeline.capture_request(iter([frame] * 100))
        self.assertEqual(captured.empty_reason, pipeline._EMPTY_NO_ONSET)

    def test_turn_skips_silent_audio_but_checks_detection_history(self):
        frame = np.zeros(1280, dtype=np.int16)
        with (
            patch.object(pipeline, "_resolve_wake_path", return_value="fake.onnx"),
            patch.object(pipeline, "_get_oww_model", return_value=object()),
            patch.object(pipeline, "stream_detect_wake", return_value=0.9),
            patch.object(pipeline, "transcribe_detailed") as transcribe,
            patch.object(
                pipeline, "recover_consumed_command", return_value=None
            ) as recover,
            self.assertLogs(pipeline.WAKE_LOGGER, logging.WARNING),
        ):
            self.assertIsNone(pipeline.run_turn(iter([frame] * 5)))
        transcribe.assert_not_called()
        recover.assert_called_once()

    def test_live_driver_skips_silent_capture_and_keeps_listening(self):
        frame = np.zeros(1280, dtype=np.int16)
        cfg = {**pipeline.DEFAULTS, "wake_chime": False}
        with (
            patch.object(live_driver, "listen_for_wake", return_value=0.9),
            patch.object(pipeline, "transcribe_detailed") as transcribe,
            patch.object(
                pipeline, "recover_consumed_command", return_value=None
            ) as recover,
            self.assertLogs(pipeline.WAKE_LOGGER, logging.WARNING),
        ):
            self.assertTrue(
                live_driver.run_turn(
                    iter([frame] * 5),
                    Mock(),
                    object(),
                    0.5,
                    "/dev/null",
                    None,
                    cfg,
                    False,
                )
            )
        transcribe.assert_not_called()
        recover.assert_called_once()

    def test_completed_silence_preserves_a_recovered_command_in_both_drivers(self):
        frame = np.zeros(1280, dtype=np.int16)
        cfg = {**pipeline.DEFAULTS, "wake_chime": False}
        for live in (False, True):
            with (
                patch.object(pipeline, "_resolve_wake_path", return_value="fake.onnx"),
                patch.object(pipeline, "_get_oww_model", return_value=object()),
                patch.object(pipeline, "stream_detect_wake", return_value=0.9),
                patch.object(live_driver, "listen_for_wake", return_value=0.9),
                patch.object(pipeline, "transcribe_detailed") as transcribe,
                patch.object(
                    pipeline,
                    "recover_consumed_command",
                    return_value=pipeline.Transcript("stop", -0.3, 0.05),
                ),
                patch.object(pipeline, "brain", return_value="Stopped.") as brain,
                patch.object(pipeline, "speak"),
                patch.object(live_driver, "_play_wav"),
                self.assertLogs(pipeline.WAKE_LOGGER, logging.WARNING),
            ):
                if live:
                    live_driver.run_turn(
                        iter([frame] * 5),
                        Mock(),
                        object(),
                        0.5,
                        "/dev/null",
                        None,
                        cfg,
                        False,
                    )
                else:
                    pipeline.run_turn(iter([frame] * 5))
            transcribe.assert_not_called()
            self.assertEqual(brain.call_args.args[0], "stop")

    def test_empty_stream_and_a_transient_are_not_all_silent(self):
        for frames in ([], [np.full(1280, 250, dtype=np.int16)]):
            with self.assertNoLogs(pipeline.WAKE_LOGGER, logging.WARNING):
                captured = pipeline.capture_request(iter(frames))
            self.assertEqual(captured.empty_reason, pipeline._EMPTY_NO_ONSET)


if __name__ == "__main__":
    unittest.main()
