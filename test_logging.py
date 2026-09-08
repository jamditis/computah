#!/usr/bin/env python3
"""Fast checks for live-turn observability and spoken-content retention (#22)."""

from __future__ import annotations

import logging
import sys
import time

import numpy as np

import brain_bridge
import live_driver
import pipeline

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str]] = []


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def check(name: str, ok: bool, detail: str) -> None:
    results.append((PASS if ok else FAIL, name))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")


def test_live_turn_logs_stage_timings_without_content() -> None:
    secret_transcript = "transfer the secret funds"
    secret_reply = "Transfer prepared."
    root = logging.getLogger("computah")
    handler = _ListHandler()
    saved_logger = (list(root.handlers), root.level, root.propagate)
    saved_pipeline = {
        name: getattr(pipeline, name)
        for name in (
            "load_config",
            "_resolve_wake_path",
            "_get_oww_model",
            "stream_detect_wake",
            "capture_request",
            "transcribe_detailed",
            "guard_transcript",
            "brain",
            "speak",
        )
    }
    cfg = dict(pipeline.DEFAULTS)
    cfg["stt_confidence_guard"] = True
    root.handlers = [handler]
    pipeline.configure_logging("INFO")
    pipeline.load_config = lambda: cfg
    pipeline._resolve_wake_path = lambda name: f"/models/{name}.onnx"
    pipeline._get_oww_model = lambda path: object()
    pipeline.stream_detect_wake = lambda *args, **kwargs: 0.91
    pipeline.capture_request = lambda *args, **kwargs: np.ones(160, dtype=np.int16)
    pipeline.transcribe_detailed = lambda audio: pipeline.Transcript(
        secret_transcript, -0.1, 0.1
    )
    pipeline.guard_transcript = lambda heard, config: (True, "")
    pipeline.brain = lambda text: secret_reply
    pipeline.speak = lambda text, path: None
    try:
        result = pipeline.run_turn(iter(()))
    finally:
        for name, value in saved_pipeline.items():
            setattr(pipeline, name, value)
        root.handlers, root.level, root.propagate = saved_logger

    info = [record for record in handler.records if record.levelno == logging.INFO]
    messages = [record.getMessage() for record in info]
    check(
        "all four stage loggers report the turn",
        {record.name for record in info}
        == {"computah.wake", "computah.stt", "computah.brain", "computah.tts"},
        f"loggers={sorted({record.name for record in info})}",
    )
    check(
        "INFO records omit score, transcript, and reply content",
        all(
            "0.9100" not in message
            and secret_transcript not in message
            and secret_reply not in message
            for message in messages
        ),
        f"messages={messages}",
    )
    expected = {"detect_wake", "capture", "transcribe", "brain", "speak", "total"}
    check(
        "run_turn returns complete non-negative timings",
        result is not None
        and set(result["timings_s"]) == expected
        and all(value >= 0 for value in result["timings_s"].values()),
        f"timings={result['timings_s'] if result else None}",
    )


def test_bridge_failures_are_warnings() -> None:
    root = logging.getLogger("computah")
    handler = _ListHandler()
    saved_logger = (list(root.handlers), root.level, root.propagate)
    root.handlers = [handler]
    pipeline.configure_logging("INFO")

    def failed_send(*args, **kwargs) -> None:
        raise OSError("offline")

    try:
        brain_bridge.brain_via_bridge(
            "private request",
            persona="syl",
            send=failed_send,
            read_reply=lambda: "",
        )
        brain_bridge.brain_via_bridge(
            "another private request",
            persona="syl",
            send=lambda *args, **kwargs: None,
            read_reply=lambda: "",
            timeout_s=0,
        )
    finally:
        root.handlers, root.level, root.propagate = saved_logger

    warnings = [
        record.getMessage()
        for record in handler.records
        if record.levelno == logging.WARNING
    ]
    check(
        "transport failure and timeout each emit a warning",
        len(warnings) == 2
        and any("send failed" in message for message in warnings)
        and any("timed out" in message for message in warnings),
        f"warnings={warnings}",
    )
    check(
        "bridge warnings omit request text",
        all("private request" not in message for message in warnings),
        f"warnings={warnings}",
    )


def test_hardware_turn_uses_shared_logging_without_info_content() -> None:
    secret_transcript = "publish the private recording"
    secret_reply = "Recording prepared."
    root = logging.getLogger("computah")
    handler = _ListHandler()
    saved_logger = (list(root.handlers), root.level, root.propagate)
    saved = (
        live_driver.listen_for_wake,
        live_driver._play_wav,
        pipeline.capture_request,
        pipeline.transcribe_detailed,
        pipeline.guard_transcript,
        pipeline.brain,
        pipeline.speak,
    )
    cfg = dict(pipeline.DEFAULTS)
    root.handlers = [handler]
    pipeline.configure_logging("INFO")
    live_driver.listen_for_wake = lambda *args, **kwargs: 0.91
    live_driver._play_wav = lambda *args, **kwargs: time.sleep(0.05)
    pipeline.capture_request = lambda *args, **kwargs: np.ones(160, dtype=np.int16)
    pipeline.transcribe_detailed = lambda audio: pipeline.Transcript(
        secret_transcript, -0.1, 0.1
    )
    pipeline.guard_transcript = lambda heard, config: (True, "")
    pipeline.brain = lambda text: secret_reply
    pipeline.speak = lambda text, path: None
    try:
        ran = live_driver.run_turn(
            iter(()), object(), object(), 0.5, "/tmp/reply.wav", None, cfg, False
        )
    finally:
        (
            live_driver.listen_for_wake,
            live_driver._play_wav,
            pipeline.capture_request,
            pipeline.transcribe_detailed,
            pipeline.guard_transcript,
            pipeline.brain,
            pipeline.speak,
        ) = saved
        root.handlers, root.level, root.propagate = saved_logger

    info = [record for record in handler.records if record.levelno == logging.INFO]
    messages = [record.getMessage() for record in info]
    check(
        "the hardware path reports all four stage loggers",
        ran
        and {record.name for record in info}
        == {"computah.wake", "computah.stt", "computah.brain", "computah.tts"},
        f"loggers={sorted({record.name for record in info})}",
    )
    check(
        "hardware INFO records omit score, transcript, and reply content",
        all(
            "0.9100" not in message
            and secret_transcript not in message
            and secret_reply not in message
            for message in messages
        ),
        f"messages={messages}",
    )
    expected_stages = {"detect_wake", "capture", "transcribe", "brain", "speak"}
    logged_stages = {
        message.split("stage=", 1)[1].split(" ", 1)[0]
        for message in messages
        if "Stage finished stage=" in message
    }
    speak_duration = next(
        float(message.rsplit("=", 1)[1])
        for message in messages
        if "stage=speak" in message
    )
    total_duration = next(
        float(message.rsplit("=", 1)[1])
        for message in messages
        if message.startswith("Turn finished total_s=")
    )
    check(
        "the hardware path reports every stage timing and the turn total",
        logged_stages == expected_stages
        and any(message.startswith("Turn finished total_s=") for message in messages),
        f"stages={sorted(logged_stages)} messages={messages}",
    )
    check(
        "speak timing excludes output playback while total timing includes it",
        total_duration - speak_duration >= 0.04,
        f"speak_s={speak_duration:.3f} total_s={total_duration:.3f}",
    )


def main() -> int:
    test_live_turn_logs_stage_timings_without_content()
    test_bridge_failures_are_warnings()
    test_hardware_turn_uses_shared_logging_without_info_content()
    failed = [name for status, name in results if status == FAIL]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
