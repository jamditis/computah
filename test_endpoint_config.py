#!/usr/bin/env python3
"""Configurable request endpointing: endpoint_silence_ms and max_request_ms (#15).

capture_request stops a request recording on a trailing-silence window and caps a
runaway with a max-request bound. Those two limits used to be fixed module constants;
now they are config keys so the live loop is tunable without a code change. These
checks are model-free and deterministic: they build frames directly (loud/silent) and
assert the frame math, so they run fast with no wake model and no whisper.

The invariant that keeps the defaults honest: the config defaults, converted to
frames, must equal the built-in constants, so the shipped config leaves capture
behavior exactly as it was before the keys existed.

Run:  .venv/bin/python test_endpoint_config.py
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import live_driver
import pipeline
from pipeline import FRAME_SIZE, Transcript

TEST_DIR = Path(__file__).resolve().parent / "test_audio"
TEST_DIR.mkdir(exist_ok=True)

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    results.append((PASS if ok else FAIL, name))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")
    return ok


class _FakeMic:
    """live_driver.run_turn calls mic.flush() on a chime; a counter is all it needs."""

    def flush(self) -> None:
        pass


def loud(n: int) -> list[np.ndarray]:
    return [np.full(FRAME_SIZE, 4000, dtype=np.int16) for _ in range(n)]


def silent(n: int) -> list[np.ndarray]:
    return [np.zeros(FRAME_SIZE, dtype=np.int16) for _ in range(n)]


def nframes(pcm: np.ndarray) -> int:
    return len(pcm) // FRAME_SIZE


def main() -> int:
    # ----- ms -> frames conversion ---------------------------------------- #
    print("=== _ms_to_frames rounds to whole 80 ms frames, floor 1 ===")
    check(
        "800 ms is 10 frames",
        pipeline._ms_to_frames(800) == 10,
        f"got {pipeline._ms_to_frames(800)}",
    )
    check(
        "8000 ms is 100 frames",
        pipeline._ms_to_frames(8000) == 100,
        f"got {pipeline._ms_to_frames(8000)}",
    )
    check(
        "a sub-frame value floors to 1, never 0 frames",
        pipeline._ms_to_frames(10) == 1,
        f"got {pipeline._ms_to_frames(10)}",
    )

    # ----- defaults must not drift from the built-in constants ------------- #
    # The shipped config has to reproduce the old hardcoded behavior exactly, or the
    # "defaults unchanged" promise is false. Pin the two defaults to the constants.
    print("\n=== config defaults equal the built-in frame constants (no drift) ===")
    check(
        "endpoint_silence_ms default maps to _ENDPOINT_SILENCE_FRAMES",
        pipeline._ms_to_frames(pipeline.DEFAULTS["endpoint_silence_ms"])
        == pipeline._ENDPOINT_SILENCE_FRAMES,
        f"{pipeline.DEFAULTS['endpoint_silence_ms']} ms vs "
        f"{pipeline._ENDPOINT_SILENCE_FRAMES} frames",
    )
    check(
        "max_request_ms default maps to _MAX_REQUEST_FRAMES",
        pipeline._ms_to_frames(pipeline.DEFAULTS["max_request_ms"])
        == pipeline._MAX_REQUEST_FRAMES,
        f"{pipeline.DEFAULTS['max_request_ms']} ms vs "
        f"{pipeline._MAX_REQUEST_FRAMES} frames",
    )

    # ----- None preserves the original behavior ---------------------------- #
    # A caller that passes neither key -- the file-fed tests, and any pre-#15 caller --
    # falls back to the constants, so nothing they see changes.
    print("\n=== None (unset) keeps the built-in endpoint and cap ===")
    base = pipeline.capture_request(iter(loud(5) + silent(15)))
    explicit_none = pipeline.capture_request(
        iter(loud(5) + silent(15)), endpoint_silence_ms=None, max_request_ms=None
    )
    check(
        "no keys captures 5 speech + 10 endpoint = 15 frames",
        nframes(base) == 15,
        f"got {nframes(base)}",
    )
    check(
        "explicit None matches the no-keys default",
        nframes(explicit_none) == 15,
        f"got {nframes(explicit_none)}",
    )

    # ----- a configured endpoint window ends sooner / later ---------------- #
    print("\n=== endpoint_silence_ms tunes the trailing-silence window ===")
    # 400 ms -> 5 frames: after 5 speech frames the same input endpoints 5 frames
    # earlier than the 800 ms default (10 frames), so 5 + 5 = 10 captured.
    short = pipeline.capture_request(
        iter(loud(5) + silent(15)), endpoint_silence_ms=400
    )
    check(
        "a 400 ms window endpoints after 5 quiet frames (10 total)",
        nframes(short) == 10,
        f"got {nframes(short)}, want 10 (5 speech + 5 quiet)",
    )
    # 1200 ms -> 15 frames: needs more trailing quiet, so it holds longer; 5 + 15 = 20.
    long = pipeline.capture_request(
        iter(loud(5) + silent(20)), endpoint_silence_ms=1200
    )
    check(
        "a 1200 ms window holds longer (20 total)",
        nframes(long) == 20,
        f"got {nframes(long)}, want 20 (5 speech + 15 quiet)",
    )

    # ----- a configured cap bounds a runaway sooner ------------------------ #
    print("\n=== max_request_ms bounds a runaway recording ===")
    # Unbroken speech never endpoints, so only the cap stops it. 800 ms -> 10 frames.
    capped = pipeline.capture_request(iter(loud(120)), max_request_ms=800)
    check(
        "a 800 ms cap stops unbroken speech at 10 frames",
        nframes(capped) == 10,
        f"got {nframes(capped)}, want 10",
    )
    check(
        f"the cap is well under the built-in default of {pipeline._MAX_REQUEST_FRAMES}",
        nframes(capped) < pipeline._MAX_REQUEST_FRAMES,
        f"{nframes(capped)} < {pipeline._MAX_REQUEST_FRAMES}",
    )

    # ----- run_turn threads the config values into capture_request --------- #
    # The live loop is only tunable if run_turn actually forwards the config it loads.
    # Both live paths do this, so both are checked: pipeline.run_turn (file/stream loop)
    # and live_driver.run_turn (the arecord hardware loop). Stub the surface each drives
    # and capture what capture_request received.
    print("\n=== pipeline.run_turn forwards cfg endpoint/cap into capture_request ===")
    seen: dict = {}
    real = (
        pipeline.load_config,
        pipeline.stream_detect_wake,
        pipeline.capture_request,
        pipeline.transcribe_detailed,
        pipeline.guard_transcript,
        pipeline.brain,
        pipeline.speak,
        pipeline._get_oww_model,
        pipeline._resolve_wake_path,
    )

    def fake_capture(
        frames,
        preroll=None,
        vad_threshold=None,
        endpoint_silence_ms=None,
        max_request_ms=None,
    ):
        seen["endpoint_silence_ms"] = endpoint_silence_ms
        seen["max_request_ms"] = max_request_ms
        seen["vad_threshold"] = vad_threshold
        return np.full(4 * FRAME_SIZE, 4000, dtype=np.int16)

    pipeline.load_config = lambda: {
        "wake_word": "hey_jarvis",
        "wake_threshold": 0.5,
        "capture_vad_threshold": 0.5,
        "endpoint_silence_ms": 440,
        "max_request_ms": 2000,
    }
    pipeline._resolve_wake_path = lambda name: "x"
    pipeline._get_oww_model = lambda path: object()
    pipeline.stream_detect_wake = lambda frames, model, threshold, preroll=None: 0.9
    pipeline.capture_request = fake_capture
    pipeline.transcribe_detailed = lambda p: pipeline.Transcript(
        "file an issue", 0.0, 0.0
    )
    pipeline.guard_transcript = lambda heard, cfg: (True, "")
    pipeline.brain = lambda t, **_: "done"
    pipeline.speak = lambda text, out, **_: out
    try:
        r = pipeline.run_turn(
            iter(loud(1)), out_wav_path=str(TEST_DIR / "endpoint_reply.wav")
        )
    finally:
        (
            pipeline.load_config,
            pipeline.stream_detect_wake,
            pipeline.capture_request,
            pipeline.transcribe_detailed,
            pipeline.guard_transcript,
            pipeline.brain,
            pipeline.speak,
            pipeline._get_oww_model,
            pipeline._resolve_wake_path,
        ) = real
    check(
        "pipeline.run_turn passes cfg endpoint_silence_ms to capture_request",
        r is not None and seen.get("endpoint_silence_ms") == 440,
        f"capture_request saw endpoint_silence_ms={seen.get('endpoint_silence_ms')} want 440",
    )
    check(
        "pipeline.run_turn passes cfg max_request_ms to capture_request",
        seen.get("max_request_ms") == 2000,
        f"capture_request saw max_request_ms={seen.get('max_request_ms')} want 2000",
    )

    # live_driver.run_turn (the hardware loop) forwards the same two keys. Its wiring is
    # a separate two-line site from pipeline.run_turn's, so the "both paths" claim is
    # only backed if this one is checked too. Stub the live_driver surface and the cfg
    # it reads, and capture what capture_request received.
    print(
        "\n=== live_driver.run_turn forwards cfg endpoint/cap into capture_request ==="
    )
    seen_ld: dict = {}
    real_ld = (
        live_driver.listen_for_wake,
        pipeline.capture_request,
        pipeline.transcribe_detailed,
        pipeline.guard_transcript,
        pipeline.brain,
        pipeline.speak,
        live_driver._play_wav,
    )

    def fake_capture_ld(
        fr,
        preroll=None,
        vad_threshold=None,
        endpoint_silence_ms=None,
        max_request_ms=None,
    ):
        seen_ld["endpoint_silence_ms"] = endpoint_silence_ms
        seen_ld["max_request_ms"] = max_request_ms
        return np.full(4 * FRAME_SIZE, 4000, dtype=np.int16)

    live_driver.listen_for_wake = lambda fr, m, t, d, preroll=None: 0.9
    pipeline.capture_request = fake_capture_ld
    pipeline.transcribe_detailed = lambda p: Transcript("file an issue", -0.2, 0.05)
    pipeline.guard_transcript = lambda heard, cfg: (True, "")
    pipeline.brain = lambda t, **_: "done"
    pipeline.speak = lambda text, out, **_: out
    live_driver._play_wav = lambda path, dev: None
    ld_cfg = {
        "capture_vad_threshold": 0.5,
        "endpoint_silence_ms": 640,
        "max_request_ms": 5000,
    }
    try:
        ld_ran = live_driver.run_turn(
            iter([]), _FakeMic(), object(), 0.5, "/dev/null", None, ld_cfg, False
        )
    finally:
        (
            live_driver.listen_for_wake,
            pipeline.capture_request,
            pipeline.transcribe_detailed,
            pipeline.guard_transcript,
            pipeline.brain,
            pipeline.speak,
            live_driver._play_wav,
        ) = real_ld
    check(
        "live_driver.run_turn passes cfg endpoint_silence_ms to capture_request",
        ld_ran is True and seen_ld.get("endpoint_silence_ms") == 640,
        f"capture_request saw endpoint_silence_ms={seen_ld.get('endpoint_silence_ms')} want 640",
    )
    check(
        "live_driver.run_turn passes cfg max_request_ms to capture_request",
        seen_ld.get("max_request_ms") == 5000,
        f"capture_request saw max_request_ms={seen_ld.get('max_request_ms')} want 5000",
    )

    n_pass = sum(1 for x in results if x[0] == PASS)
    n_total = len(results)
    print(f"\n=== {n_pass}/{n_total} checks passed ===")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
