#!/usr/bin/env python3
"""Fast, no-model checks for faster-whisper input normalization."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import pipeline

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append((PASS if ok else FAIL, name))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")


class _FakeWhisper:
    def __init__(self) -> None:
        self.source = None

    def transcribe(self, source, *, beam_size, language):
        self.source = source
        assert beam_size == 1
        assert language == "en"
        return [], object()


def call(audio):
    fake = _FakeWhisper()
    real_config, real_whisper = pipeline.load_config, pipeline._get_whisper
    pipeline.load_config = lambda: {
        "whisper_model": "tiny.en",
        "whisper_compute": "int8",
    }
    pipeline._get_whisper = lambda *_: fake
    try:
        result = pipeline.transcribe_detailed(audio)
    finally:
        pipeline.load_config, pipeline._get_whisper = real_config, real_whisper
    return fake.source, result


def raises(error, audio) -> bool:
    try:
        call(audio)
    except error:
        return True
    return False


def main() -> int:
    pcm = np.array([-32768, -1, 0, 1, 32767], dtype=np.int16)
    source, result = call(pcm)
    expected = pcm.astype(np.float32) / 32768.0
    check(
        "int16 PCM is normalized to mono float32 without a file hop",
        isinstance(source, np.ndarray)
        and source.dtype == np.float32
        and source.ndim == 1
        and np.array_equal(source, expected),
        f"dtype={getattr(source, 'dtype', None)} values={source}",
    )
    check(
        "an empty model result preserves the conservative transcript signals",
        result == pipeline.Transcript("", float("-inf"), 1.0),
        f"result={result}",
    )

    normalized = np.array([-0.5, 0.0, 0.5], dtype=np.float64)
    source, _ = call(normalized)
    check(
        "normalized floating-point audio is converted to float32",
        source.dtype == np.float32
        and np.array_equal(source, normalized.astype(np.float32)),
        f"dtype={source.dtype} values={source}",
    )

    wav = Path("request.wav")
    source, _ = call(wav)
    check(
        "file-based callers remain compatible",
        source == str(wav),
        f"source={source!r}",
    )

    check(
        "multichannel arrays fail before model transcription",
        raises(ValueError, np.zeros((8, 2), dtype=np.int16)),
        "expected ValueError",
    )
    check(
        "unnormalized floating-point arrays fail before model transcription",
        raises(ValueError, np.array([1.01], dtype=np.float32)),
        "expected ValueError",
    )
    check(
        "unsupported integer arrays fail before model transcription",
        raises(TypeError, np.zeros(8, dtype=np.int32)),
        "expected TypeError",
    )

    n_pass = sum(1 for status, _ in results if status == PASS)
    print(f"\n=== {n_pass}/{len(results)} checks passed ===")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
