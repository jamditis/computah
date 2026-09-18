#!/usr/bin/env python3
"""Fast, no-model checks for faster-whisper input normalization."""

from __future__ import annotations

import sys
from pathlib import Path

import faster_whisper
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
        self.load_args = None
        self.decode_args = None

    def transcribe(
        self,
        source,
        *,
        beam_size,
        language,
        vad_filter,
        condition_on_previous_text,
    ):
        self.source = source
        self.decode_args = (vad_filter, condition_on_previous_text)
        assert beam_size == 1
        assert language == "en"
        return [], object()


def call(audio, config_overrides=None):
    fake = _FakeWhisper()
    real_config, real_whisper = pipeline.load_config, pipeline._get_whisper
    cfg = {
        "whisper_model": "tiny.en",
        "whisper_compute": "int8",
        "whisper_cpu_threads": 0,
        "whisper_vad_filter": False,
    }
    cfg.update(config_overrides or {})
    pipeline.load_config = lambda: cfg

    def load(*args):
        fake.load_args = args
        return fake

    pipeline._get_whisper = load
    try:
        result = pipeline.transcribe_detailed(audio)
    finally:
        pipeline.load_config, pipeline._get_whisper = real_config, real_whisper
    return fake, result


def raises(error, audio) -> bool:
    try:
        call(audio)
    except error:
        return True
    return False


def test_whisper_cache_keys_cpu_threads() -> None:
    constructed: list[tuple] = []
    real_model = faster_whisper.WhisperModel
    real_cache = pipeline._whisper_cache

    class _FakeModel:
        def __init__(self, *args, **kwargs) -> None:
            constructed.append((args, kwargs))

    faster_whisper.WhisperModel = _FakeModel
    pipeline._whisper_cache = {}
    try:
        first = pipeline._get_whisper("tiny.en", "int8", 2)
        repeated = pipeline._get_whisper("tiny.en", "int8", 2)
        changed = pipeline._get_whisper("tiny.en", "int8", 4)
    finally:
        faster_whisper.WhisperModel = real_model
        pipeline._whisper_cache = real_cache

    check(
        "Whisper cache separates CPU thread counts",
        first is repeated
        and changed is not first
        and [call[1]["cpu_threads"] for call in constructed] == [2, 4],
        f"models={len(constructed)} threads={[call[1]['cpu_threads'] for call in constructed]}",
    )


def main() -> int:
    test_whisper_cache_keys_cpu_threads()
    pcm = np.array([-32768, -1, 0, 1, 32767], dtype=np.int16)
    fake, result = call(pcm)
    source = fake.source
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
    fake, _ = call(normalized)
    source = fake.source
    check(
        "normalized floating-point audio is converted to float32",
        source.dtype == np.float32
        and np.array_equal(source, normalized.astype(np.float32)),
        f"dtype={source.dtype} values={source}",
    )

    wav = Path("request.wav")
    fake, _ = call(wav)
    source = fake.source
    check(
        "file-based callers remain compatible",
        source == str(wav),
        f"source={source!r}",
    )

    fake, _ = call(
        pcm,
        {"whisper_cpu_threads": 3, "whisper_vad_filter": True},
    )
    check(
        "Whisper tuning reaches model load and decode",
        fake.load_args == ("tiny.en", "int8", 3) and fake.decode_args == (True, False),
        f"load_args={fake.load_args!r} decode_args={fake.decode_args!r}",
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
