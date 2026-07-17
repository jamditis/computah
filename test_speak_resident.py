#!/usr/bin/env python3
"""Resident-Piper checks for speak() (#12).

speak() must load the Piper voice once and reuse it across calls in one process
(the latency win), keep the missing-voice error path with its download hint, and
fall back to the CLI when the in-process synth fails instead of crashing the loop.
These run hardware-free: a fake `piper` module is injected so no ONNX voice loads
and no audio backend is touched.

Run:  .venv/bin/python test_speak_resident.py
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import sys
import tempfile
import types
import warnings
from pathlib import Path

import pipeline

PASS, FAIL = "PASS", "FAIL"
results: list[bool] = []


def check(name: str, ok: bool, detail: str) -> bool:
    results.append(bool(ok))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")
    return bool(ok)


class _FakeVoice:
    """Writes a tiny valid WAV so speak()'s wave context manager succeeds."""

    def synthesize_wav(self, text, wav_file, **kwargs):
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"\x00\x00" * 16)


def _install_fake_piper() -> tuple[types.ModuleType, list[str]]:
    """Build a fake `piper` module whose PiperVoice.load records each call."""
    calls: list[str] = []
    mod = types.ModuleType("piper")

    class _FakePiperVoice:
        @classmethod
        def load(cls, model_path, config_path=None, **kwargs):
            calls.append(str(model_path))
            return _FakeVoice()

    mod.PiperVoice = _FakePiperVoice  # type: ignore[attr-defined]
    return mod, calls


def _voices_dir(tmp: Path, stem: str = "testvoice") -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / f"{stem}.onnx").write_bytes(b"")  # only existence is checked
    return tmp


def test_voice_loads_once_across_calls() -> None:
    tmp = Path(tempfile.mkdtemp())
    voices = _voices_dir(tmp)
    fake_piper, calls = _install_fake_piper()

    saved_mod = sys.modules.get("piper")
    saved_dir = pipeline.VOICES_DIR
    pipeline.VOICES_DIR = voices
    pipeline._piper_cache.clear()
    sys.modules["piper"] = fake_piper
    try:
        outs = [str(tmp / f"out{i}.wav") for i in range(3)]
        for out in outs:
            pipeline.speak("hello there", out, voice_model="testvoice")
    finally:
        pipeline.VOICES_DIR = saved_dir
        pipeline._piper_cache.clear()
        if saved_mod is not None:
            sys.modules["piper"] = saved_mod
        else:
            sys.modules.pop("piper", None)

    check(
        "piper voice loaded exactly once across 3 speak() calls",
        len(calls) == 1,
        f"load calls={len(calls)}",
    )
    check(
        "every speak() call wrote a non-empty WAV",
        all(Path(o).exists() and Path(o).stat().st_size > 0 for o in outs),
        f"sizes={[Path(o).stat().st_size for o in outs if Path(o).exists()]}",
    )


def test_missing_voice_raises_with_hint() -> None:
    tmp = Path(tempfile.mkdtemp())  # empty: no onnx present
    saved_dir = pipeline.VOICES_DIR
    pipeline.VOICES_DIR = tmp
    raised: Exception | None = None
    try:
        pipeline.speak("hi", str(tmp / "o.wav"), voice_model="nope")
    except FileNotFoundError as e:
        raised = e
    finally:
        pipeline.VOICES_DIR = saved_dir

    check(
        "missing voice raises FileNotFoundError",
        isinstance(raised, FileNotFoundError),
        f"raised={type(raised).__name__ if raised else None}",
    )
    check(
        "missing-voice error keeps the download hint",
        raised is not None and "download" in str(raised),
        f"msg={str(raised)[:70]!r}",
    )


def test_falls_back_to_cli_when_inprocess_fails() -> None:
    tmp = Path(tempfile.mkdtemp())
    voices = _voices_dir(tmp)
    saved_dir = pipeline.VOICES_DIR
    saved_get = pipeline._get_piper
    saved_sub = pipeline._speak_subprocess
    fell_back: list[tuple] = []

    def boom(model_path):
        raise RuntimeError("simulated bad piper build")

    def fake_sub(text, onnx, out_wav_path):
        fell_back.append((text, str(onnx), out_wav_path))
        Path(out_wav_path).write_bytes(b"RIFF")  # pretend the CLI wrote a wav

    pipeline.VOICES_DIR = voices
    pipeline._get_piper = boom  # type: ignore[assignment]
    pipeline._speak_subprocess = fake_sub  # type: ignore[assignment]
    pipeline._piper_cache.clear()
    out = str(tmp / "fb.wav")
    err: Exception | None = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.speak("hello", out, voice_model="testvoice")
    except Exception as e:  # noqa: BLE001 - the point is that it should NOT raise
        err = e
    finally:
        pipeline.VOICES_DIR = saved_dir
        pipeline._get_piper = saved_get  # type: ignore[assignment]
        pipeline._speak_subprocess = saved_sub  # type: ignore[assignment]

    check("in-process failure does not crash speak()", err is None, f"err={err!r}")
    check(
        "speak() fell back to the CLI path exactly once",
        len(fell_back) == 1,
        f"fallback calls={len(fell_back)}",
    )
    check(
        "fallback produced the output wav",
        Path(out).exists(),
        f"exists={Path(out).exists()}",
    )


def test_evicts_and_falls_back_when_synth_fails() -> None:
    """Load succeeds and caches the voice, then synth raises: speak() evicts the
    entry and falls back to the CLI. Unlike the boom-on-load test above, this runs
    the real _get_piper cache and the _piper_cache.pop eviction line, which is the
    exact seam residency introduces."""
    tmp = Path(tempfile.mkdtemp())
    voices = _voices_dir(tmp)
    onnx_key = str(voices / "testvoice.onnx")

    calls: list[str] = []
    mod = types.ModuleType("piper")

    class _BoomVoice:
        def synthesize_wav(self, text, wav_file, **kwargs):
            raise RuntimeError("simulated synth failure after a good load")

    class _FakePiperVoice:
        @classmethod
        def load(cls, model_path, config_path=None, **kwargs):
            calls.append(str(model_path))
            return _BoomVoice()

    mod.PiperVoice = _FakePiperVoice  # type: ignore[attr-defined]

    saved_mod = sys.modules.get("piper")
    saved_dir = pipeline.VOICES_DIR
    saved_sub = pipeline._speak_subprocess
    fell_back: list[tuple] = []

    def fake_sub(text, onnx, out_wav_path):
        fell_back.append((text, str(onnx), out_wav_path))
        Path(out_wav_path).write_bytes(b"RIFF")  # pretend the CLI wrote a wav

    pipeline.VOICES_DIR = voices
    pipeline._speak_subprocess = fake_sub  # type: ignore[assignment]
    pipeline._piper_cache.clear()
    sys.modules["piper"] = mod
    out = str(tmp / "boom.wav")
    err: Exception | None = None
    cached_after = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.speak("hello", out, voice_model="testvoice")
        cached_after = onnx_key in pipeline._piper_cache
    except Exception as e:  # noqa: BLE001 - it must NOT raise
        err = e
    finally:
        pipeline.VOICES_DIR = saved_dir
        pipeline._speak_subprocess = saved_sub  # type: ignore[assignment]
        pipeline._piper_cache.clear()
        if saved_mod is not None:
            sys.modules["piper"] = saved_mod
        else:
            sys.modules.pop("piper", None)

    check(
        "the voice loaded once via the real cache before synth failed",
        len(calls) == 1,
        f"load calls={len(calls)}",
    )
    check(
        "synth failure after load does not crash speak()", err is None, f"err={err!r}"
    )
    check(
        "the failed voice entry was evicted from the cache",
        cached_after is False,
        f"cached_after={cached_after}",
    )
    check(
        "speak() fell back to the CLI after the synth failure",
        len(fell_back) == 1,
        f"fallback calls={len(fell_back)}",
    )
    check(
        "fallback produced the output wav",
        Path(out).exists(),
        f"exists={Path(out).exists()}",
    )


def main() -> int:
    print("=== resident-piper speak() checks ===")
    for test in (
        test_voice_loads_once_across_calls,
        test_missing_voice_raises_with_hint,
        test_falls_back_to_cli_when_inprocess_fails,
        test_evicts_and_falls_back_when_synth_fails,
    ):
        try:
            test()
        except Exception as e:  # noqa: BLE001 - report, don't abort the suite
            check(
                f"{test.__name__} ran without error", False, f"{type(e).__name__}: {e}"
            )
    n_pass = sum(1 for r in results if r)
    print(f"\n=== SUMMARY: {n_pass}/{len(results)} checks passed ===")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
