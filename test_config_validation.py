#!/usr/bin/env python3
"""Config value validation: fail clearly on bad values, fall back to defaults (#21).

load_config merges DEFAULTS, config.json, and config.local.json and used to guard
only against invalid JSON, so a bad value (a wake_threshold above 1, an unknown
whisper_compute, a zero timeout, an unknown wake_word) surfaced deep in detect_wake,
_get_whisper, or speak rather than at load. validate_config now catches those at load,
prints a clear stderr message naming the key, and falls the key back to its default.

These checks are model-free and deterministic: they call validate_config / load_config
directly and monkeypatch the wake-model library, so they load no wake or whisper model.

Run:  .venv/bin/python test_config_validation.py
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import pipeline
from pipeline import DEFAULTS

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    results.append((PASS if ok else FAIL, name))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")
    return ok


def validate(cfg: dict) -> tuple[dict, str]:
    """Run validate_config on a copy, returning (validated_cfg, stderr_text)."""
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        out = pipeline.validate_config(dict(cfg))
    return out, buf.getvalue()


# --- unit-interval floats ---------------------------------------------------- #

def test_wake_threshold_out_of_range() -> None:
    for bad in (1.5, -0.1, 2, "high", True):
        cfg, err = validate({"wake_threshold": bad})
        check(
            f"wake_threshold {bad!r} -> default",
            cfg["wake_threshold"] == DEFAULTS["wake_threshold"] and "wake_threshold" in err,
            f"got {cfg['wake_threshold']!r}, stderr names key: {'wake_threshold' in err}",
        )


def test_wake_threshold_valid_untouched() -> None:
    for good in (0.0, 0.42, 1.0):
        cfg, err = validate({"wake_threshold": good})
        check(
            f"wake_threshold {good!r} kept",
            cfg["wake_threshold"] == good and err == "",
            f"got {cfg['wake_threshold']!r}, no warning: {err == ''}",
        )


def test_capture_vad_threshold_sibling() -> None:
    cfg, err = validate({"capture_vad_threshold": 9.0})
    check(
        "capture_vad_threshold 9.0 -> default",
        cfg["capture_vad_threshold"] == DEFAULTS["capture_vad_threshold"]
        and "capture_vad_threshold" in err,
        f"got {cfg['capture_vad_threshold']!r}",
    )


def test_stt_max_no_speech_prob_sibling() -> None:
    # Same [0, 1] rule as the two thresholds: out of range falls back, in range is kept.
    cfg, err = validate({"stt_max_no_speech_prob": 1.5})
    check(
        "stt_max_no_speech_prob 1.5 -> default",
        cfg["stt_max_no_speech_prob"] == DEFAULTS["stt_max_no_speech_prob"]
        and "stt_max_no_speech_prob" in err,
        f"got {cfg['stt_max_no_speech_prob']!r}",
    )
    cfg, err = validate({"stt_max_no_speech_prob": 0.6})
    check(
        "stt_max_no_speech_prob 0.6 kept",
        cfg["stt_max_no_speech_prob"] == 0.6 and err == "",
        f"got {cfg['stt_max_no_speech_prob']!r}, no warning: {err == ''}",
    )


# --- positive-int timeouts --------------------------------------------------- #

def test_timeout_bad_values() -> None:
    for bad in (0, -5, 1.5, True, "60"):
        cfg, err = validate({"claude_timeout_s": bad})
        check(
            f"claude_timeout_s {bad!r} -> default",
            cfg["claude_timeout_s"] == DEFAULTS["claude_timeout_s"]
            and "claude_timeout_s" in err,
            f"got {cfg['claude_timeout_s']!r}",
        )


def test_timeout_valid_untouched() -> None:
    cfg, err = validate({"claude_timeout_s": 90})
    check(
        "claude_timeout_s 90 kept",
        cfg["claude_timeout_s"] == 90 and err == "",
        f"got {cfg['claude_timeout_s']!r}, no warning: {err == ''}",
    )


# --- whisper_compute enum ---------------------------------------------------- #

def test_whisper_compute_unknown() -> None:
    cfg, err = validate({"whisper_compute": "int9"})
    check(
        "whisper_compute 'int9' -> default",
        cfg["whisper_compute"] == DEFAULTS["whisper_compute"] and "whisper_compute" in err,
        f"got {cfg['whisper_compute']!r}",
    )


def test_whisper_compute_valid_untouched() -> None:
    # "default"/"auto" are always accepted (they let CTranslate2 choose), so they are the
    # host-independent valid examples: a concrete type like float16 is CPU-dependent.
    for good in ("default", "auto"):
        cfg, err = validate({"whisper_compute": good})
        check(
            f"whisper_compute {good!r} kept",
            cfg["whisper_compute"] == good and err == "",
            f"got {cfg['whisper_compute']!r}, no warning: {err == ''}",
        )


def test_whisper_compute_unsupported_backend_type_falls_back() -> None:
    # _get_whisper forces device="cpu" and CTranslate2 raises on a real type name the CPU
    # backend cannot load (float16/bfloat16/int16 on a Pi). validate_config must reject
    # those, not just typos. Host-independent: skip a type this box happens to support.
    supported = pipeline._supported_whisper_compute_types()
    if not supported:
        check("whisper_compute deferred when support is unknown", True,
              "ctranslate2 unavailable; nothing to validate against")
        return
    unsupported = next(
        (t for t in ("float16", "bfloat16", "int16", "int8_float16", "int8_bfloat16")
         if t not in supported),
        None,
    )
    if unsupported is None:
        check("whisper_compute CPU-unsupported path", True,
              "this backend supports all candidate types; skipped")
        return
    cfg, err = validate({"whisper_compute": unsupported})
    check(
        f"CPU-unsupported whisper_compute {unsupported!r} -> default",
        cfg["whisper_compute"] == DEFAULTS["whisper_compute"] and "whisper_compute" in err,
        f"got {cfg['whisper_compute']!r}, stderr names key: {'whisper_compute' in err}",
    )


def test_whisper_compute_non_string_falls_back() -> None:
    # A JSON array/object is unhashable, so a naive `x not in set` would raise TypeError
    # instead of falling back; the string guard has to run first.
    for bad in ([], {}, 5):
        cfg, err = validate({"whisper_compute": bad})
        check(
            f"whisper_compute {bad!r} -> default (non-string)",
            cfg["whisper_compute"] == DEFAULTS["whisper_compute"] and "whisper_compute" in err,
            f"got {cfg['whisper_compute']!r}",
        )


# --- wake_word resolution ---------------------------------------------------- #

def test_wake_word_unknown_falls_back(monkeypatch_lib) -> None:
    monkeypatch_lib({"hey_jarvis": "/x", "alexa": "/y"})
    cfg, err = validate({"wake_word": "not_a_real_model"})
    check(
        "unknown wake_word -> default (lib non-empty)",
        cfg["wake_word"] == DEFAULTS["wake_word"] and "wake_word" in err,
        f"got {cfg['wake_word']!r}, stderr names key: {'wake_word' in err}",
    )


def test_wake_word_known_kept(monkeypatch_lib) -> None:
    monkeypatch_lib({"hey_jarvis": "/x", "alexa": "/y"})
    cfg, err = validate({"wake_word": "alexa"})
    check(
        "known wake_word kept",
        cfg["wake_word"] == "alexa" and err == "",
        f"got {cfg['wake_word']!r}, no warning: {err == ''}",
    )


def test_wake_word_empty_lib_deferred(monkeypatch_lib) -> None:
    # A fresh clone with no installed models cannot validate the word; load must not
    # rewrite it or warn, leaving detect_wake to surface the real error at use.
    monkeypatch_lib({})
    cfg, err = validate({"wake_word": "whatever_custom"})
    check(
        "empty lib defers wake_word (no rewrite, no warning)",
        cfg["wake_word"] == "whatever_custom" and err == "",
        f"got {cfg['wake_word']!r}, no warning: {err == ''}",
    )


def test_wake_word_non_string_falls_back(monkeypatch_lib) -> None:
    # A non-string can never name a model, so it falls back regardless of the library, and
    # the unhashable array/object case must not reach the membership lookup and raise.
    monkeypatch_lib({"hey_jarvis": "/x", "alexa": "/y"})
    for bad in ([], {}, 7):
        cfg, err = validate({"wake_word": bad})
        check(
            f"wake_word {bad!r} -> default (non-string)",
            cfg["wake_word"] == DEFAULTS["wake_word"] and "wake_word" in err,
            f"got {cfg['wake_word']!r}",
        )


# --- integration: load_config wires validation ------------------------------- #

def test_load_config_integration(tmp_dir) -> None:
    bad = tmp_dir / "config.json"
    bad.write_text(json.dumps({"wake_threshold": 5, "claude_timeout_s": 0}))
    orig_cfg, orig_local = pipeline.CONFIG_PATH, pipeline.LOCAL_CONFIG_PATH
    pipeline.CONFIG_PATH = bad
    pipeline.LOCAL_CONFIG_PATH = tmp_dir / "config.local.json"  # absent
    try:
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            cfg = pipeline.load_config()
        err = buf.getvalue()
    finally:
        pipeline.CONFIG_PATH, pipeline.LOCAL_CONFIG_PATH = orig_cfg, orig_local
    check(
        "load_config falls both bad keys back to defaults",
        cfg["wake_threshold"] == DEFAULTS["wake_threshold"]
        and cfg["claude_timeout_s"] == DEFAULTS["claude_timeout_s"]
        and "wake_threshold" in err and "claude_timeout_s" in err,
        f"wake_threshold={cfg['wake_threshold']!r}, claude_timeout_s={cfg['claude_timeout_s']!r}",
    )


def main() -> int:
    saved = pipeline.available_wake_models

    def monkeypatch_lib(lib: dict) -> None:
        pipeline.available_wake_models = lambda: lib

    import tempfile
    try:
        test_wake_threshold_out_of_range()
        test_wake_threshold_valid_untouched()
        test_capture_vad_threshold_sibling()
        test_stt_max_no_speech_prob_sibling()
        test_timeout_bad_values()
        test_timeout_valid_untouched()
        test_whisper_compute_unknown()
        test_whisper_compute_valid_untouched()
        test_whisper_compute_unsupported_backend_type_falls_back()
        test_whisper_compute_non_string_falls_back()
        test_wake_word_unknown_falls_back(monkeypatch_lib)
        test_wake_word_known_kept(monkeypatch_lib)
        test_wake_word_empty_lib_deferred(monkeypatch_lib)
        test_wake_word_non_string_falls_back(monkeypatch_lib)
        with tempfile.TemporaryDirectory() as d:
            test_load_config_integration(Path(d))
    finally:
        pipeline.available_wake_models = saved

    failed = [n for r, n in results if r == FAIL]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
