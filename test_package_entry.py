#!/usr/bin/env python3
"""Fast checks for the installed ``computah`` command contract (#23).

Run:  .venv/bin/python test_package_entry.py
Exit code is 0 only if the package metadata and live-loop dispatch agree.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pipeline

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str]] = []
PROJECT_DIR = Path(__file__).resolve().parent


def check(name: str, ok: bool, detail: str) -> None:
    results.append((PASS if ok else FAIL, name))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")


def main() -> int:
    metadata = tomllib.loads((PROJECT_DIR / "pyproject.toml").read_text())
    entry_point = metadata["project"]["scripts"].get("computah")
    check(
        "package exposes the computah command",
        entry_point == "pipeline:_cli",
        f"entry point={entry_point!r}",
    )

    called: dict[str, object] = {}
    original_run_loop = pipeline.run_loop
    original_argv = sys.argv

    def record_run_loop(**kwargs: object) -> None:
        called.update(kwargs)

    pipeline.run_loop = record_run_loop
    sys.argv = [
        "computah",
        "--listen",
        "--wake-word",
        "hey_jarvis",
        "--mic",
        "USB mic",
        "--speaker",
        "USB speaker",
    ]
    try:
        exit_code = pipeline._cli()
    finally:
        pipeline.run_loop = original_run_loop
        sys.argv = original_argv

    expected = {
        "wake_word": "hey_jarvis",
        "mic_name": "USB mic",
        "output_name": "USB speaker",
    }
    check(
        "computah --listen dispatches to the live loop",
        exit_code == 0 and called == expected,
        f"exit={exit_code}, arguments={called}",
    )

    failed = [name for status, name in results if status == FAIL]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("Failed: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
