#!/usr/bin/env python3
"""Cross-platform safety checks for pipeline.py.

These guard portability defects that pass on a dev box that happens to have the
Linux paths present (for example a C:\\tmp directory) but break on a clean
Windows install:

  1. _brain_cli must run the claude subprocess in a real temp dir for THIS OS
     (tempfile.gettempdir()), not a hardcoded "/tmp".
  2. The claude binary fallback (used only when claude is not on PATH) must be
     platform-appropriate: a POSIX per-user path on POSIX, and not a meaningless
     POSIX path on Windows.

The subprocess is monkeypatched, so no claude binary is invoked and no models
load. Fast, no network.

Run:  .venv/bin/python test_cross_platform.py
      .venv\\Scripts\\python.exe test_cross_platform.py   (Windows)
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pipeline

PASS, FAIL = "PASS", "FAIL"
results: list[bool] = []


def check(name: str, ok: bool, detail: str) -> bool:
    results.append(bool(ok))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")
    return bool(ok)


class _FakeCompleted:
    """Stand-in for subprocess.CompletedProcess so _brain_cli sees a clean exit."""
    returncode = 0
    stdout = "ok"
    stderr = ""


def test_brain_cli_tempdir() -> None:
    """_brain_cli should run claude in this OS's real temp dir, not '/tmp'."""
    captured: dict[str, object] = {}
    real_run = pipeline.subprocess.run

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        return _FakeCompleted()

    pipeline.subprocess.run = fake_run
    try:
        pipeline._brain_cli("hello", pipeline.load_config())
    finally:
        pipeline.subprocess.run = real_run

    cwd = captured.get("cwd")
    check("brain cli runs in this OS's temp dir",
          cwd == tempfile.gettempdir(),
          f"cwd={cwd!r} expected={tempfile.gettempdir()!r}")
    check("brain cli temp dir actually exists",
          bool(cwd) and os.path.isdir(str(cwd)),
          f"isdir({cwd!r})={bool(cwd) and os.path.isdir(str(cwd))}")


def test_claude_bin_fallback() -> None:
    """The not-on-PATH fallback must be right for the running OS."""
    real_which = pipeline.shutil.which
    pipeline.shutil.which = lambda _name: None  # simulate claude not on PATH
    try:
        resolved = pipeline._resolve_claude_bin()
    finally:
        pipeline.shutil.which = real_which

    if os.name == "posix":
        expected = str(Path.home() / ".local/bin/claude")
        check("claude fallback is the POSIX per-user path on POSIX",
              resolved == expected, f"{resolved!r} (expected {expected!r})")
    else:
        # On Windows a POSIX path is meaningless; defer to the bare name so the
        # OS resolver (or the graceful FileNotFoundError path) handles absence.
        check("claude fallback is not a POSIX-style path on Windows",
              "/" not in resolved and "\\" not in resolved, f"{resolved!r}")


def main() -> int:
    print("=== cross-platform pipeline checks ===")
    for test in (test_brain_cli_tempdir, test_claude_bin_fallback):
        try:
            test()
        except Exception as e:  # noqa: BLE001 - report, don't abort the suite
            check(f"{test.__name__} ran without error", False,
                  f"{type(e).__name__}: {e}")
    n_pass = sum(1 for r in results if r)
    print(f"\n=== SUMMARY: {n_pass}/{len(results)} checks passed ===")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
