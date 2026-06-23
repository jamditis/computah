#!/usr/bin/env python3
"""Cross-platform safety checks for the OS-specific seams (pipeline.py, audio.py).

These guard portability and half-duplex defects that pass on a dev box that
happens to have the Linux paths present (for example a C:\\tmp directory) but
break on a clean Windows install:

  1. _brain_cli must run the claude subprocess in a real temp dir for THIS OS
     (tempfile.gettempdir()), not a hardcoded "/tmp".
  2. The claude binary fallback (used only when claude is not on PATH) must be
     platform-appropriate: a POSIX per-user path on POSIX, and not a meaningless
     POSIX path on Windows.
  3. Microphone.flush() must drop both the queued callback blocks and the
     sub-frame remainder buffered inside frames(), so the always-on loop can
     pause/resume the mic without leaking pre-pause audio into the next turn.

The subprocess is monkeypatched and the mic check feeds the queue directly, so no
claude binary is invoked, no audio device is opened, and no models load. Fast, no
network. The flush check needs the audio module (sounddevice/PortAudio); where that
native backend is absent (a headless CI runner) it is skipped, not failed, so the two
remaining checks still gate the suite.

Run:  .venv/bin/python test_cross_platform.py
      .venv\\Scripts\\python.exe test_cross_platform.py   (Windows)
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

import pipeline

# audio.py imports sounddevice, which loads the native PortAudio library at import
# time. On a headless/CI runner without that runtime the import raises (ImportError
# if the package is absent, OSError if the shared library will not load). pipeline.py
# imports audio lazily inside its live loop, so importing pipeline above stays
# hardware-free; only the flush check below needs the backend. Capture the failure
# once here so that one check can be skipped (not failed) when the backend is missing,
# leaving the two genuinely hardware-free checks to gate the suite. The catch is
# narrow on purpose: a real defect in audio.py (e.g. a SyntaxError) still propagates.
try:
    import audio
    _AUDIO_IMPORT_ERROR: Exception | None = None
except (ImportError, OSError) as e:
    audio = None  # type: ignore[assignment]
    _AUDIO_IMPORT_ERROR = e

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[bool] = []
skipped: list[str] = []


def check(name: str, ok: bool, detail: str) -> bool:
    results.append(bool(ok))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")
    return bool(ok)


def skip(name: str, reason: str) -> None:
    """Record a check that could not run for an environmental reason (no PortAudio),
    distinct from a failure. Skips never affect the suite's exit code."""
    skipped.append(name)
    print(f"  [{SKIP}] {name}: {reason}")


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


def test_flush_drops_subframe_remainder() -> None:
    """frames() keeps its sub-frame remainder on the instance so flush() can drop
    it. Without that, a paused/resumed loop would prefix the next turn with stale
    pre-pause samples that flush() (queue-only) could not reach. Device-free: feed
    blocks straight into the queue; never open a real stream."""
    if audio is None:
        skip("Microphone.flush() drops the sub-frame remainder",
             f"PortAudio backend unavailable "
             f"({type(_AUDIO_IMPORT_ERROR).__name__}: {_AUDIO_IMPORT_ERROR})")
        return
    fs = audio.FRAME_SIZE
    mic = audio.Microphone(queue_timeout_s=0.05)
    mic._open_sr = audio.TARGET_SR  # equal to target_sr -> no resample branch
    mic._open_ch = 1
    gen = mic.frames()

    # A block of 1.x frames yields one frame and leaves a sub-frame remainder.
    mic._q.put(np.full((fs + 500, 1), 1000, dtype=np.int16))
    f1 = next(gen)
    check("frames() yields a full frame", f1.shape == (fs,), f"shape={f1.shape}")
    check("a sub-frame remainder is buffered on the instance, not a local",
          mic._buf.size > 0, f"buf={mic._buf.size} samples")

    # flush() must clear the queued blocks AND the buffered remainder.
    mic._q.put(np.zeros((fs, 1), dtype=np.int16))
    mic.flush()
    check("flush drops queued callback blocks", mic._q.empty(), "queue empty")
    check("flush drops the buffered sub-frame remainder", mic._buf.size == 0,
          f"buf={mic._buf.size}")

    # The next frame must be built only from post-flush audio (the half-duplex
    # guarantee): no stale prefix from before the flush.
    fresh = np.full((fs, 1), 5000, dtype=np.int16)
    mic._q.put(fresh)
    f2 = next(gen)
    expected = audio._to_int16(audio._to_mono_float(fresh, 1))
    check("the post-flush frame is fresh audio, not the stale remainder",
          np.array_equal(f2, expected) and not np.array_equal(f2, f1),
          f"f2[0]={int(f2[0])} fresh~={int(expected[0])} stale={int(f1[0])}")


def main() -> int:
    print("=== cross-platform pipeline checks ===")
    for test in (test_brain_cli_tempdir, test_claude_bin_fallback,
                 test_flush_drops_subframe_remainder):
        try:
            test()
        except Exception as e:  # noqa: BLE001 - report, don't abort the suite
            check(f"{test.__name__} ran without error", False,
                  f"{type(e).__name__}: {e}")
    n_pass = sum(1 for r in results if r)
    skip_note = f", {len(skipped)} skipped" if skipped else ""
    print(f"\n=== SUMMARY: {n_pass}/{len(results)} checks passed{skip_note} ===")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
