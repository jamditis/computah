#!/usr/bin/env python3
"""Fast, no-audio tests for the voice-comparison metrics (#38).

The scoring half of tts_voice_eval is a pure function over timing numbers, so this
exercises it with no piper install, no voice download, and no audio, like
test_wake_eval.py. It pins the ratio the whole comparison rests on, the failure
cases that must not read as a fast voice, and the fact that a voice that failed
to run never reaches the recommendation.

Run:  .venv/bin/python test_tts_voice_eval.py
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

from tts_voice_eval import (
    LineResult,
    VoiceResult,
    config_path,
    download_error,
    download_voice,
    format_table,
    mean_rtf,
    real_time_factor,
    recommend,
    rtf_spread,
    voice_is_complete,
)

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    results.append((PASS if ok else FAIL, name))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")
    return ok


def voice(name: str, rtf: float, **kwargs) -> VoiceResult:
    """A voice whose every line costs `rtf` synthesis seconds per audio second."""
    return VoiceResult(
        voice=name,
        lines=[LineResult(label="only", synth_s=rtf * 2.0, audio_s=2.0)],
        **kwargs,
    )


def test_real_time_factor() -> None:
    check(
        "rtf is synthesis seconds per audio second",
        real_time_factor(1.0, 2.0) == 0.5 and real_time_factor(4.0, 2.0) == 2.0,
        "1s/2s = 0.5, 4s/2s = 2.0",
    )


def test_silent_synthesis_is_not_free() -> None:
    """No audio is a failed voice, not an infinitely fast one.

    Reported as a division it would be the cheapest voice on the table and would
    win the recommendation outright, which is the wrong direction to fail in.
    """
    check(
        "zero-length audio costs infinity, not zero",
        real_time_factor(1.5, 0.0) == math.inf
        and real_time_factor(0.0, 0.0) == math.inf
        and mean_rtf([]) == math.inf,
        "a voice that produced nothing cannot be the cheapest row",
    )


def test_mean_is_over_lines_not_totals() -> None:
    """A long line must not outvote a short one just for being long."""
    lines = [
        LineResult(label="short", synth_s=0.1, audio_s=1.0),
        LineResult(label="long", synth_s=9.0, audio_s=9.0),
    ]
    check(
        "mean averages per-line ratios, not summed seconds",
        mean_rtf(lines) == 0.55,
        f"(0.1 + 1.0) / 2 = 0.55, got {mean_rtf(lines)}",
    )


def test_spread_shows_every_line() -> None:
    """The mean can hide the outlier the whole cross-machine argument rests on."""
    result = VoiceResult(
        voice="v",
        lines=[
            LineResult(label="fallback", synth_s=0.1, audio_s=1.0),
            LineResult(label="status", synth_s=0.9, audio_s=1.0),
        ],
    )
    spread = rtf_spread(result)
    check(
        "per-line spread names each line and its own rtf",
        "fallback 0.100" in spread and "status 0.900" in spread,
        spread,
    )
    check(
        "the spread reaches the printed table",
        "fallback 0.100" in format_table([result]),
        "a number collected and never shown is a number nobody can act on",
    )


def test_recommend_orders_by_cost_and_names_the_ratio() -> None:
    results_in = [voice("baseline", 0.10), voice("pricey", 0.65), voice("cheap", 0.05)]
    notes = recommend(results_in, rtf_budget=0.5)

    check(
        "notes run cheapest first",
        notes[0].startswith("cheap")
        and notes[1].startswith("baseline")
        and notes[2].startswith("pricey"),
        " | ".join(note.split(":")[0] for note in notes[:3]),
    )
    # The ratio is against the first voice measured, which is the shipped one in
    # the default set, not against whichever voice happened to be cheapest.
    check(
        "the ratio is against the first voice measured, not the cheapest",
        "6.5x the cost of baseline" in notes[2]
        and "0.5x the cost of baseline" in notes[0],
        "a ratio against the winner would always read 1.0x and say nothing",
    )


def test_recommend_flags_only_what_is_over_budget() -> None:
    notes = recommend([voice("fast", 0.1), voice("slow", 0.9)], rtf_budget=0.5)
    over = [note for note in notes if note.startswith("Over the")]

    check(
        "only the over-budget voice is flagged",
        len(over) == 1 and "slow" in over[0] and "fast" not in over[0],
        over[0] if over else "no budget note at all",
    )
    # A budget note that does not send the reader to the real device would end the
    # decision on the wrong machine's numbers.
    check(
        "the budget note sends the reader to the target device",
        over and "target device" in over[0],
        "this box is not the Pi, and the absolutes do not transfer",
    )


def test_failed_voice_is_excluded_but_still_shown() -> None:
    """A voice that would not run must not be scored, and must not vanish either."""
    results_in = [
        voice("works", 0.2),
        VoiceResult(voice="broken", error="no such model"),
    ]
    notes = recommend(results_in, rtf_budget=0.5)
    table = format_table(results_in)

    check(
        "a failed voice is not scored",
        not any("broken" in note for note in notes),
        "scoring it would rank a voice that never ran",
    )
    check(
        "a failed voice still appears, with its reason",
        "broken" in table and "no such model" in table,
        "silently dropping it would read as a voice nobody tried",
    )


def test_download_error_is_actionable() -> None:
    """A misspelled voice name should not read as a urllib traceback."""
    notfound = subprocess.CalledProcessError(
        1,
        "cmd",
        stderr=b"Traceback...\nurllib.error.HTTPError: HTTP Error 404: Not Found",
    )
    message = download_error(notfound)
    check(
        "a 404 names the fix, not the traceback",
        "no voice by that name" in message
        and "piper.download_voices" in message
        and "Traceback" not in message,
        message,
    )
    check(
        "any other failure keeps its own last line",
        download_error(
            subprocess.CalledProcessError(1, "cmd", stderr="boom: disk full")
        )
        == "boom: disk full"
        and download_error(subprocess.CalledProcessError(1, "cmd"))
        == "download failed",
        "a disk-full error must not be reported as a bad voice name",
    )


def test_half_downloaded_voice_is_not_treated_as_ready() -> None:
    """A model with no config is a download to finish, not a voice to load.

    PiperVoice.load derives the config path from the model path, so a voice that
    kept its 63 MB .onnx through an interrupted setup and lost its 5 KB .onnx.json
    loads as a failure. Returning early on the .onnx alone puts that failure on the
    table as a property of the voice, when one more download call fixes it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        onnx = Path(tmp) / "en_US-lessac-medium.onnx"
        onnx.write_bytes(b"model")
        # The literal name piper writes and PiperVoice.load reads, spelled out
        # rather than taken from config_path, so a helper that derives the wrong
        # path cannot move the file and the assertion together and stay green.
        config = Path(tmp) / "en_US-lessac-medium.onnx.json"

        check(
            "the helper points at the file piper actually writes",
            config_path(onnx) == config,
            f"{config_path(onnx).name}, want {config.name}",
        )

        missing = voice_is_complete(onnx)
        config.write_bytes(b"")
        empty = voice_is_complete(onnx)
        config.write_bytes(b"{}")
        both = voice_is_complete(onnx)
        onnx.unlink()
        no_model = voice_is_complete(onnx)

    check(
        "a model without its config is not complete",
        not missing and not empty,
        "missing and empty both have to re-download, as piper's own check does",
    )
    check(
        "both files present and non-empty is complete",
        both and not no_model,
        "and a missing model is still incomplete however good the config is",
    )


def test_download_repairs_a_half_downloaded_voice_once() -> None:
    """The check above has to be the one the skip actually consults.

    Testing voice_is_complete alone would leave `if onnx.exists()` at the call site
    green, which is the whole bug. So this drives download_voice with the
    downloader stubbed out: it must call the downloader for a model with no config,
    and must not call it again once the config is there.
    """
    voice_name = "en_US-lessac-medium"
    with tempfile.TemporaryDirectory() as tmp:
        voices_dir = Path(tmp)
        onnx = voices_dir / f"{voice_name}.onnx"
        onnx.write_bytes(b"model")
        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            (voices_dir / f"{voice_name}.onnx.json").write_bytes(b"{}")
            return subprocess.CompletedProcess(cmd, 0)

        with mock.patch("tts_voice_eval.subprocess.run", fake_run):
            download_voice(voice_name, voices_dir)
            repaired = len(calls)
            download_voice(voice_name, voices_dir)
            again = len(calls)

    check(
        "a model with no config sends the voice back to the downloader",
        repaired == 1,
        f"{repaired} download calls; 0 means the skip is still checking the .onnx alone",
    )
    check(
        "a complete voice downloads nothing",
        again == 1,
        f"{again} total calls; more than one means it re-downloads 63 MB every run",
    )


def test_recommend_says_so_when_nothing_ran() -> None:
    notes = recommend([VoiceResult(voice="broken", error="boom")], rtf_budget=0.5)
    check(
        "an all-failed run says so instead of printing an empty comparison",
        len(notes) == 1 and "nothing to compare" in notes[0],
        notes[0] if notes else "no note at all",
    )


def main() -> int:
    test_real_time_factor()
    test_silent_synthesis_is_not_free()
    test_mean_is_over_lines_not_totals()
    test_spread_shows_every_line()
    test_recommend_orders_by_cost_and_names_the_ratio()
    test_recommend_flags_only_what_is_over_budget()
    test_failed_voice_is_excluded_but_still_shown()
    test_download_error_is_actionable()
    test_half_downloaded_voice_is_not_treated_as_ready()
    test_download_repairs_a_half_downloaded_voice_once()
    test_recommend_says_so_when_nothing_ran()

    failed = [name for verdict, name in results if verdict == FAIL]
    total = len(results)
    print(f"\n{total - len(failed)}/{total} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
