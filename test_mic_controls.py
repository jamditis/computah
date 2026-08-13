#!/usr/bin/env python3
"""Fast, no-model, no-mic tests for the ALSA capture-control survey (issue #46).

Everything here is stdlib-only string work: the parser turns real `amixer
scontents` output into structured controls, and the probe is exercised with an
injected `run` so no amixer or microphone is needed. Runs on a box with no numpy,
no PortAudio, and no sound card.

Run:  .venv/bin/python test_mic_controls.py
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import subprocess
import sys

from mic_controls import (
    capture_gain_control,
    parse_amixer_scontents,
    survey_card,
)

# A realistic USB capture mic: a capture volume control ('Mic'), a capture
# switch-only control ('Auto Gain Control' -- a mute lever, not a gain lever),
# and a playback control ('PCM') carrying a dB annotation and no switch.
FIXTURE = """Simple mixer control 'Mic',0
  Capabilities: cvolume cswitch
  Capture channels: Front Left - Front Right
  Limits: Capture 0 - 65536
  Front Left: Capture 32768 [50%] [on]
  Front Right: Capture 32768 [50%] [on]
Simple mixer control 'Auto Gain Control',0
  Capabilities: cswitch cswitch-joined
  Capture channels: Mono
  Mono: Capture [on]
Simple mixer control 'PCM',0
  Capabilities: pvolume
  Playback channels: Front Left - Front Right
  Limits: Playback 0 - 65536
  Front Left: Playback 52428 [80%] [-4.00dB]
  Front Right: Playback 52428 [80%] [-4.00dB]
"""

# A speakerphone-style dual control: one selem carrying both a playback and a
# capture volume. Real amixer prints both ranges on ONE Limits line and both
# states on each channel line. This is the expected shape for the PowerConf S3,
# a speaker and mic in one USB device.
DUAL_FIXTURE = """Simple mixer control 'Headset',0
  Capabilities: pvolume cvolume pswitch cswitch
  Playback channels: Front Left - Front Right
  Capture channels: Front Left - Front Right
  Limits: Playback 0 - 31 Capture 0 - 39
  Front Left: Playback 20 [65%] [-10.00dB] [on] Capture 30 [77%] [on]
  Front Right: Playback 20 [65%] [-10.00dB] [on] Capture 30 [77%] [on]
"""

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    results.append((PASS if ok else FAIL, name))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")
    return bool(ok)


def _fake_run(stdout="", stderr="", returncode=0, raises=None):
    def run(*args, **kwargs):
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(args[0], returncode, stdout, stderr)

    return run


def test_parse():
    controls = parse_amixer_scontents(FIXTURE)
    check("parse: three controls", len(controls) == 3, f"got {len(controls)}")
    names = [c.name for c in controls]
    check(
        "parse: names in order",
        names == ["Mic", "Auto Gain Control", "PCM"],
        f"{names}",
    )
    mic = controls[0]
    check(
        "parse: mic is a capture control",
        mic.is_capture,
        f"is_capture={mic.is_capture}",
    )
    check("parse: mic has a volume lever", mic.has_volume, f"caps={mic.capabilities}")
    check("parse: mic limits", mic.limits == (0, 65536), f"{mic.limits}")
    fl = mic.channels[0]
    check(
        "parse: channel value/percent/switch",
        fl.value == 32768 and fl.percent == 50 and fl.switch is True,
        f"{fl}",
    )


def test_db_and_no_switch():
    pcm = parse_amixer_scontents(FIXTURE)[2]
    fl = pcm.channels[0]
    check("parse: dB parsed", fl.db == -4.0, f"db={fl.db}")
    check("parse: absent switch is None", fl.switch is None, f"switch={fl.switch}")


def test_switch_only_capture():
    agc = parse_amixer_scontents(FIXTURE)[1]
    check(
        "parse: switch-only is capture", agc.is_capture, f"is_capture={agc.is_capture}"
    )
    check(
        "parse: switch-only has no volume",
        not agc.has_volume,
        f"caps={agc.capabilities}",
    )
    check("parse: switch-only has no limits", agc.limits is None, f"{agc.limits}")
    check(
        "parse: switch-only channel value None",
        agc.channels[0].value is None,
        f"{agc.channels[0]}",
    )


def test_gain_lever_selection():
    controls = parse_amixer_scontents(FIXTURE)
    gain = capture_gain_control(controls)
    # Must pick the capture *volume* control, never the earlier capture switch
    # ('Auto Gain Control') and never the playback control ('PCM').
    check(
        "gain: picks the capture volume control",
        gain is not None and gain.name == "Mic",
        f"{gain and gain.name}",
    )


def test_no_gain_lever_when_only_switch_and_playback():
    text = """Simple mixer control 'Auto Gain Control',0
  Capabilities: cswitch cswitch-joined
  Capture channels: Mono
  Mono: Capture [on]
Simple mixer control 'PCM',0
  Capabilities: pvolume
  Playback channels: Mono
  Limits: Playback 0 - 100
  Mono: Playback 50 [50%]
"""
    gain = capture_gain_control(parse_amixer_scontents(text))
    check(
        "gain: none when no capture volume exists",
        gain is None,
        f"{gain and gain.name}",
    )


def test_survey_ok():
    survey = survey_card("PowerConf", run=_fake_run(stdout=FIXTURE))
    check("survey: ok on good output", survey.ok, f"error={survey.error}")
    check(
        "survey: capture control surfaced",
        survey.capture_control is not None and survey.capture_control.name == "Mic",
        f"{survey.capture_control and survey.capture_control.name}",
    )


def test_survey_amixer_missing():
    survey = survey_card("PowerConf", run=_fake_run(raises=FileNotFoundError()))
    check("survey: amixer missing does not crash", not survey.ok, f"ok={survey.ok}")
    check(
        "survey: reports amixer not installed",
        "not installed" in (survey.error or ""),
        f"{survey.error}",
    )


def test_survey_bad_card():
    survey = survey_card(
        "BadCard",
        run=_fake_run(
            stderr="amixer: Mixer attach BadCard error: No such file", returncode=1
        ),
    )
    check("survey: bad card is not ok", not survey.ok, f"ok={survey.ok}")
    check(
        "survey: bad card carries amixer's reason",
        "Mixer attach" in (survey.error or ""),
        f"{survey.error}",
    )


def test_survey_empty_output():
    survey = survey_card("PowerConf", run=_fake_run(stdout="\n"))
    check(
        "survey: empty output is not ok",
        not survey.ok and survey.capture_control is None,
        f"{survey}",
    )


def test_dual_control_capture_range():
    controls = parse_amixer_scontents(DUAL_FIXTURE)
    check("dual: one control", len(controls) == 1, f"got {len(controls)}")
    ctl = controls[0]
    # The combined Limits line carries both ranges; a capture survey wants the
    # capture range (0-39), not the playback range (0-31).
    check("dual: capture range parsed", ctl.limits == (0, 39), f"{ctl.limits}")
    gain = capture_gain_control(controls)
    check(
        "dual: picked as the gain lever",
        gain is not None and gain.limits == (0, 39),
        f"{gain and gain.limits}",
    )
    # The combined channel lines carry both states; a capture survey reads the
    # capture side, so the current gain percentage survives (it is not dropped).
    first = gain.channels[0] if gain and gain.channels else None
    check(
        "dual: capture channel state read",
        first is not None and first.direction == "Capture" and first.percent == 77,
        f"{first}",
    )


def test_bare_common_volume_limits():
    # A common-volume capture control: amixer prints the range with no direction.
    text = """Simple mixer control 'Capture',0
  Capabilities: cvolume cswitch
  Capture channels: Mono
  Limits: 0 - 4096
  Mono: Capture 2048 [50%] [on]
"""
    ctl = parse_amixer_scontents(text)[0]
    check("bare: common-volume limits parsed", ctl.limits == (0, 4096), f"{ctl.limits}")
    check(
        "bare: picked as the gain lever",
        capture_gain_control([ctl]) is ctl,
        f"{ctl.name}",
    )


def test_playback_volume_is_not_a_capture_lever():
    # A capture control (it has a capture switch) whose only volume is playback
    # (pvolume, no cvolume): not a capture gain lever, even once its range
    # parses. This pins the guard against picking a playback volume for capture.
    text = """Simple mixer control 'Sidetone',0
  Capabilities: pvolume cswitch
  Playback channels: Mono
  Capture channels: Mono
  Limits: Playback 0 - 10 Capture 0 - 10
  Mono: Playback 5 [50%] Capture [on]
"""
    controls = parse_amixer_scontents(text)
    check(
        "guard: control is a capture control",
        controls[0].is_capture and not controls[0].has_capture_volume,
        f"is_capture={controls[0].is_capture} has_capture_volume={controls[0].has_capture_volume}",
    )
    gain = capture_gain_control(controls)
    check(
        "guard: playback-only volume is not a capture lever",
        gain is None,
        f"{gain and gain.name}",
    )


def main() -> int:
    test_parse()
    test_db_and_no_switch()
    test_switch_only_capture()
    test_gain_lever_selection()
    test_no_gain_lever_when_only_switch_and_playback()
    test_dual_control_capture_range()
    test_bare_common_volume_limits()
    test_playback_volume_is_not_a_capture_lever()
    test_survey_ok()
    test_survey_amixer_missing()
    test_survey_bad_card()
    test_survey_empty_output()
    n_pass = sum(1 for r in results if r[0] == PASS)
    n_total = len(results)
    print(f"\n=== {n_pass}/{n_total} checks passed ===")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
