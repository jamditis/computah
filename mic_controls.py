#!/usr/bin/env python3
"""Survey the ALSA mixer capture controls on a USB Audio Class mic (issue #46).

The Anker PowerConf S3 is a driverless USB Audio Class device, so on Linux and
the Pi its only deeper-control surface is the ALSA mixer. `amixer -c <card>
scontents` lists the simple controls, and the capture-volume control is the one
real lever for STT accuracy (#12) and response time (#14): raise the capture gain
and quiet speech transcribes; raise it too far and the onboard AGC clips. This
module parses that output into a structured survey and picks the capture-gain
control, so a Pi session can record the real control ranges and sweep gain
against transcription accuracy instead of guessing.

Split the way benchmark.py splits its ssh-hop probe: the parser is pure and
stdlib-only, testable on any box with no mic and no ALSA, and `survey_card` is
the thin probe that shells amixer on the device. numpy and sounddevice are
deliberately not imported here, so the test runs anywhere the pipeline cannot.

Run:  python mic_controls.py --card PowerConf
The exit code is 0 only when a capture-gain control was found on the card.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from typing import Callable, NamedTuple

# A "Simple mixer control 'Name',0" header. The index after the comma
# distinguishes two controls that share a name.
_HEADER = re.compile(r"^Simple mixer control '(?P<name>.*)',(?P<index>\d+)\s*$")
# A per-channel state line, e.g. "Front Left: Capture 205 [80%] [-4.00dB] [on]"
# or the switch-only "Mono: Playback [on]". A dual playback/capture control
# prints both states on one line ("Front Left: Playback 20 [65%] [-10.00dB] [on]
# Capture 30 [77%] [on]"), so the name is matched once and each direction's
# state is read on its own. The value, percent, dB, and switch are each optional
# so one pattern covers volume, volume+dB, and switch-only.
_CHAN_NAME = re.compile(r"^(?P<chan>.+?): (?P<rest>(?:Playback|Capture).*)$")
_CHAN_STATE = re.compile(
    r"(?P<dir>Playback|Capture)"
    r"(?: (?P<value>-?\d+))?"
    r"(?: \[(?P<pct>-?\d+)%\])?"
    r"(?: \[(?P<db>-?\d+(?:\.\d+)?)dB\])?"
    r"(?: \[(?P<switch>on|off)\])?"
)
# amixer prints a control's range as a "Limits:" line in one of three shapes: a
# single direction ("Limits: Capture 0 - 65536"), both directions on one line
# for a control that plays and captures ("Limits: Playback 0 - 31 Capture 0 -
# 39"), or bare with no direction for a common playback/capture volume
# ("Limits: 0 - 4096"). The PowerConf S3 is a speaker and mic in one device, so
# the combined form is the expected shape. A capture survey reads the capture
# range first.
_LIMITS_RANGE = re.compile(r"(?P<dir>Playback|Capture) (?P<lo>-?\d+) - (?P<hi>-?\d+)")
_LIMITS_BARE = re.compile(r"^Limits:\s*(?P<lo>-?\d+) - (?P<hi>-?\d+)\s*$")


class Channel(NamedTuple):
    name: str
    direction: str  # "Capture" or "Playback"
    value: int | None
    percent: int | None
    db: float | None
    switch: bool | None


class MixerControl(NamedTuple):
    name: str
    index: int
    capabilities: tuple[str, ...]
    limits: tuple[int, int] | None  # (min, max) of the capture/playback range
    channels: tuple[Channel, ...]

    @property
    def is_capture(self) -> bool:
        # A capture control either advertises a capture capability or reports at
        # least one capture channel. Either alone is enough: some firmwares list
        # only the capability, some only the channel line.
        if any(c.startswith("c") for c in self.capabilities):
            return True
        return any(ch.direction == "Capture" for ch in self.channels)

    @property
    def has_volume(self) -> bool:
        return "cvolume" in self.capabilities or "pvolume" in self.capabilities

    @property
    def has_capture_volume(self) -> bool:
        # A capture gain lever needs a capture volume: the split "cvolume" or a
        # common "volume" shared across playback and capture. Playback-only
        # "pvolume" does not count.
        return "cvolume" in self.capabilities or "volume" in self.capabilities


class CardSurvey(NamedTuple):
    card: str
    ok: bool
    error: str | None
    controls: tuple[MixerControl, ...]
    capture_control: MixerControl | None


def _parse_channel(line: str) -> Channel | None:
    head = _CHAN_NAME.match(line.strip())
    if not head:
        return None
    # Prefer the capture state on a dual line: a capture survey reads that side.
    states = {m["dir"]: m for m in _CHAN_STATE.finditer(head["rest"])}
    m = states.get("Capture") or states.get("Playback")
    if m is None:
        return None
    value = int(m["value"]) if m["value"] is not None else None
    percent = int(m["pct"]) if m["pct"] is not None else None
    db = float(m["db"]) if m["db"] is not None else None
    switch = None if m["switch"] is None else (m["switch"] == "on")
    return Channel(head["chan"].strip(), m["dir"], value, percent, db, switch)


def _parse_limits(line: str) -> tuple[int, int] | None:
    """Read a control's range from an amixer "Limits:" line.

    Prefer the capture range, fall back to the playback range, then to a bare
    common-volume range. So the one range slot holds the capture gain range
    whenever the control has one.
    """
    ranges = {
        m["dir"]: (int(m["lo"]), int(m["hi"])) for m in _LIMITS_RANGE.finditer(line)
    }
    if "Capture" in ranges:
        return ranges["Capture"]
    if "Playback" in ranges:
        return ranges["Playback"]
    bare = _LIMITS_BARE.match(line)
    if bare:
        return (int(bare["lo"]), int(bare["hi"]))
    return None


def parse_amixer_scontents(text: str) -> list[MixerControl]:
    """Parse `amixer -c <card> scontents` into structured controls.

    Unknown lines inside a block (Capture channels, Playback channels, and the
    like) are ignored, so a firmware that adds a field does not break the parse.
    """
    controls: list[MixerControl] = []
    name: str | None = None
    index = 0
    caps: tuple[str, ...] = ()
    limits: tuple[int, int] | None = None
    channels: list[Channel] = []

    def flush() -> None:
        nonlocal name, caps, limits, channels
        if name is not None:
            controls.append(MixerControl(name, index, caps, limits, tuple(channels)))
        name, caps, limits, channels = None, (), None, []

    for raw in text.splitlines():
        header = _HEADER.match(raw)
        if header:
            flush()
            name = header["name"]
            index = int(header["index"])
            continue
        if name is None:
            continue
        stripped = raw.strip()
        if stripped.startswith("Capabilities:"):
            caps = tuple(stripped[len("Capabilities:") :].split())
            continue
        if stripped.startswith("Limits:"):
            parsed = _parse_limits(stripped)
            if parsed is not None:
                limits = parsed
            continue
        channel = _parse_channel(stripped)
        if channel:
            channels.append(channel)
    flush()
    return controls


def capture_gain_control(controls: list[MixerControl]) -> MixerControl | None:
    """The capture control that carries a real gain lever.

    Requires a capture direction, a capture volume, and a range: a capture
    switch (mute button) alone is not a gain lever, and a control whose only
    volume is playback is not either, so both are skipped. Returns the first
    match, which is amixer's own order.
    """
    for control in controls:
        if (
            control.is_capture
            and control.has_capture_volume
            and control.limits is not None
        ):
            return control
    return None


def format_survey(survey: CardSurvey) -> str:
    if not survey.ok:
        return f"card {survey.card!r}: no survey ({survey.error})"
    lines = [f"card {survey.card!r}: {len(survey.controls)} mixer control(s)"]
    for control in survey.controls:
        tag = "capture" if control.is_capture else "playback"
        rng = f" {control.limits[0]}-{control.limits[1]}" if control.limits else ""
        lines.append(
            f"  [{tag}] {control.name!r}{rng}  caps={' '.join(control.capabilities) or 'none'}"
        )
    gain = survey.capture_control
    if gain is None:
        lines.append("  no capture-gain lever found")
    else:
        cur = next((c for c in gain.channels if c.percent is not None), None)
        at = f" at {cur.percent}%" if cur else ""
        lines.append(
            f"  gain lever: {gain.name!r} range {gain.limits[0]}-{gain.limits[1]}{at}"
        )
    return "\n".join(lines)


def survey_card(
    card: str,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> CardSurvey:
    """Shell `amixer -c <card> scontents` and parse it. Never raises.

    Mirrors benchmark.py: a probe that cannot read the device reports that it
    could not, rather than inventing a control. `run` is injected so the parse
    path is testable without amixer or a mic.
    """
    try:
        proc = run(
            ["amixer", "-c", card, "scontents"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return CardSurvey(
            card, False, "amixer not installed (install alsa-utils)", (), None
        )
    except subprocess.TimeoutExpired:
        return CardSurvey(card, False, "amixer timed out", (), None)
    if proc.returncode != 0:
        reason = (proc.stderr or "").strip() or f"amixer exit {proc.returncode}"
        return CardSurvey(card, False, reason, (), None)
    controls = parse_amixer_scontents(proc.stdout or "")
    if not controls:
        return CardSurvey(card, False, "no mixer controls reported", (), None)
    return CardSurvey(card, True, None, tuple(controls), capture_gain_control(controls))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Survey ALSA capture controls for a mic card."
    )
    parser.add_argument(
        "--card",
        default="PowerConf",
        help="ALSA card name or index (default: PowerConf)",
    )
    args = parser.parse_args(argv)
    survey = survey_card(args.card)
    print(format_survey(survey))
    if not survey.ok:
        return 2
    return 0 if survey.capture_control is not None else 1


if __name__ == "__main__":
    sys.exit(main())
