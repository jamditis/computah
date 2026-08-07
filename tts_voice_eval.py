#!/usr/bin/env python3
"""Compare Piper voices on the text this assistant actually speaks (#38).

The question in #38 is timbre, and timbre is a judgment call nobody can make from
a table. So this measures only what a machine can settle honestly: how much CPU a
voice costs per second of speech, how big it is, and how long it takes to load.
It writes one wav per voice per line so a person can listen and decide.

The number that matters is the real-time factor, synthesis seconds per second of
audio. It is a ratio, so it is comparable across voices measured on the same box,
and it stays meaningful when the absolute speed does not: a voice measured at
twice the RTF of another here should cost about twice as much on the Pi too,
even though both numbers move. The candidate voices share an architecture and
op mix, which is why the ratio travels at all; it is not guaranteed to, and the
absolutes plainly do not. Read the ratios here, then run this on the target
device before deciding, because the decision is absolute RTF against 1.0.

RTF above 1.0 means synthesis is slower than playback, so the reply cannot start
until well after the turn ends. Piper stays resident between turns (pipeline.py
_get_piper), so load time is paid once per process, not once per reply, and this
reports it separately rather than folding it into the per-line cost.

Usage:
  .venv/bin/python tts_voice_eval.py                       # the #38 candidate set
  .venv/bin/python tts_voice_eval.py --voice en_US-amy-medium --voice en_US-ryan-high
  .venv/bin/python tts_voice_eval.py --out-dir /tmp/voices --rtf-budget 0.5

Voices download to voices/ (the same directory pipeline.py reads) and are not
committed. List what is available with:
  .venv/bin/python -m piper.download_voices
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

VOICES_DIR = Path(__file__).parent / "voices"

# The voices #38 names, plus lessac-high: same speaker as the shipped voice, so
# it isolates quality tier from speaker identity. Everything else changes both at
# once, which is why a straight A/B against the current voice is worth having.
DEFAULT_VOICES = [
    "en_US-lessac-medium",
    "en_US-lessac-high",
    "en_US-ryan-high",
    "en_US-amy-medium",
    "en_US-hfc_female-medium",
]

# Lines this assistant really produces, not lorem ipsum. The fallback is a
# deliberate copy of pipeline.EMPTY_REPLY_FALLBACK rather than an import, because
# importing pipeline would pull in numpy and openWakeWord and cost this module its
# stdlib-only imports; keep the two in step by hand. It is the single most-spoken
# string in the system, so it is the one worth judging a voice on. The other two
# carry what a synthesizer reads worst: a coined proper noun, a clock time that
# has to be expanded aloud, an underscored identifier, and a person's name.
DEFAULT_LINES = [
    ("fallback", "Sorry, I don't have an answer for that."),
    (
        "status",
        "The build failed on landofjawn at 4:12 PM. Two tests in test_pipeline are red.",
    ),
    (
        "longer",
        "I moved the meeting with Stefanie to Thursday at ten, "
        "and left the agenda in the shared folder.",
    ),
]


# --- pure core (no piper, no audio, no network: covered by test_tts_voice_eval.py)


@dataclass
class LineResult:
    label: str
    synth_s: float
    audio_s: float

    @property
    def rtf(self) -> float:
        return real_time_factor(self.synth_s, self.audio_s)


@dataclass
class VoiceResult:
    voice: str
    model_mb: float = 0.0
    load_s: float = 0.0
    peak_rss_mb: float = 0.0
    lines: list[LineResult] = field(default_factory=list)
    error: str = ""

    @property
    def mean_rtf(self) -> float:
        return mean_rtf(self.lines)


def real_time_factor(synth_s: float, audio_s: float) -> float:
    """Synthesis seconds per second of audio.

    Zero-length audio means the voice produced nothing, which is a failure and not
    a free synthesis, so it reports infinite cost rather than dividing by zero.
    """
    if audio_s <= 0:
        return float("inf")
    return synth_s / audio_s


def mean_rtf(lines: list[LineResult]) -> float:
    if not lines:
        return float("inf")
    return sum(line.rtf for line in lines) / len(lines)


def recommend(results: list[VoiceResult], rtf_budget: float) -> list[str]:
    """Notes on the measurable half of the decision. Timbre stays with the reader.

    Returns one line per finding, ordered cheapest first, naming the cost of each
    voice relative to the first one measured, which is the shipped voice when the
    default set is used.
    """
    usable = [r for r in results if not r.error and r.lines]
    if not usable:
        return ["No voice produced audio, so there is nothing to compare."]

    baseline = usable[0]
    notes = []
    for result in sorted(usable, key=lambda r: r.mean_rtf):
        ratio = (
            result.mean_rtf / baseline.mean_rtf
            if baseline.mean_rtf not in (0, float("inf"))
            else float("inf")
        )
        verdict = "within budget" if result.mean_rtf <= rtf_budget else "over budget"
        notes.append(
            f"{result.voice}: {result.mean_rtf:.2f} RTF, "
            f"{ratio:.1f}x the cost of {baseline.voice}, {verdict}"
        )
    over = [r.voice for r in usable if r.mean_rtf > rtf_budget]
    if over:
        notes.append(
            f"Over the {rtf_budget:.2f} budget on this machine: {', '.join(over)}. "
            "Re-measure on the target device before ruling any of them out."
        )
    return notes


def download_error(exc: subprocess.CalledProcessError) -> str:
    """A sentence the reader can act on, not the tail of a urllib traceback.

    A misspelled voice name is the overwhelmingly likely cause of a 404 here, and
    the fix is one command away, so say which command.
    """
    stderr = exc.stderr
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    stderr = (stderr or "").strip()
    if "404" in stderr:
        return (
            "no voice by that name. List the real ones with "
            "`python -m piper.download_voices`"
        )
    return stderr.splitlines()[-1][:200] if stderr else "download failed"


def rtf_spread(result: VoiceResult) -> str:
    """Every line's RTF, not just the mean.

    The mean is the one statistic that can hide the thing this tool exists to
    show. Reading one voice as N times another only means something if the ratio
    holds line to line, and a single outlier moves a three-sample mean enough to
    invent a difference that is not there. It also exposes a first-line cost:
    warm_models loads Piper without synthesizing, so the first line in a process
    pays onnxruntime's one-time setup and reads slower than the rest.
    """
    return "  ".join(f"{line.label} {line.rtf:.3f}" for line in result.lines)


def format_table(results: list[VoiceResult]) -> str:
    header = "{:<26} {:>8} {:>8} {:>9} {:>9}  {}".format(
        "voice", "size MB", "load s", "mean RTF", "peak MB", "per line"
    )
    rows = [header, "-" * len(header)]
    for result in results:
        if result.error:
            rows.append(f"{result.voice:<26} failed: {result.error}")
            continue
        rows.append(
            "{:<26} {:>8.0f} {:>8.2f} {:>9.3f} {:>9.0f}  {}".format(
                result.voice,
                result.model_mb,
                result.load_s,
                result.mean_rtf,
                result.peak_rss_mb,
                rtf_spread(result),
            )
        )
    return "\n".join(rows)


# --- driver (needs piper and the network)


def download_voice(voice: str, voices_dir: Path) -> Path:
    onnx = voices_dir / f"{voice}.onnx"
    if onnx.exists():
        return onnx
    voices_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "piper.download_voices",
            voice,
            "--download-dir",
            str(voices_dir),
        ],
        check=True,
        capture_output=True,
    )
    return onnx


def measure_voice(
    voice: str, onnx: Path, lines: list[tuple[str, str]], out_dir: Path
) -> VoiceResult:
    """Load the voice once and synthesize every line, as the live loop does.

    Runs in a subprocess so each voice reports its own peak memory rather than the
    high-water mark of every voice measured before it.
    """
    script = Path(__file__).with_name("_tts_voice_eval_worker.py")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            str(onnx),
            json.dumps(lines),
            str(out_dir),
            voice,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return VoiceResult(voice=voice, error=proc.stderr.strip()[-300:] or "no output")
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    return VoiceResult(
        voice=voice,
        model_mb=onnx.stat().st_size / 1e6,
        load_s=payload["load_s"],
        peak_rss_mb=payload["peak_rss_mb"],
        lines=[LineResult(**line) for line in payload["lines"]],
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--voice",
        action="append",
        dest="voices",
        help="voice to measure; repeatable. Defaults to the #38 candidate set.",
    )
    parser.add_argument(
        "--out-dir",
        default="tts-samples",
        help="where to write the wav files to listen to (default: tts-samples)",
    )
    parser.add_argument(
        "--voices-dir",
        default=str(VOICES_DIR),
        help="where voice models live (default: voices/, the one pipeline.py reads)",
    )
    parser.add_argument(
        "--rtf-budget",
        type=float,
        default=0.5,
        help="flag a voice costing more than this many synthesis seconds per "
        "second of audio. The default leaves half the turn's speaking time as "
        "headroom before synthesis stops keeping up with playback (default: 0.5)",
    )
    args = parser.parse_args()

    voices = args.voices or DEFAULT_VOICES
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    voices_dir = Path(args.voices_dir)

    results = []
    for voice in voices:
        print(f"{voice}: downloading", flush=True)
        try:
            onnx = download_voice(voice, voices_dir)
        except subprocess.CalledProcessError as exc:
            print(f"{voice}: download failed", flush=True)
            results.append(VoiceResult(voice=voice, error=download_error(exc)))
            continue
        print(f"{voice}: measuring", flush=True)
        results.append(measure_voice(voice, onnx, DEFAULT_LINES, out_dir))

    print()
    print(format_table(results))
    print()
    for note in recommend(results, args.rtf_budget):
        print(note)

    if not any(not r.error and r.lines for r in results):
        return 1

    print()
    print(f"Listen to the samples in {out_dir}/ and pick on timbre.")
    print("Set the winner as voice_model in config.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
