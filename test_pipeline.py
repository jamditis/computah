#!/usr/bin/env python3
"""Mic-free test harness for the jawn-voice pipeline.

There is no microphone on this machine, so the harness synthesizes its own test
audio with Piper and feeds those WAVs through the real pipeline. It checks:

  1. wake detection FIRES on the active wake phrase (positive case)
  2. wake detection does NOT fire on audio without the wake phrase (negative)
  3. transcription returns the spoken text
  4. the claude brain returns a non-empty reply
  5. TTS writes a playable WAV
  6. the wake word is switchable between two pretrained models via config,
     and each model fires only on its own phrase

The harness runs against an isolated, throwaway config (see _isolate_config), so a
deployment's gitignored config.local.json cannot change the outcome.

Run:  .venv/bin/python test_pipeline.py
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import wave
from pathlib import Path

import pipeline

TEST_DIR = Path(__file__).resolve().parent / "test_audio"
TEST_DIR.mkdir(exist_ok=True)

# (name, spoken text, the wake word it should trigger or None for negative)
CLIPS = {
    "jarvis": ("hey jarvis, what is two plus two?", "hey_jarvis"),
    "alexa": ("alexa, what is the capital of France?", "alexa"),
    "negative": ("what time is it in tokyo right now?", None),
}

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")
    return ok


def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def synth_clips() -> dict[str, str]:
    print("\n=== synthesizing test audio with Piper (stands in for a mic) ===")
    paths = {}
    for key, (text, _) in CLIPS.items():
        out = str(TEST_DIR / f"clip_{key}.wav")
        t0 = time.time()
        pipeline.speak(text, out)
        dur = wav_duration(out)
        print(
            f"  {key:9s} -> {out}  ({dur:.1f}s audio, "
            f"rendered in {time.time() - t0:.2f}s)"
        )
        paths[key] = out
    return paths


def _isolate_config() -> tempfile.TemporaryDirectory:
    """Point the pipeline at a throwaway config so the test is hermetic.

    load_config() layers config.local.json over config.json, and a real deployment's
    config.local.json overrides wake_word (this box routes the brain to a persona
    session and sets its own wake word). That override silently wins over
    set_wake_word(), so detect_wake scores every clip against the wrong model and the
    positive-detection checks go red. Seeding a temp config.json from the committed
    base alone, with no local overlay, makes set_wake_word take effect as designed
    and reproduces a bare-checkout run. It also stops the test from mutating the real
    config.json (set_wake_word writes there; a failure before the restore would leave
    it on the wrong wake word).

    Returns the TemporaryDirectory so the caller can clean it up; the pipeline's
    config paths are repointed into it for the rest of the run.
    """
    base = pipeline._read_base_config()  # committed config.json, no local overlay
    tmp = tempfile.TemporaryDirectory(prefix="computah-test-config-")
    root = Path(tmp.name)
    (root / "config.json").write_text(json.dumps(base))
    pipeline.CONFIG_PATH = root / "config.json"
    pipeline.LOCAL_CONFIG_PATH = root / "config.local.json"  # intentionally absent
    print(
        "config source: isolated throwaway (committed base, no config.local.json overlay)"
    )
    return tmp


def _run_checks() -> int:
    cfg = pipeline.load_config()
    print(
        f"config: wake_word={cfg['wake_word']} threshold={cfg['wake_threshold']} "
        f"whisper={cfg['whisper_model']}/{cfg['whisper_compute']} "
        f"voice={cfg['voice_model']} brain=claude:{cfg['claude_model']}"
    )
    print(f"wake-word library: {sorted(pipeline.available_wake_models())}")

    clips = synth_clips()
    timings: dict[str, float] = {}

    # ----- check 1: positive wake detection (active = hey_jarvis) ----------- #
    print("\n=== stage 1: wake detection ===")
    pipeline.set_wake_word("hey_jarvis")
    t0 = time.time()
    fired, name, score = pipeline.detect_wake(clips["jarvis"])
    timings["detect_wake"] = time.time() - t0
    check(
        "wake fires on phrase",
        fired and name == "hey_jarvis",
        f"hey_jarvis clip -> fired={fired} score={score:.4f}",
    )

    # ----- check 2: negative case ------------------------------------------ #
    fired_n, _, score_n = pipeline.detect_wake(clips["negative"])
    check(
        "wake silent without phrase",
        not fired_n,
        f"no-wake clip -> fired={fired_n} score={score_n:.4f}",
    )

    # ----- check 3: transcription ------------------------------------------ #
    print("\n=== stage 2: transcription ===")
    t0 = time.time()
    transcript = pipeline.transcribe(clips["jarvis"])
    timings["transcribe"] = time.time() - t0
    t_low = transcript.lower()
    check(
        "transcription returns text",
        "jarvis" in t_low and ("two" in t_low or "2" in t_low),
        f"{transcript!r}",
    )

    # ----- check 4: the brain (claude CLI) --------------------------------- #
    print("\n=== stage 3: brain (claude CLI subprocess) ===")
    t0 = time.time()
    reply = pipeline.brain(transcript)
    timings["brain"] = time.time() - t0
    check(
        "brain returns a reply",
        bool(reply) and not reply.startswith("Sorry,"),
        f"{reply!r}",
    )

    # ----- check 5: TTS ---------------------------------------------------- #
    print("\n=== stage 4: text-to-speech ===")
    out_wav = str(TEST_DIR / "reply.wav")
    t0 = time.time()
    pipeline.speak(reply, out_wav)
    timings["speak"] = time.time() - t0
    dur = wav_duration(out_wav)
    check(
        "TTS writes a playable WAV",
        Path(out_wav).exists() and dur > 0.2,
        f"{out_wav} ({dur:.1f}s)",
    )

    # ----- check 6: wake-word switching ------------------------------------ #
    print("\n=== wake-word switching (config-driven, instant) ===")
    # With active = hey_jarvis, the jarvis clip fires and the alexa clip does not.
    j_fired, _, j_score = pipeline.detect_wake(clips["jarvis"])
    a_fired, _, a_score = pipeline.detect_wake(clips["alexa"])
    check(
        "active hey_jarvis: matches jarvis only",
        j_fired and not a_fired,
        f"jarvis={j_score:.3f}(fire={j_fired}) alexa={a_score:.3f}(fire={a_fired})",
    )

    # Flip the active wake word to alexa and re-test the same two clips.
    pipeline.set_wake_word("alexa")
    j_fired2, _, j_score2 = pipeline.detect_wake(clips["jarvis"])
    a_fired2, _, a_score2 = pipeline.detect_wake(clips["alexa"])
    check(
        "active alexa: matches alexa only",
        a_fired2 and not j_fired2,
        f"jarvis={j_score2:.3f}(fire={j_fired2}) alexa={a_score2:.3f}(fire={a_fired2})",
    )

    # No config restore needed: set_wake_word wrote to the throwaway config that
    # _isolate_config repointed the pipeline at, and main() discards it.

    # ----- summary --------------------------------------------------------- #
    print("\n=== timings (seconds, includes one-time model loads) ===")
    for k, v in timings.items():
        print(f"  {k:12s} {v:6.2f}")
    pipeline_total = sum(timings.values())
    print(f"  {'sum':12s} {pipeline_total:6.2f}")

    n_pass = sum(1 for r in results if r[0] == PASS)
    n_total = len(results)
    print(f"\n=== SUMMARY: {n_pass}/{n_total} checks passed ===")
    for status, name, _ in results:
        print(f"  [{status}] {name}")
    return 0 if n_pass == n_total else 1


def main() -> int:
    tmp_cfg = _isolate_config()
    try:
        return _run_checks()
    finally:
        tmp_cfg.cleanup()


if __name__ == "__main__":
    sys.exit(main())
