#!/usr/bin/env python3
"""Full mic-free chain with the BRIDGE brain backend (not claude -p).

Runs the real wake + STT + TTS models, but swaps the brain for brain_via_bridge()
backed by the sim persona. Proves wake -> STT -> bridge-brain -> TTS works end to
end without a mic, without claude -p, and without a live Claude session — the
exact shape of the eventual Syl integration, minus the ssh transport.

Loads models, so run it under a memory cap, e.g.:
  systemd-run --user --scope -p MemoryMax=1500M -p MemorySwapMax=0 \
    nice -n19 ionice -c3 .venv/bin/python test_pipeline_bridge.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import brain_bridge
import pipeline
from sim_persona import SimPersona

PROJECT = Path(__file__).resolve().parent
results: list[tuple[bool, str]] = []


def check(ok: bool, detail: str) -> None:
    results.append((ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {detail}")


def main() -> int:
    d = Path(tempfile.mkdtemp(prefix="voice-pipe-bridge-"))
    inbox = d / "manual-inbox.jsonl"
    reply = d / "voice-reply.txt"

    sim = SimPersona(inbox, reply, poll_s=0.05)
    sim.start()

    # Swap the module-level brain for the bridge backend. run_pipeline() calls
    # brain() by name in this module's namespace, so reassigning it here routes
    # the brain stage through bot-spren's send/reply contract instead of claude -p.
    pipeline.brain = lambda text, **_: brain_bridge.brain_via_bridge(
        text, persona="syl",
        send=brain_bridge.local_sim_send(inbox),
        read_reply=brain_bridge.file_reply_reader(reply),
        system_prompt=pipeline.VOICE_SYSTEM_PROMPT,
        timeout_s=20, poll_s=0.05,
    )

    clip = str(PROJECT / "test_audio" / "clip_jarvis.wav")
    if not Path(clip).exists():
        # Synthesize the prompt clip with Piper if the builder's clip is gone.
        pipeline.speak("hey jarvis, what is two plus two?", clip)

    try:
        # Pin the wake word so the test is independent of config.json's active one.
        r = pipeline.run_pipeline(clip, out_wav_path=str(d / "reply.wav"),
                                  wake_word="hey_jarvis")
    finally:
        sim.stop()

    check(r["wake_fired"], f"wake fired on the clip (score {r['wake_score']})")
    check(bool(r["transcript"]), f"STT produced text: {r['transcript']!r}")
    check(r["reply"] == "Two plus two is four.",
          f"bridge brain answered correctly: {r['reply']!r}")
    check(bool(r["output_wav"]) and Path(r["output_wav"]).exists(),
          f"TTS wrote a reply wav: {r['output_wav']}")

    print("\ntimings (s): " + ", ".join(
        f"{k}={v:.2f}" for k, v in r["timings_s"].items()))
    n_pass = sum(1 for ok, _ in results if ok)
    print(f"=== {n_pass}/{len(results)} checks passed ===")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
