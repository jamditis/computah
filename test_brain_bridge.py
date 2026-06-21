#!/usr/bin/env python3
"""Test the bridge brain backend against the sim persona.

No bot-spren CLI, no Claude session, no mic, and no model loads — pure file I/O
and threads, so it is safe to run on the RAM-fragile Pi without a cgroup cap.
Proves brain_via_bridge() sends to an inbox and reads the correct reply back,
and that a second turn does not return the first turn's stale reply.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import brain_bridge
from sim_persona import SimPersona

results: list[tuple[bool, str]] = []


def check(ok: bool, detail: str) -> None:
    results.append((ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {detail}")


def main() -> int:
    d = Path(tempfile.mkdtemp(prefix="voice-bridge-"))
    inbox = d / "manual-inbox.jsonl"
    reply = d / "voice-reply.txt"

    # One persistent persona for the whole test — this mirrors production, where a
    # single Syl session checkpoints its inbox offset and never re-replies to an
    # already-consumed message. A fresh SimPersona per turn would reset its offset
    # to 0, re-read the first question, and emit a duplicate reply that races the
    # next turn's poll (a test-harness artifact, not a brain_via_bridge bug).
    sim = SimPersona(inbox, reply, poll_s=0.05)
    sim.start()
    send = brain_bridge.local_sim_send(inbox)
    read = brain_bridge.file_reply_reader(reply)
    try:
        out1 = brain_bridge.brain_via_bridge(
            "hey jarvis, what is two plus two?", persona="syl",
            send=send, read_reply=read, timeout_s=10, poll_s=0.05)
        check(out1 == "Two plus two is four.", f"round trip reply: {out1!r}")

        # A second turn must return the new reply, not the first turn's stale block.
        out2 = brain_bridge.brain_via_bridge(
            "what is the capital of france?", persona="syl",
            send=send, read_reply=read, timeout_s=10, poll_s=0.05)
        check(out2 == "The capital of France is Paris.",
              f"second turn is fresh, not stale: {out2!r}")
    finally:
        sim.stop()

    # Timeout path: no sim running, so no reply ever appears.
    inbox2 = d / "dead-inbox.jsonl"
    reply2 = d / "dead-reply.txt"
    out3 = brain_bridge.brain_via_bridge(
        "is anyone there?", persona="syl",
        send=brain_bridge.local_sim_send(inbox2),
        read_reply=brain_bridge.file_reply_reader(reply2),
        timeout_s=1, poll_s=0.05,
    )
    check(out3.startswith("Sorry, the brain took too long"),
          f"timeout returns a spoken error: {out3!r}")

    n_pass = sum(1 for ok, _ in results if ok)
    print(f"\n=== {n_pass}/{len(results)} checks passed ===")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
