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
    # One cursor for the whole conversation, as production keeps per reply file.
    cursor = brain_bridge.ReplyCursor()
    try:
        out1 = brain_bridge.brain_via_bridge(
            "hey jarvis, what is two plus two?", persona="syl",
            send=send, read_reply=read, cursor=cursor, timeout_s=10, poll_s=0.05)
        check(out1 == "Two plus two is four.", f"round trip reply: {out1!r}")

        # A second turn must return the new reply, not the first turn's stale block.
        out2 = brain_bridge.brain_via_bridge(
            "what is the capital of france?", persona="syl",
            send=send, read_reply=read, cursor=cursor, timeout_s=10, poll_s=0.05)
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

    # Regression (the one-behind audio bug): a late reply from a timed-out turn must
    # not be spoken on the next turn. The cursor reserves a slot per send, so the
    # timed-out turn's late block fills its own reserved slot and this turn skips
    # past it. The old "next new block" logic spoke that late block — the previous
    # prompt's answer — one turn behind.
    def _blocks_text(*pairs: tuple[str, str]) -> str:
        return "".join(
            f"\n--- 2026-01-01T00:00:00+00:00 delivery_id={i} ---\n{p}\n"
            for i, p in pairs
        )

    # State after a good turn A (slot 0, on disk) and a timed-out turn B whose reply
    # is still in flight: the cursor has reserved 2 slots. During this turn (C),
    # B's late reply lands first (slot 1), then C's real reply (slot 2).
    cursor3 = brain_bridge.ReplyCursor(consumed=2)
    frames = [
        _blocks_text(("a", "A answer")),                                    # at send
        _blocks_text(("a", "A answer"), ("b", "late B answer")),            # B lands
        _blocks_text(("a", "A answer"), ("b", "late B answer"), ("c", "C answer")),
    ]
    reads = {"i": 0}

    def _scripted_read() -> str:
        text = frames[min(reads["i"], len(frames) - 1)]
        reads["i"] += 1
        return text

    out4 = brain_bridge.brain_via_bridge(
        "turn C prompt", persona="syl",
        send=lambda *_: None, read_reply=_scripted_read,
        cursor=cursor3, timeout_s=5, poll_s=0.01)
    check(out4 == "C answer",
          f"late reply from a timed-out turn is skipped, not spoken: {out4!r}")
    check(cursor3.consumed == 3,
          f"the send reserved this turn's slot: consumed={cursor3.consumed}")

    # Self-heal (the dropped-reply wedge): if a reply is never written at all — the
    # session crashed or its delivery hook failed — the reserved slot can never fill,
    # so without recovery the cursor sits one ahead forever and every later turn times
    # out. After _RESYNC_AFTER_MISSES consecutive timeouts with the cursor ahead of the
    # file, it resyncs to the live end so the next turn returns its own answer. A single
    # late reply never triggers this: the following turn reads its slot and resets the
    # miss count.
    class _DropThenAnswer:
        """First send's reply is dropped (never written); each later send writes one
        block immediately, one slot behind the reservations — the wedge signature."""

        def __init__(self) -> None:
            self.blocks: list[str] = []
            self.first = True

        def send(self, _persona: str, prompt: str) -> None:
            if self.first:
                self.first = False
                return  # dropped: no block ever appears for this turn
            self.blocks.append(prompt)

        def read(self) -> str:
            return "".join(
                f"\n--- t delivery_id={i} ---\n{p}\n"
                for i, p in enumerate(self.blocks)
            )

    brain = _DropThenAnswer()
    heal = brain_bridge.ReplyCursor(consumed=0)
    turns = [
        brain_bridge.brain_via_bridge(
            f"q{n}", persona="syl", send=brain.send, read_reply=brain.read,
            cursor=heal, timeout_s=0.3, poll_s=0.02)
        for n in range(1, 4)
    ]
    check(turns[0].startswith("Sorry, the brain took too long"),
          f"dropped reply times out (turn 1): {turns[0]!r}")
    check(turns[1].startswith("Sorry, the brain took too long"),
          f"still wedged before the resync threshold (turn 2): {turns[1]!r}")
    check(turns[2] == "q3",
          f"cursor resynced; turn 3 speaks its own answer, not a wedge: {turns[2]!r}")

    # The send argv must carry -d <working_dir> so bot-spren writes to the session's
    # inbox, not its default ~/.bot-spren/<name>/state dead-letter dir. Capture the
    # command without running anything.
    captured: list[list[str]] = []

    class _FakeProc:
        returncode = 0

    def _fake_run(cmd, *a, **k):
        captured.append(cmd)
        return _FakeProc()

    real_run = brain_bridge.subprocess.run
    brain_bridge.subprocess.run = _fake_run
    try:
        brain_bridge.cli_send("bot-spren", working_dir="/x/syl")("syl", "hi")
        check(captured[-1] == ["bot-spren", "send", "-d", "/x/syl", "syl", "hi"],
              f"cli_send with workdir inserts -d: {captured[-1]}")

        brain_bridge.cli_send("bot-spren")("syl", "hi")
        check(captured[-1] == ["bot-spren", "send", "syl", "hi"],
              f"cli_send without workdir omits -d: {captured[-1]}")

        brain_bridge.ssh_cli_send("ofj", "bot-spren", working_dir="/x/syl")("syl", "hi")
        remote = captured[-1][-1]  # the joined remote command is the last ssh arg
        check("send -d /x/syl syl hi" in remote,
              f"ssh_cli_send with workdir inserts -d into remote: {remote!r}")
    finally:
        brain_bridge.subprocess.run = real_run

    n_pass = sum(1 for ok, _ in results if ok)
    print(f"\n=== {n_pass}/{len(results)} checks passed ===")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
