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
        send=lambda *a, **k: None, read_reply=_scripted_read,
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

        def send(self, _persona: str, prompt: str, *,
                 event_id: str | None = None) -> None:
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

    # --- #19 correlation key: identity match, positional fallback -------------
    # When a reply echoes the request's event_id the bridge matches by identity, so a
    # dropped or out-of-order reply on one turn cannot shift another turn's answer --
    # the collateral damage #48 hits under pure positional correlation.
    answers = {
        "what is two plus two?": "Two plus two is four.",
        "what is the capital of france?": "The capital of France is Paris.",
    }

    class _StampedSyl:
        """Stamps each reply with the request's event_id (models the future producer).
        drop = 1-based turn numbers whose reply is never written (models #48)."""

        def __init__(self, drop=(), plain=()) -> None:
            self.blocks: list[tuple[str, str | None, str]] = []
            self.turn = 0
            self.drop = set(drop)
            self.plain = set(plain)  # turns written WITHOUT an event_id (positional)

        def send(self, _persona: str, prompt: str, *,
                 event_id: str | None = None) -> None:
            self.turn += 1
            text = prompt.split("User: ", 1)[-1] if "User: " in prompt else prompt
            if self.turn in self.drop:
                return  # reply dropped at source (#48)
            eid = None if self.turn in self.plain else event_id  # unstamped -> positional
            self.blocks.append((f"d{self.turn}", eid,
                                answers.get(text.strip(), f"echo:{text.strip()}")))

        def read(self) -> str:
            out = ""
            for did, eid, payload in self.blocks:
                hdr = f"--- t delivery_id={did}"
                if eid is not None:
                    hdr += f" event_id={eid}"
                out += f"\n{hdr} ---\n{payload}\n"
            return out

    # The parser tolerates the optional event_id token and extracts it; an unstamped
    # header still parses with event_id None. (Against the pre-#19 regex the stamped
    # block fails to match at all -- the reason the consumer must tolerate the token
    # before any producer emits it.)
    parsed = brain_bridge._delivery_blocks(
        "\n--- t delivery_id=d1 event_id=e1 ---\nstamped\n"
        "\n--- t delivery_id=d2 ---\nunstamped\n")
    check(parsed == [("d1", "e1", "stamped"), ("d2", None, "unstamped")],
          f"_delivery_blocks parses the optional event_id token: {parsed}")

    # Identity round trip: two turns, each returns its own stamped answer.
    syl = _StampedSyl()
    common = dict(persona="syl", send=syl.send, read_reply=syl.read,
                  cursor=brain_bridge.ReplyCursor(), timeout_s=0.5, poll_s=0.01)
    r1 = brain_bridge.brain_via_bridge("what is two plus two?", **common)
    r2 = brain_bridge.brain_via_bridge("what is the capital of france?", **common)
    check(r1 == answers["what is two plus two?"], f"identity turn 1: {r1!r}")
    check(r2 == answers["what is the capital of france?"], f"identity turn 2: {r2!r}")

    # The #48 fix: a dropped reply costs ONLY its own turn. Turn 1's reply is dropped;
    # turn 2 still returns its own answer by identity. Under pure positional
    # correlation turn 2 would time out too (its block sits at slot 0 while the cursor
    # polls slot 1) -- the collateral damage this closes.
    syl2 = _StampedSyl(drop={1})
    common2 = dict(persona="syl", send=syl2.send, read_reply=syl2.read,
                   cursor=brain_bridge.ReplyCursor(), timeout_s=0.3, poll_s=0.02)
    d1 = brain_bridge.brain_via_bridge("what is two plus two?", **common2)
    d2 = brain_bridge.brain_via_bridge("what is the capital of france?", **common2)
    check(d1.startswith("Sorry, the brain took too long"),
          f"dropped reply still times out its own turn: {d1!r}")
    check(d2 == answers["what is the capital of france?"],
          f"next turn keeps its own answer despite the drop (#48 fixed): {d2!r}")

    # Mixed/transition seam (the hardest case): a stamped block that is NOT this turn's
    # -- a different event_id -- sitting at the positional target slot must be SKIPPED
    # by the positional fallback, never consumed. Otherwise a dropped stamped reply
    # would just relocate the #48 theft to the next turn. event_id is an internal uuid,
    # so the literal eids below never match this turn's id.
    seam_file = (
        "\n--- t delivery_id=d1 event_id=eid-one ---\nanswer one\n"
        "\n--- t delivery_id=dx event_id=eid-x ---\nanswer x\n"
    )
    seam = brain_bridge.brain_via_bridge(
        "stamped reply dropped for this turn", persona="syl",
        send=lambda *a, **k: None, read_reply=lambda: seam_file,
        cursor=brain_bridge.ReplyCursor(consumed=1), timeout_s=0.1, poll_s=0.02)
    check(seam.startswith("Sorry, the brain took too long"),
          f"non-matching stamped block at the target slot is skipped: {seam!r}")

    # Cursor coherence across mixed resolution: an unstamped turn (positional) then a
    # stamped turn (identity) through one cursor each return their own answer, and the
    # cursor still counts one slot per send.
    mix = _StampedSyl(plain={1})
    mcur = brain_bridge.ReplyCursor()
    mcommon = dict(persona="syl", send=mix.send, read_reply=mix.read,
                   cursor=mcur, timeout_s=0.5, poll_s=0.01)
    m1 = brain_bridge.brain_via_bridge("what is two plus two?", **mcommon)
    m2 = brain_bridge.brain_via_bridge("what is the capital of france?", **mcommon)
    check(m1 == answers["what is two plus two?"],
          f"mixed: unstamped turn resolves positionally: {m1!r}")
    check(m2 == answers["what is the capital of france?"],
          f"mixed: stamped turn resolves by identity: {m2!r}")
    check(mcur.consumed == 2, f"mixed: cursor counts both sends: {mcur.consumed}")

    # Transition coherence: an identity match for a reply that is NOT at the positional
    # target must realign the cursor to the matched block, or a later UNSTAMPED turn
    # that falls back to position polls a slot past its own reply and times out. The
    # sequence: a stamped turn dropped, a stamped turn answered at slot 0, then an
    # unstamped turn answered at slot 1.
    trans = _StampedSyl(drop={1}, plain={3})
    tcur = brain_bridge.ReplyCursor()
    tcommon = dict(persona="syl", send=trans.send, read_reply=trans.read,
                   cursor=tcur, timeout_s=0.3, poll_s=0.02)
    t1 = brain_bridge.brain_via_bridge("what is two plus two?", **tcommon)
    t2 = brain_bridge.brain_via_bridge("what is the capital of france?", **tcommon)
    t3 = brain_bridge.brain_via_bridge("what is two plus two?", **tcommon)
    check(t1.startswith("Sorry, the brain took too long"),
          f"transition: dropped stamped turn times out: {t1!r}")
    check(t2 == answers["what is the capital of france?"],
          f"transition: stamped turn matches by identity off-target: {t2!r}")
    check(t3 == answers["what is two plus two?"],
          f"transition: later unstamped turn reads its own slot (cursor realigned): {t3!r}")

    # End-to-end identity match through the real components (model-free): the actual
    # local_sim_send stamps the inbox event_id, a real SimPersona echoes it into the
    # reply block, and the bridge identity-matches -- proving the wire composes, not
    # just the in-test stub.
    e2e_inbox = d / "e2e-inbox.jsonl"
    e2e_reply = d / "e2e-reply.txt"
    e2e_sim = SimPersona(e2e_inbox, e2e_reply, poll_s=0.05, echo_event_id=True)
    e2e_sim.start()
    try:
        ecommon = dict(persona="syl", send=brain_bridge.local_sim_send(e2e_inbox),
                       read_reply=brain_bridge.file_reply_reader(e2e_reply),
                       cursor=brain_bridge.ReplyCursor(), timeout_s=10, poll_s=0.05)
        e1 = brain_bridge.brain_via_bridge("what is two plus two?", **ecommon)
        e2 = brain_bridge.brain_via_bridge("what is the capital of france?", **ecommon)
        check(e1 == "Two plus two is four.", f"e2e identity turn 1: {e1!r}")
        check(e2 == "The capital of France is Paris.", f"e2e identity turn 2: {e2!r}")
    finally:
        e2e_sim.stop()

    n_pass = sum(1 for ok, _ in results if ok)
    print(f"\n=== {n_pass}/{len(results)} checks passed ===")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
