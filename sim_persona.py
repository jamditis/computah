#!/usr/bin/env python3
"""Stand-in for Syl: tail a manual-inbox.jsonl, write a reply per message.

Replaces the persona + bridge + FileOutbound for mic-free, session-free tests of
the voice loop. It tails the same inbox `bot-spren send` (or local_sim_send)
writes, and appends a reply to the reply file in FileOutbound's exact block
format, so brain_via_bridge() cannot tell it from a real persona.

The "brain" here is a trivial canned responder — enough to prove the
wake -> STT -> brain -> TTS loop end to end. The real brain is Syl.

Use as a daemon thread inside a test (SimPersona(...).start()), or as a
standalone process: `python sim_persona.py --inbox <f> --reply <f>`.
"""

from __future__ import annotations

import argparse
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


def canned_reply(text: str) -> str:
    """A trivial deterministic brain for loop tests. The real brain is Syl."""
    t = text.lower()
    if "two plus two" in t or "2 plus 2" in t or "2+2" in t:
        return "Two plus two is four."
    if "capital of france" in t:
        return "The capital of France is Paris."
    if "time" in t:
        return "I cannot check the time yet, but that is coming."
    # Echo so a test can assert the transcript made the round trip.
    return f"You said: {text.strip()}"


def file_outbound_append(reply_path: Path, payload: str, delivery_id: str,
                         event_id: str | None = None) -> None:
    """Append one block in FileOutbound's exact on-disk format. When event_id is
    given, stamp it into the header so the bridge can match the reply to its request
    by identity (#19) — models the future stamped producer."""
    ts = datetime.now(timezone.utc).isoformat()
    reply_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"--- {ts} delivery_id={delivery_id}"
    if event_id is not None:
        header += f" event_id={event_id}"
    header += " ---"
    with reply_path.open("a", encoding="utf-8") as f:
        f.write(f"\n{header}\n{payload}\n")


class SimPersona:
    """Tail an inbox file and reply to each new line. Use as a daemon thread."""

    def __init__(
        self,
        inbox_path: str | Path,
        reply_path: str | Path,
        reply_fn: Callable[[str], str] = canned_reply,
        poll_s: float = 0.2,
        echo_event_id: bool = False,
    ) -> None:
        self.inbox = Path(inbox_path)
        self.reply = Path(reply_path)
        self.reply_fn = reply_fn
        self.poll_s = poll_s
        # When set, echo each request's event_id into its reply block header so the
        # bridge can identity-match (#19). Off by default keeps the legacy positional
        # behavior the other tests exercise.
        self.echo_event_id = echo_event_id
        self._offset = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _tick(self) -> None:
        if not self.inbox.exists():
            return
        data = self.inbox.read_bytes()
        if len(data) <= self._offset:
            return
        new = data[self._offset:]
        nl = new.rfind(b"\n")
        if nl == -1:
            return  # only a partial line so far
        complete = new[: nl + 1]
        self._offset += len(complete)
        for raw in complete.split(b"\n"):
            if not raw.strip():
                continue
            try:
                entry = json.loads(raw.decode())
                payload = entry["payload"]
            except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                continue
            # Strip the optional "system\n\nUser: <text>" wrapper to get the words.
            text = payload.split("User: ", 1)[-1] if "User: " in payload else payload
            eid = entry.get("event_id") if self.echo_event_id else None
            file_outbound_append(self.reply, self.reply_fn(text),
                                 str(uuid.uuid4()), event_id=eid)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="sim-persona")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.poll_s):
            self._tick()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


def main() -> int:
    p = argparse.ArgumentParser(description="Syl stand-in for voice-loop tests")
    p.add_argument("--inbox", required=True, help="manual-inbox.jsonl to tail")
    p.add_argument("--reply", required=True, help="reply file to append to")
    args = p.parse_args()
    sim = SimPersona(args.inbox, args.reply)
    sim.start()
    print(f"sim_persona watching {args.inbox} -> {args.reply} (Ctrl-C to stop)")
    try:
        if sim._thread is not None:
            sim._thread.join()
    except KeyboardInterrupt:
        sim.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
