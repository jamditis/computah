#!/usr/bin/env python3
"""Brain backend that routes through a bot-spren persona instead of `claude -p`.

The prototype's original brain() shelled `claude -p`, a one-shot non-interactive
call. The brain is moving to a long-running interactive bot-spren persona (Syl)
so the assistant draws on the normal subscription, keeps memory across turns, and
is the same Claude Joe talks to over Telegram (one assistant, voice or text).

bot-spren is message-passing, not request/response:
  - send: `bot-spren send <persona> "<text>"` appends one JSON event to the
    persona's manual-inbox.jsonl; a ManualCLIFileInbound adapter tails it.
  - reply: when the persona finishes a turn, its Stop hook delivers the response
    to an outbound adapter. A FileOutbound appends a block to a reply file:

        --- <iso-ts> delivery_id=<id> ---
        <reply text>

There is no built-in correlation between the inbound event_id and the outbound
delivery_id, so brain_via_bridge() correlates by position: snapshot the last
delivery_id, send, then poll the reply file until a NEW block appears. Voice
turns are serialized (one utterance at a time), so the next new block is this
turn's reply.

Transport is injected so the same logic works in three settings:
  - persona on this host: cli_send + file_reply_reader
  - persona on another host (pipeline on houseofjawn, Syl on officejawn):
    ssh_cli_send + ssh_reply_reader, over Tailscale
  - tests: local_sim_send + file_reply_reader against sim_persona, so the whole
    loop runs with no CLI, no session, and no mic.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# A FileOutbound block header line: "--- <ts> delivery_id=<id> ---".
_DELIVERY_RE = re.compile(r"^--- .* delivery_id=(\S+) ---$", re.MULTILINE)

SendFn = Callable[[str, str], None]   # (persona, prompt) -> None, raises on failure
ReplyReader = Callable[[], str]       # () -> full reply-file text ("" if absent)


def _last_delivery(reply_text: str) -> tuple[str | None, str]:
    """Return (delivery_id, payload) of the last FileOutbound block.

    payload is everything after the last header line to end of text, stripped of
    the surrounding newlines FileOutbound adds. Returns (None, "") when no block
    is present yet.
    """
    matches = list(_DELIVERY_RE.finditer(reply_text))
    if not matches:
        return None, ""
    last = matches[-1]
    return last.group(1), reply_text[last.end():].strip("\n")


def brain_via_bridge(
    text: str,
    *,
    persona: str,
    send: SendFn,
    read_reply: ReplyReader,
    system_prompt: str | None = None,
    timeout_s: int = 120,
    poll_s: float = 0.5,
) -> str:
    """Send `text` to the persona and return its reply, or a spoken error string.

    Never raises for an expected failure (timeout, send error): the caller is a
    voice loop, so a short spoken sentence is more useful than a traceback.
    """
    prev_id, _ = _last_delivery(read_reply())
    prompt = text if system_prompt is None else f"{system_prompt}\n\nUser: {text}"
    try:
        send(persona, prompt)
    except Exception as e:  # transport failure (ssh down, CLI missing, ...)
        return f"Sorry, I couldn't reach the brain ({type(e).__name__})."

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        cur_id, payload = _last_delivery(read_reply())
        if cur_id is not None and cur_id != prev_id and payload:
            return payload
        time.sleep(poll_s)
    return "Sorry, the brain took too long to answer."


# --------------------------------------------------------------------------- #
# Concrete transports
# --------------------------------------------------------------------------- #
def cli_send(bot_spren_bin: str = "bot-spren") -> SendFn:
    """Send via the local bot-spren CLI (persona on this host)."""
    def _send(persona: str, prompt: str) -> None:
        subprocess.run([bot_spren_bin, "send", persona, prompt],
                       capture_output=True, text=True, timeout=30, check=True)
    return _send


def ssh_cli_send(host: str, bot_spren_bin: str = "bot-spren") -> SendFn:
    """Send via bot-spren on a remote host over ssh.

    ssh does not preserve argv boundaries past the host: everything after it is
    joined into one string and run by the remote login shell. So the remote
    command is built explicitly and every field is shell-quoted. The prompt is
    untrusted (transcribed speech), so this prevents both word-splitting and
    shell-metacharacter execution on the brain host.
    """
    def _send(persona: str, prompt: str) -> None:
        remote = " ".join(shlex.quote(p) for p in (bot_spren_bin, "send", persona, prompt))
        subprocess.run(
            ["ssh", "-o", "ConnectTimeout=15", host, remote],
            capture_output=True, text=True, timeout=40, check=True,
        )
    return _send


def file_reply_reader(reply_path: str | Path) -> ReplyReader:
    """Read the reply file directly (persona on this host).

    Honors the ReplyReader contract: never raises. A missing or unreadable file
    means "no reply yet" -> "", so the poll loop keeps going and brain_via_bridge
    falls through to its own spoken timeout rather than crashing the voice loop.
    """
    reply_path = Path(reply_path)
    def _read() -> str:
        try:
            return reply_path.read_text(encoding="utf-8")
        except OSError:
            return ""
    return _read


def ssh_reply_reader(host: str, reply_path: str) -> ReplyReader:
    """Read the reply file on a remote host over ssh.

    Honors the ReplyReader contract: never raises. A flaky/hanging remote (ssh
    timeout, non-zero exit, transport error) returns "" so the poll keeps trying
    and a bad host degrades to a spoken timeout instead of an exception.
    """
    def _read() -> str:
        try:
            proc = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=15", host, "cat", shlex.quote(reply_path)],
                capture_output=True, text=True, timeout=40,
            )
        except (subprocess.TimeoutExpired, OSError):
            return ""
        return proc.stdout if proc.returncode == 0 else ""
    return _read


# --------------------------------------------------------------------------- #
# Test/sim transport: append to a local inbox in bot-spren's `send` format.
# --------------------------------------------------------------------------- #
def local_sim_send(inbox_path: str | Path) -> SendFn:
    """Append to a local manual-inbox.jsonl exactly like `bot-spren send`.

    Lets the pipeline exercise the real send -> inbox -> poll path against the
    sim_persona watcher, with no bot-spren CLI and no persona deployed.
    """
    inbox_path = Path(inbox_path)
    def _send(persona: str, prompt: str) -> None:
        inbox_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "type": "manual", "source": "cli", "payload": prompt,
            "event_id": str(uuid.uuid4()),
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }
        with inbox_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    return _send
