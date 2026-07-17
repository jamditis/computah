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

The reply can echo the request's event_id in its header ("event_id=<id>"), so
brain_via_bridge() matches a reply to its request by identity (#19). When the field
is absent — the producer does not stamp it yet — it falls back to positional
correlation with a persistent cursor (ReplyCursor): each send reserves the next
reply slot, and the turn reads the block at that slot. Voice turns are serialized,
and a send reserves its slot even when it times out, so a late reply from a timed-out
turn fills its own reserved slot and is skipped — it is not mis-read as the next
turn's answer. The parser tolerates the optional event_id token whether or not a
producer emits it, so the bridge can ship before the producer side.

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

# A FileOutbound block header line: "--- <ts> delivery_id=<id> ---", optionally
# carrying the originating request's "event_id=<id>" so a reply can be matched to its
# request by identity instead of file position (#19). The event_id token is optional:
# a producer that does not stamp it yet (today's bot-spren) still parses, and the
# bridge falls back to positional correlation for those blocks. The consumer tolerates
# the token before any producer emits it, so the rollout needs no flag day.
_DELIVERY_RE = re.compile(
    r"^--- .* delivery_id=(\S+)(?: event_id=(\S+))? ---$", re.MULTILINE
)

# (persona, prompt, *, event_id=None) -> None, raises on failure. event_id is this
# turn's correlation id (#19); a transport that can carry it stamps it into the inbound
# event so the reply can echo it, others accept and ignore it (positional fallback).
SendFn = Callable[..., None]
ReplyReader = Callable[[], str]  # () -> full reply-file text ("" if absent)


def _delivery_blocks(reply_text: str) -> list[tuple[str, str | None, str]]:
    """Return every FileOutbound block as (delivery_id, event_id, payload), in file
    order.

    event_id is the originating request id when the producer stamped it (#19), else
    None. payload is the text from one header line to the next header (or end of
    text), stripped of the surrounding newlines FileOutbound adds. Empty list when no
    block is present yet.
    """
    matches = list(_DELIVERY_RE.finditer(reply_text))
    blocks: list[tuple[str, str | None, str]] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(reply_text)
        blocks.append((m.group(1), m.group(2), reply_text[m.end() : end].strip("\n")))
    return blocks


# Consecutive timeouts after which the cursor is assumed to have out-run the reply
# file because a reply was dropped (never written), not merely delayed — at which
# point it resyncs to the live end instead of wedging forever. A single late reply
# never reaches this: the next turn reads its own slot and resets the counter. Two
# is the smallest value that still distinguishes one slow turn from a real drop.
_RESYNC_AFTER_MISSES = 2


class ReplyCursor:
    """Persistent positional watermark for the reply file, shared across turns.

    Tracks how many FileOutbound blocks the bridge has accounted for. Initialized
    lazily to the block count present at the first send, so blocks already in the
    file (a prior session, or a reply still in flight from a timed-out turn) are
    skipped instead of being read as this turn's answer. Each send advances the
    cursor by one — reserving that turn's reply slot even when the turn times out,
    which is what stops a late reply from shifting every following turn by one.

    `misses` counts consecutive timeouts. A reply that lands resets it; once it
    reaches _RESYNC_AFTER_MISSES while the cursor sits ahead of the file, the bridge
    concludes a reply was dropped (not delayed) and resyncs to the live end, so a
    dropped reply self-heals instead of wedging the loop. The robust fix is a real
    correlation key linking request to reply; this is the bounded interim.
    """

    __slots__ = ("consumed", "misses")

    def __init__(self, consumed: int | None = None, misses: int = 0) -> None:
        self.consumed = consumed
        self.misses = misses


def brain_via_bridge(
    text: str,
    *,
    persona: str,
    send: SendFn,
    read_reply: ReplyReader,
    cursor: ReplyCursor | None = None,
    system_prompt: str | None = None,
    timeout_s: int = 120,
    poll_s: float = 0.5,
) -> str:
    """Send `text` to the persona and return its reply, or a spoken error string.

    Correlates the reply by position using `cursor` (see ReplyCursor): the send
    reserves the next slot and the turn returns the block at that slot. Pass a
    cursor that persists across turns so timeouts and in-flight replies stay
    aligned; with no cursor a fresh one is used, which is correct only for a
    single isolated, backlog-free turn.

    Never raises for an expected failure (timeout, send error): the caller is a
    voice loop, so a short spoken sentence is more useful than a traceback.
    """
    if cursor is None:
        cursor = ReplyCursor()
    blocks_now = len(_delivery_blocks(read_reply()))
    if cursor.consumed is None:
        cursor.consumed = blocks_now
    elif cursor.misses >= _RESYNC_AFTER_MISSES and cursor.consumed > blocks_now:
        # The cursor has out-run the reply file across several turns: a reply was
        # dropped (the session never wrote it), not merely delayed, so every turn
        # since has reserved a slot that can never fill. Resync to the live end —
        # skipping the dead slots — rather than time out forever. A correlation key
        # would make this exact; positional correlation cannot tell dropped from late.
        cursor.consumed = blocks_now
        cursor.misses = 0
    target = cursor.consumed

    prompt = text if system_prompt is None else f"{system_prompt}\n\nUser: {text}"
    event_id = str(uuid.uuid4())  # this turn's correlation id (#19)
    try:
        send(persona, prompt, event_id=event_id)
    except Exception as e:  # transport failure (ssh down, CLI missing, ...)
        return f"Sorry, I couldn't reach the brain ({type(e).__name__})."
    # Reserve this turn's slot even if it times out below, so a late reply fills
    # the reserved slot and is skipped next turn instead of shifting it by one.
    cursor.consumed = target + 1

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        blocks = _delivery_blocks(read_reply())
        # Identity match (#19): a reply that echoes this turn's event_id is ours
        # whatever its file position, so a dropped or reordered reply on another turn
        # cannot shift it. Active only once the producer stamps event_id; until then
        # no block carries one and this loop falls through to the positional branch.
        for _did, eid, payload in blocks:
            if eid == event_id and payload:
                # Identity match is cursor-independent and deliberately does NOT touch
                # the positional cursor. Realigning it to the matched slot would have to
                # rewind past slots reserved for earlier timed-out sends, and positional
                # correlation cannot tell a dropped reserved slot from a merely late one
                # — rewinding could let a late legacy reply be spoken as the next turn's
                # answer (the invariant that a late reply fills its own reserved slot and
                # is skipped). So coherence between a stamped match and a later unstamped
                # fallback turn is deferred to the producer step (#59), where full
                # stamping removes the unstamped fallback entirely.
                cursor.misses = 0
                return payload
        # Positional fallback, for unstamped blocks only. A stamped block that is not
        # ours is left alone (never consumed by position), so a dropped stamped reply
        # cannot make us read another turn's answer — the failure #48 hits today.
        if len(blocks) > target:
            _did, eid, payload = blocks[target]
            if eid is None and payload:
                cursor.misses = 0
                return payload
        time.sleep(poll_s)
    cursor.misses += 1
    return "Sorry, the brain took too long to answer."


# --------------------------------------------------------------------------- #
# Concrete transports
# --------------------------------------------------------------------------- #
def _send_argv(
    bot_spren_bin: str, working_dir: str | None, persona: str, prompt: str
) -> list[str]:
    """Build the `bot-spren send` argv, inserting -d when a working dir is set.

    bot-spren resolves the persona's inbox from its working directory (--working-dir,
    default ~/.bot-spren/<name>), NOT from BOT_SPREN_STATE_DIR. A send without -d
    therefore writes to ~/.bot-spren/<name>/state/manual-inbox.jsonl — a dead-letter
    file the running session (which reads BOT_SPREN_STATE_DIR) never tails, so the
    message is silently lost and the turn just times out. Passing -d <persona project
    dir> points the send at the same inbox the session consumes.
    """
    argv = [bot_spren_bin, "send"]
    if working_dir:
        argv += ["-d", working_dir]
    argv += [persona, prompt]
    return argv


def cli_send(
    bot_spren_bin: str = "bot-spren", working_dir: str | None = None
) -> SendFn:
    """Send via the local bot-spren CLI (persona on this host).

    event_id is accepted for the SendFn contract but not yet forwarded: bot-spren has
    no flag to set the inbound event_id, so stamping the reply (the #19 producer side)
    is a separate, held step. Until it lands these sends are unstamped and the bridge
    matches them positionally.
    """

    def _send(persona: str, prompt: str, *, event_id: str | None = None) -> None:
        subprocess.run(
            _send_argv(bot_spren_bin, working_dir, persona, prompt),
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )

    return _send


def ssh_cli_send(
    host: str, bot_spren_bin: str = "bot-spren", working_dir: str | None = None
) -> SendFn:
    """Send via bot-spren on a remote host over ssh.

    ssh does not preserve argv boundaries past the host: everything after it is
    joined into one string and run by the remote login shell. So the remote
    command is built explicitly and every field is shell-quoted. The prompt is
    untrusted (transcribed speech), so this prevents both word-splitting and
    shell-metacharacter execution on the brain host.
    """

    def _send(persona: str, prompt: str, *, event_id: str | None = None) -> None:
        # event_id accepted but not forwarded yet — see cli_send. The held #19 producer
        # step adds the remote --event-id flow plus the FileOutbound stamp.
        argv = _send_argv(bot_spren_bin, working_dir, persona, prompt)
        remote = " ".join(shlex.quote(p) for p in argv)
        subprocess.run(
            ["ssh", "-o", "ConnectTimeout=15", host, remote],
            capture_output=True,
            text=True,
            timeout=40,
            check=True,
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
                [
                    "ssh",
                    "-o",
                    "ConnectTimeout=15",
                    host,
                    "cat",
                    shlex.quote(reply_path),
                ],
                capture_output=True,
                text=True,
                timeout=40,
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

    def _send(persona: str, prompt: str, *, event_id: str | None = None) -> None:
        inbox_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "type": "manual",
            "source": "cli",
            "payload": prompt,
            "event_id": event_id or str(uuid.uuid4()),
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }
        with inbox_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    return _send
