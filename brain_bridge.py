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

        --- <iso-ts> delivery_id=<id> event_id=<request-id> ---
        <reply text>

The reply echoes the request's event_id in its header ("event_id=<id>"), so
brain_via_bridge() matches a reply to its request by identity (#19). When an older
producer omits the field, the bridge falls back to positional correlation with a
persistent cursor (ReplyCursor): each send reserves the next reply slot, and the
turn reads the block at that slot. Voice turns are serialized, and a send reserves
its slot even when it times out, so a late reply from a timed-out turn fills its own
reserved slot and is skipped. Keeping event_id optional supports a rolling upgrade.

Transport is injected so the same logic works in three settings:
  - persona on this host: cli_send + file_reply_reader
  - persona on another host (pipeline on houseofjawn, Syl on officejawn):
    ssh_cli_send + ssh_reply_reader, over Tailscale
  - tests: local_sim_send + file_reply_reader against sim_persona, so the whole
    loop runs with no CLI, no session, and no mic.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger("computah.brain")


class BrainFailure(str):
    """Spoken error text with a stable reason for callers that measure answers.

    It remains a string so the voice loops can speak it without an exception or
    a separate error path. The reason comes from the failing backend, never prose.
    """

    reason: str

    def __new__(cls, text: str, reason: str) -> BrainFailure:
        reply = super().__new__(cls, text)
        reply.reason = reason
        return reply


# A FileOutbound block header line: "--- <ts> delivery_id=<id> ---", optionally
# carrying the originating request's "event_id=<id>" so a reply can be matched to its
# request by identity instead of file position (#19). The event_id token is optional:
# a legacy producer still parses, and the bridge falls back to positional correlation
# for those blocks.
_DELIVERY_RE = re.compile(
    r"^--- .* delivery_id=(\S+)(?: event_id=(\S+))? ---$", re.MULTILINE
)

# (persona, prompt, *, event_id=None) -> None or bool, raises on failure. event_id is
# this turn's correlation id (#19); the concrete transports pass it to bot-spren so
# the reply producer can echo it. Test stand-ins may accept and ignore it. A send
# returns False when an older bot-spren made it drop the id (the landing probe then
# has nothing to look up); None or True means the id reached the inbox.
SendFn = Callable[..., "bool | None"]
ReplyReader = Callable[[], str]  # () -> full reply-file text ("" if absent)

# (event_id) -> True when an event carrying this turn's event_id is in the inbox the
# SESSION consumes, False when it is not there yet, or None when the inbox cannot be
# observed (a transport blip) so the caller skips the check rather than
# false-alarming. It lets the bridge confirm a send landed before it waits on a
# reply (#44): `bot-spren send` can exit 0 yet append to an inbox the session never
# reads (a working-dir/inbox mismatch), which otherwise looks identical to a slow
# brain, a full timeout with no diagnostic. Matching the exact event_id (#98) needs
# no pre-send baseline, so the check adds no probe before the send.
LandingProbe = Callable[[str], "bool | None"]


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

    `misses` counts consecutive timeouts in the legacy positional fallback. A reply
    that lands resets it; once it reaches _RESYNC_AFTER_MISSES while the cursor sits
    ahead of the file, the bridge concludes a reply was dropped and resyncs to the
    live end so a missing reply does not wedge the loop.
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
    confirm_landing: LandingProbe | None = None,
    system_prompt: str | None = None,
    timeout_s: int = 120,
    landing_timeout_s: float = 10.0,
    poll_s: float = 0.5,
) -> str:
    """Send `text` to the persona and return its reply, or a spoken error string.

    Matches a stamped reply by event_id. For an unstamped reply from a legacy
    producer, `cursor` reserves the next positional slot and the turn returns the
    block at that slot. Pass a cursor that persists across turns so fallback-mode
    timeouts and in-flight replies stay aligned; with no cursor a fresh one is used,
    which is correct only for a single isolated, backlog-free turn.

    When `confirm_landing` is given, the turn checks that the session's inbox holds
    this turn's event_id after the send, for up to `landing_timeout_s`. A send that
    exits 0 but never reached the inbox (a dead-letter working-dir mismatch, #44)
    returns a distinct, logged error instead of masquerading as a slow brain. With no
    probe, a probe that reports the inbox as unobservable, or a legacy send that
    dropped the event_id, the check is skipped.

    Never raises for an expected failure (timeout, send error, non-landing send): the
    caller is a voice loop, so a short spoken sentence is more useful than a traceback.
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
        stamped = send(persona, prompt, event_id=event_id)
    except Exception as e:  # transport failure (ssh down, CLI missing, ...)
        logger.warning("Brain send failed error=%s", type(e).__name__)
        return BrainFailure(
            f"Sorry, I couldn't reach the brain ({type(e).__name__}).", "bridge_send"
        )

    # A send can exit 0 yet dead-letter to an inbox the session never reads (#44):
    # bot-spren resolves the inbox from its working dir, so a missing -d writes to a
    # file nobody tails. That looks identical to a slow brain, a full timeout with no
    # diagnostic. When a probe is configured, look for this turn's event_id in the
    # inbox the session reads, and fail loudly and distinctly if it never appears. A
    # slow brain (the message landed, the reply is just late) still falls through to
    # the timeout below. The lookup runs only after a successful send (#98), so a down
    # host costs one transport timeout, in the send, before the spoken send error.
    if confirm_landing is not None and stamped is False:
        # An older bot-spren rejected --event-id and the send was retried without it,
        # so the inbox event carries bot-spren's own id. Looking up ours would report
        # a false non-landing, so this turn skips the check.
        logger.warning(
            "brain bridge: send to persona %r used the legacy path without "
            "--event-id; skipping the landing check. Upgrade bot-spren.",
            persona,
        )
    elif confirm_landing is not None:
        landing_deadline = time.monotonic() + landing_timeout_s
        while True:
            landed = confirm_landing(event_id)
            if landed is None or landed:
                break  # landed, or the inbox went unobservable, so do not false-alarm
            if time.monotonic() >= landing_deadline:
                logger.error(
                    "brain bridge: send to persona %r did not land; event_id %s was "
                    "not in the session's inbox after %.1fs. Likely a working-dir "
                    "mismatch. The send wrote to a dead-letter inbox the session does "
                    "not read (bot-spren -d). See computah #44.",
                    persona,
                    event_id,
                    landing_timeout_s,
                )
                return BrainFailure(
                    "Sorry, I sent that but it never reached the brain's inbox. "
                    "The message may be going to the wrong place.",
                    "bridge_not_landed",
                )
            time.sleep(poll_s)

    # Reserve this turn's slot even if it times out below, so a late reply fills the
    # reserved slot and is skipped next turn instead of shifting it by one. A
    # non-landing send returned above without reaching here, so it never reserves a
    # slot no reply can fill, so the cursor stays aligned for the next turn.
    cursor.consumed = target + 1

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        blocks = _delivery_blocks(read_reply())
        # Identity match (#19): a reply that echoes this turn's event_id is ours
        # whatever its file position, so a dropped or reordered reply on another turn
        # cannot shift it. An unstamped legacy reply falls through to the positional
        # branch.
        for _did, eid, payload in blocks:
            if eid == event_id and payload:
                # Identity match is cursor-independent and deliberately does NOT touch
                # the positional cursor. Realigning it to the matched slot would have to
                # rewind past slots reserved for earlier timed-out sends, and positional
                # correlation cannot tell a dropped reserved slot from a merely late one
                # — rewinding could let a late legacy reply be spoken as the next turn's
                # answer (the invariant that a late reply fills its own reserved slot and
                # is skipped). So coherence between a stamped match and a later unstamped
                # fallback turn remains a rollout gap (#59); a fully stamped producer
                # removes the unstamped fallback entirely.
                cursor.misses = 0
                return payload
        # Positional fallback, for unstamped blocks only. A stamped block that is not
        # ours is left alone (never consumed by position), so a dropped stamped reply
        # cannot make us read another turn's answer.
        if len(blocks) > target:
            _did, eid, payload = blocks[target]
            if eid is None and payload:
                cursor.misses = 0
                return payload
        time.sleep(poll_s)
    cursor.misses += 1
    logger.warning(
        "Brain reply timed out persona=%r timeout_s=%s consecutive_misses=%d",
        persona,
        timeout_s,
        cursor.misses,
    )
    return BrainFailure("Sorry, the brain took too long to answer.", "bridge_timeout")


# --------------------------------------------------------------------------- #
# Concrete transports
# --------------------------------------------------------------------------- #
def _send_argv(
    bot_spren_bin: str,
    working_dir: str | None,
    persona: str,
    prompt: str,
    event_id: str | None,
) -> list[str]:
    """Build the `bot-spren send` argv with its routing and correlation fields.

    bot-spren resolves the persona's inbox from its working directory (--working-dir,
    default ~/.bot-spren/<name>), NOT from BOT_SPREN_STATE_DIR. A send without -d
    therefore writes to ~/.bot-spren/<name>/state/manual-inbox.jsonl — a dead-letter
    file the running session (which reads BOT_SPREN_STATE_DIR) never tails, so the
    message is silently lost and the turn just times out. Passing -d <persona project
    dir> points the send at the same inbox the session consumes.

    The caller-generated event_id must reach that inbox unchanged. Once FileOutbound
    echoes it, the bridge can match the reply by identity instead of file position.
    """
    argv = [bot_spren_bin, "send"]
    if working_dir:
        argv += ["-d", working_dir]
    if event_id is not None:
        argv += ["--event-id", event_id]
    argv += [persona, prompt]
    return argv


_UNKNOWN_EVENT_ID_OPTION = "No such option: --event-id"


def _run_send_command(
    command: list[str], *, legacy_command: list[str] | None, timeout_s: int
) -> bool:
    """Run a send, falling back only when an older CLI rejects --event-id.

    Returns False when the legacy retry ran, so the event_id never reached the inbox.

    Click rejects an unknown option before it invokes bot-spren's send handler, so
    this exact failure cannot have appended an inbox event. Retrying without the
    option is therefore safe during a staged bot-spren upgrade. Every other failure
    remains fatal so a send with an unknown outcome is never duplicated.
    """
    try:
        subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        if legacy_command is None or _UNKNOWN_EVENT_ID_OPTION not in (exc.stderr or ""):
            raise
        subprocess.run(
            legacy_command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=True,
        )
        return False
    return True


def cli_send(
    bot_spren_bin: str = "bot-spren", working_dir: str | None = None
) -> SendFn:
    """Send via the local bot-spren CLI (persona on this host).

    bot-spren's --event-id option preserves the bridge's correlation id in the
    inbound event. An older CLI gets one safe positional-mode retry without the
    unsupported option; only replies from that compatibility path remain unstamped
    and use positional matching.
    """

    def _send(persona: str, prompt: str, *, event_id: str | None = None) -> bool:
        return _run_send_command(
            _send_argv(bot_spren_bin, working_dir, persona, prompt, event_id),
            legacy_command=(
                _send_argv(bot_spren_bin, working_dir, persona, prompt, None)
                if event_id is not None
                else None
            ),
            timeout_s=30,
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

    def _send(persona: str, prompt: str, *, event_id: str | None = None) -> bool:
        argv = _send_argv(bot_spren_bin, working_dir, persona, prompt, event_id)
        remote = " ".join(shlex.quote(p) for p in argv)
        legacy_remote = None
        if event_id is not None:
            legacy_argv = _send_argv(bot_spren_bin, working_dir, persona, prompt, None)
            legacy_remote = " ".join(shlex.quote(p) for p in legacy_argv)
        return _run_send_command(
            ["ssh", "-o", "ConnectTimeout=15", host, remote],
            legacy_command=(
                ["ssh", "-o", "ConnectTimeout=15", host, legacy_remote]
                if legacy_remote is not None
                else None
            ),
            timeout_s=40,
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


def file_inbox_probe(inbox_path: str | Path) -> LandingProbe:
    """Look up an event_id in a local inbox file (persona on this host), for #44/#98.

    `bot-spren send` appends one JSON event per line, carrying the caller's event_id.
    A missing file is False (the session's inbox has received nothing here yet, which
    is exactly the non-landing signal). A line that is not a JSON object is skipped.
    Any other read error is None so a transient failure skips the check instead of
    raising a false alarm.
    """
    inbox_path = Path(inbox_path)

    def _has(event_id: str) -> bool | None:
        try:
            with inbox_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(entry, dict) and entry.get("event_id") == event_id:
                        return True
            return False
        except FileNotFoundError:
            return False
        except (OSError, UnicodeDecodeError):
            return None

    return _has


def ssh_inbox_probe(host: str, inbox_path: str) -> LandingProbe:
    """Look up an event_id in a remote inbox over ssh (persona on another host), #98.

    Mirrors file_inbox_probe over ssh with one command per lookup: a missing file is
    False, and any failure (ssh timeout or transport error, an unreadable inbox,
    unexpected output) returns None so a flaky host skips the landing check rather
    than reporting a false non-landing. The pattern is anchored to the `event_id`
    key, like the local probe's field match, so the id quoted in another field of
    some other event does not count. grep's own status separates a miss (1) from
    a read error (2).
    """

    # Quote once here, so a malformed path fails when the brain is built, before any
    # send, instead of raising mid-turn after the prompt is already on its way.
    path = shlex.quote(inbox_path)

    def _has(event_id: str) -> bool | None:
        # The bridge's ids are uuid4s: hex and hyphens, literal inside an ERE. Any
        # other id cannot be matched safely, so it is unobservable, not absent.
        if not re.fullmatch(r"[0-9A-Za-z-]+", event_id):
            return None
        pattern = shlex.quote(f'"event_id" *: *"{event_id}"')
        remote = (
            f"if [ -f {path} ]; then grep -qE -- {pattern} {path}; "
            f"case $? in 0) echo FOUND;; 1) echo ABSENT;; *) echo ERROR;; esac; "
            f"else echo MISSING; fi"
        )
        try:
            proc = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=15", host, remote],
                capture_output=True,
                text=True,
                timeout=40,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        if proc.returncode != 0:
            return None
        out = proc.stdout.strip()
        if out == "FOUND":
            return True
        if out in ("ABSENT", "MISSING"):
            return False
        return None

    return _has


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


def build_brain(
    cfg: dict,
    *,
    system_prompt: str | None = None,
    cursor: ReplyCursor | None = None,
) -> Callable[[str], str]:
    """Build a bridge-backed ``brain(text)`` callable from configuration.

    Supported transports are ``local``, ``ssh``, and ``sim``. Deployment-specific
    hosts and paths stay in the supplied configuration, while missing or unknown
    settings degrade to a spoken error instead of escaping into the voice loop.
    """
    reply_path = cfg.get("brain_reply_path") or ""
    if not reply_path:
        return lambda _text: BrainFailure(
            "Sorry, the brain reply path is not configured.", "bridge_reply_path"
        )

    persona = cfg.get("brain_persona") or "assistant"
    transport = cfg.get("brain_transport")
    if transport is None or transport == "":
        transport = "local"
    bot_spren_bin = cfg.get("brain_bot_spren_bin") or "bot-spren"
    workdir = cfg.get("brain_bot_spren_workdir") or None
    inbox_path = cfg.get("brain_inbox_path") or ""

    if transport == "ssh":
        host = cfg.get("brain_host") or ""
        if not host:
            return lambda _text: BrainFailure(
                "Sorry, the brain host is not configured.", "bridge_host"
            )
        send = ssh_cli_send(host, bot_spren_bin, working_dir=workdir)
        read_reply = ssh_reply_reader(host, reply_path)
        confirm_landing = ssh_inbox_probe(host, inbox_path) if inbox_path else None
    elif transport == "local":
        send = cli_send(bot_spren_bin, working_dir=workdir)
        read_reply = file_reply_reader(reply_path)
        confirm_landing = file_inbox_probe(inbox_path) if inbox_path else None
    elif transport == "sim":
        if not inbox_path:
            return lambda _text: BrainFailure(
                "Sorry, the brain inbox path is not configured.", "bridge_inbox_path"
            )
        if not isinstance(inbox_path, (str, Path)):
            return lambda _text: BrainFailure(
                "Sorry, the brain inbox path is not a filesystem path.",
                "bridge_inbox_path",
            )
        send = local_sim_send(inbox_path)
        read_reply = file_reply_reader(reply_path)
        confirm_landing = file_inbox_probe(inbox_path)
    else:
        return lambda _text: BrainFailure(
            f"Sorry, brain transport {transport!r} is not supported.",
            "bridge_transport",
        )

    reply_cursor = cursor if cursor is not None else ReplyCursor()

    def _brain(text: str) -> str:
        return brain_via_bridge(
            text,
            persona=persona,
            send=send,
            read_reply=read_reply,
            cursor=reply_cursor,
            confirm_landing=confirm_landing,
            system_prompt=system_prompt,
            timeout_s=cfg.get("brain_timeout_s", 120),
            poll_s=cfg.get("brain_poll_s", 0.5),
        )

    return _brain
