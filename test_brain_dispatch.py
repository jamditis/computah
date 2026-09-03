#!/usr/bin/env python3
"""brain() dispatches on the brain_backend config key.

test_pipeline_bridge.py proves the full bridge chain through build_brain(). This
proves the step before that: config alone selects the backend and transport, so a
real deployment enables the persistent-session brain by editing config.local.json,
with no code change. No models and no bot-spren CLI are needed — config selects the
sim transport and points its paths at temp files, so the repo's own config is
untouched.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import brain_bridge
import pipeline
from sim_persona import SimPersona

results: list[tuple[bool, str]] = []


def check(ok: bool, detail: str) -> None:
    results.append((ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {detail}")


def _write(path: Path, cfg: dict) -> None:
    path.write_text(json.dumps(cfg))


def test_routing(d: Path) -> None:
    """brain_backend picks the helper; cli is the default for a bare config."""
    pipeline.CONFIG_PATH = d / "config.json"
    pipeline.LOCAL_CONFIG_PATH = d / "config.local.json"
    pipeline._brain_cli = lambda text, cfg, **kw: "CLI"
    pipeline._brain_bridge = lambda text, cfg: "BRIDGE"

    _write(pipeline.CONFIG_PATH, {"brain_backend": "cli"})
    check(pipeline.brain("hi") == "CLI", "brain_backend=cli routes to the CLI path")

    _write(pipeline.CONFIG_PATH, {"brain_backend": "bridge"})
    check(
        pipeline.brain("hi") == "BRIDGE",
        "brain_backend=bridge routes to the bridge path",
    )

    # A config with no brain_backend key falls back to the DEFAULTS value (cli),
    # so a fresh clone runs standalone.
    _write(pipeline.CONFIG_PATH, {"wake_word": "hey_jarvis"})
    check(pipeline.brain("hi") == "CLI", "missing brain_backend defaults to cli")


def test_bridge_via_config(d: Path) -> None:
    """With only config.local.json set to bridge, brain() answers via the sim."""
    inbox = d / "manual-inbox.jsonl"
    reply = d / "reply.txt"

    # config.json stays minimal; config.local.json enables the bridge. This is the
    # committed-base + gitignored-overlay split a real deployment uses.
    pipeline.CONFIG_PATH = d / "config.json"
    pipeline.LOCAL_CONFIG_PATH = d / "config.local.json"
    _write(pipeline.CONFIG_PATH, {"wake_word": "hey_jarvis"})
    _write(
        pipeline.LOCAL_CONFIG_PATH,
        {
            "brain_backend": "bridge",
            "brain_transport": "sim",
            "brain_persona": "syl",
            "brain_reply_path": str(reply),
            "brain_inbox_path": str(inbox),
            "brain_timeout_s": 5,
            "brain_poll_s": 0.05,
        },
    )

    sim = SimPersona(inbox, reply, poll_s=0.02)
    sim.start()
    try:
        out = pipeline.brain("what is two plus two?")
    finally:
        sim.stop()

    check(
        out == "Two plus two is four.",
        f"config-driven sim transport answered: {out!r}",
    )
    sent = json.loads(inbox.read_text().splitlines()[0])["payload"]
    check(
        sent == f"{pipeline.SYL_VOICE_SYSTEM_PROMPT}\n\nUser: what is two plus two?",
        "the bridge sends Syl the voice dispatcher policy before the transcript",
    )


def test_voice_dispatch_policy() -> None:
    """The voice contract pins both escalation paths and the non-blocking handoff."""
    policy = pipeline.SYL_VOICE_SYSTEM_PROMPT.casefold()
    required = {
        "explicit escalation": "explicitly asks you to hand work off",
        "implicit escalation": "needs tools, multiple steps, or more than a quick answer",
        "ephemeral background worker": "ephemeral subagent in the background",
        "immediate voice-loop return": "do not do delegated work inline or wait",
        "mid-task correction": "follow-up corrections while the subagent runs",
        "spoken result": "summarize each unreported result once",
        "one bridge reply per turn": "never emit a second reply for the original turn",
        "voice-event scope": "apply these instructions only to voice events",
        "confirmation before external change": (
            "do not start the subagent or act until joe confirms"
        ),
        "spoken reply budget": "at most 500 characters",
        "deferred overflow result": "leave other results unreported",
    }
    for behavior, instruction in required.items():
        check(instruction in policy, f"voice policy includes {behavior}")
    check(
        pipeline._bridge_voice_system_prompt("syl") == pipeline.SYL_VOICE_SYSTEM_PROMPT,
        "the Syl bridge persona receives the dispatcher policy",
    )
    check(
        pipeline._bridge_voice_system_prompt("assistant")
        == pipeline.VOICE_SYSTEM_PROMPT,
        "a non-Syl bridge persona keeps the neutral voice policy",
    )
    check(
        pipeline._bridge_voice_system_prompt(None) == pipeline.VOICE_SYSTEM_PROMPT,
        "a null bridge persona falls back to the neutral voice policy",
    )
    check(
        pipeline._bridge_voice_system_prompt(49) == pipeline.VOICE_SYSTEM_PROMPT,
        "a non-string bridge persona falls back to the neutral voice policy",
    )


def test_factory_validation() -> None:
    """Invalid deployment settings stay inside the voice error boundary."""
    cases = (
        ({}, "reply path is not configured"),
        (
            {"brain_reply_path": "/tmp/reply", "brain_transport": "ssh"},
            "host is not configured",
        ),
        (
            {"brain_reply_path": "/tmp/reply", "brain_transport": "sim"},
            "inbox path is not configured",
        ),
        (
            {"brain_reply_path": "/tmp/reply", "brain_transport": "carrier-pigeon"},
            "is not supported",
        ),
    )
    for cfg, expected in cases:
        reply = brain_bridge.build_brain(cfg)("hello")
        check(expected in reply, f"bad bridge config returns a spoken error: {reply!r}")


def test_cli_voice_prompt(real_brain_cli) -> None:
    """The tool-free CLI fallback must not promise a background handoff."""
    captured = {}
    real_run = pipeline.subprocess.run

    def fake_run(*_args, **kwargs):
        captured["prompt"] = kwargs["input"]
        return SimpleNamespace(returncode=0, stdout="Done.\n", stderr="")

    pipeline.subprocess.run = fake_run
    try:
        out = real_brain_cli("file an issue", dict(pipeline.DEFAULTS))
    finally:
        pipeline.subprocess.run = real_run

    check(out == "Done.", f"CLI fallback still returns its reply: {out!r}")
    check(
        captured["prompt"]
        == f"{pipeline.VOICE_SYSTEM_PROMPT}\n\nUser: file an issue\nAssistant:",
        "CLI fallback uses the tool-free voice prompt",
    )
    check(
        "subagent" not in captured["prompt"].casefold(),
        "CLI fallback does not claim it can delegate",
    )


def test_set_wake_word_no_leak(d: Path) -> None:
    """--set-wake-word writes only the base config, never the local overlay."""
    pipeline.CONFIG_PATH = d / "config.json"
    pipeline.LOCAL_CONFIG_PATH = d / "config.local.json"
    _write(pipeline.CONFIG_PATH, {"wake_word": "alexa"})
    _write(
        pipeline.LOCAL_CONFIG_PATH,
        {
            "brain_backend": "bridge",
            "brain_host": "secret-host",
            "brain_reply_path": "/secret/path",
            "brain_persona": "secret-persona",
        },
    )

    pipeline.set_wake_word("hey_jarvis")
    written = json.loads(pipeline.CONFIG_PATH.read_text())

    local_keys = ("brain_backend", "brain_host", "brain_reply_path", "brain_persona")
    leaked = [k for k in local_keys if k in written]
    check(not leaked, f"config.json keeps the local overlay out (leaked: {leaked})")
    check(written.get("wake_word") == "hey_jarvis", "config.json wake_word was updated")
    check(
        pipeline.load_config()["brain_backend"] == "bridge",
        "effective config still reads the local overlay (runtime unaffected)",
    )


def test_set_wake_word_local(d: Path) -> None:
    """--set-wake-word --local writes config.local.json, leaving the base intact."""
    pipeline.CONFIG_PATH = d / "config.json"
    pipeline.LOCAL_CONFIG_PATH = d / "config.local.json"
    _write(pipeline.CONFIG_PATH, {"wake_word": "hey_jarvis"})
    _write(
        pipeline.LOCAL_CONFIG_PATH,
        {"brain_backend": "bridge", "brain_host": "secret-host"},
    )

    pipeline.set_wake_word("alexa", local=True)

    base = json.loads(pipeline.CONFIG_PATH.read_text())
    overlay = json.loads(pipeline.LOCAL_CONFIG_PATH.read_text())
    check(
        base.get("wake_word") == "hey_jarvis",
        "committed config.json default is left untouched by a local set",
    )
    check(
        overlay.get("wake_word") == "alexa",
        "config.local.json carries the local wake-word override",
    )
    check(
        overlay.get("brain_host") == "secret-host",
        "existing config.local.json keys survive the local set",
    )
    check(
        pipeline.load_config()["wake_word"] == "alexa",
        "effective config reads the local wake word (overlay wins)",
    )


def test_set_wake_word_local_malformed_no_clobber(d: Path) -> None:
    """--set-wake-word --local refuses to overwrite a malformed config.local.json."""
    pipeline.CONFIG_PATH = d / "config.json"
    pipeline.LOCAL_CONFIG_PATH = d / "config.local.json"
    _write(pipeline.CONFIG_PATH, {"wake_word": "hey_jarvis"})
    # A truncated overlay an operator left mid-edit, still carrying live settings.
    truncated = '{"brain_host": "secret-host",'
    pipeline.LOCAL_CONFIG_PATH.write_text(truncated)

    raised = False
    try:
        pipeline.set_wake_word("alexa", local=True)
    except ValueError:
        raised = True

    check(raised, "a malformed config.local.json raises instead of clobbering")
    check(
        pipeline.LOCAL_CONFIG_PATH.read_text() == truncated,
        "the malformed overlay is left intact for the operator to fix",
    )


def test_set_wake_word_base_shadowed_by_local(d: Path) -> None:
    """A base set stays shadowed by a local override — the invariant the CLI note keys off."""
    pipeline.CONFIG_PATH = d / "config.json"
    pipeline.LOCAL_CONFIG_PATH = d / "config.local.json"
    _write(pipeline.CONFIG_PATH, {})
    _write(pipeline.LOCAL_CONFIG_PATH, {"wake_word": "alexa"})

    cfg = pipeline.set_wake_word("hey_jarvis", local=False)

    base = json.loads(pipeline.CONFIG_PATH.read_text())
    check(base.get("wake_word") == "hey_jarvis", "the base write lands in config.json")
    check(
        cfg["wake_word"] == "alexa",
        "effective wake word stays the local override, so the CLI flags it as shadowed",
    )


def main() -> int:
    # Snapshot module state so the test restores the repo's real wiring on exit.
    saved = (
        pipeline.CONFIG_PATH,
        pipeline.LOCAL_CONFIG_PATH,
        pipeline._brain_cli,
        pipeline._brain_bridge,
    )
    try:
        with tempfile.TemporaryDirectory(prefix="brain-dispatch-") as tmp:
            d = Path(tmp)
            test_routing(d)
            test_voice_dispatch_policy()
            test_factory_validation()
            # Restore the real helpers before the live-ish bridge test.
            pipeline._brain_cli, pipeline._brain_bridge = saved[2], saved[3]
            test_cli_voice_prompt(saved[2])
            test_bridge_via_config(d)
            test_set_wake_word_no_leak(d)
            test_set_wake_word_local(d)
            test_set_wake_word_local_malformed_no_clobber(d)
            test_set_wake_word_base_shadowed_by_local(d)
    finally:
        (
            pipeline.CONFIG_PATH,
            pipeline.LOCAL_CONFIG_PATH,
            pipeline._brain_cli,
            pipeline._brain_bridge,
        ) = saved

    n_pass = sum(1 for ok, _ in results if ok)
    print(f"=== {n_pass}/{len(results)} checks passed ===")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
