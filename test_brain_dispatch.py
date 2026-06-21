#!/usr/bin/env python3
"""brain() dispatches on the brain_backend config key.

test_pipeline_bridge.py proves the bridge mechanics by monkeypatching
pipeline.brain directly. This proves the step before that: config alone selects
the backend, so a real deployment enables the persistent-session brain by editing
config.local.json, with no code change. No models and no bot-spren CLI are needed
-- the local transport is swapped for the sim sender, and config paths are pointed
at temp files so the repo's own config is untouched.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

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
    check(pipeline.brain("hi") == "BRIDGE",
          "brain_backend=bridge routes to the bridge path")

    # A config with no brain_backend key falls back to the DEFAULTS value (cli),
    # so a fresh clone runs standalone.
    _write(pipeline.CONFIG_PATH, {"wake_word": "hey_jarvis"})
    check(pipeline.brain("hi") == "CLI", "missing brain_backend defaults to cli")


def test_bridge_via_config(d: Path, real_brain_bridge) -> None:
    """With only config.local.json set to bridge, brain() answers via the sim."""
    inbox = d / "manual-inbox.jsonl"
    reply = d / "reply.txt"

    # config.json stays minimal; config.local.json enables the bridge. This is the
    # committed-base + gitignored-overlay split a real deployment uses.
    pipeline.CONFIG_PATH = d / "config.json"
    pipeline.LOCAL_CONFIG_PATH = d / "config.local.json"
    _write(pipeline.CONFIG_PATH, {"wake_word": "hey_jarvis"})
    _write(pipeline.LOCAL_CONFIG_PATH, {
        "brain_backend": "bridge",
        "brain_transport": "local",
        "brain_persona": "sim",
        "brain_reply_path": str(reply),
        "brain_timeout_s": 5,
        "brain_poll_s": 0.05,
    })

    # The local transport normally shells the real bot-spren CLI. Swap just the
    # sender for the sim writer; the reply reader is the real file reader.
    real_brain_bridge.cli_send = lambda _bin: real_brain_bridge.local_sim_send(inbox)

    sim = SimPersona(inbox, reply, poll_s=0.02)
    sim.start()
    try:
        out = pipeline.brain("what is two plus two?")
    finally:
        sim.stop()

    check(out == "Two plus two is four.",
          f"config-driven local bridge answered correctly: {out!r}")


def test_set_wake_word_no_leak(d: Path) -> None:
    """--set-wake-word writes only the base config, never the local overlay."""
    pipeline.CONFIG_PATH = d / "config.json"
    pipeline.LOCAL_CONFIG_PATH = d / "config.local.json"
    _write(pipeline.CONFIG_PATH, {"wake_word": "alexa"})
    _write(pipeline.LOCAL_CONFIG_PATH, {
        "brain_backend": "bridge",
        "brain_host": "secret-host",
        "brain_reply_path": "/secret/path",
        "brain_persona": "secret-persona",
    })

    pipeline.set_wake_word("hey_jarvis")
    written = json.loads(pipeline.CONFIG_PATH.read_text())

    local_keys = ("brain_backend", "brain_host", "brain_reply_path", "brain_persona")
    leaked = [k for k in local_keys if k in written]
    check(not leaked,
          f"config.json keeps the local overlay out (leaked: {leaked})")
    check(written.get("wake_word") == "hey_jarvis",
          "config.json wake_word was updated")
    check(pipeline.load_config()["brain_backend"] == "bridge",
          "effective config still reads the local overlay (runtime unaffected)")


def main() -> int:
    # Snapshot module state so the test restores the repo's real wiring on exit.
    saved = (pipeline.CONFIG_PATH, pipeline.LOCAL_CONFIG_PATH,
             pipeline._brain_cli, pipeline._brain_bridge, brain_bridge.cli_send)
    try:
        with tempfile.TemporaryDirectory(prefix="brain-dispatch-") as tmp:
            d = Path(tmp)
            test_routing(d)
            # Restore the real helpers before the live-ish bridge test.
            pipeline._brain_cli, pipeline._brain_bridge = saved[2], saved[3]
            test_bridge_via_config(d, brain_bridge)
            test_set_wake_word_no_leak(d)
    finally:
        (pipeline.CONFIG_PATH, pipeline.LOCAL_CONFIG_PATH,
         pipeline._brain_cli, pipeline._brain_bridge,
         brain_bridge.cli_send) = saved

    n_pass = sum(1 for ok, _ in results if ok)
    print(f"=== {n_pass}/{len(results)} checks passed ===")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
