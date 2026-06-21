# CLAUDE.md

Guidance for Claude Code (and any contributor) working in the computah repository.

## What this is

computah is a local voice assistant with a changeable, custom-trainable wake word.
Audio comes in, a wake word triggers it, speech is transcribed, a persistent
assistant session answers, and the answer is spoken back. It runs CPU-only on a
Raspberry Pi, with no cloud speech API and no PyTorch.

The design goal: the voice assistant and the user's existing text assistant are the
same session with shared memory. So the brain is a long-running session reached over
a bridge, not a stateless per-utterance API call.

## Architecture

Four stages, each independently swappable (`pipeline.py`):

1. `detect_wake` — openWakeWord (ONNX). 80 ms frames; peak score vs threshold.
   For a live mic, `stream_detect_wake` + `capture_request` drive detection and
   energy-based endpointing over a continuous frame stream instead of a whole
   file, and `run_turn` runs one full turn from that stream.
2. `transcribe` — faster-whisper (CTranslate2, int8).
3. `brain` — the transcript goes to a persistent assistant; the reply is short
   spoken text.
4. `speak` — Piper (ONNX) renders the reply to a WAV.

Module-level caches keep models warm across calls in one process. Audio is
normalized to 16 kHz mono int16 (`_load_pcm16`).

### The brain bridge (`brain_bridge.py`)

The brain stage routes through a persistent session using a file-based contract (the
bot-spren framework's): send appends one event to an inbox file; the session writes
its reply to a reply file as a delimited block; `brain_via_bridge` snapshots the last
block, sends, then polls for the next new one.

Transport is injected, so the same logic serves three settings: assistant on the
same host (`cli_send` + `file_reply_reader`), on another host (`ssh_cli_send` +
`ssh_reply_reader`), or a test stand-in (`local_sim_send` against `sim_persona.py`).
Deployment-specific values — the persona name, the transport host, the reply-file
path — belong in configuration or environment, not hardcoded, and anything sensitive
stays out of version control.

`brain()` in `pipeline.py` dispatches on the `brain_backend` config key: `cli` (the
`claude -p` fallback, the default so a fresh clone runs standalone) or `bridge`.
Bridge settings live in `config.local.json`, which is gitignored and overrides
`config.json`. `load_config` returns the merged view; `set_wake_word` writes only the
committed base, so persisting the wake word never copies local overrides into
`config.json`.

## Invariants

- The production brain is a persistent assistant session via the bridge, not a
  one-shot `claude -p` call. `pipeline.brain()` dispatches on `brain_backend`; the
  CLI path (`_brain_cli`) is a development fallback only.
- The assistant session is a shared, long-running resource (the user also talks to
  it by text). Treat any change to its configuration as deliberate and tested; do not
  break its existing function.
- Keep model binaries out of git (`voices/`, `whisper_models/`, `*.onnx`). Reproduce
  the environment from `requirements.txt` plus the README setup steps.
- No emojis in code, logs, or commit messages. Sentence case for headings and UI
  text.
- No AI attribution in commits, code, or docs.

## Memory discipline

The pipeline loads several models. On a memory-constrained host, run the model tests
under a cap and check free memory first:

```bash
systemd-run --user --scope -p MemoryMax=1500M -p MemorySwapMax=0 \
  .venv/bin/python test_pipeline_bridge.py
```

The persistent assistant session is the large, always-resident consumer; keep it on
the host with the most headroom. The pipeline's own footprint is small (~500 MB
warm).

## Dev commands

```bash
# environment
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m piper.download_voices en_US-lessac-medium --download-dir voices

# tests
.venv/bin/python test_brain_bridge.py      # bridge logic — fast, no models
.venv/bin/python test_brain_dispatch.py    # config selects the backend — fast, no models
.venv/bin/python test_pipeline_bridge.py   # full chain + bridge brain (loads models)
.venv/bin/python test_pipeline.py          # full chain + fallback CLI brain
.venv/bin/python test_stream_turn.py       # streaming detect + endpointing + run_turn (loads models)

# wake words
.venv/bin/python pipeline.py --list-wake-words
.venv/bin/python pipeline.py --set-wake-word <name>
```

## Testing strategy

Tests run mic-free: audio is synthesized with Piper, so the whole chain is exercised
with no microphone, no live session, and (for the bridge test) no models. The sim
persona must model a real persistent session — one instance across many turns,
tracking its inbox offset — so tests do not create a fresh persona per turn. A fresh
persona resets its read offset to 0, re-reads old inbox lines, and emits a duplicate
reply that races the next turn's poll. That is a test-harness artifact, not a bridge
bug.

## Known limitations

- Positional reply correlation in the bridge: `brain_via_bridge` returns the next new
  reply block after sending. A late reply from a timed-out turn can be mis-attributed
  on the following turn. The reply format carries no correlation key; mitigate with a
  generous timeout. Voice turns are serialized, so this is rare.
- `speak()` shells a fresh Piper process per call and reloads the voice model
  (~3.9 s). Keeping Piper resident is the planned fix.

## File index

- `pipeline.py` — stages, chain, CLI, and the live-streaming turn (`stream_detect_wake`, `capture_request`, `run_turn`).
- `brain_bridge.py` — bridge plus transports.
- `sim_persona.py` — test stand-in for the assistant.
- `test_*.py` — see Dev commands.
- `config.json` — wake word, thresholds, model choices, brain backend toggle.
- `config.local.json` (gitignored) / `config.local.example.json` — deployment bridge settings.
