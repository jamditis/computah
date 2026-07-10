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

## Primary use case

Voice-first and eyes-free: the user is away from a keyboard (often just looking at a
phone) and wants to act on something without getting to a computer — most often "file
a GitHub issue about this" or capture a quick note. So computah is a voice-driven
action executor, not a chatbot. The brain (Syl) has a real toolset and permissions,
so it carries the instruction out — it files the issue — rather than only talking
about it.

That makes faithful execution of explicit spoken instructions the core requirement,
not raw reasoning. Syl runs on a fast, low-cycle model (Sonnet 4.6 medium) and stays
in the conversation: anything that is not pure voice back-and-forth — any real task or
tool use — it hands to a subagent immediately rather than doing inline. Two reasons.
Interruptibility: while a subagent works, Syl is still listening, so a follow-up
correction ("no, change the title") can interrupt and redirect the subagent instead of
waiting for it to finish and redoing the work. Capability: the subagent's model scales
with the task — a routine action at the default tier, a genuinely hard one on a higher
tier (Opus 4.8). Either way the subagent is ephemeral (spawn, work, exit), not a
second resident session.

Two consequences for the pipeline. Eyes-free means a spoken confirmation is the only
feedback channel: the brain must say what it did ("filed issue 46 in computah") and
read an external or destructive action back before doing it. And because speech is
misheard, a mishear guard belongs in front of any action that creates or changes
something, so a garbled word never files a garbage issue.

## Architecture

Four stages, each independently swappable (`pipeline.py`):

1. `detect_wake` — openWakeWord (ONNX). 80 ms frames; peak score vs threshold.
   For a live mic, `stream_detect_wake` + `capture_request` drive detection and
   energy-based endpointing over a continuous frame stream instead of a whole
   file, and `run_turn` runs one full turn from that stream. `stream_detect_wake`
   keeps a `_PREROLL_FRAMES`-deep ring buffer of the most recent frames and
   `capture_request` prepends it, so a command spoken with no pause after the wake
   word is not clipped by detection latency (the pre-roll is dropped on an
   abandoned wake, so it never becomes a phantom request). The pre-roll size is
   tuned to the deployed mic's real wake-detection latency. On a live mic the wake
   fire can also play an acknowledgment chime (`chime.py`, `wake_chime` config key)
   before capture, fired through `run_turn`'s `on_wake` hook; the loop handles it
   half-duplex so the cue is not captured into the request, and the detection
   pre-roll is dropped with it so the pre-cue wake-word tail does not prepend the
   post-cue command. The chime is opt-in,
   default off: on a half-duplex device the cue and capture cannot overlap, so when
   it is on a command spoken in one breath with the wake word loses its leading
   audio (it lands in the cue window) -- the no-pause pre-roll case. Off keeps that
   primary flow intact; gating the cue on a pause so both coexist is the deferred fix.
2. `transcribe` — faster-whisper (CTranslate2, int8). `transcribe_detailed` also
   returns the decoder's confidence (`avg_logprob`, `no_speech_prob`). Both live
   paths (`run_turn` and `live_driver`) gate on it through `guard_transcript` so a
   misheard or silence-derived command never reaches the action-capable brain (the
   signal-level mishear guard); a rejected turn speaks a short re-prompt instead of
   dispatching.
3. `brain` — the transcript goes to a persistent assistant; the reply is short
   spoken text.
4. `speak` — Piper (ONNX) renders the reply to a WAV.

Module-level caches keep models warm across calls in one process. Audio is
normalized to 16 kHz mono int16 (`_load_pcm16`).

### The brain bridge (`brain_bridge.py`)

The brain stage routes through a persistent session using a file-based contract (the
bot-spren framework's): send appends one event to an inbox file; the session writes
its reply to a reply file as a delimited block; `brain_via_bridge` reserves the next
reply slot with a persistent cursor, sends, then polls for the block at that slot — so
a late reply from a timed-out turn fills its own reserved slot instead of being read
as the next turn's answer.

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
.venv/bin/python test_confidence_guard.py  # mishear guard decision + aggregation — fast, no models
.venv/bin/python test_preroll.py           # pre-roll buffer keeps a no-pause request's leading audio — fast, no models
.venv/bin/python test_endpoint_config.py   # endpoint_silence_ms / max_request_ms tune capture endpointing — fast, no models
.venv/bin/python test_chime.py             # wake-acknowledgment cue generator + both-loop wiring — fast, no models
.venv/bin/python test_live_driver.py       # live_driver hardware path honors the guard — fast, no models
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

- Positional reply correlation in the bridge: the reply format carries no correlation
  key, so `brain_via_bridge` correlates by position with a persistent cursor that
  reserves one reply slot per send. A late reply from a timed-out turn fills its own
  reserved slot and is skipped, so it is not mis-attributed to the next turn; a reply
  dropped entirely (never written) would otherwise wedge the cursor one ahead, so after
  a couple of consecutive timeouts it resyncs to the live end and recovers. Positional
  correlation still cannot tell a slow reply from a dropped one, and a relaunch across
  an in-flight prompt can re-seed a one-turn lag — drain the session to quiescence
  before relaunching. The robust fix is a real correlation key: the reply echoing the
  request's `event_id` so the match is by identity, not position.
- `speak()` shells a fresh Piper process per call and reloads the voice model
  (~3.9 s). Keeping Piper resident is the planned fix.

## File index

- `pipeline.py` — stages, chain, CLI, and the live-streaming turn (`stream_detect_wake`, `capture_request`, `run_turn`). `run_turn` exposes `on_wake` (fires at the wake->capture boundary, for the chime) and `on_capture` (fires after capture, for half-duplex mic pause); `run_loop` is the desktop live loop and wires both.
- `live_driver.py` — always-on live loop: real mic (arecord on stdin) -> wake -> chime -> STT -> brain -> spoken reply, re-arming after each turn. Gates the transcript through `pipeline.guard_transcript` before dispatch, the same mishear guard as `run_turn`.
- `chime.py` — the wake-acknowledgment cue (issue #41): a pure-DSP generator for the two-tone rising chime played the instant the wake fires, before capture. Backend-free (no sounddevice/aplay) so both live loops can render the cue and play it through their own output. Gated by the `wake_chime` config key (opt-in, default off — it regresses the no-pause case on a half-duplex device); half-duplex handling keeps the cue out of the captured request when it is on.
- `brain_bridge.py` — bridge plus transports.
- `sim_persona.py` — test stand-in for the assistant.
- `test_*.py` — see Dev commands.
- `config.json` — wake word, thresholds, the wake-chime toggle (`wake_chime`,
  opt-in/default off), model choices, brain backend toggle, the mishear-guard
  thresholds (`stt_confidence_guard`, `stt_min_avg_logprob`, `stt_max_no_speech_prob`),
  and the request-endpointing knobs (`endpoint_silence_ms`, `max_request_ms`).
- `config.local.json` (gitignored) / `config.local.example.json` — deployment bridge settings.
