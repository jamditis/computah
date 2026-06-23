# Changelog

All notable changes to computah are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Mishear confidence guard: a live turn now gates the transcript on faster-whisper
  confidence (`avg_logprob`, `no_speech_prob`) before dispatching to the brain, so a
  garbled or silence-derived command cannot trigger an action. A rejected turn speaks
  a short re-prompt instead of calling the brain. New config keys
  `stt_confidence_guard`, `stt_min_avg_logprob`, and `stt_max_no_speech_prob` (the
  last two default to faster-whisper's own thresholds). `transcribe_detailed` returns
  the new `Transcript` (text plus the two signals); `transcribe` still returns text.
- `test_confidence_guard.py`: fast, no-model coverage of the gate decision and
  segment aggregation, plus two model-tier checks in `test_stream_turn.py` proving a
  low-confidence transcript skips the brain and a confident one passes through.
- Config-selectable brain backend: the `brain_backend` key chooses `cli` (the
  `claude -p` fallback, default) or `bridge` (a persistent assistant session), so the
  bridge is reachable from a real run, not only from tests.
- `config.local.json` overlay (gitignored) for deployment-specific bridge settings —
  persona, transport, ssh host, reply path — overriding the committed `config.json`.
  `config.local.example.json` is a template.
- `test_brain_dispatch.py`: proves config alone selects the backend and that
  persisting the wake word does not leak local config into `config.json`.

### Changed
- `brain()` now dispatches on `brain_backend`; the `claude` CLI path moved to
  `_brain_cli`. `load_config()` merges `config.local.json` over `config.json`.

### Fixed
- `set_wake_word()` wrote the merged config (defaults plus the local overlay) back to
  `config.json`. It now round-trips only the committed base, so the gitignored bridge
  settings can no longer leak into the tracked file.

## [0.1.0] - 2026-06-21

First working core. The full voice chain runs mic-free, driven by audio files.

### Added
- Four-stage pipeline (`pipeline.py`): openWakeWord wake detection, faster-whisper
  speech-to-text (int8, no PyTorch), the brain stage, and Piper text-to-speech, plus
  an end-to-end chain and a CLI.
- Changeable wake word from `config.json`, with `--list-wake-words` and
  `--set-wake-word`.
- Brain bridge (`brain_bridge.py`): routes the transcript to a persistent assistant
  session over a file-based message-passing contract, with injected transports
  (local CLI, ssh, and a test stand-in).
- Simulated persona (`sim_persona.py`) for exercising the brain loop with no live
  session and no microphone.
- Tests: bridge round-trip (`test_brain_bridge.py`), full chain with the bridge
  brain (`test_pipeline_bridge.py`), and full chain with the fallback CLI brain
  (`test_pipeline.py`).
- Pinned dependencies (`requirements.txt`), CPU-only, no PyTorch.

[Unreleased]: https://github.com/jamditis/computah/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jamditis/computah/releases/tag/v0.1.0
