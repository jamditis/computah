# Changelog

All notable changes to computah are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
