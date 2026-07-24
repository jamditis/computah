# Changelog

All notable changes to computah are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `prep_wake_samples.py --clean` (#8): removes leftover clips of a take the run
  just re-recorded (a `<stem>_NNN.wav` clip whose stem this run wrote) from
  `--output`, so re-recording with fewer utterances no longer strands orphaned
  higher-numbered clips that training and eval would still read. Off by default
  (deletion is opt-in). The source manifest below broadens cleanup to the
  prep-owned clips of an intentionally omitted take while still sparing source
  recordings and hand-curated audio. With no manifest or an unreadable one, the
  narrow same-stem rule remains as a warned legacy fallback; readable older
  records without source ownership fail closed until removed. Without `--clean`
  the run still warns and names the leftover count. `test_prep_wake_samples.py`
  covers the re-run-with-fewer-clips case with and without `--clean`, and that
  source-aware `--clean` spares files prep did not record.
- `prep_wake_samples.py` output manifest (#84): prep records each resolved source
  path and the clips it owns in `.prep-manifest.json` in the output dir.
  `--clean` uses that provenance to remove orphans the stem match cannot reach --
  clips from an input dropped between runs, and clips under a disambiguated stem
  (`take-1_000.wav`) that a later run no longer produces. Before a destructive
  run, prep names every omitted source and how many of its clips `--clean` will
  remove. A run must share an exact recorded source path before it may write to
  that dataset, so a different dataset with the same label and generic basename
  cannot overwrite or clean it.
  Malformed ownership maps, including clip names that are paths instead of safe
  output filenames, are refused before any audio is decoded or written.
  Manifested stems remain reserved for their source while new output names are
  assigned, so a new same-basename recording cannot overwrite the only good
  clips from a source whose rerun is silent. A source this run attempted but got
  no clips from retains those prior clips. With a readable source map,
  hand-curated audio stays outside it and is never deleted. A non-clean legacy
  refresh that leaves ambiguous same-stem files does not write a manifest; the
  next `--clean` remains on the documented filename fallback instead of turning
  a guess into durable provenance. A no-output run cannot claim an unrecorded
  directory, and a first-time source that writes no clips cannot gain cleanup
  authority over an existing dataset. Manifest writes use atomic replacement.
  Only an absent manifest degrades `--clean` to the narrow stem match with an
  explicit warning; that fallback cannot distinguish a prep leftover from a
  hand-added `<same-stem>_NNN.wav`. An unreadable, malformed, or non-v2 manifest
  fails closed until the user removes it to bootstrap ownership explicitly.
  Existing pre-manifest directories must bootstrap with the complete source set;
  clips under stems omitted from a partial first run remain deliberately unowned.
- Configurable request endpointing (#15): the trailing-silence window that ends a
  captured request and the max-request cap that bounds a runaway are now config keys
  (`endpoint_silence_ms`, `max_request_ms`, milliseconds) instead of fixed constants,
  so the live loop is tunable without a code change. `run_turn` and the `live_driver`
  loop thread the config through `capture_request`; both keys default to the built-in
  values (800 ms endpoint, 8000 ms cap), so the shipped config leaves capture behavior
  unchanged. `test_endpoint_config.py` covers the conversion, the no-drift default
  invariant, the tuned endpoint and cap, and the `run_turn` wiring, model-free.
- Mishear confidence guard: both live paths (`pipeline.run_turn` and the
  `live_driver` hardware loop) now gate the transcript on faster-whisper confidence
  before dispatching to the brain, through a shared `guard_transcript`, so a garbled
  command cannot trigger an action on either path. `avg_logprob` is the gate;
  following faster-whisper's own no-speech rule, a high `no_speech_prob` only marks a
  reject as silence when the decode is also unconfident, so a clear command is never
  dropped for it alone. A rejected turn speaks a short re-prompt instead of calling
  the brain. New config keys `stt_confidence_guard`, `stt_min_avg_logprob`, and
  `stt_max_no_speech_prob` (defaults mirror faster-whisper's own thresholds).
  `transcribe_detailed` returns the new `Transcript` (text plus the two signals);
  `transcribe` still returns text.
- `test_confidence_guard.py`: fast, no-model coverage of the gate decision, segment
  aggregation, and the shared `guard_transcript`. `test_live_driver.py`: fast,
  no-model proof the hardware path honors the guard. Two model-tier checks in
  `test_stream_turn.py` prove a low-confidence transcript skips the brain and a
  confident one passes through.
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
