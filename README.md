# computah

A local voice assistant for people who want the assistant they already use by text to answer by voice too.

computah listens for a wake word, transcribes the request, sends the transcript to a persistent assistant session, and speaks the answer back. The speech path runs locally on CPU with openWakeWord, faster-whisper, and Piper. The brain can run on the same host or behind a small file bridge on another machine.

<img width="1213" height="667" alt="computah screenshot" src="https://github.com/user-attachments/assets/7fbd1b26-35bf-490f-8fac-edb73c74b7b6" />

## What it does

computah is the mic-free core of a local voice assistant. Feed it a wav file and it runs the same stages a live loop will use later:

1. Detect the wake word in the audio.
2. Transcribe the spoken request.
3. Send the transcript to an assistant session that keeps context.
4. Render the answer as a wav file.

The project is built so each stage can be replaced. The wake-word detector, transcriber, brain bridge, and text-to-speech layer have narrow boundaries in `pipeline.py`.

## Why it exists

Most voice assistants bundle wake-word detection, speech recognition, memory, and speech output into one service. That makes the assistant easy to start but hard to own. computah is shaped around different constraints:

- The wake word is configurable today, and a trainable `computah` wake word is the goal.
- Speech recognition and speech synthesis stay local.
- The voice interface talks to the same long-running assistant session used for text.
- Tests can run without a microphone by passing audio files through the pipeline.
- Model files, voices, local session paths, and hostnames stay outside git.

The name is also the intended wake word: “computah,” said in your own voice.

## Current status

v0.1.0 is a working file-based pipeline, not a live microphone appliance yet.

| Area | Status |
| --- | --- |
| Wake-word detection | Works with installed openWakeWord models and custom `.onnx` files in `models/`. |
| Speech-to-text | Works through faster-whisper with CTranslate2 int8. |
| Brain | Supports a fallback CLI backend and the persistent file bridge. |
| Text-to-speech | Works through Piper by writing a reply wav. |
| Live loop | Planned. |
| Custom `computah` wake word | Planned; recording notes live in `docs/recording-computah.md`. |

## How the pipeline works

```text
audio in
  └─▶ wake word       openWakeWord, ONNX, 80 ms frames
      └─▶ transcript  faster-whisper, CTranslate2 int8
          └─▶ brain   persistent assistant bridge or CLI fallback
              └─▶ wav Piper, ONNX voice model
```

| Stage | Default implementation | Boundary |
| --- | --- | --- |
| Wake word | openWakeWord | `detect_wake` returns whether the configured phrase fired. |
| Speech-to-text | faster-whisper | `transcribe` returns text from the audio file. |
| Brain | bridge or CLI | `brain` returns short spoken text. |
| Text-to-speech | Piper | `speak` writes the answer to a wav file. |

Module-level caches keep the wake-word and Whisper models warm inside one process. Wake-word detection normalizes audio to 16 kHz mono int16; transcription passes the wav file to faster-whisper.

## The brain bridge

The bridge is the main design choice. computah does not need to create a fresh assistant call for every voice turn. It can append a user event to an inbox file and wait for the next reply block from an already-running assistant session.

`brain_bridge.py` keeps the transport injectable:

- `cli_send` and `file_reply_reader` talk to a session on the same host.
- `ssh_cli_send` and `ssh_reply_reader` use the same file contract on another host.
- `local_sim_send` and `sim_persona.py` let tests exercise bridge behavior without a live assistant.

`brain_via_bridge` snapshots the latest reply block, sends one transcript, then polls until a newer block appears. Voice turns are serialized, so this simple positional contract is enough for the current prototype.

A fresh clone uses `brain_backend: "cli"` so it can run without bridge setup. To use the persistent session path, copy `config.local.example.json` to `config.local.json`, set `brain_backend` to `bridge`, and keep deployment values there. `config.local.json` is gitignored and overrides `config.json` at runtime.

## Repository layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | Pipeline stages, config loading, and CLI entry point. |
| `brain_bridge.py` | Bridge contract plus local, ssh, and simulated transports. |
| `sim_persona.py` | Test stand-in for a long-running assistant session. |
| `prep_wake_samples.py` | Converts wake-word recordings into training clips. |
| `config.json` | Committed defaults for wake word, model choices, and backend selection. |
| `config.local.example.json` | Template for gitignored local bridge settings. |
| `requirements.txt` | Python dependencies for the CPU-only speech path. |
| `docs/` | GitHub Pages site and recording notes. |
| `models/`, `voices/`, `whisper_models/` | Local model directories; generated or downloaded files stay out of git. |
| `test_*.py` | Mic-free tests for the bridge, dispatch logic, sample prep, and pipeline. |

## Setup

The project is developed on Linux/ARM64 with Python 3.13 on a Raspberry Pi 5. Other Linux hosts should work if the same dependencies are available.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m piper.download_voices en_US-lessac-medium --download-dir voices
```

faster-whisper downloads its model into `whisper_models/` on first use. Piper voices live in `voices/`. Custom wake-word models live in `models/`. These files are local artifacts and are not committed.

## Usage

List available wake words:

```bash
.venv/bin/python pipeline.py --list-wake-words
```

Switch the active wake word and persist it to `config.json`:

```bash
.venv/bin/python pipeline.py --set-wake-word hey_jarvis
```

Run the pipeline on a wav file:

```bash
.venv/bin/python pipeline.py clip.wav -o reply.wav
```

The output path receives the spoken reply.

## Configuration

`config.json` contains safe defaults that can be committed:

| Key | Meaning |
| --- | --- |
| `wake_word` | Active openWakeWord model name. |
| `wake_threshold` | Detection score required before the pipeline continues. |
| `whisper_model` | faster-whisper model size or path. |
| `whisper_compute` | CTranslate2 compute type, usually `int8` on the target device. |
| `voice_model` | Piper voice name/stem; `speak` loads `voices/<voice_model>.onnx`. |
| `brain_backend` | `cli` for standalone fallback or `bridge` for the persistent session path. |
| `claude_model` | Model name for the fallback CLI brain. |
| `claude_timeout_s` | Timeout for the fallback CLI brain. |

Use `config.local.json` for machine-specific bridge values. It is gitignored and merged over `config.json`, so private hostnames, usernames, and assistant paths do not leak into commits.

## Custom wake words

Built-in openWakeWord phrases appear in `--list-wake-words`. To add a custom wake word, place a trained `<name>.onnx` model in `models/`. It appears as `<name>` and can be selected the same way as a built-in model:

```bash
.venv/bin/python pipeline.py --set-wake-word <name>
```

A custom model overrides a built-in model with the same name. Keep custom model files local unless you intend to publish them.

## Testing

Start with the fast tests. They do not load speech models:

```bash
.venv/bin/python test_brain_bridge.py
.venv/bin/python test_brain_dispatch.py
.venv/bin/python test_prep_wake_samples.py
```

Run model-dependent tests when the voice and Whisper models are present:

```bash
.venv/bin/python test_pipeline_bridge.py
.venv/bin/python test_pipeline.py
```

On a memory-constrained host, cap the full bridge test:

```bash
systemd-run --user --scope -p MemoryMax=1500M -p MemorySwapMax=0 \
  .venv/bin/python test_pipeline_bridge.py
```

## Latency notes

Measured on a Raspberry Pi 5 with warm models and a simulated brain:

| Stage | Approximate time |
| --- | --- |
| Wake detection | 0.8 s |
| Speech-to-text | 3.3 s |
| Text-to-speech | 3.9 s |

The largest known cost is `speak()`: it shells out to Piper for each reply, so the voice model reloads every turn. Keeping Piper resident is the likely next latency win.

## Roadmap

- Train and ship a custom `computah` openWakeWord model.
- Add a live microphone loop with endpointing and playback.
- Keep Piper loaded between turns.
- Exercise the bridge against a live assistant session over the network.
- Explore a small wake-word satellite that streams audio only after detection.

## Known limitations

The bridge correlates replies by position. It returns the next reply block after the transcript is sent. If one turn times out and its late reply arrives during the next turn, that reply can be misattributed. Voice turns are serialized, which lowers the risk, but the reply format needs an explicit correlation key before this is fully solved.

## Contributing

Keep changes small and tested:

1. Read `CLAUDE.md` before changing architecture or public docs.
2. Use sentence case for headings and user-facing text.
3. Avoid filler and hype; say what the project does in plain language.
4. Update docs when behavior changes.
5. Run the fastest relevant test first, then broader tests when local models are available.
6. Keep generated models, voices, local config, and assistant session data out of git.

## GitHub Pages

The site in `docs/` is ready for GitHub Pages. In repository settings, deploy Pages from the `docs` folder on the current branch. The page includes an SVG favicon and PNG social preview metadata.

## License

MIT. See `LICENSE`.
