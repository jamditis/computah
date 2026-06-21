# computah

A local voice assistant for people who want the assistant they already talk to by text to also answer by voice.

computah listens for a wake word, transcribes the request, sends it to a persistent assistant session, and speaks the answer back. The speech stack is local and CPU-only: openWakeWord, faster-whisper, and Piper. The brain can live on the same machine or across the network behind a small file-based bridge.

<img width="1213" height="667" alt="computah screenshot" src="https://github.com/user-attachments/assets/7fbd1b26-35bf-490f-8fac-edb73c74b7b6" />

## Why this exists

Most voice assistants make you pick a fixed wake phrase and route every request through a stateless cloud turn. computah is built around a different shape:

- The wake word is yours and can be swapped from configuration today.
- The voice turn reaches the same long-running assistant session you already use by text.
- Speech recognition and speech synthesis stay local.
- The pipeline can be tested without a microphone by feeding audio files through the same stages.
- Each stage is small enough to replace without rewriting the rest of the project.

The project name is also the intended wake word: “computah,” said the way you say it. A custom trained wake-word model is on the roadmap; the current release can already switch between installed openWakeWord models.

## Project status

v0.1.0 is the mic-free core. The end-to-end path works with audio files:

1. Wake-word detection reads a wav file.
2. Speech-to-text turns the spoken request into text.
3. The brain bridge sends the transcript to a persistent assistant session or a test stand-in.
4. Text-to-speech writes the spoken reply to a wav file.

Live microphone capture, a custom “computah” wake-word model, and lower-latency resident text-to-speech are next.

## How it works

```text
audio in ──▶ wake word ──▶ speech-to-text ──▶ brain bridge ──▶ text-to-speech ──▶ audio out
            openWakeWord   faster-whisper     persistent       Piper
            (ONNX)         (CTranslate2 int8) assistant        (ONNX)
```

| Stage | Default implementation | Why it is here |
| --- | --- | --- |
| wake word | openWakeWord | Scans 80 ms frames locally using ONNX models. |
| speech-to-text | faster-whisper | Runs Whisper through CTranslate2 with int8 weights. |
| brain | file bridge | Talks to a long-running assistant session instead of creating one stateless request per turn. |
| text-to-speech | Piper | Produces local spoken replies through an ONNX voice model. |

All speech stages run CPU-only and are intended to fit on an 8 GB Raspberry Pi.

## The brain bridge

The bridge is the project’s main design choice. A normal voice assistant sends one request and gets one response. computah sends the transcript into a persistent assistant session so the voice assistant and the text assistant can share memory, tone, and context.

The transport is injected in `brain_bridge.py`:

- Local transport appends a turn to an inbox file and reads a reply file.
- SSH transport does the same work on another host.
- Simulated transport uses `sim_persona.py` so tests can exercise the bridge without a real assistant session.

The bridge snapshots the last reply block, sends the new transcript, then polls for the next block. That keeps the production path simple while preserving the same contract in tests.

## Repository layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | The wake-word, transcription, brain, speech, and CLI pipeline. |
| `brain_bridge.py` | File-based bridge and local, ssh, and simulated transports. |
| `sim_persona.py` | Test assistant that tails an inbox and writes replies in the production reply format. |
| `config.json` | Active wake word, thresholds, model names, and paths. |
| `test_brain_bridge.py` | Fast bridge round-trip test with no speech models. |
| `test_pipeline.py` | End-to-end pipeline test using the fallback CLI brain. |
| `test_pipeline_bridge.py` | End-to-end pipeline test using the bridge and simulated persona. |
| `assets/og-image.html` | Source document for the repository social preview image. |
| `docs/` | GitHub Pages site. |

## Setup

The project is developed on Linux/ARM64 with Python 3.13 on a Raspberry Pi 5. It should also run on other Linux hosts with the same dependencies.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# download a Piper voice, about 63 MB
.venv/bin/python -m piper.download_voices en_US-lessac-medium --download-dir voices
```

faster-whisper downloads its model into `whisper_models/` on first use. Model binaries and generated voice assets are intentionally not committed.

## Usage

List installed wake words:

```bash
.venv/bin/python pipeline.py --list-wake-words
```

Switch the active wake word and persist it to `config.json`:

```bash
.venv/bin/python pipeline.py --set-wake-word hey_jarvis
```

Run the full chain on a wav file:

```bash
.venv/bin/python pipeline.py clip.wav -o reply.wav
```

The output file contains the spoken reply.

## Configuration

`config.json` holds the runtime defaults the pipeline reads:

- `wake_word` chooses the active openWakeWord model.
- `wake_threshold` sets how confident detection must be before the pipeline continues.
- `whisper_model` and `whisper_compute` choose the transcription model size and compute type.
- `voice_model` chooses the Piper voice used for replies.
- `claude_model` and `claude_timeout_s` configure the fallback CLI brain.

The brain bridge is wired in code rather than from `config.json`: callers such as `test_pipeline_bridge.py` construct the local, ssh, or simulated transport and point it at the assistant inbox and reply file. Keep hostnames, usernames, and private session paths out of commits when adapting the bridge for a real deployment.

## Testing

Fast bridge test:

```bash
.venv/bin/python test_brain_bridge.py
```

Full bridge pipeline test:

```bash
.venv/bin/python test_pipeline_bridge.py
```

Fallback CLI brain pipeline test:

```bash
.venv/bin/python test_pipeline.py
```

On a memory-constrained host, cap the full model run:

```bash
systemd-run --user --scope -p MemoryMax=1500M -p MemorySwapMax=0 \
  .venv/bin/python test_pipeline_bridge.py
```

The bridge test is the best first check because it avoids large speech models. The pipeline tests synthesize and consume audio, so they are slower and depend on local model availability.

## Latency notes

Measured on a Raspberry Pi 5 with warm models and a simulated brain:

| Stage | Approximate time |
| --- | --- |
| wake detection | 0.8 s |
| speech-to-text | 3.3 s |
| text-to-speech | 3.9 s |

Piper currently reloads its voice model in a fresh subprocess per reply. Keeping Piper resident should cut most of the text-to-speech time.

## Roadmap

- Train a custom “computah” openWakeWord model on real recordings.
- Add a live microphone loop with endpointing and playback.
- Keep Piper resident between turns.
- Point the bridge at a live assistant session over the network.
- Build a small network satellite that listens for the wake word locally and streams audio only after activation.
- Add reply correlation if the upstream reply format gains a turn id.

## Known limitations

The bridge correlates replies by position. It returns the next new reply block after sending a transcript. If one turn times out and its late reply arrives during the next turn, that reply can be misattributed. Voice turns are serialized, so this should be rare, but the limitation is real until replies carry an explicit correlation key.

## Contributing

Keep changes small and testable:

1. Read `CLAUDE.md` for architecture notes and repository rules.
2. Update docs when behavior changes.
3. Run the fastest relevant test first, then broader tests when models are available.
4. Keep generated model files, voices, and local assistant session data out of git.
5. Use sentence case for headings and user-facing text, and keep product names like GitHub and GitHub Pages in their official casing.

## GitHub Pages

The site in `docs/` is ready for GitHub Pages. In the repository settings, set Pages to deploy from the `docs` folder on the current branch. The page includes a matching SVG favicon and social preview metadata.

## License

MIT. See `LICENSE`.
