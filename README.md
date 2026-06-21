# COMPUTAH

<img width="1213" height="667" alt="image" src="https://github.com/user-attachments/assets/7fbd1b26-35bf-490f-8fac-edb73c74b7b6" />

A local, self-hosted voice assistant with a wake word you set and change yourself.
Speak to it and it answers in a spoken voice with your own assistant — the same
one you already talk to by text, with the same memory. No cloud speech API, no
PyTorch, runs CPU-only on a Raspberry Pi.

The name is the wake word: "computah," said the way you say it. The point is that
the wake word is yours and retrainable, not a fixed brand phrase.

## Status

v0.1.0 — the mic-free core works end to end. Wake-word detection, speech-to-text,
the brain (via a bridge to a persistent assistant session), and text-to-speech all
run and are tested by feeding audio files in, with no microphone required yet. Live
microphone capture and the custom "computah" wake-word model are the next
milestones (see [Roadmap](#roadmap)).

## How it works

Four stages, each swappable on its own:

```
audio in ──▶ wake word ──▶ speech-to-text ──▶ brain ──▶ text-to-speech ──▶ audio out
            openWakeWord   faster-whisper     bridge    Piper
            (ONNX)         (CTranslate2 int8)           (ONNX)
```

1. Wake word — openWakeWord (ONNX, CPU). Scans audio in 80 ms frames for the active
   wake phrase. The phrase is changeable from config and can be custom-trained.
2. Speech-to-text — faster-whisper on CTranslate2, int8 quantized (tiny.en by
   default). No PyTorch; fits in a few hundred MB.
3. Brain — the transcript is handed to a persistent assistant session over a
   file-based bridge (see below), not a one-shot API call. The reply is short, plain
   text meant to be spoken.
4. Text-to-speech — Piper (ONNX), a natural local voice.

All four are CPU-only and fit in the RAM of an 8 GB Pi.

### The brain bridge

Most voice assistants make a fresh, stateless request to an LLM for every utterance.
computah instead routes the transcript to a long-running assistant session, so the
voice and the existing text chat are the same assistant with the same memory.

The bridge is message-passing over files, matching the bot-spren framework's
contract:

- send — append the transcript as one event to the session's inbox.
- reply — the session writes its answer to a reply file as a delimited block; the
  bridge polls for the next new block.

The transport is injected (`brain_bridge.py`), so the same logic runs three ways:
the assistant on the same host, on another host over ssh, or a simulated stand-in
for tests. Moving where the brain lives changes two functions, not the pipeline.

## Repository layout

| File | What it is |
|------|-----------|
| `pipeline.py` | The four stages, the end-to-end chain, and the CLI. |
| `brain_bridge.py` | Routes the transcript to a persistent assistant session; injected transports (local / ssh / sim). |
| `sim_persona.py` | A stand-in assistant for tests — tails an inbox, writes canned replies in the real reply format. |
| `test_brain_bridge.py` | Bridge round-trip test (no models, no session, no mic). |
| `test_pipeline_bridge.py` | Full chain with real models plus the bridge brain, mic-free. |
| `test_pipeline.py` | Full chain with the fallback CLI brain. |
| `config.json` | Active wake word, thresholds, model choices. |
| `requirements.txt` | Pinned dependencies — CPU-only, no PyTorch. |

## Setup

Python 3.13 on Linux/ARM64 (developed on a Raspberry Pi 5).

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# Piper voice (about 63 MB)
.venv/bin/python -m piper.download_voices en_US-lessac-medium --download-dir voices

# faster-whisper downloads its model into whisper_models/ on first run
```

The model binaries (Piper voice, Whisper weights, openWakeWord ONNX files) are not
committed — they download on setup and first run.

## Usage

Mic-free today: feed it a WAV.

```bash
# list the wake words you can switch between right now
.venv/bin/python pipeline.py --list-wake-words

# switch the active wake word (persists to config.json)
.venv/bin/python pipeline.py --set-wake-word hey_jarvis

# run the full chain on a clip
.venv/bin/python pipeline.py clip.wav -o reply.wav
```

Run the tests:

```bash
.venv/bin/python test_brain_bridge.py     # bridge logic, fast, no models
.venv/bin/python test_pipeline_bridge.py  # full chain, loads models
```

On a memory-constrained host, cap the model run:

```bash
systemd-run --user --scope -p MemoryMax=1500M -p MemorySwapMax=0 \
  .venv/bin/python test_pipeline_bridge.py
```

## Latency

Measured on a Raspberry Pi 5, warm models, with a stand-in brain:

| Stage | Time |
|-------|------|
| wake detection | ~0.8 s |
| speech-to-text | ~3.3 s |
| text-to-speech | ~3.9 s |

The text-to-speech cost is almost entirely Piper reloading its voice model in a
fresh subprocess each call; keeping Piper resident drops it to well under a second.
The real brain adds the assistant's own thinking time on top.

## Roadmap

- Custom "computah" wake word — train an openWakeWord model on real recordings so it
  fires on the exact pronunciation, not a stock phrase.
- Live microphone loop — continuous capture, voice-activity endpointing, and
  playback (needs a USB mic and speaker, or a network voice satellite).
- Resident Piper — keep the voice model loaded to cut the spoken-reply latency.
- Live assistant integration — point the bridge at the running session over the
  network.
- Network satellite — an on-device wake-word puck that streams audio only when
  summoned, so the always-listening cost stays off the small host.

## Known limitations

Reply correlation is positional: the bridge returns the next new reply block after
sending. If a turn times out and a late reply lands during the next turn, it can be
mis-attributed. The reply format carries no correlation key, so the mitigation is a
generous timeout. Voice turns are serialized, so this is rare.

## License

MIT — see [LICENSE](LICENSE).
