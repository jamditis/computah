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

In a live turn, a mishear guard sits between transcription and the brain: it reads faster-whisper's own confidence signals (`avg_logprob`, `no_speech_prob`) and, when a transcript looks garbled or silence-derived, speaks a short re-prompt instead of dispatching. Because the brain acts on what it hears, this keeps a misheard command from triggering an action. Both live paths (`run_turn` and the `live_driver` hardware loop) gate through the same `guard_transcript`; the file-based `run_pipeline` reports the same signals for inspection but does not gate on them.

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

Use `--local` to persist the choice to the gitignored `config.local.json`
instead, so a deployment can activate its own wake word (often a gitignored
`computah` model) without dirtying the tracked `config.json` a fresh clone
runs with:

```bash
.venv/bin/python pipeline.py --set-wake-word computah --local
```

`config.local.json` overrides `config.json` at runtime, so the local choice
wins while the committed default stays a built-in.

Run the pipeline on a wav file:

```bash
.venv/bin/python pipeline.py clip.wav -o reply.wav
```

The output path receives the spoken reply.

## Choosing a microphone

The live loop needs a capture device that delivers full-band audio. Use a USB
Audio Class mic (or another wired/USB mic). **Do not use a Bluetooth mic on
Windows for continuous speech**, including a conference mic like the Anker
PowerConf paired over Bluetooth.

Windows captures from a Bluetooth mic over the hands-free profile (HFP), which
downsamples and recompresses the stream to narrowband and drops frames.
faster-whisper then receives degraded audio and returns garbled transcripts. The
same PowerConf over USB delivers its DSP-cleaned full-band stream intact and
transcribes cleanly.

This does not look like an audio problem. Wake detection keeps working over HFP --
a live "computah ..." still scores 0.998 -- because the wake model matches a short
fixed pattern that survives the degradation. Only the sentence after it comes back
wrong, so the loop appears healthy right up to the transcript.

To check a device before relying on it:

```bash
.venv/bin/python audio.py --list                  # flags unsuitable input devices
.venv/bin/python audio.py --test-mic "powerconf"  # see the exit codes below
```

`--list` tags any input device it can tell is unsuitable and prints why. A
hands-free marker in the name works on every host. A low default rate is used only
for WASAPI, where it represents the shared-mode capture format; ALSA and CoreAudio
defaults are preferences and do not prove the device is narrowband. The live loop
prints the same warning at startup and keeps running, since the device is still
good enough for wake detection.

`--test-mic` goes further, because it has audio to look at: after capturing, it
inspects the spectrum for the cliff that upsampling leaves behind. Audio that
arrived at 16 kHz but started at 8 kHz carries almost nothing above 4 kHz, and no
amount of resampling puts it back. That catches the common Linux case where
PipeWire hides an 8 kHz transport behind a 16 kHz stream.

The spectral check has a strict boundary: energy above 4 kHz rules out that one
upsampling signature, but it does not rule out wideband HFP. A wideband HFP codec
can carry energy above 4 kHz while recompression and frame drops still garble
speech. A capture without the narrowband cliff is therefore reported as
inconclusive, as is a capture that is too short, silent, or constant.

`--test-mic` exit codes, which point at different fixes:

| Code | Meaning |
| --- | --- |
| 2 | Nothing arrived (only zeros). The mic is muted, unplugged, or the wrong source. |
| 3 | Frames arrived, but the device is unsuitable for speech. It works; pick a different one. |
| 4 | Frames arrived, but the available checks cannot establish suitability. Prefer USB or verify with transcription. |

The two layers fail differently. `--list` sees only what the device advertises:
wideband HFP negotiates 16 kHz and passes the rate test, and on Linux the
hands-free marker is usually absent from the name. `--test-mic` can flag an
upsampled narrowband stream from the audio, while leaving wideband HFP and a
resampler noisy enough to fill the empty band unresolved. A clean `--list` is not
proof a Bluetooth link is fine. Prefer USB.

## Configuration

`config.json` contains safe defaults that can be committed:

| Key | Meaning |
| --- | --- |
| `wake_word` | Active openWakeWord model name. |
| `wake_threshold` | Detection score required before the pipeline continues. |
| `whisper_model` | faster-whisper model size or path. |
| `whisper_compute` | CTranslate2 compute type, usually `int8` on the target device. |
| `stt_confidence_guard` | When true, a live turn drops a low-confidence transcript before the brain and speaks a re-prompt. |
| `stt_min_avg_logprob` | Floor for the transcript's mean per-token log-probability; below it the turn is rejected. This is the gate. |
| `stt_max_no_speech_prob` | How silence-like the audio looked. Combined with a low `avg_logprob` it labels a reject as silence; following faster-whisper's own rule, a confident decode is never rejected for this alone. |
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
.venv/bin/python test_confidence_guard.py
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

Those are one manual measurement. Regenerate them with `benchmark.py`, which runs the
file pipeline repeatedly on one fixed clip with warm models and prints a median/p95
table to paste over the one above, along with peak RSS:

```bash
systemd-run --user --scope -p MemoryMax=1500M -p MemorySwapMax=0 \
  .venv/bin/python benchmark.py --runs 20
```

The clip is a fixed Piper utterance built from the configured wake word, synthesized
into `test_audio/benchmark_clip_<wake_word>.wav` on first run and reused after that, so
the input does not drift between runs. The wake word is in the filename because the clip
speaks it: switching `--wake-word` synthesizes a new clip rather than scoring the old
phrase against the new model. Only that default clip is synthesized on demand; an
explicit `--wav` that does not exist is an error rather than something to fill in, so a
typo cannot quietly benchmark generated audio in place of your sample.

The table above predates the script and was taken against a simulated brain, so it has
no Brain reply row. `benchmark.py` has no simulator: it measures whichever backend the
config names, so its Brain reply and End-to-end turn rows carry real assistant latency
(the `claude` CLI under the default `brain_backend: "cli"`, or a live session under
`bridge` with `--live-brain`). Those rows depend on the model and the load at the time.

Text-to-speech is the one to watch, because it looks like a stage row and is not
reproducible either: `run_pipeline` times `speak(reply, ...)`, so it renders the
assistant's answer, and its duration tracks how long that answer happens to be. Only
wake detection and speech-to-text are driven by the fixed clip. If you want a
comparable Text-to-speech number across runs, quote the reply length beside it or hold
the reply fixed.

So when you paste, replace the "simulated brain" caption too: say which backend and
model produced the run, and note that only the first two rows are input-driven. Keeping
the old caption over a pasted table would claim a simulated brain for rows that
measured a real one.

On a working bridge (`brain_backend: "bridge"` with its reply path set, and a host when
the transport is `ssh`), every run sends the clip's transcript into the persistent
assistant session, so the benchmark refuses to start and says so. Pass `--live-brain` to
measure the bridge on purpose, or set `brain_backend` to `cli` to leave that session
alone. A half-configured bridge answers locally and sends nothing, so it still runs
without the flag.

It times the ssh hop to the brain host as its own row, and only when `brain_backend` is
`bridge`: a `cli` backend answers locally, so a leftover `brain_transport: "ssh"` buys no
hop. The brain stage is transport plus however long the assistant took to answer, and the
transport half is not one hop per turn: the floor is three (the pre-send read, the send,
and an immediate first poll), then `ssh_reply_reader` opens another connection for every
later poll with nothing multiplexing them, so it grows with the answer. Warming Piper or
whisper cannot reduce it.

The 3.9 s text-to-speech figure predates the resident voice: `speak()` now synthesizes
in process from a cached Piper voice, with the CLI only as a warned fallback, and
`warm_models` loads it before the first turn. Re-running the benchmark is what settles
whether text-to-speech is still the largest cost.

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
