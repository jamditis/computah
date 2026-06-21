# Recording the "computah" wake word

Custom wake-word detection in computah has two trained parts:

1. The main openWakeWord model, trained on synthesized speech (many TTS voices and
   augmentations). This needs no real recordings.
2. A custom verifier — openWakeWord's per-speaker layer that tailors detection to
   one person's voice and pronunciation. This is trained on real recordings of the
   target speaker saying the wake word, plus negatives.

Your recordings train the verifier and double as the real-world evaluation set. The
point of a custom wake word is that it fires on your pronunciation, not a stock
phrase, so the recordings should sound like you actually say it.

## How to say it

Say "computah" the way you naturally say it. Natural variation across clips is the
goal, not consistency — vary pace, pitch, volume, and mood as you would calling out
to it from across a room. Each variation you record is one it learns to recognize.

Leave a clear silent gap of about 2 seconds between each utterance. The prep script
splits on silence, so clean gaps mean clean clips. Do not clip the start or end of a
word.

## The recordings

About 10 minutes of effort. You can stop after the first file and still get a usable
verifier; the rest sharpen it.

| File | What to do | Reps |
|------|-----------|------|
| `computah_normal.wav` | Quiet room, normal voice, at the mic | ~30 |
| `computah_styles.wav` | Soft, loud, fast, slow drawl, questioning, flat | ~20 |
| `computah_distance.wav` | 8-10 feet from the mic, normal volume | ~15 |
| `negatives.wav` | "computer", "commuter", "computing", "compute" (~5 each) | ~20 |
| `background.wav` | Talk or read for ~2 min; do NOT say "computah" | n/a |

The negatives and background teach the model what not to fire on, which is what
prevents false triggers.

## Format

Record with whatever is easiest — Audacity (shows the waveform, best for clean gaps),
a built-in voice recorder, or a phone. Any sample rate, mono or stereo, and common
formats (wav, m4a, mp3, flac) are fine. `prep_wake_samples.py` resamples everything
to 16 kHz mono, the rate openWakeWord uses.

Recording with the same microphone used at inference time gives the best match. If a
different mic will be used later, adding a handful of clips from that mic improves the
verifier.

## Processing

Put the files in one folder, then process each class on its own. Each `--label`
writes exactly the files you point it at, so list the wake-word files for `positive` —
do not point the whole mixed folder at `positive`, or the negatives and background get
labeled as wake words.

```bash
# positive wake-word clips — list the wake-word files only (shell glob)
.venv/bin/python prep_wake_samples.py --input <folder>/computah_*.wav \
  --output samples/positive --label positive

# hard negatives (segmented on silence)
.venv/bin/python prep_wake_samples.py --input <folder>/negatives.wav \
  --output samples/negative --label negative

# continuous background (normalized, not segmented)
.venv/bin/python prep_wake_samples.py --input <folder>/background.wav \
  --output samples/background --label background
```

`--input` takes one or more files, a shell glob, or a folder (a folder globs every
audio file in it, so only point at a folder when it holds a single class). The script
reports per-file segment counts and a duration summary, and writes 16 kHz mono int16
WAVs ready for verifier training. `samples/` is gitignored — the recordings are
personal data and stay out of version control.

## Training

Training runs externally (openWakeWord 0.4.0 ships no train submodule) on a GPU host.
The main model is trained from synthetic data; the custom verifier is trained from the
processed `samples/positive` and `samples/negative` clips. See the training notes once
the recordings exist.
