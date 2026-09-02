# Recording the "computah" wake word

Custom wake-word detection in computah can have two trained parts:

1. The main openWakeWord model, trained on synthesized speech (many TTS voices and
   augmentations). This needs no real recordings.
2. A custom verifier — openWakeWord's per-speaker layer that tailors detection to
   one person's voice and pronunciation. This is trained on real recordings of the
   target speaker saying the wake word, plus negatives.

computah currently loads only the main ONNX model; it does not pass a verifier to
openWakeWord yet. Your recordings are the real-world evaluation set now and can train
the verifier after that integration exists. A custom wake word must fire on your
pronunciation, so the recordings should sound like you actually say it.

## How to say it

Say "computah" the way you naturally say it. Natural variation across clips is the
goal, not consistency — vary pace, pitch, volume, and mood as you would calling out
to it from across a room. Each variation you record is one it learns to recognize.

Leave a clear silent gap of about 2 seconds between each utterance. The prep script
splits on silence, so clean gaps mean clean clips. Do not clip the start or end of a
word.

## The recordings

About 10 minutes of effort produces a smoke-test set. Record each positive style and
the near-word negatives in at least two separate sessions, with a new source file for
each session. A session boundary matters: clips from the same raw recording share the
same room, microphone, noise, and gain, so they must stay together when the recipe
splits training from evaluation.

| File | What to do | Reps |
|------|-----------|------|
| `computah_normal_sNN.wav` | Quiet room, normal voice, at the mic | ~30 total |
| `computah_styles_sNN.wav` | Soft, loud, fast, slow drawl, questioning, flat | ~20 total |
| `computah_distance_sNN.wav` | 8-10 feet from the mic, normal volume | ~15 total |
| `negatives_sNN.wav` | "computer", "commuter", "computing", "compute" (~5 each) | ~20 total |
| `background_sNN.wav` | Talk, read, or capture normal room audio; do NOT say "computah" | 5+ hr held out |

The negatives and background measure false triggers now. They can also teach a
future verifier what not to fire on. A two-minute background take is enough only to
smoke-test the evaluator. The deployment recipe requires at least five hours of
held-out non-wake audio: at that duration one false accept equals 0.2 per hour. Use
substantially more audio, across several sessions and environments, before using
the rate to select a production threshold. This tuning set is not independent
deployment evidence; [issue #113](https://github.com/jamditis/computah/issues/113)
tracks the untouched test set and separate near-word checks needed for that claim.

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
.venv/bin/python prep_wake_samples.py --input <folder>/negatives_*.wav \
  --output samples/negative --label negative

# continuous background (normalized, not segmented)
.venv/bin/python prep_wake_samples.py --input <folder>/background_*.wav \
  --output samples/background --label background
```

`--input` takes one or more files, a shell glob, or a folder (a folder globs every
audio file in it, so only point at a folder when it holds a single class). The script
reports per-file segment counts and a duration summary, and writes 16 kHz mono int16
WAVs ready for verifier training. `samples/` is gitignored — the recordings are
personal data and stay out of version control.

When adding manifests to an existing `samples/` directory, run the first refresh
with the complete source list for that label. A partial bootstrap cannot prove
ownership of legacy clips under omitted stems, so those files remain manual-only
leftovers. Move or clear the old output first if the complete source set is no
longer available.

Once an output directory has a source manifest, an incremental run must include at
least one recording already listed there along with any new recording. Keep added
microphone takes under the same positive-file glob, or list them beside an existing
source explicitly. With `--clean`, pass every source whose clips the dataset should
keep; omitted sources are treated as intentionally dropped.

## Training

Training runs externally (openWakeWord 0.4.0 ships no train submodule) on a GPU host.
The main model is trained from synthetic data. See [the reproducible training
recipe](training-computah.md) for its pinned source, configuration, three phases,
held-out split, and validation target.
