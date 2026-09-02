# Training the "computah" wake-word model

This guide records the custom model recipe that was previously kept under the
gitignored `experiments/` directory. It uses relative paths and public inputs so a
new training host can repeat the run. Generated audio, features, recordings, and
model binaries stay outside git.

## Reproduction boundary

computah pins `openwakeword==0.4.0` for inference. That package does not ship the
training module. Run training from the upstream openWakeWord source at commit
[`368c037`](https://github.com/dscripka/openWakeWord/tree/368c03716d1e92591906a84949bc477f3a834455),
which contains the [automated training notebook](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/notebooks/automatic_model_training.ipynb),
[`train.py`](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/train.py),
and the [base configuration](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/examples/custom_model.yml)
used by this recipe. Use a Linux GPU host and follow that pinned notebook's
environment setup.

The private recordings and the original ignored helper scripts are not available
from a clone. The old run also did not record a dependency lock or input checksums.
This guide reproduces its parameters and split rule, not a byte-identical model or
the exact private audio and historical file membership.

## Training inputs

Create one working directory, clone the two pinned source trees into it, and keep
the other public inputs beside those checkouts:

```bash
mkdir computah-wake-training
cd computah-wake-training
git clone https://github.com/dscripka/openWakeWord.git openwakeword
git -C openwakeword checkout 368c03716d1e92591906a84949bc477f3a834455
git clone https://github.com/dscripka/piper-sample-generator.git piper-sample-generator
git -C piper-sample-generator checkout f1988a4d54eddb23d99e86f0adfef6226a85acc7
curl -L https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt \
  --output piper-sample-generator/models/en_US-libritts_r-medium.pt
```

| Path | Input |
| --- | --- |
| `piper-sample-generator/` | The [openWakeWord Piper sample generator at `f1988a4`](https://github.com/dscripka/piper-sample-generator/tree/f1988a4d54eddb23d99e86f0adfef6226a85acc7) and its `en_US-libritts_r-medium.pt` v2.0.0 voice model, installed by the commands above. |
| `mit_rirs/` | The [MIT environmental impulse responses](https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses), converted to 16 kHz PCM WAV as shown in the pinned notebook. |
| `openwakeword_features_ACAV100M_2000_hrs_16bit.npy` | The 2,000-hour negative feature set from [openwakeword_features](https://huggingface.co/datasets/davidscripka/openwakeword_features). |
| `validation_set_features.npy` | The false-positive validation features from the same dataset. |

Download the two feature arrays at the pinned dataset revision and verify their
LFS SHA-256 digests. The ACAV100M array is about 17.3 GB.

```bash
curl -L --fail \
  https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/985bf1b47e7f19c07741af82bfe32d5a9dc56096/openwakeword_features_ACAV100M_2000_hrs_16bit.npy \
  --output openwakeword_features_ACAV100M_2000_hrs_16bit.npy
curl -L --fail \
  https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/985bf1b47e7f19c07741af82bfe32d5a9dc56096/validation_set_features.npy \
  --output validation_set_features.npy
printf '%s  %s\n' \
  721a66d0682c65a1b5c1da0aa109409cede1d20e28b15235c344b000cbb7654f \
  openwakeword_features_ACAV100M_2000_hrs_16bit.npy \
  a56a8a0f8e0efb91900acc6de4c0cdf4c564842e8475a7d49b36c039e17a690f \
  validation_set_features.npy | sha256sum --check
```

Create `mit_rirs/` from the pinned dataset revision with the same conversion as
the upstream notebook:

```bash
python - <<'PY'
from pathlib import Path

import numpy as np
from datasets import load_dataset
from scipy.io.wavfile import write

output = Path("mit_rirs")
output.mkdir(exist_ok=True)
rows = load_dataset(
    "davidscripka/MIT_environmental_impulse_responses",
    revision="b824a1ef2821f112fda0b9cb26e4278c62b425bb",
    split="train",
    streaming=True,
)
for row in rows:
    audio = row["audio"]
    write(
        output / Path(audio["path"]).name,
        16000,
        (audio["array"] * 32767).astype(np.int16),
    )
PY
```

The recovered prototype deliberately did not mix a background-audio dataset. Room
simulation still runs through the MIT impulse responses. Preserve that distinction:
`background_paths` is empty and `rir_paths` is not.

## Configuration

Save this as `computah.yaml` in the working directory:

```yaml
model_name: "computah"
target_phrase:
  - "computah"
custom_negative_phrases:
  - "computer"
  - "commuter"
  - "computing"

n_samples: 1000
n_samples_val: 200
tts_batch_size: 50
augmentation_batch_size: 16
piper_sample_generator_path: "./piper-sample-generator"
output_dir: "./training-output"

rir_paths:
  - "./mit_rirs"
background_paths: []
background_paths_duplication_rate: []
false_positive_validation_data_path: "./validation_set_features.npy"
augmentation_rounds: 1

feature_data_files:
  "ACAV100M_sample": "./openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
batch_n_per_class:
  "ACAV100M_sample": 1024
  "adversarial_negative": 50
  "positive": 50

model_type: "dnn"
layer_size: 32
steps: 10000
max_negative_weight: 1500
target_false_positives_per_hour: 0.2
```

The 1,000/200 sample counts reproduce the quick prototype. The original recipe
marked 30,000 or more training samples and 2,000 validation samples as the next
production-size run. Change only those two counts for that run; keep its result
separate so the prototype and production evidence cannot be confused.

## Run the three phases

From the working directory that contains `computah.yaml`, the inputs above, and an
`openwakeword/` source checkout, point each phase at the same configuration:

```bash
python openwakeword/openwakeword/train.py --training_config computah.yaml --generate_clips
python openwakeword/openwakeword/train.py --training_config computah.yaml --augment_clips
python openwakeword/openwakeword/train.py --training_config computah.yaml --train_model
```

The phases generate synthetic positives and near-word negatives, apply room
augmentation, create openWakeWord features, and train the DNN. A successful final
phase writes `training-output/computah.onnx`. Keep the full training output local.
Copy only the model needed for an inference check:

```bash
cp training-output/computah.onnx <computah-checkout>/models/computah.onnx
<computah-checkout>/.venv/bin/python <computah-checkout>/pipeline.py --list-wake-words
<computah-checkout>/.venv/bin/python <computah-checkout>/pipeline.py \
  --set-wake-word computah --local
```

`models/*.onnx` and `config.local.json` are gitignored. Do not commit either file.

## Real recordings and held-out evaluation

Follow [the recording protocol](recording-computah.md): collect normal, style, and
distance positives; the near-word negatives; and continuous background speech. For
this held-out workflow, prepare the full set under a staging directory instead of
the evaluator's default directories:

```bash
.venv/bin/python prep_wake_samples.py --input <folder>/computah_*.wav \
  --output samples/computah-prepared/positive --label positive
.venv/bin/python prep_wake_samples.py --input <folder>/negatives.wav \
  --output samples/computah-prepared/negative --label negative
.venv/bin/python prep_wake_samples.py --input <folder>/background.wav \
  --output samples/computah-prepared/background --label background
```

The recovered split rule reserves 15% from every positive recording style before
the remaining clips are pooled. Apply the same 15% holdout to the real negative
clips. The continuous background recording is evaluation-only and is not split.

The following deterministic split ranks each filename by SHA-256, holds out 15% of
each positive style and the negative set, and refuses to mix with an earlier split.
The prepared clip names preserve the source recording stems used here.

```bash
python - <<'PY'
from hashlib import sha256
from math import ceil
from pathlib import Path
from shutil import copy2

root = Path("samples")
prepared = root / "computah-prepared"
train = root / "computah-train"
held_out = root / "computah-heldout"
for destination in (train, held_out):
    if destination.exists():
        raise SystemExit(f"refusing to mix with existing split: {destination}")

positive_groups = []
for stem in ("computah_normal", "computah_styles", "computah_distance"):
    clips = sorted((prepared / "positive").glob(f"{stem}_*.wav"))
    if not clips:
        raise SystemExit(f"no prepared clips for {stem}")
    positive_groups.append(clips)
all_positive = set((prepared / "positive").glob("*.wav"))
if set().union(*map(set, positive_groups)) != all_positive:
    raise SystemExit("positive staging contains clips outside the three style groups")

negative = sorted((prepared / "negative").glob("*.wav"))
background = sorted((prepared / "background").glob("*.wav"))
if not negative or not background:
    raise SystemExit("negative and background staging must both contain audio")

def split_group(clips, label):
    ranked = sorted(clips, key=lambda path: sha256(path.name.encode()).hexdigest())
    held = set(ranked[: max(1, ceil(len(ranked) * 0.15))])
    for clip in clips:
        destination = held_out if clip in held else train
        target = destination / label
        target.mkdir(parents=True, exist_ok=True)
        copy2(clip, target / clip.name)

for group in positive_groups:
    split_group(group, "positive")
split_group(negative, "negative")
for clip in background:
    target = held_out / "background"
    target.mkdir(parents=True, exist_ok=True)
    copy2(clip, target / clip.name)
PY
```

Keep the resulting file lists with the training receipt so no held-out clip can
enter a verifier or later tuning run by accident. The historical private split
contained 50 positive and 19 negative training clips, but those counts do not replace
a manifest for a new recording set.

The base ONNX model above is trained on synthetic positives. Real clips provide the
deployment check and can later train openWakeWord's optional speaker verifier.
computah does not yet pass a verifier model to `openwakeword.Model`, so a verifier
artifact is not part of this reproduction.

Run the evaluator against the isolated holdout root with the same false-positive
target as training:

```bash
.venv/bin/python eval_wake_threshold.py --model computah \
  --samples samples/computah-heldout \
  --min-threshold 0.1 --max-threshold 0.9 --step 0.05 \
  --max-fa-per-hour 0.2 --max-latency-s 1.0
```

Record the selected threshold, false rejects per activation, false accepts per hour,
sample counts, openWakeWord commit, configuration checksum, input checksums, and
model checksum. The recovered recipe does not define a false-reject ceiling. Meeting
0.2 false accepts per hour is necessary, but deployment remains blocked until a
false-reject ceiling is selected and the held-out run meets it. See [threshold
tuning](wake-threshold-tuning.md) for the evaluator's full contract.

Record the selected threshold, false rejects per activation, false accepts per hour,
sample counts, openWakeWord commit, configuration checksum, and model checksum. The
recovered recipe does not define a false-reject ceiling. Meeting 0.2 false accepts
per hour is necessary, but deployment remains blocked until a false-reject ceiling
is selected and the held-out run meets it. See [threshold tuning](wake-threshold-tuning.md)
for the evaluator's full contract.
