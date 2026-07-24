# Tuning the wake threshold

`wake_threshold` in `config.json` decides how high the wake-word score must be to
fire. Set it too low and background speech triggers the assistant; too high and real
activations are missed. This picks it from measured behavior on your own recordings
instead of the shipped default (0.5), which has no data behind it.

Two numbers drive the choice:

- **false-rejects per activation**: how often a real "computah" is missed. One
  positive recording is one activation.
- **false-accepts per hour**: how often non-wake audio (near-words and ambient
  sound) spuriously fires, per hour of that audio. Counting per hour makes the rate
  comparable no matter how much negative audio you recorded.

## 1. Record and prep the samples

Follow [recording-computah.md](recording-computah.md) for the positive, negative,
and background takes, then split them into clips with `prep_wake_samples.py`:

```bash
.venv/bin/python prep_wake_samples.py --input computah_normal.wav computah_styles.wav \
    computah_distance.wav --output samples/positive --label positive
.venv/bin/python prep_wake_samples.py --input negatives.wav \
    --output samples/negative --label negative
.venv/bin/python prep_wake_samples.py --input background.wav \
    --output samples/background --label background
```

`samples/` is gitignored: these are personal voice recordings and never go in the
repo. The evaluation script below is committed; the audio it reads is not.

Re-running prep into a populated `samples/` dir refreshes it in place. If a
re-recording yields fewer clips than last time, the extra clips from the old run
would linger and the training globs would still read them, so prep warns and
names the count. Pass `--clean` to remove those leftovers in the same run.

With a readable source-aware manifest, `--clean` only deletes clips prep recorded
making. It removes the now-unused higher-numbered clips of a take this run
re-recorded, and the clips a previous run recorded for a take that is no longer
in the inputs. The `.prep-manifest.json` in the output dir maps each resolved
source path to the clips prep wrote for it. A source recording, a hand-curated
clip, and anything else absent from that map is named but left in place.
Recorded output stems also remain reserved for their source while the next run
assigns names. A new same-basename recording therefore cannot overwrite another
source's manifested clips before `--clean` has a chance to apply its ownership
rules.

That record proves prep made a file, not that the file belongs to what you are
refreshing. A `--label` that disagrees with the record already in the directory
is refused outright, before anything is decoded: that is a mistyped `--output`,
and waiting until `--clean` would be too late, since the run would already have
overwritten that dataset's clips. Past that, a run must contain at least one
exact source path from the record before it may delete from the dataset or write
to it. A shared basename is not ownership. A directory with no record yet counts
as yours. If a non-clean legacy refresh leaves same-stem files behind, prep does
not write a manifest because it cannot prove whether those files are old prep
output or audio curated by hand. The follow-up `--clean` stays on the legacy
filename rule rather than recording that guess as provenance. A run that
produces no clips does not claim an unrecorded directory. Prep also spares the
prior clips of a source it tried and got nothing from (a silent or bad
re-recording): those are the only copy, and it will not trade them for a run
that produced nothing.

To add a new recording to an existing dataset without `--clean`, include at
least one source already recorded there in the same invocation. This gives prep
explicit ownership proof while it adds the new source to the manifest. If you
use `--clean`, include every source whose clips the dataset should keep: omitted
sources are treated as intentionally dropped and their prep-owned clips are
removed. A safe incremental workflow is to add recordings without `--clean`,
then run a full-input `--clean` pass after reviewing the complete source list.

Manifest updates use atomic replacement. An absent or unreadable manifest falls
back to the narrow same-stem cleanup rule with a warning that source ownership
protection is unavailable. That fallback has no provenance: `--clean` can remove any
`<same-stem>_NNN.wav` file, including a hand-added file with that shape. Check the
warning list and move any such file before cleaning. A readable older manifest
without source ownership is refused; after confirming the output directory,
remove that manifest once to bootstrap the source map from the next successful
run.

```bash
.venv/bin/python prep_wake_samples.py --input computah_normal.wav computah_styles.wav \
    computah_distance.wav --output samples/positive --label positive --clean
```

## 2. Run the sweep

```bash
.venv/bin/python eval_wake_threshold.py --model computah \
    --min-threshold 0.1 --max-threshold 0.9 --step 0.05 --max-fa-per-hour 1.0
```

`--model` must match the wake word you recorded in step 1: score the `computah`
takes with the `computah` model. Scoring them with a different model (the shipped
`hey_jarvis`, say) rejects every real activation and the recommendation collapses to
a 100% miss rate. Omit `--model` to fall back to the `wake_word` in `config.json`.

It scores every clip once with the same openWakeWord model the live loop uses, then
sweeps the threshold and prints a table:

```
 thresh   FR/act     FA/hr   rejects   accepts
  --------------------------------------------
  0.300    0.000     12.40     0/40          62
  ...
  0.550    0.050      0.40     2/40           2
  ...
```

`--max-fa-per-hour` is the false-accept budget. The script recommends the lowest
threshold whose false-accept rate stays within that budget while missing the fewest
real activations; ties go to the lower threshold, which leaves more margin for a
quiet or clipped activation. If no threshold meets the budget it says so and reports
the lowest achievable false-accept rate rather than a silently-worse default. That
is the signal to record cleaner negatives or accept a higher rate.

## 3. Record the result

Put the recommended value in `config.json` under `wake_threshold`, and fill in the
row below so the shipped default is traceable to a run. Re-run whenever the model or
the microphone changes.

| Model | Recommended `wake_threshold` | FR/activation | FA/hour | Sample set | Date |
|-------|------------------------------|---------------|---------|------------|------|
| `hey_jarvis` (shipped default) | 0.5 (unmeasured placeholder) | run eval | run eval | run eval | run eval |

Until a row is filled from a real sweep, the shipped 0.5 stays a placeholder, not a
tuned value.

## Notes

- False accepts are counted per wake event, not per frame. After a fire the live
  loop does not re-arm until it has consumed the request (at least ~1.6 s when no
  speech follows), so `--refractory-s` defaults to that floor and one sustained
  near-word or noise burst counts as a single false accept, the way production would
  incur it. Lower it only to study raw crossings.
- The metrics core (`wake_eval.py`) is separate from the scoring driver
  (`eval_wake_threshold.py`) so the math is unit-tested with plain values and no
  audio (`test_wake_eval.py`), the way the confidence guard is tested.
- The core also supports an activation-latency tolerance (a fire that arrives too
  late counts as a reject). The driver scores on peak only today; wire a measured
  latency in once a latency budget is set for the deployed device.
