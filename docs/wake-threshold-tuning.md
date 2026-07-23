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

`--clean` only ever deletes clips prep itself made. It removes the now-unused
higher-numbered clips of a take this run re-recorded, and the clips a previous
run recorded for a take that is no longer in the inputs -- the second kind is
matched against `.prep-manifest.json`, a record prep keeps in the output dir of
what it wrote there. A source recording, a hand-curated clip, and anything else
absent from that record is named but left in place.

That record proves prep made a file, not that the file belongs to what you are
refreshing. A `--label` that disagrees with the record already in the directory
is refused outright, before anything is decoded: that is a mistyped `--output`,
and waiting until `--clean` would be too late, since the run would already have
overwritten that dataset's clips. Past that, a run has to be refreshing the
record it found (at least one take in common) before it may delete from it or
write to it. A directory with no record yet counts as yours.
Prep also spares the prior clips of a take it
tried and got nothing from (a silent or bad re-recording): those are the only
copy, and it will not trade them for a run that produced nothing.

```bash
.venv/bin/python prep_wake_samples.py --input computah_normal.wav \
    --output samples/positive --label positive --clean
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
