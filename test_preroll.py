#!/usr/bin/env python3
"""Pre-roll buffer: the audio just before a wake fire is preserved (issue #30).

stream_detect_wake consumes every frame up to and including the one whose score
crosses threshold, and there is detection latency, so when a request is spoken with
no pause after the wake word the first fraction of the request is consumed during
detection and lost. The fix keeps a small ring buffer of the most recent frames in
the detector and prepends it to the captured request.

These checks are model-free and deterministic: they exercise the
stream_detect_wake -> capture_request handoff with synthetic frames and a tiny fake
model, so they run fast with no wake model and no whisper. The end-to-end "first
word survives transcription" proof depends on the real wake-detection latency for a
given mic and is a hardware voice-test; here we prove the plumbing that makes it
possible -- that frames consumed during detection are kept and prepended, and that an
abandoned wake (no speech) still captures nothing.

Run:  .venv/bin/python test_preroll.py
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np

import pipeline
from pipeline import FRAME_SIZE

TEST_DIR = Path(__file__).resolve().parent / "test_audio"
TEST_DIR.mkdir(exist_ok=True)

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    results.append((PASS if ok else FAIL, name))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")
    return ok


def frame(value: int) -> np.ndarray:
    """A constant int16 frame whose fill value tags it for identity checks."""
    return np.full(FRAME_SIZE, value, dtype=np.int16)


def loud(n: int) -> list[np.ndarray]:
    return [np.full(FRAME_SIZE, 4000, dtype=np.int16) for _ in range(n)]


def silent(n: int) -> list[np.ndarray]:
    return [np.zeros(FRAME_SIZE, dtype=np.int16) for _ in range(n)]


def nframes(pcm: np.ndarray) -> int:
    return len(pcm) // FRAME_SIZE


class _FakePreproc:
    """The preprocessor surface _reset_oww touches, nothing more."""

    def __init__(self):
        self.melspectrogram_buffer = np.zeros(1, dtype=np.float32)
        self.feature_buffer = np.zeros(1, dtype=np.float32)
        self.raw_data_buffer = collections.deque()
        self.accumulated_samples = 0


class _FiringModel:
    """Scores 0 until the fire_on-th predict, then 1.0. Carries just enough surface
    for _reset_oww, so stream_detect_wake can reset and drive it like a real model
    without loading onnxruntime."""

    def __init__(self, fire_on: int):
        self.fire_on = fire_on
        self.calls = 0
        self.reset_calls = 0
        self.preprocessor = _FakePreproc()

    def reset(self):
        self.reset_calls += 1

    def predict(self, frame_in):
        self.calls += 1
        return {"w": 1.0 if self.calls >= self.fire_on else 0.0}


def main() -> int:
    # ----- stream_detect_wake keeps a pre-roll ring buffer ----------------- #
    print("=== stream_detect_wake fills a caller-owned pre-roll buffer ===")
    n = pipeline._PREROLL_FRAMES
    fire_on = n + 5  # fire well after the ring buffer has filled
    model = _FiringModel(fire_on=fire_on)
    frames = iter([frame(i) for i in range(fire_on + 10)])
    preroll = collections.deque(maxlen=n)
    score = pipeline.stream_detect_wake(frames, model, 0.5, preroll=preroll)
    fired_frame_value = fire_on - 1  # 0-based: the fire_on-th predict is frame index fire_on-1
    check("detector fires and returns the peak score", score == 1.0, f"score={score}")
    check("pre-roll holds exactly _PREROLL_FRAMES recent frames",
          len(preroll) == n, f"len={len(preroll)} want {n}")
    check("pre-roll ends with the firing frame (the consumed-during-detection frame)",
          len(preroll) > 0 and int(preroll[-1][0]) == fired_frame_value,
          f"last pre-roll frame value={int(preroll[-1][0]) if preroll else None} "
          f"want {fired_frame_value}")
    check("pre-roll starts _PREROLL_FRAMES back from the firing frame",
          len(preroll) > 0 and int(preroll[0][0]) == fired_frame_value - n + 1,
          f"first pre-roll frame value={int(preroll[0][0]) if preroll else None} "
          f"want {fired_frame_value - n + 1}")

    # ----- stream_detect_wake resets the warm model once, not per frame ---- #
    # Issue #9 AC #2: a continuous stream reuses the warm model with a SINGLE reset,
    # unlike detect_wake's per-clip reset+pad. The reset is what separates the two
    # paths; if _reset_oww ever moved inside stream_detect_wake's frame loop, the
    # model's ~2 s context would be wiped every 80 ms and the live wake path would
    # silently stop firing. Feed a long non-firing stream and pin reset==1 while
    # predict still runs once per frame.
    print("\n=== stream_detect_wake resets the model once for the whole stream ===")
    n_frames = 25
    warm = _FiringModel(fire_on=10_000)  # threshold never reached within the stream
    none_score = pipeline.stream_detect_wake(
        iter([frame(i) for i in range(n_frames)]), warm, 0.5)
    check("a non-firing stream returns None", none_score is None, f"score={none_score}")
    check("the warm model is reset exactly once for the whole stream, not per frame",
          warm.reset_calls == 1, f"reset_calls={warm.reset_calls} want 1")
    check("every frame is scored against that one warm model",
          warm.calls == n_frames, f"predict calls={warm.calls} want {n_frames}")

    # ----- capture_request prepends the pre-roll seed ---------------------- #
    print("\n=== capture_request prepends a pre-roll seed ===")
    seed = [frame(9), frame(9), frame(9)]
    cap = pipeline.capture_request(iter(loud(5) + silent(15)), preroll=seed)
    # Without pre-roll this is 5 speech + 10 trailing-silence endpoint = 15 frames;
    # the 3 seed frames go in front, so 18.
    check("captured request includes the pre-roll frames",
          nframes(cap) == 18, f"got {nframes(cap)} frames, want 18 (3 seed + 15)")
    check("pre-roll frames sit at the very front, in order",
          cap.size >= 3 * FRAME_SIZE
          and np.array_equal(cap[:3 * FRAME_SIZE], np.concatenate(seed)),
          "first 3 frames equal the seed")

    # The endpoint/onset decision must ignore the seed: an abandoned wake (a wake
    # with no speech after it) still returns empty, so the wake word held in the
    # pre-roll is never transcribed as a phantom request.
    cap_empty = pipeline.capture_request(iter(silent(60)), preroll=loud(3))
    check("abandoned wake stays empty even with a pre-roll seed",
          cap_empty.size == 0, f"got {cap_empty.size} samples, want 0")

    # A transient after the wake -- a click, a cough edge, a speaker echo -- is a frame
    # or two of energy, not speech. It must not count as speech onset: otherwise a
    # wake-only turn would prepend its pre-roll (which holds the wake word) and whisper
    # could decode a confident phantom command the mishear guard cannot catch. Onset
    # requires a sustained run of speech frames, so a transient leaves the wake abandoned
    # and empty (energy gate is not a speech gate -- found reviewing #54).
    for k in (1, 2):
        cap_transient = pipeline.capture_request(iter(loud(k) + silent(30)),
                                                  preroll=loud(3))
        check(f"a {k}-frame transient after the wake stays empty (no phantom command)",
              cap_transient.size == 0, f"got {cap_transient.size} samples, want 0")
    # The fix must not over-reject: a short but sustained command (3 frames) still
    # captures, with its pre-roll seed prepended.
    cap_short = pipeline.capture_request(iter(loud(3) + silent(15)), preroll=loud(2))
    check("a sustained 3-frame command still captures with its pre-roll",
          nframes(cap_short) == 15, f"got {nframes(cap_short)} frames, want 15 (2 seed + 3 + 10)")

    # A pre-roll of None or [] behaves exactly like the original (no prepend).
    cap_none = pipeline.capture_request(iter(loud(5) + silent(15)), preroll=None)
    check("no pre-roll behaves like the original capture", nframes(cap_none) == 15,
          f"got {nframes(cap_none)} frames, want 15")

    # ----- capture_request confirms speech (VAD) before prepending --------- #
    # The energy onset rejects a 1-2 frame transient, but a *sustained* >=240 ms noise
    # (a long cough, a fan spinning up, room rumble) clears the energy gate exactly like
    # speech does -- energy cannot tell them apart. When a vad_threshold is supplied,
    # capture_request runs voice-activity detection over the POST-FIRE audio only and
    # keeps the pre-roll (which holds the wake word) only if real speech is confirmed,
    # closing the phantom path the energy gate leaves open (#54). The VAD verdict is
    # faked here so the check stays model-free; the real-model proof is test_vad_gate.py.
    print("\n=== capture_request confirms speech (VAD) before prepending the pre-roll ===")
    real_confirm = pipeline._confirm_speech
    try:
        pipeline._confirm_speech = lambda pcm, threshold: True
        cap_ok = pipeline.capture_request(iter(loud(5) + silent(15)),
                                          preroll=loud(2), vad_threshold=0.5)
        check("confirmed speech captures and keeps its pre-roll",
              nframes(cap_ok) == 17, f"got {nframes(cap_ok)} frames, want 17 (2 seed + 5 + 10)")

        pipeline._confirm_speech = lambda pcm, threshold: False
        cap_noise = pipeline.capture_request(iter(loud(5) + silent(15)),
                                             preroll=loud(2), vad_threshold=0.5)
        check("sustained noise the VAD rejects stays empty (no phantom command)",
              cap_noise.size == 0, f"got {cap_noise.size} samples, want 0")

        # The crux: the gate must run on the command audio ONLY, never the pre-roll. The
        # pre-roll holds the wake word, which IS speech, so feeding it to the VAD would
        # always-confirm and defeat the gate.
        seen = {}
        pipeline._confirm_speech = lambda pcm, threshold: (seen.update(n=len(pcm)) or True)
        pipeline.capture_request(iter(loud(5) + silent(15)), preroll=loud(2), vad_threshold=0.5)
        check("VAD sees only the post-fire command audio, not the pre-roll",
              seen.get("n") == 15 * FRAME_SIZE,
              f"VAD saw {seen.get('n')} samples, want {15 * FRAME_SIZE} (5 speech + 10 endpoint, no pre-roll)")

        # vad_threshold=None (the default) skips the gate entirely, so the energy-only
        # callers above keep their existing behavior.
        pipeline._confirm_speech = lambda pcm, threshold: False  # would reject if consulted
        cap_skip = pipeline.capture_request(iter(loud(5) + silent(15)), preroll=loud(2))
        check("no vad_threshold skips the VAD gate (energy-only path unchanged)",
              nframes(cap_skip) == 17, f"got {nframes(cap_skip)} frames, want 17")
    finally:
        pipeline._confirm_speech = real_confirm

    # ----- run_turn wires detection pre-roll into capture ------------------ #
    print("\n=== run_turn threads the detection pre-roll into capture_request ===")
    seen = {"preroll": "unset"}
    real = (pipeline.stream_detect_wake, pipeline.capture_request,
            pipeline.transcribe_detailed, pipeline.guard_transcript,
            pipeline.brain, pipeline.speak,
            pipeline._get_oww_model, pipeline._resolve_wake_path)

    def fake_detect(frames, model, threshold, preroll=None):
        if preroll is not None:
            preroll.extend(loud(2))  # two frames "consumed during detection"
        return 0.9

    def fake_capture(frames, preroll=None, vad_threshold=None, **_):
        seen["preroll"] = preroll
        return np.full(4 * FRAME_SIZE, 4000, dtype=np.int16)

    pipeline._resolve_wake_path = lambda name: "x"
    pipeline._get_oww_model = lambda path: object()
    pipeline.stream_detect_wake = fake_detect
    pipeline.capture_request = fake_capture
    pipeline.transcribe_detailed = lambda p: pipeline.Transcript("file an issue", 0.0, 0.0)
    pipeline.guard_transcript = lambda heard, cfg: (True, "")
    pipeline.brain = lambda t, **_: "done"
    pipeline.speak = lambda text, out: out
    try:
        r = pipeline.run_turn(iter(loud(1)), model_name="x", threshold=0.5,
                              out_wav_path=str(TEST_DIR / "preroll_reply.wav"))
    finally:
        (pipeline.stream_detect_wake, pipeline.capture_request,
         pipeline.transcribe_detailed, pipeline.guard_transcript,
         pipeline.brain, pipeline.speak,
         pipeline._get_oww_model, pipeline._resolve_wake_path) = real
    got = seen["preroll"]
    check("run_turn passes the frames gathered during detection to capture_request",
          r is not None and got not in ("unset", None) and len(got) == 2,
          f"capture_request saw preroll={('unset' if got == 'unset' else (None if got is None else len(got)))}")

    n_pass = sum(1 for x in results if x[0] == PASS)
    n_total = len(results)
    print(f"\n=== {n_pass}/{n_total} checks passed ===")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
