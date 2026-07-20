#!/usr/bin/env python3
"""Fast, no-model tests for the one-shot wake padding and framing (#28).

wake_frame_scores is the single scoring path behind both detect_wake and the
threshold-tuning eval, and the part of it that shapes the peak score is pure:
prepend 1.5s of silence, append 0.5s, then hand the model consecutive
1280-sample frames. Only model.predict needs weights. So this feeds a recording
stand-in through the real function and pins the framing contract with no
openWakeWord model, no Piper, and no microphone, like test_confidence_guard.py.

What it pins, and why each one can regress silently:

  1. A clip shorter than one 1280-sample frame is still scored. Without the pad
     the frame loop would produce zero frames and detect_wake would report a
     confident 0.0, which reads identically to "the model heard nothing."
  2. An empty or silent clip still yields a full padded frame sequence, and
     detect_wake reduces an all-zero score list to a non-firing 0.0.
  3. A wake word at the very start of a clip is preceded by a full window of
     silence, so a freshly reset detector has context before the real audio.
  4. Frame truncation only ever discards trailing silence. The loop drops the
     final partial frame; that is safe only while the tail pad is longer than
     one frame. Shrinking _WAKE_TAIL_SAMPLES below 1280 would start eating the
     end of the clip, and every score would still look plausible.
  5. The model is reset before the first frame, so one clip cannot inherit the
     previous clip's feature buffer.
  6. detect_wake's peak is the max of wake_frame_scores, and its fire test is
     inclusive of the threshold.

The stand-in returns scores this file plants, so nothing here says anything about
what the model does with real sound. Whether a near-word (computer/commuter/
computing) actually stays below threshold is scored from recorded speech by
eval_wake_threshold.py, which reuses this same padded path.

Run:  .venv/bin/python test_wake_framing.py
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import sys
import tempfile
import wave
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import pipeline

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str]] = []

# Hardcoded rather than read from pipeline.FRAME_SIZE: wake_frame_scores keeps its
# own local `step`, and pinning the frame width independently is the point. If the
# two ever drift apart, that is a finding, not something to paper over.
STEP = 1280  # 80ms at 16kHz, openWakeWord's frame size
LEAD = pipeline._WAKE_LEAD_SAMPLES
TAIL = pipeline._WAKE_TAIL_SAMPLES

# A sample value the silence pad can never produce, so a frame's origin is
# unambiguous: anything non-zero came from the clip.
CLIP_TONE = 12000


def check(name: str, ok: bool, detail: str) -> bool:
    results.append((PASS if ok else FAIL, name))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")
    return ok


class _FakePreprocessor:
    """The preprocessor attributes _reset_oww clears or restores.

    raw_data_buffer and accumulated_samples start dirty so that clearing them is
    observable, unlike test_preroll.py's _FakePreproc, which starts clean because
    the streaming path it drives only counts resets. Reusing that one here would
    make the "buffers were cleared" check pass against an implementation that
    clears nothing. The mel and feature buffers start blank because _reset_oww
    snapshots them on first reset for a model it did not build, so their restore
    is not observable from here either way.
    """

    def __init__(self) -> None:
        self.raw_data_buffer: deque = deque([1, 2, 3])
        self.accumulated_samples = 99
        self.melspectrogram_buffer = np.zeros((1, 32), dtype=np.float32)
        self.feature_buffer = np.zeros((1, 96), dtype=np.float32)


class _RecordingModel:
    """Stands in for an openWakeWord Model, recording exactly what it is fed.

    Records every frame and whether reset() ran before the first predict(), which
    is the ordering wake_frame_scores relies on for a clean slate per clip. Frame
    scores are planted by the caller, so any assertion about a score is an
    assertion about how the pipeline reduces scores, never about acoustics.
    """

    def __init__(self, scores: list[float] | None = None) -> None:
        self.frames: list[np.ndarray] = []
        self.resets = 0
        self.reset_before_first_predict = False
        self.preprocessor = _FakePreprocessor()
        self._scores = scores or []

    def predict(self, frame) -> dict[str, float]:
        if not self.frames:
            self.reset_before_first_predict = self.resets > 0
        self.frames.append(np.asarray(frame).copy())
        i = len(self.frames) - 1
        return {"fake": self._scores[i] if i < len(self._scores) else 0.0}

    def reset(self) -> None:
        self.resets += 1


def _write_wav(path: Path, pcm: np.ndarray) -> str:
    """Write int16 mono PCM at 16kHz. Synthesized in-process, so no mic and no TTS."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.astype(np.int16).tobytes())
    return str(path)


@contextmanager
def _fake_model(model: _RecordingModel):
    """Point the wake path at the stand-in instead of a real .onnx model.

    A context manager rather than paired install/restore helpers: one site that
    forgot the restore would leave pipeline._get_oww_model patched for the rest of
    the process, quietly deciding every later check in the same run.
    """
    saved = (pipeline._resolve_wake_path, pipeline._get_oww_model)
    pipeline._resolve_wake_path = lambda name: f"/nonexistent/{name}.onnx"
    pipeline._get_oww_model = lambda path: model
    try:
        yield model
    finally:
        pipeline._resolve_wake_path, pipeline._get_oww_model = saved


def _score(clip: np.ndarray, tmp: Path, name: str, scores=None):
    """Run one synthesized clip through the real padded scoring path."""
    model = _RecordingModel(scores)
    with _fake_model(model):
        wav = _write_wav(tmp / f"{name}.wav", clip)
        frame_scores = pipeline.wake_frame_scores(wav, model_name="fake_wake")
    return model, frame_scores, wav


def _expected_frames(clip_len: int) -> int:
    """Frames the loop yields for a clip of clip_len samples, from the constants."""
    return (LEAD + clip_len + TAIL) // STEP


def _run_checks(tmp: Path) -> int:
    print("\n=== padding rescues clips too short to frame on their own ===")
    # 100 samples is 6.25ms, well under one 1280-sample frame. Unpadded this clip
    # yields no frames at all.
    short = np.full(100, CLIP_TONE, dtype=np.int16)
    model, scores, _ = _score(short, tmp, "subframe")
    # Compare against unpadded framing rather than a figure derived from the pad
    # constants, so this still fails if the pad is removed or shrunk to nothing.
    check(
        "sub-frame clip is still scored",
        len(scores) > 0 and len(short) // STEP == 0,
        f"{len(short)}-sample clip -> {len(scores)} frames "
        f"(unpadded it would yield {len(short) // STEP})",
    )
    check(
        "the sub-frame clip's audio reaches the model",
        any(f.any() for f in model.frames),
        f"{sum(1 for f in model.frames if f.any())} of {len(model.frames)} "
        f"frames carry clip audio",
    )

    print("\n=== empty and silent clips still frame, and reduce to a non-fire ===")
    for label, clip in (
        ("empty", np.zeros(0, dtype=np.int16)),
        ("silent", np.zeros(16000, dtype=np.int16)),
    ):
        model, scores, wav = _score(clip, tmp, label)
        check(
            f"{label} clip yields the padded frame count",
            len(scores) == _expected_frames(len(clip))
            and len(scores) > len(clip) // STEP,
            f"{len(clip)} samples -> {len(scores)} frames "
            f"(unpadded it would yield {len(clip) // STEP})",
        )

        # The stand-in scores every frame 0.0, so this pins the reduction and the
        # fire test, not the model's behaviour on silence. Named for what it pins.
        with _fake_model(_RecordingModel()):
            fired, _, peak = pipeline.detect_wake(
                wav, model_name="fake_wake", threshold=0.5
            )
        check(
            f"{label} clip: an all-zero score list reduces to a non-firing 0.0",
            not fired and peak == 0.0,
            f"fired={fired} peak={peak:.4f}",
        )

    print("\n=== a wake at the very start of a clip gets a full silence window ===")
    # The whole clip is non-silent, so the first frame carrying any clip audio is
    # the earliest point a wake word at offset 0 could be scored.
    from_start = np.full(16000, CLIP_TONE, dtype=np.int16)
    model, _, _ = _score(from_start, tmp, "wake_at_start")
    first_audio = next(i for i, f in enumerate(model.frames) if f.any())
    last_audio = max(i for i, f in enumerate(model.frames) if f.any())
    trailing_silent = len(model.frames) - 1 - last_audio
    silent_prefix = all(not f.any() for f in model.frames[:first_audio])
    check(
        "every frame before the clip is pure silence",
        silent_prefix and first_audio == LEAD // STEP,
        f"frames 0..{first_audio - 1} silent, clip audio starts at frame {first_audio}",
    )
    # 1.4 rather than 1.5: the 24000-sample lead is 18.75 frames, and the clip's
    # first sample lands inside frame 18, so 18 whole silent frames (1.44s) precede
    # it. Raising this to 1.5 would turn the check red without anything regressing.
    check(
        "the silence window is at least 1.4s of context",
        first_audio * STEP / 16000 >= 1.4,
        f"{first_audio} frames = {first_audio * STEP / 16000:.2f}s before the clip",
    )
    # The trailing pad matters for a wake spoken at the very end of a clip: the
    # detector scores a frame from accumulated context, so the last real frame
    # needs frames after it to be scored against. 4 rather than 6: the 8000-sample
    # tail is 6.25 frames, the clip's end shares a frame with the tail's start, and
    # the final partial frame is dropped, which leaves 5 whole silent frames here.
    check(
        "silence also follows the clip, not just precedes it",
        trailing_silent >= 4,
        f"{trailing_silent} silent frames after the clip "
        f"({trailing_silent * STEP / 16000:.2f}s)",
    )

    print("\n=== frame truncation only ever discards trailing silence ===")
    # The loop drops the final partial frame. That is safe only while the tail pad
    # outlasts one frame; otherwise the drop reaches back into the clip.
    check(
        "the tail pad outlasts one frame",
        TAIL > STEP,
        f"tail={TAIL} samples vs frame={STEP}",
    )
    # Which clip length leaves the largest partial frame depends on both pad
    # constants, so sweep a full frame period instead of guessing one. Picking a
    # single length hides the regression: with a 1000-sample tail a 400-sample
    # clip loses audio while a 1279-sample clip does not.
    lost: list[tuple[int, int]] = []
    for clip_len in range(0, 2 * STEP, 97):
        ragged = np.full(clip_len, CLIP_TONE, dtype=np.int16)
        model, _, _ = _score(ragged, tmp, "ragged")
        delivered = int(sum(int((f != 0).sum()) for f in model.frames))
        if delivered != clip_len:
            lost.append((clip_len, clip_len - delivered))
    swept = len(range(0, 2 * STEP, 97))
    check(
        "no clip length loses audio to the dropped partial frame",
        not lost,
        f"{swept} clip lengths swept across two frame periods, "
        f"{len(lost)} lost audio"
        + (f" (worst: {max(lost, key=lambda x: x[1])})" if lost else ""),
    )

    print("\n=== the model is reset before the first frame of a clip ===")
    model, _, _ = _score(np.full(4000, CLIP_TONE, dtype=np.int16), tmp, "reset")
    check(
        "reset runs before any frame is scored",
        model.reset_before_first_predict and model.resets == 1,
        f"resets={model.resets} before_first_predict="
        f"{model.reset_before_first_predict}",
    )
    check(
        "the preprocessor buffers are cleared, not just the score buffer",
        len(model.preprocessor.raw_data_buffer) == 0
        and model.preprocessor.accumulated_samples == 0,
        f"raw_data_buffer={len(model.preprocessor.raw_data_buffer)} "
        f"accumulated_samples={model.preprocessor.accumulated_samples}",
    )
    check(
        "every frame handed to the model is exactly one frame wide",
        all(len(f) == STEP for f in model.frames),
        f"{len(model.frames)} frames, widths {sorted({len(f) for f in model.frames})}",
    )

    print("\n=== detect_wake reduces the frame scores correctly ===")
    clip = np.full(8000, CLIP_TONE, dtype=np.int16)
    wav = _write_wav(tmp / "peak.wav", clip)
    n = _expected_frames(len(clip))
    planted = [0.0] * n
    planted[n // 2] = 0.61  # the peak
    planted[n // 3] = 0.44  # a lower score earlier in the clip

    for label, threshold, want_fired in (
        ("below the peak", 0.5, True),
        ("exactly at the peak", 0.61, True),  # the fire test is >=, not >
        ("above the peak", 0.62, False),
    ):
        with _fake_model(_RecordingModel(planted)):
            fired, name, peak = pipeline.detect_wake(
                wav, model_name="fake_wake", threshold=threshold
            )
        check(
            f"threshold {label}: fired={want_fired}",
            fired is want_fired and abs(peak - 0.61) < 1e-9,
            f"threshold={threshold} peak={peak:.4f} fired={fired} name={name}",
        )

    with _fake_model(_RecordingModel(planted)):
        frame_scores = pipeline.wake_frame_scores(wav, model_name="fake_wake")
    check(
        "detect_wake's peak is the max of wake_frame_scores",
        abs(max(frame_scores) - 0.61) < 1e-9,
        f"max of {len(frame_scores)} frame scores = {max(frame_scores):.4f}",
    )

    n_pass = sum(1 for r in results if r[0] == PASS)
    n_total = len(results)
    print(f"\n=== {n_pass}/{n_total} checks passed ===")
    for status, name in results:
        if status == FAIL:
            print(f"  [{FAIL}] {name}")
    return 0 if n_pass == n_total else 1


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="computah-wake-framing-") as tmp:
        return _run_checks(Path(tmp))


if __name__ == "__main__":
    sys.exit(main())
