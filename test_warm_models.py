#!/usr/bin/env python3
"""Fast, no-model tests for pipeline.warm_models (issue #13).

warm_models pre-loads the wake, whisper, and Piper voice models at live-loop startup
so the first user turn does not pay the model-load cost mid-conversation. These checks
stub the three model loaders so they run with no models and no audio device: they pin
that warm_models loads each stage exactly once from config, returns a per-model warm
time, survives one stage failing, that run_loop warms before it opens the mic, and that
the file pipeline stays lazy (it never warms).

Run:  .venv/bin/python test_warm_models.py
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import sys
import types

import pipeline

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    results.append((PASS if ok else FAIL, name))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")
    return ok


# A minimal config with exactly the keys warm_models reads, so these tests never
# touch this host's real config.json / config.local.json.
CFG = {
    "wake_word": "hey_jarvis",
    "whisper_model": "base.en",
    "whisper_compute": "int8",
    "voice_model": "en_US-lessac-medium",
}


def _stub_loaders(record: dict) -> tuple:
    """Swap the three model loaders + the wake-path resolver for call recorders.

    Returns the tuple of originals so the caller restores them in a finally.
    """
    real = (pipeline._get_oww_model, pipeline._get_whisper, pipeline._get_piper,
            pipeline._get_vad, pipeline._resolve_wake_path)
    pipeline._resolve_wake_path = lambda name: f"/models/{name}.onnx"
    pipeline._get_oww_model = lambda path: record["wake"].append(path)
    pipeline._get_whisper = lambda model, compute: record["whisper"].append((model, compute))
    pipeline._get_piper = lambda path: record["piper"].append(path)
    pipeline._get_vad = lambda: record["vad"].append("vad")
    return real


def _restore_loaders(real: tuple) -> None:
    (pipeline._get_oww_model, pipeline._get_whisper, pipeline._get_piper,
     pipeline._get_vad, pipeline._resolve_wake_path) = real


def test_warms_each_once() -> None:
    print("\n=== warm_models loads each stage once, from config ===")
    record = {"wake": [], "whisper": [], "piper": [], "vad": []}
    real = _stub_loaders(record)
    try:
        timings = pipeline.warm_models(CFG, "hey_jarvis")
    finally:
        _restore_loaders(real)

    check("wake model loaded once, resolved from the active wake word",
          record["wake"] == ["/models/hey_jarvis.onnx"], f"{record['wake']}")
    check("capture-time VAD loaded once",
          record["vad"] == ["vad"], f"{record['vad']}")
    check("whisper loaded once with the configured model and compute",
          record["whisper"] == [("base.en", "int8")], f"{record['whisper']}")
    check("piper voice loaded once from the configured voice model",
          record["piper"] == [str(pipeline.VOICES_DIR / "en_US-lessac-medium.onnx")],
          f"{record['piper']}")
    check("returns a warm time for every stage that loaded",
          set(timings) == {"wake", "vad", "whisper", "piper"}
          and all(isinstance(v, float) and v >= 0 for v in timings.values()),
          f"{timings}")


def test_one_failure_skips_only_that_stage() -> None:
    print("\n=== a stage that fails to warm is skipped; the others still warm ===")
    record = {"wake": [], "whisper": [], "piper": [], "vad": []}
    real = _stub_loaders(record)

    def boom(*_a, **_k):
        raise RuntimeError("no whisper build")

    pipeline._get_whisper = boom
    try:
        timings = pipeline.warm_models(CFG, "hey_jarvis")
    finally:
        _restore_loaders(real)

    check("one stage failing does not raise or stop the others",
          record["wake"] == ["/models/hey_jarvis.onnx"]
          and record["vad"] == ["vad"]
          and record["piper"] == [str(pipeline.VOICES_DIR / "en_US-lessac-medium.onnx")],
          f"wake={record['wake']} vad={record['vad']} piper={record['piper']}")
    check("the failed stage is absent from the returned warm times",
          "whisper" not in timings and set(timings) == {"wake", "vad", "piper"},
          f"{timings}")


def test_run_loop_warms_before_listening() -> None:
    print("\n=== run_loop warms the models before it opens the mic to listen ===")
    order: list[str] = []
    saved_load = pipeline.load_config
    saved_warm = pipeline.warm_models
    saved_run_turn = pipeline.run_turn
    saved_audio = sys.modules.get("audio")

    # wake_chime off so the loop never touches the (guarded) cue path.
    pipeline.load_config = lambda: {"wake_word": "hey_jarvis", "wake_chime": False}
    pipeline.warm_models = lambda cfg=None, wake_word=None: (order.append("warm"), {})[1]
    # No turn is captured, so the loop reaches the inactive-mic branch and breaks.
    pipeline.run_turn = lambda frames, **kw: None

    class _FakeMic:
        device_label = "fake-mic"

        def __enter__(self):
            # Opening the mic is the moment the loop begins listening.
            order.append("listen")
            return self

        def __exit__(self, *exc):
            return False

        def frames(self):
            return iter(())

        def pause(self):
            pass

        def resume(self):
            pass

        def flush(self):
            pass

        def active(self) -> bool:
            return False

    fake_audio = types.ModuleType("audio")
    fake_audio.Microphone = lambda name=None: _FakeMic()
    fake_audio.play_wav = lambda *a, **k: None
    sys.modules["audio"] = fake_audio

    try:
        pipeline.run_loop(wake_word="hey_jarvis")
    finally:
        pipeline.load_config = saved_load
        pipeline.warm_models = saved_warm
        pipeline.run_turn = saved_run_turn
        if saved_audio is None:
            sys.modules.pop("audio", None)
        else:
            sys.modules["audio"] = saved_audio

    check("run_loop warms models, then opens the mic (warm before listen)",
          order == ["warm", "listen"], f"order={order}")


def test_file_pipeline_stays_lazy() -> None:
    print("\n=== the file pipeline never warms models (it stays lazy) ===")
    calls: list[str] = []
    saved_warm = pipeline.warm_models
    saved_detect = pipeline.detect_wake

    pipeline.warm_models = lambda *a, **k: calls.append("warm")
    # Wake does not fire, so run_pipeline returns after detection without loading the
    # later models -- and, the point here, without ever warming.
    pipeline.detect_wake = lambda wav, model_name=None: (False, "hey_jarvis", 0.0)
    try:
        pipeline.run_pipeline("/dev/null")
    finally:
        pipeline.warm_models = saved_warm
        pipeline.detect_wake = saved_detect

    check("run_pipeline does not call warm_models", calls == [], f"warm calls={calls}")


def main() -> int:
    test_warms_each_once()
    test_one_failure_skips_only_that_stage()
    test_run_loop_warms_before_listening()
    test_file_pipeline_stays_lazy()

    n_pass = sum(1 for r in results if r[0] == PASS)
    n_total = len(results)
    print(f"\n=== {n_pass}/{n_total} checks passed ===")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
