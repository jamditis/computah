#!/usr/bin/env python3
"""jawn-voice pipeline: wake word -> speech-to-text -> claude brain -> text-to-speech.

The load-bearing core of a local voice assistant, built to run mic-free by
feeding it audio files. No microphone, no network LLM API, no PyTorch.

Stages:
  detect_wake(wav)  openWakeWord (ONNX) — is the wake phrase present?
  transcribe(wav)   faster-whisper (CTranslate2, int8) — what was said?
  brain(text)       the `claude` CLI as a subprocess — what to reply?
  speak(text, out)  Piper TTS (ONNX) — render the reply to a WAV.

All four are CPU-only and fit in the Pi's RAM. The wake word is changeable
through config.json (see set_wake_word / available_wake_models).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from math import gcd
from pathlib import Path

# Silence onnxruntime's CUDA-provider chatter before it imports. There is no GPU
# on the Pi; CPU fallback is expected and correct.
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config.json"
VOICES_DIR = PROJECT_DIR / "voices"
WHISPER_DIR = PROJECT_DIR / "whisper_models"
CLAUDE_BIN = str(Path.home() / ".local/bin/claude")

# openWakeWord ships its pretrained models inside the installed package.
import openwakeword  # noqa: E402

_OWW_MODELS_DIR = Path(openwakeword.__file__).parent / "resources" / "models"
# Files that are feature extractors / VAD, not wake-word classifiers.
_NON_WAKE = {"melspectrogram", "embedding_model", "silero_vad"}
# Pretrained classifiers that are not actually wake *phrases* (they are intent
# models bundled with the library). Keep them out of the wake-word library.
_NON_PHRASE = {"timer", "weather"}

DEFAULTS = {
    "wake_word": "hey_jarvis",
    "wake_threshold": 0.5,
    "whisper_model": "tiny.en",
    "whisper_compute": "int8",
    "voice_model": "en_US-lessac-medium",
    "claude_model": "haiku",
    "claude_timeout_s": 60,
}

VOICE_SYSTEM_PROMPT = (
    "You are a local voice assistant. Answer in one or two short, plain spoken "
    "sentences. No markdown, no bullet points, no code blocks, no emoji."
)

# Module-level caches so repeated calls in one process do not reload models.
_oww_cache: dict[str, object] = {}
_whisper_cache: dict[tuple[str, str], object] = {}


# --------------------------------------------------------------------------- #
# Config + wake-word library
# --------------------------------------------------------------------------- #
def load_config() -> dict:
    """Read config.json, filling any missing key from DEFAULTS."""
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text()))
        except json.JSONDecodeError as e:
            print(f"warning: config.json is not valid JSON ({e}); using defaults",
                  file=sys.stderr)
    return cfg


def _strip_version(filename: str) -> str:
    """alexa_v0.1.onnx -> alexa, hey_jarvis_v0.1.onnx -> hey_jarvis."""
    stem = filename[:-5] if filename.endswith(".onnx") else filename
    if "_v" in stem:
        stem = stem.rsplit("_v", 1)[0]
    return stem


def available_wake_models() -> dict[str, str]:
    """Map friendly wake-word name -> absolute .onnx path for installed models.

    Excludes the feature-extraction/VAD models and the non-phrase intent models
    (timer, weather). These are the words you can switch between instantly.
    """
    out: dict[str, str] = {}
    for f in sorted(_OWW_MODELS_DIR.glob("*.onnx")):
        # Filter feature/VAD models on the raw filename (before version-stripping,
        # since "silero_vad" would otherwise split on its "_v").
        if f.stem in _NON_WAKE:
            continue
        name = _strip_version(f.name)
        if name in _NON_PHRASE:
            continue
        out[name] = str(f)
    return out


def _resolve_wake_path(name: str) -> str:
    lib = available_wake_models()
    if name not in lib:
        raise ValueError(
            f"unknown wake word {name!r}. available: {sorted(lib)}"
        )
    return lib[name]


def set_wake_word(name: str) -> dict:
    """Change the active wake word in config.json and return the new config."""
    _resolve_wake_path(name)  # validate before writing
    cfg = load_config()
    cfg["wake_word"] = name
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n")
    return cfg


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #
def _load_pcm16(wav_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load a WAV as mono int16 PCM at target_sr (openWakeWord's input format)."""
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        g = gcd(int(target_sr), int(sr))
        audio = resample_poly(audio, target_sr // g, sr // g)
    pcm = np.clip(audio, -1.0, 1.0) * 32767.0
    return pcm.astype(np.int16)


# --------------------------------------------------------------------------- #
# Stage 1: wake-word detection
# --------------------------------------------------------------------------- #
def _get_oww_model(model_path: str):
    if model_path not in _oww_cache:
        from openwakeword.model import Model

        _oww_cache[model_path] = Model(wakeword_model_paths=[model_path])
    return _oww_cache[model_path]


def detect_wake(wav_path: str, model_name: str | None = None,
                threshold: float | None = None) -> tuple[bool, str, float]:
    """Run audio through the active openWakeWord model.

    Returns (fired, model_name, peak_score). fired is peak_score >= threshold.
    The audio is scanned in 80ms (1280-sample) frames; the highest score over
    the clip is the detection score.
    """
    cfg = load_config()
    model_name = model_name or cfg["wake_word"]
    threshold = cfg["wake_threshold"] if threshold is None else threshold
    path = _resolve_wake_path(model_name)

    model = _get_oww_model(path)
    model.reset()  # clear streaming buffers between independent clips
    pcm = _load_pcm16(wav_path)

    step = 1280  # 80ms at 16kHz — openWakeWord's frame size
    peak = 0.0
    for i in range(0, len(pcm) - step + 1, step):
        preds = model.predict(pcm[i:i + step])
        for score in preds.values():
            peak = max(peak, float(score))
    return peak >= threshold, model_name, peak


# --------------------------------------------------------------------------- #
# Stage 2: speech-to-text
# --------------------------------------------------------------------------- #
def _get_whisper(model: str, compute: str):
    key = (model, compute)
    if key not in _whisper_cache:
        from faster_whisper import WhisperModel

        _whisper_cache[key] = WhisperModel(
            model, device="cpu", compute_type=compute,
            download_root=str(WHISPER_DIR),
        )
    return _whisper_cache[key]


def transcribe(wav_path: str) -> str:
    """Transcribe a WAV with faster-whisper (int8). Returns the text."""
    cfg = load_config()
    model = _get_whisper(cfg["whisper_model"], cfg["whisper_compute"])
    segments, _info = model.transcribe(wav_path, beam_size=1, language="en")
    return " ".join(seg.text for seg in segments).strip()


# --------------------------------------------------------------------------- #
# Stage 3: the brain (claude CLI subprocess — never an HTTP LLM API)
# --------------------------------------------------------------------------- #
def brain(text: str, model: str | None = None,
          timeout_s: int | None = None) -> str:
    """Send transcribed text to the local `claude` CLI and return its reply.

    Uses the host Claude Code subscription via subprocess. Tools are disabled so
    untrusted spoken input cannot drive local actions; session persistence is
    off so audio transcripts are not written to disk. This is the only LLM call
    in the pipeline and it is a CLI call, not a network API request.
    """
    cfg = load_config()
    model = model or cfg["claude_model"]
    timeout_s = cfg["claude_timeout_s"] if timeout_s is None else timeout_s

    prompt = f"{VOICE_SYSTEM_PROMPT}\n\nUser: {text}\nAssistant:"
    try:
        result = subprocess.run(
            [CLAUDE_BIN, "-p", "--model", model,
             "--tools", "", "--no-session-persistence"],
            input=prompt, capture_output=True, text=True,
            timeout=timeout_s, cwd="/tmp",
        )
    except subprocess.TimeoutExpired:
        return "Sorry, I timed out thinking about that."
    if result.returncode != 0:
        err = (result.stderr or "").strip().splitlines()
        tail = err[-1] if err else f"exit {result.returncode}"
        return f"Sorry, the brain call failed: {tail}"
    return result.stdout.strip()


# --------------------------------------------------------------------------- #
# Stage 4: text-to-speech
# --------------------------------------------------------------------------- #
def speak(text: str, out_wav_path: str, voice_model: str | None = None) -> str:
    """Render text to a WAV with Piper. Returns the output path."""
    cfg = load_config()
    voice_model = voice_model or cfg["voice_model"]
    onnx = VOICES_DIR / f"{voice_model}.onnx"
    if not onnx.exists():
        raise FileNotFoundError(
            f"voice model not found: {onnx}. download with "
            f"`python -m piper.download_voices {voice_model} --download-dir voices`"
        )
    out_wav_path = str(out_wav_path)
    subprocess.run(
        [sys.executable, "-m", "piper", "-m", str(onnx), "-f", out_wav_path],
        input=text, capture_output=True, text=True, check=True,
    )
    return out_wav_path


# --------------------------------------------------------------------------- #
# End-to-end chain
# --------------------------------------------------------------------------- #
def run_pipeline(wav_path: str, out_wav_path: str | None = None,
                 wake_word: str | None = None) -> dict:
    """Full chain on one input WAV. Returns a dict of stage results + timings."""
    timings: dict[str, float] = {}
    t0 = time.time()
    fired, wname, score = detect_wake(wav_path, model_name=wake_word)
    timings["detect_wake"] = time.time() - t0

    result = {
        "input": wav_path,
        "wake_word": wname,
        "wake_fired": fired,
        "wake_score": round(score, 4),
        "transcript": None,
        "reply": None,
        "output_wav": None,
        "timings_s": timings,
    }
    if not fired:
        timings["total"] = time.time() - t0
        return result

    t1 = time.time()
    transcript = transcribe(wav_path)
    timings["transcribe"] = time.time() - t1
    result["transcript"] = transcript

    t2 = time.time()
    reply = brain(transcript)
    timings["brain"] = time.time() - t2
    result["reply"] = reply

    if out_wav_path is None:
        out_wav_path = str(PROJECT_DIR / "test_audio" / "reply.wav")
    t3 = time.time()
    speak(reply, out_wav_path)
    timings["speak"] = time.time() - t3
    result["output_wav"] = out_wav_path

    timings["total"] = time.time() - t0
    return result


def _cli() -> int:
    p = argparse.ArgumentParser(description="jawn-voice pipeline (mic-free)")
    p.add_argument("input_wav", nargs="?", help="input WAV to process")
    p.add_argument("-o", "--output", help="output WAV path for the spoken reply")
    p.add_argument("-w", "--wake-word", help="override the active wake word")
    p.add_argument("--list-wake-words", action="store_true",
                   help="list installed wake-word models and exit")
    p.add_argument("--set-wake-word", metavar="NAME",
                   help="persist a new active wake word to config.json and exit")
    args = p.parse_args()

    if args.list_wake_words:
        cfg = load_config()
        for name in sorted(available_wake_models()):
            mark = " (active)" if name == cfg["wake_word"] else ""
            print(f"{name}{mark}")
        return 0
    if args.set_wake_word:
        cfg = set_wake_word(args.set_wake_word)
        print(f"active wake word is now: {cfg['wake_word']}")
        return 0
    if not args.input_wav:
        p.error("input_wav is required unless using --list/--set-wake-word")

    r = run_pipeline(args.input_wav, out_wav_path=args.output,
                     wake_word=args.wake_word)
    print(f"wake word   : {r['wake_word']}")
    print(f"wake fired  : {r['wake_fired']} (score {r['wake_score']})")
    if not r["wake_fired"]:
        print("not woken — pipeline stopped (no transcription/brain/tts)")
    else:
        print(f"transcript  : {r['transcript']}")
        print(f"reply       : {r['reply']}")
        print(f"output wav  : {r['output_wav']}")
    print("timings (s) : " + ", ".join(
        f"{k}={v:.2f}" for k, v in r["timings_s"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
