#!/usr/bin/env python3
"""computah pipeline: wake word -> speech-to-text -> brain -> text-to-speech.

The load-bearing core of a local voice assistant, built to run mic-free by
feeding it audio files. No microphone, no network LLM API, no PyTorch.

Stages:
  detect_wake(wav)  openWakeWord (ONNX) — is the wake phrase present?
  transcribe(wav)   faster-whisper (CTranslate2, int8) — what was said?
  brain(text)       dispatches to a persistent assistant session (the bridge)
                    or, as a dev fallback, the local `claude` CLI subprocess.
  speak(text, out)  Piper TTS (ONNX) — render the reply to a WAV.

All four are CPU-only and fit in the Pi's RAM. The wake word is changeable
through config.json (see set_wake_word / available_wake_models). The brain
backend is selected by the brain_backend config key; deployment-specific and
sensitive bridge settings live in config.local.json (gitignored), which
overrides config.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from math import gcd
from pathlib import Path
from typing import NamedTuple

import brain_bridge

# Silence onnxruntime's CUDA-provider chatter before it imports. There is no GPU
# on the Pi; CPU fallback is expected and correct.
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config.json"
# Gitignored overrides for this deployment (persona, ssh host, reply path).
# Keeps infra names and paths out of the published config.json.
LOCAL_CONFIG_PATH = PROJECT_DIR / "config.local.json"
VOICES_DIR = PROJECT_DIR / "voices"
WHISPER_DIR = PROJECT_DIR / "whisper_models"
CUSTOM_MODELS_DIR = PROJECT_DIR / "models"  # custom-trained wake words live here
def _resolve_claude_bin() -> str:
    """Locate the claude CLI: prefer PATH, otherwise fall back per OS.

    The fallback applies only when claude is not on PATH. On POSIX (the Pi) the
    common per-user install is ~/.local/bin/claude. On Windows that POSIX path is
    meaningless, so defer to the bare name and let the OS resolver -- or the
    graceful FileNotFoundError path in _brain_cli -- handle absence.
    """
    found = shutil.which("claude")
    if found:
        return found
    if os.name == "posix":
        return str(Path.home() / ".local/bin/claude")
    return "claude"


# Prefer the claude binary on PATH; otherwise a platform-appropriate fallback.
CLAUDE_BIN = _resolve_claude_bin()

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
    # Mishear guard: gate a transcript on faster-whisper's confidence before it
    # reaches the brain, so a garbled command never drives an action. avg_logprob is
    # the gate (it must stay at or above the floor); following faster-whisper's own
    # no-speech rule, a high no_speech_prob only marks a reject as silence when the
    # decode is also unconfident, so a clear command is never dropped for it alone.
    # The defaults mirror faster-whisper's log_prob and no_speech thresholds. Set
    # stt_confidence_guard false to dispatch every turn.
    "stt_confidence_guard": True,
    "stt_min_avg_logprob": -1.0,
    "stt_max_no_speech_prob": 0.6,
    # Capture-time speech gate (#54): after a wake fires, energy endpointing decides
    # when the command ends, but energy alone cannot tell a sustained noise from speech,
    # so a >=240 ms cough/clap/hum after a bare wake could prepend the wake-word pre-roll
    # as a phantom command. capture_request runs the silero VAD over the POST-FIRE audio
    # and keeps the pre-roll only if the peak per-chunk speech probability reaches this
    # threshold. Lower to confirm quieter/marginal speech; raise to reject more noise.
    "capture_vad_threshold": 0.5,
    "voice_model": "en_US-lessac-medium",
    # Live audio device selection by case-insensitive name substring (see audio.py).
    # Empty string means the system default device. Per-host values belong in
    # config.local.json -- e.g. Legion: mic "shure", output "desktop speakers";
    # the Pi: "powerconf" over USB.
    "mic_device": "",
    "output_device": "",
    # Brain backend: "cli" (the claude -p dev fallback, default so a fresh clone
    # runs standalone) or "bridge" (a persistent assistant session).
    "brain_backend": "cli",
    "claude_model": "haiku",
    "claude_timeout_s": 60,
    # Bridge settings (used when brain_backend == "bridge"). Put real values in
    # config.local.json — persona/host/path are deployment-specific and stay out
    # of the published config.json.
    "brain_persona": "assistant",
    "brain_transport": "local",   # "local" (this host) or "ssh" (another host)
    "brain_host": "",             # ssh host alias, required for transport "ssh"
    "brain_bot_spren_bin": "bot-spren",
    "brain_bot_spren_workdir": "",  # bot-spren --working-dir; set so the send lands
                                    # in the session's inbox, not a dead-letter one
    "brain_reply_path": "",       # path to the persona's FileOutbound reply file
    "brain_timeout_s": 120,
    "brain_poll_s": 0.5,
}

VOICE_SYSTEM_PROMPT = (
    "You are a local voice assistant. Answer in one or two short, plain spoken "
    "sentences. No markdown, no bullet points, no code blocks, no emoji."
)

# Spoken when the mishear guard rejects a low-confidence transcript. Short, so the
# user simply repeats the command; the guard never sends this turn to the brain.
STT_REPROMPT = "Sorry, I didn't catch that. Please say that again."

# Module-level caches so repeated calls in one process do not reload models.
_oww_cache: dict[str, object] = {}
_whisper_cache: dict[tuple[str, str], object] = {}


# --------------------------------------------------------------------------- #
# Config + wake-word library
# --------------------------------------------------------------------------- #
def load_config() -> dict:
    """Read config, filling any missing key from DEFAULTS.

    Layering, lowest to highest precedence: DEFAULTS, then config.json (committed,
    non-sensitive), then config.local.json (gitignored deployment overrides).
    """
    cfg = dict(DEFAULTS)
    for path, label in ((CONFIG_PATH, "config.json"),
                        (LOCAL_CONFIG_PATH, "config.local.json")):
        if not path.exists():
            continue
        try:
            cfg.update(json.loads(path.read_text()))
        except json.JSONDecodeError as e:
            print(f"warning: {label} is not valid JSON ({e}); ignoring it",
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
    # Custom-trained models in the repo's models/ dir are named directly by file
    # stem (e.g. computah.onnx -> "computah") and override built-ins on collision.
    if CUSTOM_MODELS_DIR.is_dir():
        for f in sorted(CUSTOM_MODELS_DIR.glob("*.onnx")):
            out[f.stem] = str(f)
    return out


def _resolve_wake_path(name: str) -> str:
    lib = available_wake_models()
    if name not in lib:
        raise ValueError(
            f"unknown wake word {name!r}. available: {sorted(lib)}"
        )
    return lib[name]


def _read_base_config() -> dict:
    """Read only config.json — no DEFAULTS, no config.local.json overlay.

    set_wake_word writes back to config.json, so it must round-trip the committed
    base alone. Writing the merged dict (load_config) would persist DEFAULTS and,
    worse, the gitignored config.local.json overlay (persona, ssh host, reply
    path) into the tracked file, defeating the committed/local split.
    """
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def set_wake_word(name: str) -> dict:
    """Change the active wake word in config.json and return the effective config."""
    _resolve_wake_path(name)  # validate before writing
    base = _read_base_config()
    base["wake_word"] = name
    CONFIG_PATH.write_text(json.dumps(base, indent=2) + "\n")
    return load_config()


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

        model = Model(wakeword_model_paths=[model_path])
        # A fresh model's preprocessor holds blank audio/feature buffers. Snapshot
        # them now so _reset_oww can restore a true clean slate cheaply (see there).
        pp = model.preprocessor
        model._blank_buffers = (
            pp.melspectrogram_buffer.copy(), pp.feature_buffer.copy())
        _oww_cache[model_path] = model
    return _oww_cache[model_path]


def _reset_oww(model) -> None:
    """Reset an openWakeWord model to a clean slate between independent streams.

    Model.reset() clears only the prediction (score) buffer; the preprocessor's
    raw-audio, melspectrogram, and feature buffers (~10 s of history) survive. A
    warm model reused across turns therefore carries the previous turn's audio
    forward and can false-fire on the next stream. This restores the pristine
    buffers snapshotted at model creation — a clean slate without rebuilding the
    feature buffer (~0.6 s of embedding compute) on every turn.
    """
    model.reset()
    pp = model.preprocessor
    pp.raw_data_buffer.clear()
    pp.accumulated_samples = 0
    blank = getattr(model, "_blank_buffers", None)
    if blank is None:
        # A model not built by _get_oww_model (e.g. a mic adapter's own Model).
        # Snapshot on first reset: a freshly constructed model is in the blank
        # state, which is the only safe moment to capture it.
        blank = (pp.melspectrogram_buffer.copy(), pp.feature_buffer.copy())
        model._blank_buffers = blank
    mel, feat = blank
    pp.melspectrogram_buffer = mel.copy()
    pp.feature_buffer = feat.copy()


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
    _reset_oww(model)  # full clean slate between independent clips
    pcm = _load_pcm16(wav_path)
    # A one-shot clip file must fill the detector's ~2s context window; live audio
    # fills it naturally. The wake word can sit at the very start of a clip of any
    # length, so always prepend leading silence (plus a short trailing pad) rather
    # than gating on total duration: a long recording whose request runs past 2.5s
    # still starts the freshly-reset detector with an empty window. Padding gives it
    # the context to score the wake word the way it was trained; silence scores ~0,
    # so the extra frames never raise the peak for clips that already had context.
    lead = np.zeros(24000, dtype=np.int16)  # 1.5s leading silence
    tail = np.zeros(8000, dtype=np.int16)  # 0.5s trailing silence
    pcm = np.concatenate([lead, pcm, tail])

    step = 1280  # 80ms at 16kHz — openWakeWord's frame size
    peak = 0.0
    for i in range(0, len(pcm) - step + 1, step):
        preds = model.predict(pcm[i:i + step])
        for score in preds.values():
            peak = max(peak, float(score))
    return peak >= threshold, model_name, peak


# --------------------------------------------------------------------------- #
# Stage 1b: live streaming — continuous detection + request capture
# --------------------------------------------------------------------------- #
# These drive a turn from a frame stream instead of a whole file. A live mic and
# iter_wav_frames yield the same thing — consecutive 80 ms int16 frames — so the
# detection and capture logic is identical and testable with no hardware. The
# physical mic and speaker adapters land with the microphone (issues #9, #11).
FRAME_SIZE = 1280  # 80 ms at 16 kHz — openWakeWord's frame size
_SILENCE_RMS = 250.0           # int16 RMS below this is room tone, not speech
_ENDPOINT_SILENCE_FRAMES = 10  # ~0.8 s of trailing quiet ends a captured request
_NO_SPEECH_ONSET_FRAMES = 20   # ~1.6 s; abandon a wake that no speech follows
_SPEECH_ONSET_FRAMES = 3       # consecutive above-threshold frames before energy counts
                               # as speech onset; a click/cough/echo is 1-2 frames, so a
                               # sustained run keeps a transient from tripping onset and
                               # prepending the wake-word pre-roll as a phantom (#54)
_MAX_REQUEST_FRAMES = 100      # ~8 s cap so a stuck stream cannot record forever
_PREROLL_FRAMES = 2            # ~0.16 s of audio kept before the wake fire and
                               # prepended to the request, so a command spoken with
                               # no pause after the wake word is not clipped by
                               # detection latency (issue #30). Tuned on the deployed
                               # PowerConf with a live voice test: at 8 the wake word
                               # itself bled into the transcript ("computer file an
                               # issue..."); at 2 (~160 ms) the no-pause command keeps a
                               # clip-margin while the wake word stays out. The command
                               # onset never clipped at any size here, so the real risk
                               # was over-capture, not under-capture. Larger recovers
                               # more leading audio but bleeds more of the wake word in.


def iter_wav_frames(wav_path: str):
    """Yield consecutive 80 ms int16 frames from a WAV: a file-backed stand-in for
    a live mic. A microphone source yields frames of the same size and dtype, so
    everything downstream is identical."""
    pcm = _load_pcm16(wav_path)
    for i in range(0, len(pcm) - FRAME_SIZE + 1, FRAME_SIZE):
        yield pcm[i:i + FRAME_SIZE]


def _frame_rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))


def stream_detect_wake(frames, model, threshold: float, preroll=None) -> float | None:
    """Watch a continuous frame stream for the wake word.

    detect_wake handles a one-shot file by resetting the model and padding the clip
    so its ~2 s context window fills. A live stream fills that window naturally, so
    here the model is reset ONCE and fed frame by frame with no padding. Stops at
    and consumes the first frame whose score crosses threshold, returning the peak
    score so far; returns None if the stream ends with no detection. Leaves the
    iterator positioned right after the wake word so the caller can capture the
    request that follows.

    `preroll`, if given, is a bounded collection (a deque(maxlen=N)) that each frame
    is appended to as it is consumed, so on a fire it holds the most recent N frames
    -- including the firing frame. capture_request prepends those to the request,
    recovering the leading audio that detection latency consumes, so a command spoken
    with no pause after the wake word is not clipped (issue #30).
    """
    _reset_oww(model)
    peak = 0.0
    for frame in frames:
        if preroll is not None:
            preroll.append(frame)
        score = max(float(s) for s in model.predict(frame).values())
        peak = max(peak, score)
        if score >= threshold:
            return peak
    return None


_vad_model = None


def _get_vad():
    """Cache the bundled silero VAD across calls (one resident instance, like
    _get_oww_model / _get_whisper). The ONNX ships with openWakeWord, so there is no
    download and no extra dependency."""
    global _vad_model
    if _vad_model is None:
        from openwakeword.vad import VAD
        _vad_model = VAD()
    return _vad_model


def _confirm_speech(pcm: np.ndarray, threshold: float) -> bool:
    """True if `pcm` (post-fire command audio, int16 16 kHz) holds real speech.

    Runs the silero VAD over 480-sample (30 ms) chunks and keeps the PEAK chunk
    probability, not the mean the model returns over a whole buffer -- a short command
    must not be diluted to a non-speech average by trailing silence or noise in the
    capture. Reset the RNN state once, then feed chunks in order so state carries across
    them. Pass the command audio ONLY, never the pre-roll: the pre-roll holds the wake
    word, which is speech, so including it would always confirm and defeat the gate.
    silero rejects coughs, claps, echoes, and sustained noise where an energy threshold
    cannot (#54).
    """
    vad = _get_vad()
    vad.reset_states()
    frame = 480
    pad = (-len(pcm)) % frame  # silero needs a length that is a multiple of frame
    if pad:
        pcm = np.concatenate([pcm, np.zeros(pad, dtype=np.int16)])
    peak = 0.0
    for i in range(0, len(pcm), frame):
        prob = float(vad.predict(pcm[i:i + frame], frame_size=frame))
        if prob > peak:
            peak = prob
    return peak >= threshold


def capture_request(frames, preroll=None, vad_threshold=None) -> np.ndarray:
    """Collect request frames after a wake fire until a trailing-silence endpoint.

    Energy-based endpointing: once speech has been seen, stop after
    _ENDPOINT_SILENCE_FRAMES consecutive quiet frames. If no speech ever starts,
    stop after the shorter _NO_SPEECH_ONSET_FRAMES window so a false or abandoned
    wake frees the listener in ~1.6 s instead of holding it for the full
    _MAX_REQUEST_FRAMES cap. Returns the captured int16 PCM, or an empty array if
    no speech followed the wake word. Transcribing that silence would feed whisper
    room tone, which it tends to hallucinate words for, so the empty return lets
    the caller ignore the turn instead.

    `preroll`, if given, is a list of frames captured just before the wake fired
    (see stream_detect_wake). They are prepended to the returned PCM so a request
    spoken immediately after the wake word keeps its leading audio, which detection
    latency would otherwise have consumed (issue #30). The pre-roll takes no part in
    the speech-onset / endpoint decision -- that runs only over the frames after the
    wake word. Onset requires a sustained run of above-threshold frames
    (_SPEECH_ONSET_FRAMES), not a single loud frame, so a transient after the wake --
    a click, a cough edge, a speaker echo -- does not count as speech. A wake with no
    sustained speech after it still returns empty, so the wake word held in the pre-roll
    is never prepended and transcribed as a phantom request (#54).

    The energy onset is a cheap first filter: it rejects a 1-2 frame transient without
    running a model and bounds how long an abandoned wake holds the mic. It cannot tell
    a *sustained* noise (a long cough, a fan, room rumble past _SPEECH_ONSET_FRAMES)
    from speech, though, so when `vad_threshold` is given the captured command audio is
    confirmed by the silero VAD (_confirm_speech) before the pre-roll is kept. Energy
    that the VAD rejects is treated as an abandoned wake and returns empty, closing the
    phantom path the energy gate alone leaves open (#54). With `vad_threshold` None the
    VAD step is skipped and only the energy gate runs.
    """
    captured: list[np.ndarray] = []
    quiet = 0
    voiced_run = 0
    speech_seen = False
    for frame in frames:
        captured.append(frame)
        if _frame_rms(frame) < _SILENCE_RMS:
            quiet += 1
            voiced_run = 0
        else:
            quiet = 0
            voiced_run += 1
            if voiced_run >= _SPEECH_ONSET_FRAMES:
                speech_seen = True  # sustained energy, not a lone transient
        if speech_seen and quiet >= _ENDPOINT_SILENCE_FRAMES:
            break
        if not speech_seen and quiet >= _NO_SPEECH_ONSET_FRAMES:
            break  # nothing said after the wake — abandon fast, don't wait the cap
        if len(captured) >= _MAX_REQUEST_FRAMES:
            break
    if not speech_seen:
        # No sustained speech after the wake fired (silence, or only a transient like a
        # click or cough). The pre-roll is dropped here on purpose: by post-fire audio
        # alone an abandoned wake is indistinguishable from a short command consumed
        # entirely during detection, and the wake word always sits in the pre-roll.
        # Prepending it whenever the pre-roll holds speech would dispatch a bare wake
        # word as a phantom command, against the mishear-guard rule. Dropping is the safe
        # side; a genuine command with fewer than _SPEECH_ONSET_FRAMES of post-fire speech
        # (the wake consumed the rest) is dropped here too -- the cost of not being able to
        # tell it from a transient -- and is tracked as its own problem (#53).
        return np.zeros(0, dtype=np.int16)
    command = np.concatenate(captured)
    if vad_threshold is not None and not _confirm_speech(command, vad_threshold):
        # Energy was sustained but the VAD says it is not speech (a long cough, a fan,
        # room rumble). Treat it as an abandoned wake: drop the pre-roll so the wake word
        # it holds is never prepended and transcribed as a phantom command (#54).
        return np.zeros(0, dtype=np.int16)
    if preroll:
        return np.concatenate([*preroll, command])
    return command


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


class Transcript(NamedTuple):
    """A transcription plus the faster-whisper signals the mishear guard reads.

    avg_logprob is the mean per-token log-probability (closer to 0 is more
    confident); no_speech_prob is how silence-like the audio looked (higher means
    less likely to be real speech).
    """
    text: str
    avg_logprob: float
    no_speech_prob: float


def _aggregate_segments(segments) -> Transcript:
    """Collapse faster-whisper segments into one Transcript.

    avg_logprob is averaged across segments weighted by their duration, so a long
    confident clause outweighs a short hesitant one; no_speech_prob is the max
    across segments, the most conservative summary -- any silence-like span pulls
    confidence down. Pure over the segment objects (it reads only .text, .start,
    .end, .avg_logprob, .no_speech_prob), so it is unit-testable with lightweight
    stand-ins and no model.
    """
    segs = list(segments)
    text = " ".join(s.text for s in segs).strip()
    if not segs:
        # No segments at all: empty text that fails the guard on both signals.
        return Transcript(text, float("-inf"), 1.0)
    weights = [max(float(s.end) - float(s.start), 1e-3) for s in segs]
    total = sum(weights)
    avg_logprob = sum(float(s.avg_logprob) * w
                      for s, w in zip(segs, weights)) / total
    no_speech_prob = max(float(s.no_speech_prob) for s in segs)
    return Transcript(text, avg_logprob, no_speech_prob)


def transcribe_detailed(wav_path: str) -> Transcript:
    """Transcribe a WAV with faster-whisper (int8), returning the text plus the
    confidence signals the mishear guard needs. transcribe() wraps this for callers
    that want only the text."""
    cfg = load_config()
    model = _get_whisper(cfg["whisper_model"], cfg["whisper_compute"])
    segments, _info = model.transcribe(wav_path, beam_size=1, language="en")
    return _aggregate_segments(segments)


def transcribe(wav_path: str) -> str:
    """Transcribe a WAV with faster-whisper (int8). Returns the text."""
    return transcribe_detailed(wav_path).text


def transcript_confident(t: Transcript, *, min_avg_logprob: float,
                         max_no_speech_prob: float) -> tuple[bool, str]:
    """Is a transcript trustworthy enough to act on?

    The brain acts on voice commands -- it files issues, changes things -- so a
    misheard transcript must be stopped before it dispatches. The decoder's
    avg_logprob is the gate: below the floor the words are unreliable and the turn
    is rejected. Following faster-whisper's own no-speech rule, a confidently decoded
    command is never rejected for a high no_speech_prob alone (a confident decode
    overrides it); no_speech_prob only distinguishes a silence-derived reject (it
    over the ceiling and avg_logprob under the floor) from a plain garbled one, which
    sharpens the logged reason without ever re-prompting a clear command. Returns
    (ok, reason); reason names why a turn was dropped.
    """
    if not t.text.strip():
        return False, "empty transcript"
    if t.avg_logprob < min_avg_logprob:
        kind = ("silence" if t.no_speech_prob > max_no_speech_prob
                else "low confidence")
        return False, (f"{kind}: avg_logprob {t.avg_logprob:.2f} under floor "
                       f"{min_avg_logprob:.2f}")
    return True, "ok"


def guard_transcript(heard: Transcript, cfg: dict) -> tuple[bool, str]:
    """Apply the configured mishear guard to a transcription.

    Returns (ok, reason): ok is True when the guard is disabled or the transcript
    clears it, and False when a low-confidence transcript should be re-prompted
    rather than dispatched. Both live paths -- run_turn here and live_driver's own
    turn loop -- gate through this one function, so the brain cannot be reached
    unguarded on one path while the other is protected.
    """
    if not cfg["stt_confidence_guard"]:
        return True, "guard disabled"
    return transcript_confident(
        heard, min_avg_logprob=cfg["stt_min_avg_logprob"],
        max_no_speech_prob=cfg["stt_max_no_speech_prob"])


# --------------------------------------------------------------------------- #
# Stage 3: the brain (persistent session via the bridge, or the claude CLI)
# --------------------------------------------------------------------------- #
def brain(text: str, model: str | None = None,
          timeout_s: int | None = None) -> str:
    """Answer transcribed text using the configured brain backend.

    brain_backend == "bridge" routes to a persistent assistant session (so voice
    and the user's text assistant are one session with shared memory); anything
    else uses the local `claude` CLI fallback. Either way the reply is short
    spoken text, and no network LLM API is called. Never raises for an expected
    failure — the caller is a voice loop, so it returns a spoken error instead.
    """
    cfg = load_config()
    if cfg["brain_backend"] == "bridge":
        return _brain_bridge(text, cfg)
    return _brain_cli(text, cfg, model=model, timeout_s=timeout_s)


# One reply cursor per reply file, kept for the life of the process so the
# positional correlation survives across turns (a fresh cursor each turn would
# re-snapshot and lose the slot reservation that keeps timeouts aligned). A loop
# relaunch starts empty, so the first turn drains whatever backlog is on disk.
_bridge_cursors: dict[str, brain_bridge.ReplyCursor] = {}


def _brain_bridge(text: str, cfg: dict) -> str:
    """Route the transcript to a persistent assistant session over the bridge.

    Transport is built from config: "local" talks to bot-spren on this host,
    "ssh" to bot-spren on another host (pipeline here, assistant elsewhere). A
    missing required setting degrades to a spoken error, never a crash.
    """
    reply_path = cfg["brain_reply_path"]
    if not reply_path:
        return "Sorry, the brain reply path is not configured."

    cursor = _bridge_cursors.setdefault(reply_path, brain_bridge.ReplyCursor())

    persona = cfg["brain_persona"]
    bot_spren_bin = cfg["brain_bot_spren_bin"]
    workdir = cfg.get("brain_bot_spren_workdir") or None
    if cfg["brain_transport"] == "ssh":
        host = cfg["brain_host"]
        if not host:
            return "Sorry, the brain host is not configured."
        send = brain_bridge.ssh_cli_send(host, bot_spren_bin, working_dir=workdir)
        read_reply = brain_bridge.ssh_reply_reader(host, reply_path)
    else:
        send = brain_bridge.cli_send(bot_spren_bin, working_dir=workdir)
        read_reply = brain_bridge.file_reply_reader(reply_path)

    return brain_bridge.brain_via_bridge(
        text, persona=persona, send=send, read_reply=read_reply, cursor=cursor,
        system_prompt=VOICE_SYSTEM_PROMPT,
        timeout_s=cfg["brain_timeout_s"], poll_s=cfg["brain_poll_s"],
    )


def _brain_cli(text: str, cfg: dict, model: str | None = None,
               timeout_s: int | None = None) -> str:
    """Dev fallback: send the transcript to the local `claude` CLI.

    Uses the host Claude Code subscription via subprocess. Tools are disabled so
    untrusted spoken input cannot drive local actions; session persistence is
    off so audio transcripts are not written to disk. A CLI call, not a network
    API request, and stateless — no shared memory with the text assistant.
    """
    model = model or cfg["claude_model"]
    timeout_s = cfg["claude_timeout_s"] if timeout_s is None else timeout_s

    prompt = f"{VOICE_SYSTEM_PROMPT}\n\nUser: {text}\nAssistant:"
    try:
        result = subprocess.run(
            [CLAUDE_BIN, "-p", "--model", model,
             "--tools", "", "--no-session-persistence"],
            input=prompt, capture_output=True, text=True,
            timeout=timeout_s, cwd=tempfile.gettempdir(),
        )
    except subprocess.TimeoutExpired:
        return "Sorry, I timed out thinking about that."
    except (FileNotFoundError, OSError):
        # The claude binary is not installed / not on PATH. Speak, do not crash.
        return "Sorry, the brain is not available right now."
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
        "transcript_avg_logprob": None,
        "transcript_no_speech_prob": None,
        "reply": None,
        "output_wav": None,
        "timings_s": timings,
    }
    if not fired:
        timings["total"] = time.time() - t0
        return result

    # The file path is an inspection tool, so it surfaces the confidence signals
    # but does not gate on them: the live turn path (run_turn) is where the mishear
    # guard stops a low-confidence command from reaching the brain.
    t1 = time.time()
    heard = transcribe_detailed(wav_path)
    timings["transcribe"] = time.time() - t1
    result["transcript"] = heard.text
    result["transcript_avg_logprob"] = round(heard.avg_logprob, 3)
    result["transcript_no_speech_prob"] = round(heard.no_speech_prob, 3)

    t2 = time.time()
    reply = brain(heard.text)
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


def run_turn(frames, model_name: str | None = None,
             threshold: float | None = None,
             out_wav_path: str | None = None,
             on_capture=None) -> dict | None:
    """One live turn over a frame stream: wait for the wake word, capture the
    request to its endpoint, then transcribe -> brain -> speak.

    `frames` is any iterator of 80 ms int16 frames (iter_wav_frames for tests, a
    mic source on hardware). Returns a result dict, or None when there is nothing
    to act on: the stream ended before the wake word fired, the wake fired but no
    speech followed it, or the captured audio transcribed to no words (noise). When
    the mishear guard rejects a low-confidence transcript the brain is skipped, but
    a result dict is still returned (reply set to a spoken re-prompt, "rejected" set
    to "low_confidence") so the loop speaks the re-prompt. This is the unit an
    always-on loop calls repeatedly; the loop wrapper and mic/speaker I/O land with
    the microphone (issues #10, #11).

    `on_capture`, if given, is called once the request has been captured (listening
    for this turn is over). A half-duplex live loop uses it to stop the mic before
    the slow transcribe/brain/speak stages, so they do not buffer input the loop
    would only discard. It fires whether or not speech was found, so the caller can
    pair every call with a resume.
    """
    cfg = load_config()
    model_name = model_name or cfg["wake_word"]
    threshold = cfg["wake_threshold"] if threshold is None else threshold
    model = _get_oww_model(_resolve_wake_path(model_name))

    frames = iter(frames)
    # Keep the most recent frames during detection so the request's leading audio,
    # consumed while the detector was crossing threshold, is recovered (issue #30).
    preroll = collections.deque(maxlen=_PREROLL_FRAMES)
    score = stream_detect_wake(frames, model, threshold, preroll=preroll)
    if score is None:
        return None

    request_pcm = capture_request(frames, preroll=list(preroll),
                                  vad_threshold=cfg["capture_vad_threshold"])
    # Listening for this turn is done. Let a live loop stop the mic now, before the
    # slow stages below, so it does not buffer (then throw away) input recorded
    # while the assistant is thinking and speaking.
    if on_capture is not None:
        on_capture()
    if request_pcm.size == 0:
        return None  # wake fired but only silence followed — ignore the turn

    # whisper's wrapper takes a path, so stage the captured request to a temp WAV.
    fd, req_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(req_wav, request_pcm, 16000, subtype="PCM_16")
        heard = transcribe_detailed(req_wav)
    finally:
        os.unlink(req_wav)

    if not heard.text.strip():
        return None  # captured audio whisper read as no words (noise) — ignore

    if out_wav_path is None:
        out_wav_path = str(PROJECT_DIR / "test_audio" / "reply.wav")

    # Mishear guard: the brain acts on the command (files an issue, changes
    # something), so a low-confidence transcript must never reach it. On a reject,
    # speak a short re-prompt and return the turn marked rejected -- the loop still
    # plays the re-prompt, since spoken feedback is the only channel, but the brain
    # is never called, so a garbled word cannot trigger an action.
    ok, reason = guard_transcript(heard, cfg)
    if not ok:
        speak(STT_REPROMPT, out_wav_path)
        return {
            "wake_word": model_name,
            "wake_score": round(score, 4),
            "transcript": heard.text,
            "reply": STT_REPROMPT,
            "output_wav": out_wav_path,
            "rejected": "low_confidence",
            "reject_reason": reason,
        }

    reply = brain(heard.text)
    speak(reply, out_wav_path)
    return {
        "wake_word": model_name,
        "wake_score": round(score, 4),
        "transcript": heard.text,
        "reply": reply,
        "output_wav": out_wav_path,
    }


def run_loop(wake_word: str | None = None, mic_name=None, output_name=None,
             threshold: float | None = None) -> None:
    """Always-on voice loop: listen, wake, capture, answer, speak -- repeat.

    Drives run_turn over a live microphone (audio.Microphone) and plays each reply
    through the configured output device. Half-duplex: run_turn's on_capture hook
    pauses the mic the instant the request is captured, so the slow transcribe /
    brain / speak stages and the reply playback never run while the mic is live.
    After each turn the mic is resumed and flushed, dropping anything that leaked in
    so the assistant never transcribes its own voice -- raw devices have no echo
    cancellation, so the loop avoids capturing and playing at once rather than
    relying on the hardware to cancel the speaker.

    Device names come from config (mic_device / output_device) unless overridden.
    `audio` is imported lazily so file-mode use and the tests never require the
    PortAudio backend.
    """
    import audio  # lazy: only the live loop needs the PortAudio backend

    cfg = load_config()
    wake_word = wake_word or cfg["wake_word"]
    if mic_name is None:
        mic_name = cfg.get("mic_device") or None
    if output_name is None:
        output_name = cfg.get("output_device") or None
    out_wav = str(PROJECT_DIR / "test_audio" / "reply.wav")
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)

    print(f"computah listening -- wake word: {wake_word}. Ctrl-C to stop.")
    with audio.Microphone(mic_name) as mic:
        print(f"mic: {mic.device_label}")
        frames = mic.frames()
        paused = False

        def pause_for_turn():
            nonlocal paused
            mic.pause()
            paused = True

        try:
            while True:
                paused = False
                result = run_turn(frames, model_name=wake_word,
                                  threshold=threshold, out_wav_path=out_wav,
                                  on_capture=pause_for_turn)
                if result is not None:
                    print(f"  heard: {result['transcript']!r}")
                    if result.get("rejected") == "low_confidence":
                        print("  low confidence, re-prompting "
                              f"({result.get('reject_reason')})")
                    print(f"  reply: {result['reply']!r}")
                    audio.play_wav(out_wav, output_name)
                if paused:
                    # A turn was captured (mic stopped before the slow stages).
                    # Flush while the stream is still stopped: no fresh audio is
                    # arriving yet, so this drops only the stale audio buffered
                    # during thinking/speaking. Resume after, so the next utterance
                    # is captured from a clean buffer even if the user starts
                    # speaking the instant playback ends. Flushing after resume()
                    # would race the callback and discard the start of that speech.
                    mic.flush()
                    mic.resume()
                elif not mic.active():
                    # run_turn returned without capturing a request and the stream
                    # is no longer delivering frames: the device ended. Stop.
                    print("mic stream ended; stopping")
                    break
        except KeyboardInterrupt:
            print("\nstopped.")


def _cli() -> int:
    p = argparse.ArgumentParser(description="jawn-voice pipeline (mic-free)")
    p.add_argument("input_wav", nargs="?", help="input WAV to process")
    p.add_argument("-o", "--output", help="output WAV path for the spoken reply")
    p.add_argument("-w", "--wake-word", help="override the active wake word")
    p.add_argument("--list-wake-words", action="store_true",
                   help="list installed wake-word models and exit")
    p.add_argument("--set-wake-word", metavar="NAME",
                   help="persist a new active wake word to config.json and exit")
    p.add_argument("--listen", action="store_true",
                   help="run the always-on live mic loop (needs sounddevice)")
    p.add_argument("--mic", metavar="NAME",
                   help="mic device name substring (overrides config mic_device)")
    p.add_argument("--speaker", metavar="NAME",
                   help="output device name substring (overrides config output_device)")
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
    if args.listen:
        run_loop(wake_word=args.wake_word, mic_name=args.mic,
                 output_name=args.speaker)
        return 0
    if not args.input_wav:
        p.error("input_wav is required unless using "
                "--listen/--list-wake-words/--set-wake-word")

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
