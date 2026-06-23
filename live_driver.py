#!/usr/bin/env python3
"""Always-on live voice loop for computah: real PowerConf mic -> wake -> STT ->
brain (bridge -> Syl) -> spoken reply, looping turn after turn.

Reads raw S16_LE 16 kHz mono frames from stdin (piped from `arecord -t raw`), so
the mic driver stays out of this process. Runs continuously: after each turn it
resets the wake model and resumes listening, so repeated "computah ..." turns work
without relaunching the process.

  sudo arecord -D plughw:CARD=PowerConf,DEV=0 -f S16_LE -r 16000 -c 1 -t raw \
    | .venv/bin/python live_driver.py --debug

-v/--debug adds per-frame rms+score telemetry to stderr (use it to see why a wake
did or did not fire). Without it, only a one-line summary per turn is printed.

This is the real-hardware counterpart to experiments/live_loop.py, which fakes the
mic with a WAV file and a sim persona. Keep them distinct: that one is a no-hardware
proof of the streaming/endpointing logic; this one drives the actual device against
the real persistent brain.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline

FRAME_BYTES = pipeline.FRAME_SIZE * 2  # 1280 int16 = 2560 bytes (80 ms)
_DRAIN_FRAMES = 25  # ~2 s discarded after a turn: clears stale/echo audio from the
                    # pipe buffer so the spoken reply is not re-heard as the next wake


def log(msg: str) -> None:
    print(f"[computah] {msg}", file=sys.stderr, flush=True)


def _play_wav(path: str, device: str | None) -> None:
    """Play a WAV through ALSA's aplay — the Pi's native output, symmetric with the
    arecord capture side and dependency-free. (The sounddevice-based audio.play_wav
    is the cross-platform/dev path and is not installed in this venv.)

    Tries unelevated first, then retries under sudo: on this Pi /dev/snd needs root in
    a non-login launch context. Only the playback subprocess is elevated, never the
    whole driver — the brain stage shells `ssh officejawn`, which must run as the
    launching user so it uses that user's ssh config and keys, not root's."""
    cmd = ["aplay", "-q"]
    if device:
        cmd += ["-D", device]
    cmd.append(path)
    try:
        subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        # Retry once under non-interactive sudo: /dev/snd needs root in some launch
        # contexts. -n so a missing or expired sudo timestamp fails fast instead of
        # prompting, and stdin from /dev/null so a prompt can never consume the raw
        # mic pipe that is this process's stdin. Playback then degrades to the logged
        # error in run_turn rather than wedging the loop.
        subprocess.run(["sudo", "-n", *cmd], check=True,
                       stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def frames_from_stdin(stream):
    """Yield writable 80 ms int16 frames from a raw S16_LE stdin stream.

    .copy() because np.frombuffer is read-only and aliases the pipe buffer; the wake
    model's predict path and capture_request both expect a normal writable array.
    Ends (returns) on EOF or a short final read, i.e. when arecord exits.
    """
    while True:
        buf = stream.read(FRAME_BYTES)
        if not buf or len(buf) < FRAME_BYTES:
            return
        yield np.frombuffer(buf, dtype=np.int16).copy()


def listen_for_wake(frames, model, threshold: float, debug: bool):
    """Feed frames through the wake model until it crosses threshold.

    Resets the model ONCE at the start of each listen phase (the streaming
    contract), then runs continuously. Returns the firing score, or None if the
    input stream ended before any wake.
    """
    pipeline._reset_oww(model)
    i = 0
    for frame in frames:
        i += 1
        score = max(float(s) for s in model.predict(frame).values())
        if debug and (score > 0.2 or i % 100 == 0):
            rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
            log(f"f{i} rms={rms:.0f} score={score:.3f}")
        if score >= threshold:
            return score
    return None


def drain(frames, n: int) -> None:
    """Discard up to n frames — clears audio captured during the turn (TTS echo,
    the tail of the user's speech) so it cannot retrigger the next wake."""
    for _ in range(n):
        if next(frames, None) is None:
            return


def run_turn(frames, model, threshold: float, out_wav: str,
             output_device, debug: bool) -> bool:
    """Run one full turn off the live frame stream.

    Returns True if a turn ran (or was correctly skipped as noise), False only when
    the input stream ended and the loop should stop.
    """
    score = listen_for_wake(frames, model, threshold, debug)
    if score is None:
        return False
    log(f"wake fired (score={score:.3f})")

    request_pcm = pipeline.capture_request(frames)
    if request_pcm.size == 0:
        log("wake fired but no speech followed — ignoring")
        return True
    log(f"captured {request_pcm.size / 16000:.2f}s of speech")

    fd, req_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(req_wav, request_pcm, 16000, subtype="PCM_16")
        transcript = pipeline.transcribe(req_wav)
    finally:
        os.unlink(req_wav)
    if not transcript.strip():
        log("empty transcript (noise) — ignoring")
        return True
    log(f"you said: {transcript!r}")

    t0 = time.monotonic()
    reply = pipeline.brain(transcript)
    log(f"brain ({time.monotonic() - t0:.1f}s): {reply!r}")

    pipeline.speak(reply, out_wav)
    try:
        _play_wav(out_wav, output_device)
    except Exception as e:  # noqa: BLE001 - degrade to a saved WAV, never crash
        log(f"playback failed ({type(e).__name__}: {e}); reply WAV at {out_wav}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="computah always-on live voice loop")
    ap.add_argument("out_wav", nargs="?", default=None,
                    help="reply WAV path (default: a temp file reused each turn)")
    ap.add_argument("-v", "--debug", action="store_true",
                    help="per-frame rms+score telemetry to stderr")
    ap.add_argument("-o", "--output-device", default=None,
                    help="ALSA output PCM for aplay -D (e.g. "
                         "plughw:CARD=PowerConf,DEV=0); default: ALSA default device")
    args = ap.parse_args()

    out_wav = args.out_wav
    auto_wav = out_wav is None
    if auto_wav:
        fd, out_wav = tempfile.mkstemp(prefix="computah-reply-", suffix=".wav")
        os.close(fd)

    cfg = pipeline.load_config()
    name = cfg["wake_word"]
    threshold = cfg["wake_threshold"]
    model = pipeline._get_oww_model(pipeline._resolve_wake_path(name))

    frames = frames_from_stdin(sys.stdin.buffer)
    log(f"listening; wake={name!r} thr={threshold} (say '{name} ...'); ctrl-c to stop")

    turn = 0
    try:
        while True:
            if not run_turn(frames, model, threshold, out_wav,
                            args.output_device, args.debug):
                log("input stream ended — exiting")
                break
            turn += 1
            drain(frames, _DRAIN_FRAMES)
            log(f"--- turn {turn} done; listening again ---")
    except KeyboardInterrupt:
        log("stopped")
    finally:
        # Remove the reply WAV only when we created it; a user-supplied path is theirs.
        if auto_wav:
            try:
                os.unlink(out_wav)
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
