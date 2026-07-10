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
import collections
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline
import chime

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


class StdinMic:
    """The arecord stdin pipe as a flushable stream of 80 ms / 16 kHz / mono int16
    frames — live_driver's counterpart to audio.Microphone, with the same
    frames()/flush() contract on the raw-pipe path.

    Reads the raw fd directly (os.read), not sys.stdin.buffer, so there is a single
    place to flush: the OS pipe, plus a sub-frame byte remainder kept on the instance
    (self._buf) so flush() can drop it too. Reading through a BufferedReader would add a
    second, Python-level buffer that an fd-level flush cannot reach, leaving stale bytes
    to prefix the next frame. flush() is the flush-to-now the post-cue drain needs
    (issue #56); it matches run_loop's mic.flush() semantics.
    """

    def __init__(self, fd: int, frame_size: int = pipeline.FRAME_SIZE):
        self.fd = fd
        self.frame_bytes = frame_size * 2  # int16: 1280 samples = 2560 bytes (80 ms)
        self._buf = b""  # sub-frame remainder, on the instance so flush() reaches it

    def frames(self):
        """Yield writable 80 ms int16 frames from the raw pipe. Ends (returns) on EOF,
        i.e. when arecord exits. A short os.read is accumulated toward a full frame
        rather than ending the stream, so a mid-stream partial read is not mistaken
        for EOF (BufferedReader hid these; the raw fd surfaces them).

        .copy() because np.frombuffer aliases a read-only buffer; the wake model's
        predict path and capture_request both expect a writable array.
        """
        while True:
            chunk = os.read(self.fd, self.frame_bytes - len(self._buf))
            if not chunk:  # EOF: arecord exited
                return
            self._buf += chunk
            if len(self._buf) >= self.frame_bytes:
                frame = np.frombuffer(self._buf[:self.frame_bytes], dtype=np.int16).copy()
                self._buf = self._buf[self.frame_bytes:]
                yield frame

    def flush(self) -> None:
        """Flush-to-now: drop everything buffered up to this instant — the sub-frame
        remainder and the whole OS pipe backlog — without blocking, so the next frame
        is built only from audio captured after the flush. Mirrors
        audio.Microphone.flush.

        Sets the fd non-blocking and reads until EAGAIN (no data buffered) or EOF, then
        restores blocking mode. Safe to call only while frames() is suspended (between
        detection and capture, or between turns), which is how run_turn uses it — there
        is no concurrent os.read to race the non-blocking window.
        """
        self._buf = b""
        os.set_blocking(self.fd, False)
        try:
            while True:
                try:
                    if not os.read(self.fd, 65536):
                        break  # EOF: the writer (arecord) closed
                except BlockingIOError:
                    break  # pipe drained to now (EAGAIN)
        finally:
            os.set_blocking(self.fd, True)


def listen_for_wake(frames, model, threshold: float, debug: bool, preroll=None):
    """Feed frames through the wake model until it crosses threshold.

    Resets the model ONCE at the start of each listen phase (the streaming
    contract), then runs continuously. Returns the firing score, or None if the
    input stream ended before any wake.

    `preroll`, if given, is a bounded collection (a deque(maxlen=N)) that each frame is
    appended to as it is consumed, so on a fire it holds the most recent N frames
    including the firing frame; capture_request prepends them so a request spoken
    with no pause after the wake word is not clipped (issue #30). Mirrors
    pipeline.stream_detect_wake's pre-roll contract on the hardware path.
    """
    pipeline._reset_oww(model)
    i = 0
    for frame in frames:
        i += 1
        if preroll is not None:
            preroll.append(frame)
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


def run_turn(frames, mic, model, threshold: float, out_wav: str,
             output_device, cfg: dict, debug: bool) -> bool:
    """Run one full turn off the live frame stream.

    `mic` is the StdinMic that owns `frames`; its flush() is used to drop the cue's
    mic bleed (issue #56). Returns True if a turn ran (or was correctly skipped as
    noise), False only when the input stream ended and the loop should stop.
    """
    # Keep the most recent frames during detection so the request's leading audio,
    # consumed while the detector crossed threshold, is recovered (issue #30).
    preroll = collections.deque(maxlen=pipeline._PREROLL_FRAMES)
    score = listen_for_wake(frames, model, threshold, debug, preroll=preroll)
    if score is None:
        return False
    log(f"wake fired (score={score:.3f})")

    # Acknowledge the wake with a cue before capture (issue #41). arecord cannot be
    # paused, so on a shared mic/speaker device (the PowerConf) the cue bleeds straight
    # back into the mic; drop the frames buffered while it played so the cue is not
    # captured as part of the request. Flush-to-now does this (issue #56): mic.flush()
    # drops, in one non-blocking shot, exactly what is buffered the instant capture
    # starts -- cue bleed and ambient -- then capture reads only fresh audio.
    #
    # This replaces sizing the drop by _play_wav's wall-clock span, which conflated the
    # cue's audio with subprocess overhead (a failed unprivileged aplay then a sudo
    # retry, plus device-open and teardown latency). That frame-count guess could
    # under-drain, leaking a cue tail into the transcript, or -- because drain() blocked
    # per frame -- over-drain past the buffer and consume the start of a command spoken
    # after the cue. flush() carries no time estimate and never blocks on or consumes
    # fresh frames, so a command spoken once _play_wav returns is captured intact.
    #
    # It does NOT rescue speech uttered in the narrow window between the cue audio
    # ending and _play_wav returning (teardown latency): that audio is already buffered
    # when flush() runs and is dropped with the bleed. That is inherent to a blocking
    # cue on a half-duplex mic -- capture cannot begin until playback returns -- and
    # capturing through the cue is the deferred half-duplex fix (the #41 no-pause
    # regression that keeps the chime opt-in). On a cue failure nothing played and
    # nothing bled in, so the flush is skipped (the else) and the request is untouched.
    if cfg.get("wake_chime", False):  # opt-in, default off (issue #41): the cue
        # regresses the no-pause case on this half-duplex device, so off unless enabled
        try:
            _play_wav(chime.wake_cue_wav(), output_device)
        except Exception as e:  # noqa: BLE001 - the chime is a nicety, not core
            log(f"wake chime failed ({type(e).__name__}: {e})")
        else:
            mic.flush()
            # The flush established a clean post-cue capture boundary (it dropped the
            # frames buffered while the cue played). Drop the detection pre-roll with it:
            # it holds the pre-cue wake-word tail, kept only to recover a no-pause
            # command's leading audio (issue #30) -- which does not apply once the user
            # waits for the cue. Without this the stale tail would prepend the post-cue
            # command and reach the brain (issue #41).
            preroll.clear()

    request_pcm = pipeline.capture_request(frames, preroll=list(preroll),
                                           vad_threshold=cfg["capture_vad_threshold"],
                                           endpoint_silence_ms=cfg["endpoint_silence_ms"],
                                           max_request_ms=cfg["max_request_ms"])
    if request_pcm.size == 0:
        log("wake fired but no speech followed — ignoring")
        return True
    log(f"captured {request_pcm.size / 16000:.2f}s of speech")

    fd, req_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(req_wav, request_pcm, 16000, subtype="PCM_16")
        heard = pipeline.transcribe_detailed(req_wav)
    finally:
        os.unlink(req_wav)
    if not heard.text.strip():
        log("empty transcript (noise) — ignoring")
        return True
    log(f"you said: {heard.text!r}")

    # Mishear guard: this is the real-hardware path to the action-capable brain, so
    # a low-confidence transcript must not be dispatched. On a reject, speak the
    # re-prompt and skip the brain, so a garbled command never triggers an action.
    ok, reason = pipeline.guard_transcript(heard, cfg)
    if not ok:
        log(f"low-confidence transcript ({reason}) — re-prompting, not dispatching")
        reply = pipeline.STT_REPROMPT
    else:
        t0 = time.monotonic()
        reply = pipeline.brain(heard.text)
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

    # Read the raw stdin fd directly (not sys.stdin.buffer) so the mic's flush() owns
    # the only buffer between the pipe and a frame — see StdinMic (issue #56).
    mic = StdinMic(sys.stdin.fileno())
    frames = mic.frames()
    log(f"listening; wake={name!r} thr={threshold} (say '{name} ...'); ctrl-c to stop")

    turn = 0
    try:
        while True:
            if not run_turn(frames, mic, model, threshold, out_wav,
                            args.output_device, cfg, args.debug):
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
