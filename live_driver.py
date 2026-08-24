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

This is the real-hardware counterpart to the mic-free streaming proof in
test_stream_turn.py, which drives pipeline.stream_detect_wake and capture_request
with synthetic audio instead of a live mic. Keep them distinct: that one
proves the streaming/endpointing logic with no hardware; this one drives the actual
device against the real persistent brain.
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline
import chime

_DRAIN_FRAMES = 25  # ~2 s discarded after a turn: clears stale/echo audio from the
# pipe buffer so the spoken reply is not re-heard as the next wake


def log(msg: str) -> None:
    print(f"[computah] {msg}", file=sys.stderr, flush=True)


def _resolve_output_pcm(cli_output_device: str | None, cfg: dict) -> str | None:
    """Pick the ALSA PCM live_driver plays through, `-o` winning over config (issue #50).

    Precedence: an explicit `-o/--output-device` first, else the config `live_output_pcm`,
    else None (the ALSA default device). Only `live_output_pcm` is read here, never the
    sounddevice `output_device`: that is a friendly-substring for the audio.py path and
    would fail as an `aplay -D` argument, so the two naming conventions stay separate.
    An empty-string value is treated as unset, matching the DEFAULTS sentinel.

    `-o` present (including an explicit `-o ''`) always wins, so `None` (option omitted)
    and `""` (explicit override to the ALSA default) stay distinct: a live_output_pcm in
    config cannot override an `-o ''` on the command line."""
    if cli_output_device is not None:
        return cli_output_device or None
    return cfg.get("live_output_pcm") or None


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
        subprocess.run(
            cmd, check=True, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        # Retry once under non-interactive sudo: /dev/snd needs root in some launch
        # contexts. -n so a missing or expired sudo timestamp fails fast instead of
        # prompting, and stdin from /dev/null so a prompt can never consume the raw
        # mic pipe that is this process's stdin. Playback then degrades to the logged
        # error in run_turn rather than wedging the loop.
        subprocess.run(
            ["sudo", "-n", *cmd],
            check=True,
            stdin=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


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
                frame = np.frombuffer(
                    self._buf[: self.frame_bytes], dtype=np.int16
                ).copy()
                self._buf = self._buf[self.frame_bytes :]
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


def run_turn(
    frames,
    mic,
    model,
    threshold: float,
    out_wav: str,
    output_device,
    cfg: dict,
    debug: bool,
) -> bool:
    """Run one full turn off the live frame stream.

    `mic` is the StdinMic that owns `frames`; its flush() is used to drop the cue's
    mic bleed (issue #56). Returns True if a turn ran (or was correctly skipped as
    noise), False only when the input stream ended and the loop should stop.
    """
    # Keep the most recent frames during detection so the request's leading audio,
    # consumed while the detector crossed threshold, is recovered (issue #30).
    wake_audio = pipeline.WakeAudioBuffer()
    score = listen_for_wake(frames, model, threshold, debug, preroll=wake_audio)
    if score is None:
        return False
    log(f"wake fired (score={score:.3f})")

    # Acknowledge the wake with a cue before capture (issue #41), but only when the
    # user paused after the wake word (issue #55). arecord cannot be paused, so on a
    # shared mic/speaker device (the PowerConf) the cue bleeds straight back into the
    # mic, and a command spoken in one breath with the wake word would be lost to the
    # cue window. peek_cue_gate reads the post-wake audio first: a no-pause command has
    # speech there, a user waiting for the cue has room tone.
    #
    # No-pause command: skip the cue so it cannot clip the command, and prepend the
    # peeked command frames to capture -- the pre-roll stays, recovering the leading
    # audio detection ate (issue #30). Paused: play the cue, then drop the frames
    # buffered while it played so it is not captured as part of the request. Flush-to-now
    # does this (issue #56): mic.flush() drops, in one non-blocking shot, exactly what is
    # buffered the instant capture starts -- cue bleed, the peeked pre-cue room tone, and
    # ambient -- then capture reads only fresh audio.
    #
    # flush() replaces sizing the drop by _play_wav's wall-clock span, which conflated
    # the cue's audio with subprocess overhead (a failed unprivileged aplay then a sudo
    # retry, plus device-open and teardown latency). That frame-count guess could
    # under-drain, leaking a cue tail into the transcript, or -- because drain() blocked
    # per frame -- over-drain past the buffer and consume the start of a command spoken
    # after the cue. flush() carries no time estimate and never blocks on or consumes
    # fresh frames, so a command spoken once _play_wav returns is captured intact. On a
    # cue failure nothing played and nothing bled in, so the flush is skipped (the else)
    # and the request is untouched.
    capture_frames = frames
    if cfg.get("wake_chime", False):  # opt-in, default off (issue #41) until the
        # pause-gate (issue #55) is validated on the PowerConf hardware
        play_cue, peeked = pipeline.peek_cue_gate(
            frames, vad_threshold=cfg["capture_vad_threshold"]
        )
        cue_boundary = False
        if play_cue:
            try:
                _play_wav(chime.wake_cue_wav(), output_device)
            except Exception as e:  # noqa: BLE001 - the chime is a nicety, not core
                log(f"wake chime failed ({type(e).__name__}: {e})")
            else:
                mic.flush()
                # The flush established a clean post-cue capture boundary (it dropped the
                # frames buffered while the cue played). The peeked pre-cue room tone was
                # already pulled off the stream and is dropped here by not being chained
                # back into capture. Drop the detection pre-roll too: it holds the pre-cue
                # wake-word tail, kept only to recover a no-pause command's leading audio
                # (issue #30) -- which does not apply once the user waits for the cue.
                # Without this the stale tail would prepend the post-cue command and reach
                # the brain (issue #41).
                wake_audio.clear()
                cue_boundary = True
                log(
                    f"wake cue gate: play ({len(peeked)} peeked frames; "
                    "capture boundary established)"
                )
        else:
            log(f"wake cue gate: skip ({len(peeked)} peeked frames; speech detected)")
        if not cue_boundary:
            # Either a no-pause command (cue skipped so it cannot clip the command) or a
            # cue that failed to establish a boundary (nothing flushed, buffered audio
            # kept). Both keep the pre-roll and feed the peeked frames to capture ahead of
            # the rest of the stream, so no leading audio is clipped (issues #30, #55).
            capture_frames = itertools.chain(peeked, frames)

    request_pcm = pipeline.capture_request(
        capture_frames,
        preroll=list(wake_audio.preroll),
        vad_threshold=cfg["capture_vad_threshold"],
        endpoint_silence_ms=cfg["endpoint_silence_ms"],
        max_request_ms=cfg["max_request_ms"],
    )
    if request_pcm.size == 0:
        if getattr(request_pcm, "empty_reason", None) != pipeline._EMPTY_NO_ONSET:
            log("post-wake audio was rejected — ignoring")
            return True
        heard = pipeline.recover_consumed_command(wake_audio.history, cfg["wake_word"])
        if heard is None:
            log("wake fired but no recoverable command followed — ignoring")
            return True
        log(f"recovered consumed command: {heard.text!r}")
    else:
        log(f"captured {request_pcm.size / 16000:.2f}s of speech")
        # The captured int16 PCM is normalized by transcribe_detailed and sent straight
        # to faster-whisper; no request-side temporary WAV is needed.
        heard = pipeline.transcribe_detailed(request_pcm)
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
    ap.add_argument(
        "out_wav",
        nargs="?",
        default=None,
        help="reply WAV path (default: a temp file reused each turn)",
    )
    ap.add_argument(
        "-v",
        "--debug",
        action="store_true",
        help="per-frame rms+score telemetry to stderr",
    )
    ap.add_argument(
        "-o",
        "--output-device",
        default=None,
        help="ALSA output PCM for aplay -D (e.g. "
        "plughw:CARD=PowerConf,DEV=0); default: config live_output_pcm, "
        "then the ALSA default device",
    )
    args = ap.parse_args()

    out_wav = args.out_wav
    auto_wav = out_wav is None
    if auto_wav:
        fd, out_wav = tempfile.mkstemp(prefix="computah-reply-", suffix=".wav")
        os.close(fd)

    cfg = pipeline.load_config()
    output_pcm = _resolve_output_pcm(args.output_device, cfg)
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
            if not run_turn(
                frames,
                mic,
                model,
                threshold,
                out_wav,
                output_pcm,
                cfg,
                args.debug,
            ):
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
