#!/usr/bin/env python3
"""Cross-platform live audio I/O for computah (the mic source and speaker sink).

The pipeline core is hardware-free: detect_wake / capture_request / run_turn
consume any iterator of 80 ms, 16 kHz, mono int16 frames (see pipeline.py). This
module is the one OS-specific seam -- it turns a real microphone into exactly
that frame stream, and plays a reply WAV back through a real speaker. It is built
on sounddevice (PortAudio), which is the same API on Windows (WASAPI), macOS
(CoreAudio) and Linux/Raspberry Pi (ALSA).

Devices are chosen by case-insensitive name substring from config, never
hardcoded, so each host names its own hardware:
  - Raspberry Pi : PowerConf over USB (mono 16 kHz natively)
  - Legion (Win) : Shure MV7 mic + desktop speakers (raw input/output)

Sample-rate / channel conversion is done by the OS audio engine when possible
(WASAPI auto_convert on Windows; native 16 kHz on the Pi) so the 80 ms frames are
clean. A Python resample_poly path is the fallback for devices that cannot open
at 16 kHz mono directly.

CLI:
  python audio.py --list                       # list audio devices
  python audio.py --test-mic "nvidia broadcast"  # capture a few seconds, report level
  python audio.py --play reply.wav --out "desktop speakers"
"""

from __future__ import annotations

import argparse
import queue
import sys
from math import gcd
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly

# Must match pipeline.py: openWakeWord's frame is 80 ms at 16 kHz = 1280 samples.
TARGET_SR = 16000
FRAME_SIZE = 1280

# Host-API preference when a device name matches under several APIs. WASAPI is the
# low-latency modern path on Windows; DirectSound/MME are fallbacks; ALSA is the
# native path on the Pi. Lower rank wins.
_HOST_RANK = (("WASAPI", 0), ("DirectSound", 1), ("ALSA", 2), ("MME", 4))


def _host_rank(host: str) -> int:
    for key, rank in _HOST_RANK:
        if key in host:
            return rank
    return 3  # unknown host: better than MME, worse than ALSA


def _norm_subs(name) -> list[str] | None:
    """Normalize a device-name selector to a lowercase substring list, or None
    (meaning: use the system default device)."""
    if not name:
        return None
    items = name if isinstance(name, (list, tuple)) else [name]
    subs = [s.lower() for s in items if s]
    return subs or None


def find_device(name, kind: str) -> tuple[int, dict, str]:
    """Resolve a device by name substring. kind is 'input' or 'output'.

    With no name, returns the system default device for that direction. Otherwise
    returns the best name match (preferring WASAPI > DirectSound > ALSA). Returns
    (index, info, host_name). Raises RuntimeError if nothing matches.
    """
    want_in = kind == "input"
    subs = _norm_subs(name)
    if subs is None:
        default = sd.default.device[0 if want_in else 1]
        if default is None or default < 0:
            raise RuntimeError(f"no default {kind} device available")
        info = sd.query_devices(default)
        return int(default), info, sd.query_hostapis(info["hostapi"])["name"]

    best = None
    for i, d in enumerate(sd.query_devices()):
        channels = d["max_input_channels"] if want_in else d["max_output_channels"]
        if channels <= 0 or not any(s in d["name"].lower() for s in subs):
            continue
        host = sd.query_hostapis(d["hostapi"])["name"]
        rank = _host_rank(host)
        if best is None or rank < best[0]:
            best = (rank, i, d, host)
    if best is None:
        raise RuntimeError(
            f"no {kind} device matching {subs}. run `python audio.py --list`")
    return best[1], best[2], best[3]


def _wasapi_autoconvert(host: str):
    """WASAPI extra_settings that let the audio engine convert rate/channels, or
    None on non-WASAPI hosts / older sounddevice without the flag."""
    if "WASAPI" not in host or not hasattr(sd, "WasapiSettings"):
        return None
    try:
        return sd.WasapiSettings(auto_convert=True)
    except TypeError:
        return None  # sounddevice too old to support auto_convert


def _to_mono_float(block: np.ndarray, channels: int) -> np.ndarray:
    """int16 callback block (n, channels) -> mono float32 in [-1, 1]."""
    a = block.astype(np.float32)
    if a.ndim == 2:
        a = a.mean(axis=1) if a.shape[1] > 1 else a[:, 0]
    return a / 32768.0


def _to_int16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


class Microphone:
    """A live microphone as a stream of 80 ms / 16 kHz / mono int16 frames.

    Use as a context manager; iterate `frames()` to drive run_turn. The audio
    engine resamples to 16 kHz mono when it can (clean); otherwise frames are
    resampled in Python. `flush()` drops buffered audio -- the always-on loop
    pauses the mic while the assistant speaks and flushes on resume, so the
    assistant never transcribes its own voice bleeding back in (half-duplex;
    raw devices have no echo cancellation, so don't capture and play at once).
    """

    def __init__(self, name=None, frame_size: int = FRAME_SIZE,
                 target_sr: int = TARGET_SR, queue_timeout_s: float = 0.5):
        self.name = name
        self.frame_size = frame_size
        self.target_sr = target_sr
        self.queue_timeout_s = queue_timeout_s
        self._q: queue.Queue = queue.Queue()
        self._buf = np.zeros(0, dtype=np.float32)  # sub-frame remainder for frames()
        self._stream = None
        self._open_sr = None
        self._open_ch = None
        self.device_label = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"mic status: {status}", file=sys.stderr)
        self._q.put(indata.copy())

    def _open(self, idx: int, host: str):
        """Open the input stream, preferring 16 kHz mono via the OS engine, with
        native-rate fallbacks. Returns (stream, open_sr, open_channels)."""
        extra = _wasapi_autoconvert(host)
        info = sd.query_devices(idx)
        native_sr = int(info["default_samplerate"])
        native_ch = max(1, int(info["max_input_channels"]))
        # Order matters: clean engine-converted path first, then native fallbacks.
        attempts = [
            (self.target_sr, 1, extra),
            (native_sr, 1, extra),
            (native_sr, native_ch, extra),
            (native_sr, native_ch, None),
        ]
        last = None
        for sr, ch, ex in attempts:
            try:
                stream = sd.InputStream(
                    device=idx, samplerate=sr, channels=ch, dtype="int16",
                    blocksize=0, callback=self._callback, extra_settings=ex)
                return stream, sr, ch
            except Exception as e:  # noqa: BLE001 - try the next fallback config
                last = e
        raise RuntimeError(f"could not open input device {idx}: {last}")

    def __enter__(self):
        idx, info, host = find_device(self.name, "input")
        self.device_label = f"{info['name']} ({host})"
        self._stream, self._open_sr, self._open_ch = self._open(idx, host)
        self._stream.start()
        return self

    def __exit__(self, *exc):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        return False

    def active(self) -> bool:
        """True while the input stream is open and running."""
        return self._stream is not None and self._stream.active

    def pause(self) -> None:
        """Stop capturing (e.g. while the assistant speaks) without closing the
        device. Makes the loop half-duplex at the device level: no input+output
        run at once, which sidesteps concurrent-stream driver conflicts."""
        if self._stream is not None and self._stream.active:
            self._stream.stop()

    def resume(self) -> None:
        """Resume capturing after pause()."""
        if self._stream is not None and not self._stream.active:
            self._stream.start()

    def flush(self) -> None:
        """Drop all buffered audio: the queued callback blocks and the partial
        frame left over inside frames(). After flush(), the next frame yielded is
        built only from audio captured after the flush -- so a half-duplex loop can
        resume() + flush() and be sure no pre-pause samples leak into the next turn.
        Safe to call only while frames() is not being advanced (e.g. between turns,
        when its generator is suspended), which is how the live loop uses it."""
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._buf = np.zeros(0, dtype=np.float32)

    def frames(self):
        """Yield consecutive frame_size int16 mono frames at target_sr.

        Ends if the stream delivers nothing for queue_timeout_s (stream stopped).

        The sub-frame remainder lives on the instance (self._buf), not in a local,
        so flush() can drop it too. That keeps one long-lived generator usable
        across pause/resume: without it, the leftover samples from before a pause
        would prefix the next turn and flush() could not reach them.
        """
        need_resample = self._open_sr != self.target_sr
        if need_resample:
            g = gcd(self.target_sr, self._open_sr)
            up, down = self.target_sr // g, self._open_sr // g
        while True:
            try:
                block = self._q.get(timeout=self.queue_timeout_s)
            except queue.Empty:
                # A finite-timeout read keeps Ctrl-C responsive on Windows. Empty
                # during a quiet gap is normal (the callback still feeds silence,
                # so this is rare); only a stopped/closed stream actually ends it.
                if self._stream is None or not self._stream.active:
                    return
                continue
            mono = _to_mono_float(block, self._open_ch)
            if need_resample:
                mono = resample_poly(mono, up, down)
            self._buf = mono if self._buf.size == 0 else np.concatenate([self._buf, mono])
            while self._buf.size >= self.frame_size:
                yield _to_int16(self._buf[:self.frame_size])
                self._buf = self._buf[self.frame_size:]


def play_wav(path, name=None) -> None:
    """Play a WAV through the named output device (default device if name is None).

    The clip is resampled to the device's rate in Python before playback, so this
    does not depend on the engine's converter being available. Blocks until done.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    idx, info, host = find_device(name, "output")
    dev_sr = int(info["default_samplerate"])
    if sr != dev_sr:
        g = gcd(dev_sr, sr)
        data = resample_poly(data, dev_sr // g, sr // g)
        sr = dev_sr
    sd.play(data, sr, device=idx, extra_settings=_wasapi_autoconvert(host))
    sd.wait()


def list_devices() -> None:
    """Print the audio device table (index, I/O, host API, default rate)."""
    for i, d in enumerate(sd.query_devices()):
        io = []
        if d["max_input_channels"] > 0:
            io.append(f"IN x{d['max_input_channels']}")
        if d["max_output_channels"] > 0:
            io.append(f"OUT x{d['max_output_channels']}")
        host = sd.query_hostapis(d["hostapi"])["name"]
        print(f"[{i:2d}] {d['name'][:44]:44s} {','.join(io):10s} "
              f"{host:18s} @{int(d['default_samplerate'])}")


def _test_mic(name: str, seconds: float) -> int:
    """Capture from a mic for a few seconds and report frame shape + level."""
    subs = _norm_subs(name)
    print(f"opening mic {subs or '(default)'} ...")
    n_frames = 0
    peak = 0
    sumsq = 0.0
    n_samples = 0
    target = int(seconds * TARGET_SR / FRAME_SIZE)
    with Microphone(subs) as mic:
        print(f"device: {mic.device_label}  open_sr={mic._open_sr} "
              f"open_ch={mic._open_ch}")
        for frame in mic.frames():
            assert frame.dtype == np.int16 and frame.shape == (FRAME_SIZE,), \
                f"bad frame: dtype={frame.dtype} shape={frame.shape}"
            n_frames += 1
            peak = max(peak, int(np.max(np.abs(frame))))
            sumsq += float(np.sum(frame.astype(np.float32) ** 2))
            n_samples += frame.size
            if n_frames >= target:
                break
    rms = (sumsq / n_samples) ** 0.5 if n_samples else 0.0
    print(f"captured {n_frames} frames ({n_frames * FRAME_SIZE / TARGET_SR:.1f}s "
          f"at {TARGET_SR} Hz)  RMS={rms:.1f}  peak={peak}")
    if peak == 0:
        print("RESULT: dead stream (only zeros) - check the mic's source")
        return 2
    print("RESULT: live frame stream - shape and dtype correct for the pipeline")
    return 0


def _cli() -> int:
    p = argparse.ArgumentParser(description="computah live audio I/O")
    p.add_argument("--list", action="store_true", help="list audio devices")
    p.add_argument("--test-mic", metavar="NAME", nargs="?", const="",
                   help="capture from a mic (name substring; empty = default)")
    p.add_argument("--play", metavar="WAV", help="play a WAV through an output device")
    p.add_argument("--out", metavar="NAME", default=None,
                   help="output device name substring for --play")
    p.add_argument("--seconds", type=float, default=3.0,
                   help="capture duration for --test-mic")
    args = p.parse_args()

    if args.list:
        list_devices()
        return 0
    if args.test_mic is not None:
        return _test_mic(args.test_mic, args.seconds)
    if args.play:
        play_wav(args.play, _norm_subs(args.out))
        print(f"played {args.play}")
        return 0
    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
