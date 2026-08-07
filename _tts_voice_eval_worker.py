#!/usr/bin/env python3
"""One voice, loaded once, every line synthesized (#38). Driven by tts_voice_eval.py.

Separate process per voice so peak memory is that voice's own, not the high-water
mark of everything measured before it. Prints one JSON object on stdout.
"""

from __future__ import annotations

import json
import resource
import sys
import time
import wave


def main() -> int:
    onnx, lines_json, out_dir, voice = sys.argv[1:5]
    lines = json.loads(lines_json)

    from piper import PiperVoice

    started = time.perf_counter()
    loaded = PiperVoice.load(onnx)
    load_s = time.perf_counter() - started

    measured = []
    for label, text in lines:
        path = f"{out_dir}/{voice}__{label}.wav"
        started = time.perf_counter()
        with wave.open(path, "wb") as handle:
            loaded.synthesize_wav(text, handle)
        synth_s = time.perf_counter() - started
        with wave.open(path) as handle:
            audio_s = handle.getnframes() / handle.getframerate()
        measured.append({"label": label, "synth_s": synth_s, "audio_s": audio_s})

    print(
        json.dumps(
            {
                "load_s": load_s,
                "lines": measured,
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
