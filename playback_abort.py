"""Pure abort primitive for barge-in playback (issue #16).

The live loop plays a full reply before it listens again, so a long or wrong
answer cannot be cut short. Barge-in turns blocking playback into "start the
clip, then poll a stop signal while it plays, and abort the stream on a detected
wake/speech event." This module is only that poll loop, with every side effect
injected, so it carries no audio dependency and runs anywhere.

A caller supplies four callables:
  is_active()   -> bool  True while the output stream is still playing.
  should_stop() -> bool  True when a wake/speech event asks to interrupt.
  stop()        -> None  Aborts the output stream. Called at most once, only on
                         a barge-in.
  sleep(ms)     -> None  Sleeps the given milliseconds between polls, injected so
                         a caller sets the cadence and a test stays fast.

The mechanism is left to the caller: audio.play_wav drives it from a sounddevice
stream (stream.active / stream.abort / sd.sleep); an aplay or streamed path can
drive the same loop from its own handle. Wiring a live wake detector into should_stop,
and the self-trigger echo guard, are the deferred half of issue #16.
"""

from __future__ import annotations

from typing import Callable


def wait_or_abort(
    is_active: Callable[[], bool],
    should_stop: Callable[[], bool],
    stop: Callable[[], None],
    sleep: Callable[[int], None],
    poll_ms: int = 50,
) -> bool:
    """Wait for playback to finish, or abort it on a barge-in.

    Returns True if the clip played to completion, False if should_stop fired and
    the stream was aborted. If playback has already finished when called, returns
    True without ever consulting should_stop, so a reply that ended on its own is
    never reported as interrupted.
    """
    while is_active():
        if should_stop():
            stop()
            return False
        sleep(poll_ms)
    return True
