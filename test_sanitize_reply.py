#!/usr/bin/env python3
"""sanitize_reply() makes a brain reply safe to speak.

A reply is model output handed straight to Piper, so it must be bounded,
control-free, and plain before TTS, and an empty reply must become a short
spoken sentence rather than silence. This proves those guarantees directly on
sanitize_reply(), and proves that brain() routes both backends through it so no
raw reply reaches speak() (issue #24).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pipeline

results: list[tuple[bool, str]] = []


def check(ok: bool, detail: str) -> None:
    results.append((ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {detail}")


def test_empty_and_whitespace() -> None:
    """An empty or whitespace-only reply yields the spoken fallback, not silence."""
    for reply in ("", "   ", "\n\t  \n", None):
        check(pipeline.sanitize_reply(reply) == pipeline.EMPTY_REPLY_FALLBACK,
              f"empty reply {reply!r} -> spoken fallback")
    # A reply that is only a fenced code block has nothing speakable, so it too
    # becomes the fallback rather than reaching TTS as raw code or as silence.
    check(pipeline.sanitize_reply("```\nprint('do not speak me')\n```")
          == pipeline.EMPTY_REPLY_FALLBACK,
          "code-block-only reply -> spoken fallback")


def test_oversized_is_capped() -> None:
    """A very long reply is capped and does not end mid-word."""
    long_reply = ("This is a sentence. " * 200).strip()  # ~3800 chars
    out = pipeline.sanitize_reply(long_reply)
    check(len(out) <= pipeline.MAX_SPOKEN_CHARS,
          f"oversized reply capped to <= {pipeline.MAX_SPOKEN_CHARS} (got {len(out)})")
    check(not out.endswith("senten") and " " not in out[-1:],
          "cap lands on a clean boundary, not mid-word")
    # A custom lower cap is honored.
    check(len(pipeline.sanitize_reply("word " * 100, max_chars=40)) <= 40,
          "custom max_chars is honored")


def test_truncation_uses_the_last_sentence_end() -> None:
    """Mixed sentence punctuation cuts at the latest end, not the first kind found.

    A reply whose last period comes well before its last exclamation would, if the
    delimiters were tried in order rather than compared, cut at that early period
    and silently drop the speakable text in between.
    """
    # The period clears the halfway mark (so trying ". " first would return there),
    # the exclamation sits later and still inside the cap, and the whole reply runs
    # past the cap so truncation actually fires.
    head = "word " * 24                                   # 120 chars, the halfway mark
    text = head + "period. " + ("beta " * 12) + "gamma! " + ("tail " * 20)
    out = pipeline.sanitize_reply(text, max_chars=240)
    check(out.endswith("gamma!"),
          f"cut lands on the last sentence end, not the first kind: {out[-24:]!r}")
    check(len(out) <= 240, f"still within the cap (got {len(out)})")
    # Every sentence-end kind is eligible, not just the period.
    q = pipeline.sanitize_reply(
        head + "period. " + ("beta " * 12) + "gamma? " + ("tail " * 20), max_chars=240)
    check(q.endswith("gamma?"), f"a question mark ends a cut too: {q[-24:]!r}")


def test_control_chars_stripped() -> None:
    """Control characters are removed before TTS."""
    dirty = "Hello\x00 there\x07, friend\x1b[0m."
    out = pipeline.sanitize_reply(dirty)
    check(all(ord(c) >= 0x20 or c == " " for c in out),
          f"control characters stripped: {out!r}")
    check("Hello there" in out.replace("  ", " "),
          "readable text survives the control strip")


def test_ansi_escapes_stripped() -> None:
    """A whole ANSI sequence goes, not just its ESC byte.

    _brain_cli returns the claude CLI's stdout, so a colourised reply is
    realistic. Dropping only the ESC byte would leave "[31m" to be read aloud.
    """
    out = pipeline.sanitize_reply("\x1b[31mThe build failed.\x1b[0m")
    check(out == "The build failed.",
          f"CSI colour codes leave no payload behind: {out!r}")
    check("[31m" not in out and "[0m" not in out,
          "no escape payload survives as spoken text")
    # Cursor moves and a window-title (OSC) sequence go the same way.
    moves = pipeline.sanitize_reply("\x1b[2J\x1b[HDone.\x1b]0;title\x07")
    check(moves == "Done.", f"CSI cursor moves and OSC stripped: {moves!r}")
    # A reply that is nothing but escapes has nothing speakable left.
    check(pipeline.sanitize_reply("\x1b[31m\x1b[0m") == pipeline.EMPTY_REPLY_FALLBACK,
          "escape-only reply -> spoken fallback")


def test_markdown_stripped() -> None:
    """The markdown the VOICE_SYSTEM_PROMPT forbids is reduced to plain text."""
    md = (
        "# Heading\n"
        "Here is **bold** and *italic* and `code` and ~~gone~~.\n"
        "- first bullet\n"
        "- second bullet\n"
        "See [the docs](https://example.com/x) for more.\n"
        "```\nprint('do not speak me')\n```"
    )
    out = pipeline.sanitize_reply(md)
    for marker in ("#", "*", "`", "~", "]", "(http"):
        check(marker not in out, f"markdown marker {marker!r} stripped")
    check("bold" in out and "italic" in out and "the docs" in out,
          "link and emphasis text is kept")
    check("print('do not speak me')" not in out,
          "fenced code block content is dropped")
    check("\n" not in out, "newlines collapsed to spaces")


def test_markdown_edge_cases() -> None:
    """Emphasis stripping keeps identifiers, and a broken fence never gets spoken."""
    # Underscore emphasis is removed, but snake_case identifiers, filenames, and env
    # vars in a plain reply survive so they are spoken intact, not run together.
    ident = pipeline.sanitize_reply("Set BOT_SPREN_STATE_DIR in config_local.json.")
    check("BOT_SPREN_STATE_DIR" in ident and "config_local.json" in ident,
          f"snake_case identifiers keep their underscores: {ident!r}")
    emph = pipeline.sanitize_reply("This is _important_ and __very__ so.")
    check("_" not in emph and "important" in emph and "very" in emph,
          f"paired underscore emphasis is stripped: {emph!r}")
    # A literal asterisk that is not emphasis (a glob or multiplication) is spoken
    # as written, not deleted along with the emphasis markers.
    glob = pipeline.sanitize_reply("List the *.py files, then compute 5 * 6.")
    check("*.py" in glob and "5 * 6" in glob,
          f"literal asterisks (glob, math) are preserved: {glob!r}")
    star = pipeline.sanitize_reply("Use **bold** and *italic* sparingly.")
    check("*" not in star and "bold" in star and "italic" in star,
          f"paired asterisk emphasis is stripped: {star!r}")

    # An unterminated fenced block (truncated reply, opening fence, no close) must be
    # dropped entirely, not just have its backticks peeled off, or code gets spoken.
    broken = pipeline.sanitize_reply("Run this:\n```\nrm -rf / --no-preserve-root")
    check("rm -rf" not in broken and "no-preserve-root" not in broken,
          f"unterminated fenced code is dropped, not spoken: {broken!r}")
    check("Run this" in broken, "text before the broken fence survives")


def test_unpaired_emphasis_runs_are_literal() -> None:
    """An unpaired delimiter run is literal text, so it survives to be spoken.

    Markdown only reads a delimiter run as emphasis when a matching run closes it;
    CommonMark renders "*args", "**kwargs", "_private" and "VALUE_" verbatim. A reply
    about code is full of these, and dropping the delimiter renames the thing spoken.
    """
    for reply, kept in (("Pass *args to it.", "*args"),
                        ("Pass **kwargs to it.", "**kwargs"),
                        ("Set _private on the class.", "_private"),
                        ("Use VALUE_ as the prefix.", "VALUE_")):
        out = pipeline.sanitize_reply(reply)
        check(kept in out, f"unpaired run stays literal: {reply!r} -> {out!r}")


def test_code_span_content_is_literal() -> None:
    """A code span's content is literal markdown, so emphasis never applies inside it.

    "`__init__`" is the identifier, not bold "init". The ticks still come off, but only
    after emphasis, which is the order markdown itself resolves them in.
    """
    for reply, kept in (("The `__init__` method runs first.", "__init__"),
                        ("The `_private` field is internal.", "_private"),
                        ("Read `VALUE_` from the env.", "VALUE_")):
        out = pipeline.sanitize_reply(reply)
        check(kept in out, f"code span stays literal: {reply!r} -> {out!r}")
        check("`" not in out, f"backticks still stripped: {out!r}")


def test_paired_dunder_speaks_as_its_content() -> None:
    """A bare "__init__" speaks as "init", because that is what the grammar makes it.

    Nothing separates "__init__" from "__bold__": each is one paired "__" run hugging
    content, and CommonMark renders both as strong emphasis. So the delimiters cannot
    be kept around the identifier without also reading them aloud around real bold
    text, and speech loses nothing by dropping them -- an underscore is not
    pronounced. A reply that needs the identifier verbatim asks with a code span
    (test_code_span_content_is_literal), which is the only thing that distinguishes
    the two cases. Both are pinned here together because they are one decision:
    whatever keeps the underscores on "__init__" puts them on "__bold__" too.
    """
    for reply, spoken in (("__init__", "init"),
                          ("The __init__ method runs first.",
                           "The init method runs first."),
                          ("__bold__", "bold"),
                          ("__really important__", "really important")):
        out = pipeline.sanitize_reply(reply)
        check(out == spoken, f"paired dunder speaks as its content: {reply!r} -> {out!r}")


def test_nested_emphasis_fully_stripped() -> None:
    """Emphasis inside emphasis leaves no delimiter behind for TTS to read."""
    out = pipeline.sanitize_reply("This is **bold with *italic* inside** it.")
    check("*" not in out and "bold with italic inside" in out,
          f"nested emphasis fully stripped: {out!r}")


def test_ansi_csi_takes_all_parameter_bytes() -> None:
    """A CSI sequence ends at its final byte, whatever parameter bytes it carries.

    ECMA-48 allows the whole 0x30-0x3f range as parameter bytes, so a colon-separated
    true-colour SGR is valid. A pattern stopping at digits and semicolons fails to
    match it, and the later control-character pass takes only the ESC, leaving the
    parameters to be read aloud.
    """
    out = pipeline.sanitize_reply("\x1b[38:2::255:0:0mred\x1b[0m")
    check(out == "red", f"colon-separated true-colour SGR stripped whole: {out!r}")
    check("38:2" not in out, f"no parameter bytes survive as speech: {out!r}")


def test_tilde_fenced_code_is_dropped() -> None:
    """A tilde fence opens a code block exactly as a backtick fence does."""
    closed = pipeline.sanitize_reply('Before.\n~~~python\nprint("hi")\n~~~\nAfter.')
    check('print("hi")' not in closed, f"tilde-fenced code not spoken: {closed!r}")
    check("~" not in closed, f"no fence characters survive: {closed!r}")
    check("Before." in closed and "After." in closed,
          f"prose around the fence survives: {closed!r}")
    # An unterminated tilde fence (a truncated reply) drops to the end, as a backtick
    # fence does, rather than spilling the code body into speech.
    broken = pipeline.sanitize_reply("Run this:\n~~~\nrm -rf / --no-preserve-root")
    check("rm -rf" not in broken, f"unterminated tilde fence dropped: {broken!r}")
    check("Run this" in broken, "text before the broken tilde fence survives")


def test_inline_fence_mention_is_not_a_block() -> None:
    """A fence delimiter named mid-sentence opens nothing.

    Fences are block constructs: an opener sits at the start of its line. Matching one
    anywhere reads a reply that merely mentions ``` as opening a block, and the
    unterminated-fence rule then drops every word after it -- so an answer explaining how
    to open a code block loses the explanation. Deleting speech is worse than leaving
    markup in it, because nothing downstream can tell the reply was cut.
    """
    out = pipeline.sanitize_reply("To open a block, type ``` and then your code. That is the whole trick.")
    check("the whole trick" in out, f"prose after an inline fence mention survives: {out!r}")
    check("`" not in out, f"no fence characters survive: {out!r}")
    # Two mentions in one sentence must not pair into a block that eats the words between.
    pair = pipeline.sanitize_reply("Use ``` to start and ``` to end. Got it?")
    check("to start" in pair and "to end" in pair, f"text between two mentions survives: {pair!r}")
    # An indented mention is still a mention: only up to 3 spaces makes a real opener, and
    # this sits mid-line regardless.
    tilde = pipeline.sanitize_reply("A ~~~ marks a fence too. Remember that.")
    check("Remember that" in tilde, f"prose after an inline tilde mention survives: {tilde!r}")
    # The guarantee the anchor must not cost: a real block still goes.
    real = pipeline.sanitize_reply("Before.\n```python\nprint('hi')\n```\nAfter.")
    check("print" not in real, f"a real fenced block is still dropped: {real!r}")
    check("Before." in real and "After." in real, f"prose around a real block survives: {real!r}")


def test_keycap_emoji_fully_stripped() -> None:
    """A keycap sequence leaves its digit, not a combining enclosure.

    '1' + U+FE0F + U+20E3 is one glyph. Removing only the variation selector leaves the
    digit wearing U+20E3, so the emoji-free guarantee hands Piper a combining mark anyway.
    """
    out = pipeline.sanitize_reply("Step 1️⃣ then step 2️⃣.")
    check("⃣" not in out, f"no combining keycap survives: {out!r}")
    check("️" not in out, f"no variation selector survives: {out!r}")
    check("1" in out and "2" in out, f"the digits are still spoken: {out!r}")
    # A bare keycap mark, with the variation selector already absent, goes too.
    bare = pipeline.sanitize_reply("Step 3⃣.")
    check("⃣" not in bare, f"no bare keycap mark survives: {bare!r}")


def test_link_destination_with_parens_consumed_whole() -> None:
    """A destination containing parentheses is consumed to its real closing paren.

    Markdown allows balanced parens in a destination, so a Wikipedia-style
    '.../Foo_(bar)' stops a non-nesting pattern at the inner ')' and leaves the outer one
    behind to be spoken as punctuation after the label.
    """
    out = pipeline.sanitize_reply("See [Wikipedia](https://en.wikipedia.org/wiki/Foo_(bar)) for more.")
    check(out == "See Wikipedia for more.", f"paren-bearing link reduced to its label: {out!r}")
    check("wikipedia.org" not in out, f"no URL survives: {out!r}")
    plain = pipeline.sanitize_reply("See [Wikipedia](https://en.wikipedia.org/wiki/Foo) for more.")
    check(plain == "See Wikipedia for more.", f"a plain link still reduces to its label: {plain!r}")


def test_truncated_csi_is_dropped() -> None:
    """A CSI sequence cut off mid-parameters leaves nothing to speak.

    Output truncated inside an escape carries no final byte, so the CSI pattern misses it
    and the control pass takes only the ESC -- leaving '[31' to be read aloud. The OSC
    branch already tolerates an unterminated sequence; CSI is the gap.
    """
    for cut in ("\x1b[31", "\x1b[38:2::255:0:0", "\x1b["):
        out = pipeline.sanitize_reply(f"Done{cut}")
        check(out == "Done", f"truncated CSI {cut!r} left nothing behind: {out!r}")
    # A complete sequence is still taken whole, so the tolerance cost nothing.
    check(pipeline.sanitize_reply("\x1b[31mred\x1b[0m") == "red", "a terminated CSI still goes")


def test_emoji_stripped() -> None:
    """Emoji are stripped without touching accented letters."""
    out = pipeline.sanitize_reply("All done \U0001f44d café résumé")
    check("\U0001f44d" not in out, "emoji stripped")
    check("café" in out and "résumé" in out,
          "accented letters untouched")


def test_clean_text_unchanged() -> None:
    """Clean short text, including the spoken error strings, passes through as-is."""
    for reply in (
        "Two plus two is four.",
        "Sorry, I timed out thinking about that.",
        "Sorry, the brain is not available right now.",
    ):
        check(pipeline.sanitize_reply(reply) == reply,
              f"clean reply unchanged: {reply!r}")


def test_brain_routes_both_paths(d: Path) -> None:
    """brain() sanitizes whichever backend produced the reply, before speak()."""
    saved = (pipeline.CONFIG_PATH, pipeline.LOCAL_CONFIG_PATH,
             pipeline._brain_cli, pipeline._brain_bridge)
    try:
        pipeline.CONFIG_PATH = d / "config.json"
        pipeline.LOCAL_CONFIG_PATH = d / "config.local.json"
        dirty = "**Loud** and `long`:\n" + ("blah " * 400)

        pipeline._brain_cli = lambda text, cfg, **kw: dirty
        pipeline.CONFIG_PATH.write_text('{"brain_backend": "cli"}')
        cli_out = pipeline.brain("hi")
        check(len(cli_out) <= pipeline.MAX_SPOKEN_CHARS and "**" not in cli_out
              and "`" not in cli_out,
              "brain() sanitizes the cli path output")

        pipeline._brain_bridge = lambda text, cfg: dirty
        pipeline.CONFIG_PATH.write_text('{"brain_backend": "bridge"}')
        bridge_out = pipeline.brain("hi")
        check(len(bridge_out) <= pipeline.MAX_SPOKEN_CHARS and "**" not in bridge_out,
              "brain() sanitizes the bridge path output")

        # An empty backend reply still yields a spoken sentence through brain().
        pipeline._brain_cli = lambda text, cfg, **kw: ""
        pipeline.CONFIG_PATH.write_text('{"brain_backend": "cli"}')
        check(pipeline.brain("hi") == pipeline.EMPTY_REPLY_FALLBACK,
              "brain() turns an empty reply into the spoken fallback")
    finally:
        (pipeline.CONFIG_PATH, pipeline.LOCAL_CONFIG_PATH,
         pipeline._brain_cli, pipeline._brain_bridge) = saved


def main() -> int:
    test_empty_and_whitespace()
    test_oversized_is_capped()
    test_truncation_uses_the_last_sentence_end()
    test_control_chars_stripped()
    test_ansi_escapes_stripped()
    test_markdown_stripped()
    test_markdown_edge_cases()
    test_unpaired_emphasis_runs_are_literal()
    test_code_span_content_is_literal()
    test_paired_dunder_speaks_as_its_content()
    test_nested_emphasis_fully_stripped()
    test_ansi_csi_takes_all_parameter_bytes()
    test_tilde_fenced_code_is_dropped()
    test_inline_fence_mention_is_not_a_block()
    test_keycap_emoji_fully_stripped()
    test_link_destination_with_parens_consumed_whole()
    test_truncated_csi_is_dropped()
    test_emoji_stripped()
    test_clean_text_unchanged()
    with tempfile.TemporaryDirectory(prefix="sanitize-reply-") as tmp:
        test_brain_routes_both_paths(Path(tmp))

    n_pass = sum(1 for ok, _ in results if ok)
    print(f"=== {n_pass}/{len(results)} checks passed ===")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
