"""Confirm-before-act readback handshake (issue #42).

The voice loop is fire-and-forget: whatever whisper heard goes to the brain and the
brain acts. The mishear guard in pipeline.py catches a garbled *decode*, but a clean
decode of the wrong words looks exactly like a clean decode of the right ones. This
module is the layer that makes the loop say back what it understood and wait for a
yes before anything happens.

Hardware-free and stdlib-only, like brain_bridge: the framing, the classifier, and
the loop's termination rule are decisions, and decisions are testable without a
microphone. The audio half -- opening a second listening window with no wake word,
which needs capture_request and so needs numpy -- is not here. See "What is not here"
at the bottom of this docstring.

The classifier is the part worth reading carefully, because its two failure
directions are not symmetric. Reading a revision or a refusal as a confirmation
executes something the user did not approve, which is the failure this whole handshake
exists to prevent. Reading a confirmation as a revision costs one more spoken turn.
So confirmation is the narrow class: a reply confirms only when it is *nothing but* a
confirmation, and anything carrying additional substance is a revision even when it
opens with "yes".

What is not here, deliberately:

- The second listening window. capture_request already endpoints on silence, so the
  window is a call with no wake-word wait in front of it, not new capture logic.
- Config keys (readback_confirm, readback_window_ms). They belong with the wiring
  that reads them; adding them ahead of it would ship defaults nothing consults.
- Correlating the readback reply with the request that produced it. That is #19, the
  bridge correlation key, and #59, its producer side. A real dependency for the live
  loop, not for these decisions.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# Stage 1: framing
# --------------------------------------------------------------------------- #
# The brain is a general assistant that acts on what it is told, so "do not act yet"
# has to be the frame around the request rather than a suffix after it -- a long
# request with the restraint tacked on the end reads as an instruction followed by an
# afterthought. The request is delimited so its own imperative voice cannot be read as
# addressed to the assistant, which is the same reason a prompt-injection boundary is
# drawn: "delete the draft" inside the block is a thing to restate, not a thing to do.
READBACK_FRAMING = (
    "Do not act on the following request yet. Restate it as a single short "
    "confirmation question in plain spoken words, so the speaker can confirm you "
    "understood before anything happens. Ask only about what the request says; do "
    "not add steps, do not answer it, and do not do it.\n\n"
    "Request:\n<<<\n{request}\n>>>"
)


def frame_for_readback(request: str) -> str:
    """Wrap a transcribed request in the restate-do-not-act framing."""
    return READBACK_FRAMING.format(request=request.strip())


# --------------------------------------------------------------------------- #
# Stage 2: classifying the answer
# --------------------------------------------------------------------------- #
CONFIRM = "confirm"
REVISE = "revise"
CANCEL = "cancel"

# Refusal words, and the same all-or-nothing rule the confirm set gets: a reply is a
# refusal only when it is nothing but one. "no" and "never mind" end the turn; "don't
# send it to Bob, send it to Alice" carries a correction and goes back around instead.
#
# Cancelling on any negation looks like the cautious reading and is not, because the
# two outcomes are not both risky. A cancel drops the request, so the speaker has to
# wake the loop and say the whole thing again; a revision re-reads it and executes
# nothing either way. There is no safety bought by the stricter rule, only a lost
# correction.
#
# "wait" is deliberately absent from the set. It leads a correction ("wait, make it
# three") far more often than a refusal.
CANCEL_WORDS = frozenset(
    {
        "no",
        "nope",
        "nah",
        "negative",
        "cancel",
        "cancelled",
        "canceled",
        "stop",
        "abort",
        "quit",
        "dont",
        "never",
        "nevermind",
        "forget",
        "scratch",
    }
)

# Bare acknowledgment: words that approve on their own, said alone, by someone who
# cannot see a screen. That last part is the bar, and it rules out the pieces of the
# phrases below. "do", "it", and "that" are not approval; they are fragments of "do
# it" and "that's right", and a clipped transcript that ends up as the bare word "it"
# would have been read as a go. The phrases are folded to one token before this set is
# consulted, so the phrase confirms and the fragment does not.
CONFIRM_WORDS = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "yup",
        "yea",
        "aye",
        "correct",
        "right",
        "exactly",
        "affirmative",
        "alright",
        "confirm",
        "confirmed",
        "approved",
        "approve",
        "proceed",
        "continue",
        "go",
        "sure",
        "ok",
        "okay",
    }
)

# Words that carry no decision either way, so they neither confirm nor block a
# confirmation. Politeness and hedges mostly: "yes please", "um, yeah", "just do it".
FILLER_WORDS = frozenset(
    {
        "um",
        "uh",
        "er",
        "erm",
        "hmm",
        "mhm",
        "please",
        "thanks",
        "well",
        "so",
        "and",
        "then",
        "just",
        "now",
        "i",
        "mean",
        "guess",
        "think",
        "would",
        "say",
        "is",
        "was",
        "be",
        "the",
        "a",
    }
)

# Speech-to-text writes contractions with an apostrophe and sometimes a unicode one.
# Folding them out means "don't" and "dont" reach the sets as one token rather than
# two spellings that have to be listed twice and stay in sync.
_APOSTROPHES = "'’ʼ"
_WORD_SPLIT = re.compile(r"[^a-z0-9]+")

# Decisions that are phrases, folded to one token before the word sets are consulted,
# for two different reasons.
#
# "never mind", "do it", and "that's right" fold because their halves cannot be listed
# on their own: "mind" would cancel "mind the order", and "it" would approve a clipped
# transcript that came back as the bare word "it". Folding first lets the phrase decide
# while the fragment falls through to a revision, the safe direction for a fragment.
#
# "go ahead", "call it off", and "thank you" fold to absorb their trailing word, which
# is otherwise unlisted and would leave the reply looking unfinished. Their first word
# already decides on its own.
#
# The negated form comes first and the order is load-bearing, because these apply in
# sequence: without it "don't do it" reaches the rule below, folds to "dont confirm",
# and lands in revise, so the plainest spoken refusal there is fails to cancel.
#
# "cancel that" and "stop it" fold for a third reason, and it is the whitelist tail
# check that creates it. Their object pronoun is not a decision word, so the reply ends
# on one and never reaches the refusal branch, which turns an unambiguous cancellation
# into a re-prompt. Absorbing the object is narrower than letting trailing pronouns
# through generally: "yes that" has to stay a revision, since it is "yes, that... one
# instead" cut off at the pause.
_PHRASES = (
    (r"\bnever\s+mind\b", "nevermind"),
    (r"\bcall\s+it\s+off\b", "cancel"),
    (r"\b(?:dont|do\s+not)\s+do\s+(?:it|that)\b", "no"),
    (r"\bdo\s+it\b", "confirm"),
    (r"\b(cancel|stop|forget|scratch)\s+(?:it|that)\b", r"\1"),
    (r"\bthats\s+(right|correct|it)\b", "confirm"),
    (r"\ball\s+right\b", "alright"),
    (r"\bgo\s+ahead\b", "go"),
    (r"\bthank\s+you\b", "thanks"),
)

# Filler that can end a finished sentence, so it is stripped from the tail before the
# check below looks at what the reply really ended on. "go ahead then" and "never mind
# then" are finished; "okay and then" is the same truncation as "okay and" wearing one.
#
# Every word here is an exception to that check, and exceptions run in the unsafe
# direction: each one lets a reply reach confirm on a word that is not itself a
# decision. So the bar for adding one is that it ends a spoken turn more often than it
# opens a hedge. "well" was here and does the opposite ("yeah, well..." is someone
# winding up to disagree), which confirmed the request they were about to change.
_TERMINAL_FILLER = frozenset({"then", "now", "please", "thanks"})

# What a finished decision is allowed to end on. Capture endpoints on silence
# (endpoint_silence_ms), so "yes, and..." said with a pause to think arrives as exactly
# the two words "yes and", and approving that runs the request the speaker was still
# amending. A reply that does not end on an actual decision word is treated as one of
# those truncations.
#
# This is a whitelist on purpose, and it replaced a list of words that cannot end a
# sentence. That list had to be complete to be safe, and it was not: "and", "but" and
# "just" were on it while "the", "a" and "i" were not, so "yes, the..." clipped before
# "first one" read as approval and executed. A word nobody thought of now falls to
# revise, which costs a turn, instead of to execute, which costs the wrong action.
#
# Position is the whole signal, so these words stay ordinary filler elsewhere: "just do
# it" is a complete answer and "yeah just" is not, from the same word. The trade is
# real and worth naming: a hesitant tail like "yes um" or "yeah i guess" now goes back
# around for one more turn rather than confirming.
_CAN_END_A_DECISION = CANCEL_WORDS | CONFIRM_WORDS


def _tokens(text: str) -> list[str]:
    """Lowercase words, apostrophes and multi-word decisions folded, punctuation dropped."""
    if not text:
        return []
    lowered = text.lower()
    for mark in _APOSTROPHES:
        lowered = lowered.replace(mark, "")
    for pattern, replacement in _PHRASES:
        lowered = re.sub(pattern, replacement, lowered)
    return [t for t in _WORD_SPLIT.split(lowered) if t]


def classify_confirmation(text: str | None) -> str:
    """Read a spoken answer to a readback as CONFIRM, REVISE, or CANCEL.

    Empty input is a cancel, not a revision. Nothing heard means the window closed on
    silence or the transcript was noise, and neither is permission to act. The floor in
    handshake_step would stop a REVISE reading from looping forever, so this is about
    what to say rather than about termination: an empty room gets "cancelled" now
    instead of two more readbacks first.

    Confirmations and refusals are both all-or-nothing, for different reasons. A reply
    confirms only when every word in it is an acknowledgment or filler, because "yes
    but make it three" means the speaker has more to say and approving it would run
    the request they were correcting. A reply refuses only when every word is a
    refusal or filler, because "don't send it to Bob, send it to Alice" is a
    correction wearing a negation and dropping it costs the speaker the whole turn.

    A reply carrying both ("no, that's right") lands in REVISE. It is genuinely
    ambiguous, and reading it back is how an ambiguous answer gets resolved.
    """
    tokens = _tokens(text or "")
    if not tokens:
        return CANCEL
    tail = list(tokens)
    while tail and tail[-1] in _TERMINAL_FILLER:
        tail.pop()
    if tail and tail[-1] not in _CAN_END_A_DECISION:
        return REVISE
    if any(t in CANCEL_WORDS for t in tokens):
        if all(t in CANCEL_WORDS or t in FILLER_WORDS for t in tokens):
            return CANCEL
        return REVISE
    if all(t in CONFIRM_WORDS or t in FILLER_WORDS for t in tokens):
        # An answer made only of filler ("um, please") acknowledged nothing. It is not
        # a refusal either, so send it back around rather than acting on it.
        if any(t in CONFIRM_WORDS for t in tokens):
            return CONFIRM
        return REVISE
    return REVISE


# --------------------------------------------------------------------------- #
# Stage 3: the loop, and where it stops
# --------------------------------------------------------------------------- #
# Spoken lines. Short, because they are heard rather than read, and every one of them
# lands between the speaker and the thing they asked for.
CANCELLED_REPLY = "Okay, cancelled."
GAVE_UP_REPLY = "Let's start over. Say the wake word when you're ready."

# A revision loop with no floor runs as long as the readback keeps missing, and a
# readback that keeps missing is usually a microphone problem rather than a wording
# problem, so more turns do not converge. Two corrections is the point where handing
# the turn back beats trying again.
MAX_REVISIONS = 2

EXECUTE = "execute"
REPROMPT = "reprompt"
ABANDON = "abandon"


def handshake_step(
    answer: str | None,
    revisions: int = 0,
    max_revisions: int = MAX_REVISIONS,
) -> tuple[str, str | None]:
    """Decide what the loop does with one answer to a readback.

    `revisions` is how many corrections this turn has already been through, so the
    caller counts and this stays a pure function of (answer, count).

    Returns (action, spoken):

    - (EXECUTE, None)          approved; send the request on and speak the result.
    - (REPROMPT, None)         a correction; re-frame and read back again.
    - (ABANDON, CANCELLED_REPLY)  refused, or nothing heard.
    - (ABANDON, GAVE_UP_REPLY)    corrected too many times without landing.

    ABANDON covers both endings because the loop treats them the same way: speak the
    line, drop the pending request, return to scanning for the wake word. They differ
    only in what is said, which is why the line comes back rather than a reason code
    the caller would have to map to one.
    """
    decision = classify_confirmation(answer)
    if decision == CONFIRM:
        return EXECUTE, None
    if decision == CANCEL:
        return ABANDON, CANCELLED_REPLY
    if revisions >= max_revisions:
        return ABANDON, GAVE_UP_REPLY
    return REPROMPT, None
