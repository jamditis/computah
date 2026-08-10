"""Tests for the confirm-before-act readback handshake (issue #42).

Stdlib-only, like test_brain_bridge: conversation.py imports nothing heavier, so
these run on any host without the model stack.

The cases are grouped by what they protect. The ones that matter most are under
"a confirmation has to be nothing but a confirmation" -- every one of those is a
sentence that opens like approval and is not, and reading any of them as approval
executes something the speaker was in the middle of changing.
"""

import unittest

import conversation as c


class TestFraming(unittest.TestCase):
    def test_the_request_is_delimited(self):
        # The request keeps its own imperative voice ("delete the draft"), so it has
        # to be fenced off from the instruction or it reads as a second instruction.
        framed = c.frame_for_readback("delete the draft")
        self.assertIn("<<<\ndelete the draft\n>>>", framed)

    def test_it_says_not_to_act(self):
        framed = c.frame_for_readback("send the email")
        self.assertIn("Do not act", framed)
        self.assertIn("do not do it", framed)

    def test_surrounding_whitespace_is_dropped(self):
        framed = c.frame_for_readback("  turn on the lights \n")
        self.assertIn("<<<\nturn on the lights\n>>>", framed)


class TestConfirm(unittest.TestCase):
    def test_bare_yes(self):
        for answer in ["yes", "yeah", "yep", "yup", "correct", "right", "okay", "sure"]:
            with self.subTest(answer=answer):
                self.assertEqual(c.classify_confirmation(answer), c.CONFIRM)

    def test_punctuation_and_case(self):
        self.assertEqual(c.classify_confirmation("Yes!"), c.CONFIRM)
        self.assertEqual(c.classify_confirmation("YES."), c.CONFIRM)

    def test_confirmation_with_only_filler_around_it(self):
        for answer in [
            "yes please",
            "um, yeah",
            "just do it",
            "yeah go ahead",
            "thats right",
            "that's right",
            "go ahead then",
            "okay sure",
        ]:
            with self.subTest(answer=answer):
                self.assertEqual(c.classify_confirmation(answer), c.CONFIRM)


class TestConfirmationMustBeOnlyConfirmation(unittest.TestCase):
    """The asymmetry the module is built around.

    Each of these opens with an acknowledgment and then says something else. Reading
    any of them as approval runs the un-corrected request.
    """

    def test_yes_followed_by_a_correction(self):
        for answer in [
            "yes but make it three",
            "yeah, at four instead",
            "right, the second one",
            "correct, and add milk",
            "okay but tomorrow",
        ]:
            with self.subTest(answer=answer):
                self.assertEqual(c.classify_confirmation(answer), c.REVISE)

    def test_a_confirmation_word_inside_a_different_instruction(self):
        # "go" confirms alone and "do it" confirms as a phrase; both are also ordinary
        # speech. The unmatched words are what keep these out of the confirm class.
        for answer in ["go to the store", "do it again with three", "that one instead"]:
            with self.subTest(answer=answer):
                self.assertEqual(c.classify_confirmation(answer), c.REVISE)

    def test_a_fragment_of_a_confirming_phrase_is_not_a_confirmation(self):
        # "do it" and "that's right" confirm; "do", "it", "that", and "ahead" are what
        # a clipped transcript leaves behind, and none of them is a word someone who
        # cannot see a screen says to approve an action.
        for answer in ["it", "that", "do", "ahead", "this"]:
            with self.subTest(answer=answer):
                self.assertEqual(c.classify_confirmation(answer), c.REVISE)

    def test_a_reply_that_acknowledges_nothing(self):
        # Filler only. Not a refusal, so not a cancel, but it approved nothing.
        for answer in ["um", "please", "uh well", "i mean"]:
            with self.subTest(answer=answer):
                self.assertEqual(c.classify_confirmation(answer), c.REVISE)


class TestClippedByTheSilenceEndpoint(unittest.TestCase):
    """The failure the capture path makes ordinary.

    capture_request endpoints on silence, so a speaker who says "yes, and..." and
    pauses to think hands the classifier the two words "yes and". Every reply here is
    a real sentence truncated at a real pause, and reading any of them as approval
    runs the request the speaker was still amending.
    """

    def test_a_reply_ending_on_a_conjunction(self):
        for answer in [
            "yes and",
            "okay and then",
            "yeah so",
            "right so",
            "sure but",
            "correct or",
            "yes also",
        ]:
            with self.subTest(answer=answer):
                self.assertEqual(c.classify_confirmation(answer), c.REVISE)

    def test_a_reply_ending_on_a_hedge(self):
        for answer in ["yeah just", "okay with", "yes to"]:
            with self.subTest(answer=answer):
                self.assertEqual(c.classify_confirmation(answer), c.REVISE)

    def test_the_check_is_final_position_only(self):
        # Same words, complete sentences. Only a reply that ENDS on one is unfinished.
        self.assertEqual(c.classify_confirmation("just do it"), c.CONFIRM)
        self.assertEqual(c.classify_confirmation("and yes"), c.CONFIRM)

    def test_filler_that_does_end_a_sentence_is_stripped_first(self):
        # "then" finishes a reply, so it must not hide the truncation under it.
        self.assertEqual(c.classify_confirmation("go ahead then"), c.CONFIRM)
        self.assertEqual(c.classify_confirmation("never mind then"), c.CANCEL)
        self.assertEqual(c.classify_confirmation("okay and then"), c.REVISE)
        self.assertEqual(c.classify_confirmation("yes and please"), c.REVISE)

    def test_a_question_back_is_not_approval(self):
        # "sure" confirms; "you sure" is the speaker asking, not approving.
        self.assertEqual(c.classify_confirmation("you sure"), c.REVISE)
        self.assertEqual(c.classify_confirmation("thank you"), c.REVISE)
        self.assertEqual(c.classify_confirmation("yes thank you"), c.CONFIRM)


class TestCancel(unittest.TestCase):
    def test_bare_refusals(self):
        for answer in ["no", "nope", "nah", "stop", "cancel", "nevermind"]:
            with self.subTest(answer=answer):
                self.assertEqual(c.classify_confirmation(answer), c.CANCEL)

    def test_contractions_fold_to_one_spelling(self):
        for answer in ["don't", "dont", "don’t"]:
            with self.subTest(answer=answer):
                self.assertEqual(c.classify_confirmation(answer), c.CANCEL)

    def test_never_mind_as_two_words(self):
        self.assertEqual(c.classify_confirmation("never mind"), c.CANCEL)

    def test_a_refusal_with_filler_still_ends_the_turn(self):
        for answer in ["no thanks", "um, no", "no please stop", "never mind then"]:
            with self.subTest(answer=answer):
                self.assertEqual(c.classify_confirmation(answer), c.CANCEL)

    def test_the_word_sets_stay_disjoint(self):
        # Both branches are all-or-nothing over CANCEL/CONFIRM plus FILLER, so a word
        # in two sets would change which branch claims a reply, silently and in
        # whichever direction the code happens to read. Fail here, where the cause is
        # visible, rather than in whichever classification test drifts.
        self.assertEqual(c.CANCEL_WORDS & c.CONFIRM_WORDS, frozenset())
        self.assertEqual(c.CANCEL_WORDS & c.FILLER_WORDS, frozenset())
        self.assertEqual(c.CONFIRM_WORDS & c.FILLER_WORDS, frozenset())


class TestNegationCarryingACorrection(unittest.TestCase):
    """A refusal has to be nothing but a refusal, for the mirror-image reason.

    Cancelling on any negation reads as the cautious choice and is not. Neither
    outcome executes anything, so the strict rule buys no safety; it just drops the
    correction and makes the speaker say the whole request again.
    """

    def test_a_negation_with_a_replacement(self):
        for answer in [
            "don't send it to Bob, send it to Alice",
            "no, make it tomorrow",
            "never mind the email, call her instead",
            "not the second one, the first one",
        ]:
            with self.subTest(answer=answer):
                self.assertEqual(c.classify_confirmation(answer), c.REVISE)

    def test_a_reply_carrying_both_a_yes_and_a_no(self):
        # Ambiguous. Reading it back is how an ambiguous answer gets resolved, and it
        # runs nothing in the meantime.
        self.assertEqual(c.classify_confirmation("no, that's right"), c.REVISE)
        self.assertEqual(c.classify_confirmation("yes, no, stop"), c.REVISE)

    def test_nothing_heard(self):
        for answer in ["", "   ", None, "...", "!!"]:
            with self.subTest(answer=answer):
                self.assertEqual(c.classify_confirmation(answer), c.CANCEL)

    def test_wait_leads_a_correction_not_a_refusal(self):
        # The one word deliberately kept out of CANCEL_WORDS. Asserted bare, because
        # "wait, make it three" is a revision under either reading -- the trailing
        # words fail the all-or-nothing cancel check on their own, so it would pass
        # with "wait" in the set and pin nothing.
        self.assertEqual(c.classify_confirmation("wait"), c.REVISE)
        self.assertEqual(c.classify_confirmation("wait, make it three"), c.REVISE)


class TestHandshakeStep(unittest.TestCase):
    def test_approval_executes(self):
        self.assertEqual(c.handshake_step("yes"), (c.EXECUTE, None))

    def test_refusal_abandons_and_says_so(self):
        self.assertEqual(c.handshake_step("no"), (c.ABANDON, c.CANCELLED_REPLY))

    def test_silence_abandons(self):
        self.assertEqual(c.handshake_step(None), (c.ABANDON, c.CANCELLED_REPLY))

    def test_a_correction_goes_around_again(self):
        self.assertEqual(
            c.handshake_step("make it three", revisions=0), (c.REPROMPT, None)
        )

    def test_the_revision_loop_has_a_floor(self):
        # Two corrections re-prompt, the third ends the turn rather than looping. A
        # readback that has missed twice is usually a microphone problem, and more
        # turns do not converge on one.
        self.assertEqual(
            c.handshake_step("make it three", revisions=0), (c.REPROMPT, None)
        )
        self.assertEqual(
            c.handshake_step("make it three", revisions=1), (c.REPROMPT, None)
        )
        self.assertEqual(
            c.handshake_step("make it three", revisions=2), (c.ABANDON, c.GAVE_UP_REPLY)
        )

    def test_the_floor_is_configurable(self):
        self.assertEqual(
            c.handshake_step("make it three", revisions=0, max_revisions=1),
            (c.REPROMPT, None),
        )
        self.assertEqual(
            c.handshake_step("make it three", revisions=1, max_revisions=1),
            (c.ABANDON, c.GAVE_UP_REPLY),
        )

    def test_the_floor_never_blocks_an_approval_or_a_refusal(self):
        # Running out of corrections must not swallow a decision the speaker did make.
        self.assertEqual(c.handshake_step("yes", revisions=99), (c.EXECUTE, None))
        self.assertEqual(
            c.handshake_step("no", revisions=99), (c.ABANDON, c.CANCELLED_REPLY)
        )


if __name__ == "__main__":
    unittest.main()
