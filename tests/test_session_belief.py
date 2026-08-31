"""Unit tests for starter/session_belief.py's Bayesian belief update.

The team's own validation for the latest rewrite of this module was an A/B
score match on the 200-sample public set ("bit-identical to 6 decimals").
That's a good end-to-end check but doesn't independently verify the odds-
space math is correct in isolation -- these tests do that directly.
"""
from __future__ import annotations

import pytest

from starter.config import ANSWERABILITY_PRIOR, BUCKET_ANSWER_LR, EXHAUSTED_THRESHOLD
from starter.session_belief import SessionBelief, _bayes_update


# ---------------------------------------------------------------------------
# _bayes_update
# ---------------------------------------------------------------------------

def test_bayes_update_matches_hand_computed_odds():
    # odds(0.5) = 1.0; ratio 2.0 -> new odds 2.0 -> prob = 2/3
    assert _bayes_update(0.5, 2.0) == pytest.approx(2.0 / 3.0)


def test_bayes_update_ratio_greater_than_one_increases_probability():
    result = _bayes_update(0.3, 3.0)
    assert result > 0.3


def test_bayes_update_ratio_less_than_one_decreases_probability():
    result = _bayes_update(0.3, 0.2)
    assert result < 0.3


def test_bayes_update_reciprocal_ratio_is_not_a_no_op():
    # Applying a ratio and then its reciprocal should not return exactly to
    # the start (odds-space composition is nonlinear in probability space),
    # but it must move back toward it.
    up = _bayes_update(0.4, 2.0)
    back = _bayes_update(up, 0.5)
    assert back == pytest.approx(0.4)


def test_bayes_update_no_op_when_prob_is_zero():
    assert _bayes_update(0.0, 5.0) == 0.0


def test_bayes_update_no_op_when_ratio_is_one():
    assert _bayes_update(0.42, 1.0) == 0.42


def test_bayes_update_prob_one_stays_one():
    # odds(1.0) is infinite; any positive ratio must leave it saturated.
    assert _bayes_update(1.0, 0.1) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SessionBelief.observe
# ---------------------------------------------------------------------------

def test_observe_answered_zeroes_the_asked_attribute():
    belief = SessionBelief()
    belief.observe("material", was_answered=True)
    assert belief.answerable["material"] == 0.0


def test_observe_non_answer_zeroes_the_asked_attribute_too():
    # A non-answer means *this* attribute is exhausted for the session,
    # regardless of what it implies about the others.
    belief = SessionBelief()
    belief.observe("material", was_answered=False)
    assert belief.answerable["material"] == 0.0


def test_observe_none_attribute_is_a_no_op():
    belief = SessionBelief()
    before = dict(belief.answerable)
    belief.observe(None, was_answered=True)
    assert belief.answerable == before
    assert belief.asks == 0


def test_observe_increments_asks_and_non_answers():
    belief = SessionBelief()
    belief.observe("material", was_answered=True)
    belief.observe("color", was_answered=False)
    assert belief.asks == 2
    assert belief.non_answers == 1


def test_observe_updates_other_attributes_by_measured_ratio():
    # material -> color ratio is documented as < 1: answering material
    # should *lower* color's odds, not raise them.
    belief = SessionBelief()
    before_color = belief.answerable["color"]
    belief.observe("material", was_answered=True)
    assert belief.answerable["color"] < before_color


def test_observe_non_answer_applies_reciprocal_direction():
    # A material non-answer should raise color's odds (reciprocal of the
    # answered-branch ratio, since material's LR for color is < 1).
    belief = SessionBelief()
    before_color = belief.answerable["color"]
    belief.observe("material", was_answered=False)
    assert belief.answerable["color"] > before_color


def test_observe_pair_absent_from_table_gets_no_update():
    # "budget" never appears as a row key in BUCKET_ANSWER_LR, so observing
    # it must not perturb any other attribute's belief.
    assert "budget" not in BUCKET_ANSWER_LR
    belief = SessionBelief()
    before = {k: v for k, v in belief.answerable.items() if k != "budget"}
    belief.observe("budget", was_answered=True)
    after = {k: v for k, v in belief.answerable.items() if k != "budget"}
    assert after == before


def test_observe_does_not_update_an_already_zeroed_attribute():
    # Once an attribute has been asked (and its belief zeroed), a later
    # observation of a correlated attribute must not resurrect it -- the
    # `> 0.0` guard in observe() exists for exactly this.
    belief = SessionBelief()
    belief.observe("material", was_answered=True)  # zeroes material
    assert belief.answerable["material"] == 0.0
    belief.observe("color", was_answered=True)  # color correlates with style/size/use_case, not material
    assert belief.answerable["material"] == 0.0


def test_observe_multiple_updates_compose_multiplicatively_in_odds_space():
    belief = SessionBelief()
    single = SessionBelief()
    single.answerable["style"] = ANSWERABILITY_PRIOR["style"]
    single_after = _bayes_update(
        _bayes_update(ANSWERABILITY_PRIOR["style"], BUCKET_ANSWER_LR["color"]["style"]),
        BUCKET_ANSWER_LR["material"]["style"],
    )
    belief.observe("color", was_answered=True)
    belief.observe("material", was_answered=True)
    assert belief.answerable["style"] == pytest.approx(single_after)


# ---------------------------------------------------------------------------
# SessionBelief.rank / best / exhausted
# ---------------------------------------------------------------------------

def test_rank_orders_most_answerable_first():
    belief = SessionBelief()
    ranked = belief.rank(excluded=set())
    assert ranked[0] == max(belief.answerable, key=belief.answerable.get)


def test_rank_excludes_given_attributes():
    belief = SessionBelief()
    ranked = belief.rank(excluded={"feature"})
    assert "feature" not in ranked


def test_best_returns_none_when_nothing_clears_threshold():
    belief = SessionBelief()
    belief.answerable = dict.fromkeys(belief.answerable, 0.0)
    assert belief.best(excluded=set()) is None


def test_best_skips_attributes_below_exhausted_threshold():
    belief = SessionBelief()
    belief.answerable = {"material": EXHAUSTED_THRESHOLD / 2, "color": 0.5}
    assert belief.best(excluded=set()) == "color"


def test_exhausted_true_when_all_below_threshold():
    belief = SessionBelief()
    belief.answerable = dict.fromkeys(belief.answerable, 0.0)
    assert belief.exhausted is True


def test_exhausted_false_when_one_attribute_remains():
    belief = SessionBelief()
    assert belief.exhausted is False  # fresh prior has attributes above threshold


def test_fresh_belief_starts_at_population_marginals():
    belief = SessionBelief()
    assert belief.answerable == ANSWERABILITY_PRIOR
    assert belief.answerable is not ANSWERABILITY_PRIOR  # must be a copy, not aliased
