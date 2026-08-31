"""Short-term, within-session belief about what this customer can still answer.

The competition spec's Pillar III ("Self-Evolution: Dynamic Context
Programming") asks for runtime adaptation: distilling accumulated dialog
history into state that changes the agent's own strategy mid-session. The
*long-term* half of that is answered and closed -- `user_profile.py` persists
cross-session disclosures, and `scripts/eval_profile_signal.py` measures that
the provided profile carries no signal to act on (same-key sessions share a
coarse category 0.5% of the time against a 1.2% +- 0.5% random baseline), so
`SessionState.profile_hint` is deliberately inert. See
`.claude/skills/retrieval-experiments/SKILL.md` #5 for the full write-up.

This module is the short-term half, and it has an actual signal behind it.

Derived from the evaluator's own reply policy over all 200 public samples, a
session has a mean of **2.09** distinct answerable attribute buckets left after
turn 1 (22 sessions have one, 138 have two, 40 have three). Measured MTTC is
~5. So most sessions run out of answerable questions well before they run out
of turns, and every ask past that point returns a contentless reply that
`Agent._build_query` then feeds to BM25 as if it were customer content.

`ANSWERABILITY_PRIOR` starts every session at the population marginals and the
belief updates from what this particular customer actually does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from .config import (ANSWERABILITY_PRIOR, BUCKET_ANSWER_LR, EXHAUSTED_THRESHOLD)


def _bayes_update(prob: float, likelihood_ratio: float) -> float:
    """Multiply `prob`'s ODDS by `likelihood_ratio`, return the new probability.

    Odds, not probability, is what a likelihood ratio composes correctly
    with: this is what lets several observations across a session multiply
    together instead of the raw-probability approach it replaced, which had
    no principled way to combine more than one non-answer.
    """
    if prob <= 0.0 or likelihood_ratio == 1.0:
        return prob
    odds = prob / (1.0 - prob)
    new_odds = odds * likelihood_ratio
    return new_odds / (1.0 + new_odds)





@dataclass
class SessionBelief:
    """Running estimate of which attributes this customer can still answer.

    Initialized from the population marginals and updated only from observed
    outcomes. Deliberately carries no cross-session state: seeding this from
    `profile_hint` is the fourth variant of an idea whose first three were each
    measured to regress the full public set (retrieval-experiments SKILL.md #5).
    """

    answerable: dict[str, float] = field(
        default_factory=lambda: dict(ANSWERABILITY_PRIOR)
    )
    asks: int = 0
    non_answers: int = 0

    def observe(self, attribute: str | None, was_answered: bool) -> None:
        """Record the outcome of one clarifying question.

        Both branches update every *other* remaining attribute's belief by
        the measured likelihood ratio in `BUCKET_ANSWER_LR[attribute]`
        (answered: multiply odds by the ratio; non-answer: by its
        reciprocal) -- an actual per-pair Bayesian update from the public
        set's card composition, not a single hand-set constant applied
        uniformly. A pair absent from that table (insufficient public-set
        support, or `attribute` not a row in it) gets no update at all.
        """
        if attribute is None:
            return
        self.asks += 1
        ratios = BUCKET_ANSWER_LR.get(attribute, {})
        if was_answered:
            self.answerable[attribute] = 0.0
            for other, ratio in ratios.items():
                if self.answerable.get(other, 0.0) > 0.0:
                    self.answerable[other] = _bayes_update(self.answerable[other], ratio)
            return
        self.non_answers += 1
        self.answerable[attribute] = 0.0
        for other, ratio in ratios.items():
            if self.answerable.get(other, 0.0) > 0.0:
                self.answerable[other] = _bayes_update(self.answerable[other], 1.0 / ratio)

    def rank(self, excluded: set[str]) -> list[str]:
        """Remaining attributes, most-answerable first."""
        return sorted(
            (a for a in self.answerable if a not in excluded),
            key=lambda a: self.answerable[a],
            reverse=True,
        )

    def best(self, excluded: set[str]) -> str | None:
        """The most-answerable attribute not yet excluded, if any is worth asking."""
        for attribute in self.rank(excluded):
            if self.answerable[attribute] >= EXHAUSTED_THRESHOLD:
                return attribute
        return None

    @property
    def exhausted(self) -> bool:
        """True once no remaining attribute looks answerable."""
        return all(v < EXHAUSTED_THRESHOLD for v in self.answerable.values())
