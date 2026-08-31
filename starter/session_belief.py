"""Short-term, within-session belief about what this customer can still answer.

Most sessions run out of answerable attribute buckets well before they run
out of turns (mean ~2.09 left after turn 1), and every ask past that point
returns a contentless reply that gets fed to BM25 as if it were customer
content. `ANSWERABILITY_PRIOR` starts each session at the population
marginals and updates from what this customer actually does -- see the
retrieval-experiments skill (#5) for why the long-term, cross-session
profile signal (user_profile.py) is deliberately unused instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from .config import (ANSWERABILITY_PRIOR, BUCKET_ANSWER_LR, EXHAUSTED_THRESHOLD)


def _bayes_update(prob: float, likelihood_ratio: float) -> float:
    """Multiply `prob`'s ODDS by `likelihood_ratio`, return the new probability.

    Odds, not probability, composes correctly across several observations
    in a session -- unlike the raw-probability approach it replaced.
    """
    if prob <= 0.0 or prob >= 1.0 or likelihood_ratio == 1.0:
        return prob
    odds = prob / (1.0 - prob)
    new_odds = odds * likelihood_ratio
    return new_odds / (1.0 + new_odds)





@dataclass
class SessionBelief:
    """Running estimate of which attributes this customer can still answer.

    Initialized from population marginals, updated only from observed
    outcomes. Deliberately carries no cross-session state (see module docstring).
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
        (answered: multiply odds by it; non-answer: by its reciprocal).
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
