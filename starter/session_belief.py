"""Short-term, within-session belief about what this customer can still answer.

The competition spec's Pillar III ("Self-Evolution: Dynamic Context
Programming") asks for runtime adaptation: distilling accumulated dialog
history into state that changes the agent's own strategy mid-session. The
*long-term* half of that is answered and closed -- `user_profile.py` persists
cross-session disclosures, and `scripts/eval_profile_signal.py` measures that
the provided profile carries no signal to act on (same-key sessions share a
coarse category 0.5% of the time against a 1.2% +- 0.5% random baseline), so
`SessionState.profile_hint` is deliberately inert. See CLAUDE.md #5.

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
from .config import (ANSWERABILITY_PRIOR, EXHAUSTED_THRESHOLD, NON_ANSWER_SPILLOVER)





@dataclass
class SessionBelief:
    """Running estimate of which attributes this customer can still answer.

    Initialized from the population marginals and updated only from observed
    outcomes. Deliberately carries no cross-session state: seeding this from
    `profile_hint` is the fourth variant of an idea whose first three were each
    measured to regress the full public set (CLAUDE.md #5).
    """

    answerable: dict[str, float] = field(
        default_factory=lambda: dict(ANSWERABILITY_PRIOR)
    )
    asks: int = 0
    non_answers: int = 0

    def observe(self, attribute: str | None, was_answered: bool) -> None:
        """Record the outcome of one clarifying question."""
        if attribute is None:
            return
        self.asks += 1
        if was_answered:
            # An answer says nothing reliable about the other buckets -- the
            # card's composition is not observable from one hit -- so nothing
            # else is touched. Resisting the urge to invent a correlation
            # here is deliberate: there is no measurement behind one.
            self.answerable[attribute] = 0.0
            return
        self.non_answers += 1
        self.answerable[attribute] = 0.0
        for other in self.answerable:
            if self.answerable[other] > 0.0:
                self.answerable[other] *= NON_ANSWER_SPILLOVER

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
