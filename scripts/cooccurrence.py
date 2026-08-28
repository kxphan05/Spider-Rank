"""Attribute co-occurrence priors learned from the frozen catalog.

The idea, in the shape a shopper would state it: *someone buying makeup
probably wants pink*. Attribute values are not independent -- a category
implies materials, a material implies colors -- so a value the customer never
stated can still be predicted from one they did. Measured on this catalog the
effect is large and nothing like subtle:

    P(material=stainless steel | category=Watches Wrist Watches) = 0.483   vs 0.023   (21x)
    P(material=mesh            | category=Running Road Running)  = 0.425   vs 0.032   (13x)
    P(color=silver             | category=Necklaces Pendant)     = 0.229   vs 0.036   ( 6x)
    P(color=brown              | material=leather)               = 0.065   vs 0.018   ( 4x)

That is a genuinely different signal from the one in `user_profile.py`, which
is measured dead (`scripts/eval_profile_signal.py`): this conditions on what
the customer said *this session*, not on a cross-session identity guess, so
it needs no assumption that two sessions sharing a profile hash are the same
shopper -- an assumption that check showed to be false.

Two uses, both supported here, deliberately kept boost-only (see
`Agent._boost_by_disclosed`): an inferred value may raise a candidate, never
sink one. CLAUDE.md #5 records why that asymmetry is not optional -- any
nonzero mismatch penalty drops a candidate below every neutral
(unknown-attribute) candidate regardless of the weight attached to it, so a
wrong *inference* penalizing candidates would be unbounded in exactly the way
a wrong disclosure already is.

Built on top of `AttributeIndex`, so it inherits that extractor's word-
boundary matching. This ordering matters and was not free: built against the
older substring extractor, the strongest "signal" this module reported was
P(material=lace | category=Necklaces Pendant Necklaces) = 0.867, which is not
a fact about jewelry -- it is "necklace" containing "lace". Co-occurrence
amplifies extraction bugs into confident nonsense, so the extractor was fixed
first (see attributes.py `_vocab_matcher`).

**This lives in `scripts/`, not `starter/`, because no agent code path reaches
it.** The signal is real and measured, but wiring it into
`Agent._infer_attributes` alongside the masked LM is an experiment that has not
been run (NEXT_STEPS #4). Keeping it out of the package keeps 200 lines of
unreachable code out of the submitted bundle, which copies `starter/` verbatim
into `src/`. Move it back when it earns its place by measurement.

Usage:
    uv run python3 scripts/cooccurrence.py
    uv run python3 scripts/cooccurrence.py --attribute color --top 15
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path

from _common import DEFAULT_CATALOG  # noqa: F401

# _common puts the repo root on sys.path as an import side effect, so the
# starter import below resolves when this script is run directly.
from starter.attributes import AttributeIndex  # noqa: E402

logger = logging.getLogger(__name__)

# Attributes usable as evidence or as a prediction target. `budget` is
# included as evidence only in the sense that it is stored -- price is missing
# on ~79% of the catalog, so conditioning on it mostly conditions on
# "unknown".
COOCCURRENCE_ATTRIBUTES = ("category", "material", "color")

# A conditional distribution is only trusted with at least this many products
# behind the conditioning value. Below it, a "3 of 4 products are silver"
# accident reads as a 20x lift.
MIN_EVIDENCE_SUPPORT = 50
# ...and the prediction is only emitted if it beats the attribute's marginal
# by this factor. A prediction that merely restates the marginal ("most things
# are black") carries no information and would just re-rank the pool by global
# color frequency.
MIN_LIFT = 1.5
# Laplace smoothing on the conditional counts, so one unseen pairing does not
# produce a zero that annihilates a naive-Bayes product.
SMOOTHING = 1.0

# Categories arrive as a breadcrumb list ("Clothing, Shoes & Jewelry" > Women
# > Jewelry > Earrings > Hoop). The first entries are constant across this
# single-department catalog, so the last two carry all the discriminating
# power. This mirrors the evaluator's own `coarse_category()`; it is
# reimplemented rather than imported because `starter/` must not depend on
# `evaluator/` (the graded agent ships without it).
_GENERIC_CATEGORIES = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}


def coarse_category(values: list[str] | None) -> str | None:
    if not values:
        return None
    cleaned: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in _GENERIC_CATEGORIES:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else None


class CooccurrenceIndex:
    """P(target attribute value | one observed attribute value), from the catalog."""

    def __init__(
        self,
        joint: dict[tuple[str, str, str], Counter],
        marginals: dict[str, Counter],
        evidence_support: dict[tuple[str, str], int],
    ) -> None:
        self._joint = joint              # (evidence_attr, evidence_value, target_attr) -> Counter(target_value)
        self._marginals = marginals      # target_attr -> Counter(target_value)
        self._support = evidence_support  # (evidence_attr, evidence_value) -> products with that value

    @classmethod
    def build(cls, catalog_path: Path, attribute_index) -> CooccurrenceIndex:
        """Count co-occurrences across the catalog.

        `attribute_index` supplies material/color so the two indexes can never
        drift apart on extraction rules; category is read straight off the
        catalog line.
        """
        joint: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
        marginals: dict[str, Counter] = defaultdict(Counter)
        support: dict[tuple[str, str], int] = Counter()

        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                observed = {
                    "category": coarse_category(product.get("categories")),
                    "material": attribute_index.value_for("material", parent_asin),
                    "color": attribute_index.value_for("color", parent_asin),
                }
                for attribute, value in observed.items():
                    if value is None:
                        continue
                    marginals[attribute][value] += 1
                    support[(attribute, value)] += 1
                for evidence_attr, evidence_value in observed.items():
                    if evidence_value is None:
                        continue
                    for target_attr, target_value in observed.items():
                        if target_attr == evidence_attr or target_value is None:
                            continue
                        joint[(evidence_attr, evidence_value, target_attr)][target_value] += 1

        logger.info(
            "CooccurrenceIndex: %d conditional distributions over %s",
            len(joint), ", ".join(COOCCURRENCE_ATTRIBUTES),
        )
        return cls(dict(joint), dict(marginals), dict(support))

    def evidence_pairs(self, min_support: int = MIN_EVIDENCE_SUPPORT) -> list[tuple[str, str, str]]:
        """(evidence_attr, evidence_value, target_attr) triples with enough support.

        Public so callers can enumerate what the index actually learned without
        reaching into its counts.
        """
        return [key for key, counts in self._joint.items()
                if sum(counts.values()) >= min_support]

    def marginal(self, attribute: str, value: str) -> float:
        totals = self._marginals.get(attribute)
        if not totals:
            return 0.0
        return totals[value] / sum(totals.values())

    def conditional(self, target_attr: str, target_value: str, evidence: dict[str, str]) -> float:
        """P(target_attr = target_value | evidence), naive-Bayes over evidence.

        Naive Bayes because the evidence attributes are not independent
        (category already implies a lot about material) -- treating them as if
        they were overstates confidence when several agree. That is tolerable
        here only because the result is used to *rank*, never as a calibrated
        probability, and because MIN_LIFT gates on the final combined value.
        """
        prior = self.marginal(target_attr, target_value)
        if prior <= 0.0:
            return 0.0
        log_score = math.log(prior)
        used = 0
        for evidence_attr, evidence_value in evidence.items():
            if evidence_attr == target_attr:
                continue
            counts = self._joint.get((evidence_attr, evidence_value, target_attr))
            if counts is None:
                continue
            total = sum(counts.values())
            if total < MIN_EVIDENCE_SUPPORT:
                continue
            distinct = len(self._marginals.get(target_attr, ()))
            likelihood = (counts[target_value] + SMOOTHING) / (total + SMOOTHING * max(distinct, 1))
            log_score += math.log(likelihood) - math.log(prior)
            used += 1
        return math.exp(log_score) if used else 0.0

    def predict(
        self,
        target_attr: str,
        evidence: dict[str, str],
        min_lift: float = MIN_LIFT,
    ) -> tuple[str, float] | None:
        """Most likely `target_attr` value given `evidence`, or None.

        Returns None rather than a weak guess whenever the best candidate
        fails to beat its own marginal by `min_lift` -- a prediction that only
        restates the global frequency would re-rank the pool by "what colors
        are common", which is not personalization.
        """
        totals = self._marginals.get(target_attr)
        if not totals or not evidence:
            return None
        best_value, best_lift = None, min_lift
        for value in totals:
            prior = self.marginal(target_attr, value)
            posterior = self.conditional(target_attr, value, evidence)
            if prior <= 0.0 or posterior <= 0.0:
                continue
            lift = posterior / prior
            if lift > best_lift:
                best_value, best_lift = value, lift
        return (best_value, best_lift) if best_value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report the strongest attribute co-occurrence priors in the catalog.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", type=Path, default=Path(DEFAULT_CATALOG))
    parser.add_argument("--attribute", choices=COOCCURRENCE_ATTRIBUTES, default=None,
                        help="target attribute to predict (default: material and color -- "
                             "'category' is excluded because the agent never needs to "
                             "predict one, and its thousands of tiny buckets produce huge "
                             "lifts over near-zero priors)")
    parser.add_argument("--top", type=int, default=20, help="rows to print (default 20)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    index = CooccurrenceIndex.build(args.catalog, AttributeIndex.build(args.catalog))

    targets = [args.attribute] if args.attribute else ["material", "color"]
    rows = []
    for evidence_attr, evidence_value, target_attr in index.evidence_pairs():
        if target_attr not in targets:
            continue
        prediction = index.predict(target_attr, {evidence_attr: evidence_value})
        if prediction is None:
            continue
        value, lift = prediction
        rows.append((lift, evidence_attr, evidence_value, target_attr, value,
                     index.conditional(target_attr, value, {evidence_attr: evidence_value}),
                     index.marginal(target_attr, value)))

    # Sorted by posterior, not lift. Lift over a near-zero marginal is huge and
    # meaningless -- ranking by it put P=0.002 predictions at the top.
    rows.sort(key=lambda row: row[5], reverse=True)
    print(f"\n{len(rows)} predictions of {'/'.join(targets)} clear MIN_LIFT={MIN_LIFT} "
          f"and MIN_EVIDENCE_SUPPORT={MIN_EVIDENCE_SUPPORT}, ranked by posterior\n")
    header = f"{'lift':>6}  {'P(target|evidence)':>18}  {'marginal':>9}  prediction"
    print(header)
    print("-" * (len(header) + 40))
    for lift, ev_attr, ev_value, tgt_attr, tgt_value, posterior, prior in rows[: args.top]:
        print(f"{lift:6.1f}x {posterior:18.3f}  {prior:9.3f}  "
              f"{tgt_attr}={tgt_value}  <-  {ev_attr}={ev_value}")

    print("\nNot wired into the agent -- see the module docstring.")


if __name__ == "__main__":
    main()
