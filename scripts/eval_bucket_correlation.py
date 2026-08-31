"""Does a non-answer on one attribute predict more, or less, chance the
customer can answer about a *different* attribute?

`SessionBelief` used to answer this with a single hand-set constant,
`NON_ANSWER_SPILLOVER`, applied uniformly to every other attribute regardless
of which one was asked. This script instead computes the real conditional
probabilities from the evaluator's own card-generation logic over the 200
public samples: for every ordered pair (A, B), P(B answerable | A answerable)
vs P(B answerable | A NOT answerable), as an odds ratio. Read-only, no Agent
required.

Finding: the correlation is NOT one direction. `material` behaves as a
fixed-budget consumer -- an answerable material is a strong NEGATIVE
predictor for every other attribute (OR 0.11-0.21, i.e. others become
5-9x less likely), because a detected material almost always becomes
`hard_constraints[0]` and gets pre-disclosed in `initial_message()` for
buying-scenario sessions, consuming the "slot" that would otherwise carry a
minor attribute. Among the *other* attributes (color/style/size/use_case),
once material's effect is set aside, the correlation runs the other way --
POSITIVE (OR 1.2-4.7): a customer whose card discloses one minor attribute
is more likely, not less, to have others too (richer catalog listings surface
several minor attributes together). `size`, `use_case`, `feature` and
`budget` don't get an entry as the conditioning attribute: their marginals
(0.045, 0.020, 0.960, 0.000) leave too few sessions on one side of the split
(< MIN_SUPPORT) to trust a ratio from 200 samples.

`starter/config.py`'s `BUCKET_ANSWER_LR` is this script's output, pasted in.
Re-run this if the hidden set's card composition looks different from the
public one -- it would invalidate the table the same way a shifted
`ANSWERABILITY_PRIOR` would.

    uv run python3 scripts/eval_bucket_correlation.py
"""
from __future__ import annotations

from _common import DEFAULT_CATALOG, DEFAULT_DATASET  # noqa: F401

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index, classify_constraint, coarse_category,
    initial_message, load_jsonl, materialize_hidden_fields,
)

ASKABLE = ("material", "color", "feature", "style", "size", "use_case", "budget")
MIN_SUPPORT = 15  # minimum sessions required on EACH side of the A=1/A=0 split


def answerable_buckets(dataset: str, catalog: str) -> list[dict[str, bool]]:
    samples = load_jsonl(dataset)
    _, categories, products = catalog_index(catalog)
    rows = []
    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        initial_message(effective, coarse_category(categories.get(target, [])), disclosed)
        constraints = [
            *[str(v) for v in card.get("hard_constraints", [])],
            *[str(v) for v in card.get("soft_preferences", [])],
        ]
        buckets = {classify_constraint(c) for c in constraints if c not in disclosed}
        rows.append({b: (b in buckets) for b in ASKABLE})
    return rows


def _odds(p: float, n: int) -> float:
    k = p * n
    return (k + 0.5) / (n - k + 0.5)  # Haldane-Anscombe correction for k in {0, n}


def main() -> None:
    rows = answerable_buckets(str(DEFAULT_DATASET), str(DEFAULT_CATALOG))
    n = len(rows)
    print(f"n={n} sessions\n")

    table: dict[str, dict[str, float]] = {}
    for a in ASKABLE:
        if a in ("feature", "budget"):
            continue
        a1 = [r for r in rows if r[a]]
        a0 = [r for r in rows if not r[a]]
        if len(a1) < MIN_SUPPORT or len(a0) < MIN_SUPPORT:
            print(f"  skip A={a}: support n1={len(a1)} n0={len(a0)} < {MIN_SUPPORT}")
            continue
        table[a] = {}
        for b in ASKABLE:
            if b == a or b in ("feature", "budget"):
                continue
            p1 = sum(r[b] for r in a1) / len(a1)
            p0 = sum(r[b] for r in a0) / len(a0)
            odds_ratio = _odds(p1, len(a1)) / _odds(p0, len(a0))
            table[a][b] = round(odds_ratio, 3)
            print(f"  A={a:<10} n1={len(a1):3d} n0={len(a0):3d}  B={b:<10} "
                  f"P(B|A answered)={p1:.3f}  P(B|A non-answer)={p0:.3f}  "
                  f"OR(answered)={odds_ratio:.3f}  OR(non-answer)={1 / odds_ratio:.3f}")

    print("\nBUCKET_ANSWER_LR = " + repr(table))


if __name__ == "__main__":
    main()
