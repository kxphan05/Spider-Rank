"""Sweep the BM25F field weights -- hand-picked at index-build time, never tuned.

`bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)` weights the FTS5 columns

    parent_asin(UNINDEXED), title, categories, features, details, store, description

and those six free numbers were guessed. The IR literature is consistent that
BM25F field weights are collection- and query-dependent and need a grid
search, so guessed values on a 50k clothing catalog are unlikely to be right.
This matters more than usual here because BM25 is the leg that actually
carries this benchmark: the dense leg adds no recall at all (CLAUDE.md #14),
and the phrase leg -- the largest win measured so far, #20 -- rides the *same*
ORDER BY clause, so a better field weighting improves two of the three legs at
once.

A full grid is unaffordable: 6 free weights at a full 200-sample eval per
point. This does **coordinate descent** instead -- one field at a time, each
tried at multiples of its current value, keeping any improvement before moving
to the next field. That finds a local optimum, not a global one, and the order
of the fields biases the result; both caveats are real and are the price of
the budget.

Rows stream as they complete, so a partial run is still usable. The identity
row must reproduce the shipped score before any other row is believed
(CLAUDE.md's verification rule -- a mis-set identity has silently corrupted an
A/B here before).

    uv run python3 scripts/sweep_bm25_fields.py
"""
from __future__ import annotations

import argparse
import sys

from _common import DEFAULT_CATALOG, DEFAULT_DATASET, isolate_profile_store, score_once  # noqa: F401

from evaluator.local_evaluator import catalog_index, load_jsonl  # noqa: E402
from starter import retrieval as retrieval_module  # noqa: E402

# Index 0 is parent_asin, which is UNINDEXED -- its weight is inert, so it is
# never swept.
FIELD_NAMES = ("parent_asin", "title", "categories", "features", "details", "store", "description")
FREE_FIELDS = (1, 2, 3, 4, 5, 6)
# Two multipliers, not four. Each point is a full 200-sample evaluation, and
# coordinate descent over six fields multiplies that by the number of
# multipliers: 4 gives 25 points, which is a 13-hour run when the box is also
# carrying other sweeps. Halve-or-double over all six fields is 13 points and
# answers the question that actually matters first -- whether any field is
# badly mis-weighted -- leaving a finer sweep for whichever field moves.
DEFAULT_MULTIPLIERS = (0.5, 2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--fields", type=int, nargs="+", default=list(FREE_FIELDS),
                        help="column indices to sweep, in the order they are swept")
    parser.add_argument("--multipliers", type=float, nargs="+", default=list(DEFAULT_MULTIPLIERS))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    isolate_profile_store()
    samples = load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog = catalog_index(args.catalog)
    n = len(samples)

    state = {"baseline": None}

    def run(weights: tuple[float, ...]) -> float:
        retrieval_module.FIELD_WEIGHTS = weights
        report, agent = score_once(samples, catalog)
        if agent.dense is None:
            print("  WARNING: dense dark -- not comparable to shipped numbers (#15)",
                  file=sys.stderr)
        tech = report["recommended_technical_score"]
        hits = round(report["hit_rate_at_10"] * n)
        label = ", ".join(f"{w:g}" for w in weights[1:])
        base = state["baseline"]
        delta = "--" if base is None else f"{tech - base:+.4f}"
        print(f"{label:<28} {report['hit_rate_at_10']:8.4f} {hits:4d}/{n}"
              f" {report['mrr']:7.4f} {report['mttc']:7.3f} {tech:9.4f} {delta:>8}",
              flush=True)
        return tech

    print(f"{'title,cat,feat,det,store,desc':<28} {'HitRate':>8} {'hits':>9}"
          f" {'MRR':>7} {'MTTC':>7} {'Technical':>9} {'delta':>8}", flush=True)

    current = tuple(retrieval_module.DEFAULT_FIELD_WEIGHTS)
    best = run(current)
    state["baseline"] = best
    print("# identity above must reproduce the shipped score", file=sys.stderr, flush=True)

    for index in args.fields:
        name = FIELD_NAMES[index]
        print(f"\n# --- sweeping {name} (current {current[index]:g}) ---", flush=True)
        for multiplier in args.multipliers:
            candidate = list(current)
            candidate[index] = round(current[index] * multiplier, 4)
            if candidate[index] == current[index]:
                continue
            tech = run(tuple(candidate))
            if tech > best:
                best = tech
                current = tuple(candidate)
        print(f"# {name} kept at {current[index]:g}  (best so far {best:.4f})", flush=True)

    print(f"\n# final weights: {', '.join(f'{w:g}' for w in current)}   technical {best:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
