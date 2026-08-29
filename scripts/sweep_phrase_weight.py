"""Sweep PHRASE_WEIGHT -- the verbatim-span retrieval leg's RRF weight.

`BM25Index.search` ORs the query's unique tokens, so "Buckle closure" is two
independent terms against 50k products. `BM25Index.phrase_search` matches the
span itself. That distinction should matter a lot here: 89.7% of the local
simulator's turn-1 hard constraints are verbatim substrings of the target
product's own catalog text (CLAUDE.md #14), so the exact span is usually
present and usually rare.

This is the one retrieval-side idea that runs *with* that confound rather than
against it. Every previous retrieval change that lost score (#14 dense weight,
#17 override rewrite, #19 query pruning) lost by adding tolerance to an
exact-match benchmark.

`PHRASE_WEIGHT = 0.0` is the identity and must reproduce 0.6182 exactly. Check
that row before believing any other.

    uv run python3 scripts/sweep_phrase_weight.py
"""
from __future__ import annotations

import argparse
import sys

from _common import DEFAULT_CATALOG, DEFAULT_DATASET, isolate_profile_store, score_once  # noqa: F401

from evaluator.local_evaluator import catalog_index, load_jsonl  # noqa: E402
from starter import agent as agent_module  # noqa: E402

DEFAULT_VALUES = (0.0, 0.5, 1.0, 2.0, 4.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--values", type=float, nargs="+", default=list(DEFAULT_VALUES))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    isolate_profile_store()
    samples = load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog = catalog_index(args.catalog)

    print(f"{'weight':>7}  {'HitRate':>8} {'hits':>9}  {'MRR':>7}  {'MTTC':>7}  {'Technical':>9}  {'delta':>8}")
    baseline = None
    for value in args.values:
        agent_module.PHRASE_WEIGHT = value
        report, agent = score_once(samples, catalog)
        if agent.dense is None:
            print("  WARNING: dense index dark -- not comparable to shipped numbers (#15)",
                  file=sys.stderr)
        technical = report["recommended_technical_score"]
        hits = round(report["hit_rate_at_10"] * len(samples))
        if baseline is None:
            baseline = technical
        delta = f"{technical - baseline:+.4f}" if value != args.values[0] else "--"
        print(f"{value:7.2f}  {report['hit_rate_at_10']:8.4f} {hits:4d}/{len(samples)}"
              f"  {report['mrr']:7.4f}  {report['mttc']:7.3f}  {technical:9.4f}  {delta:>8}",
              flush=True)


if __name__ == "__main__":
    main()
