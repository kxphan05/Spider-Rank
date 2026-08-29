"""Sweep PHRASE_WEIGHT, the phrase-match leg's RRF weight. Results in CLAUDE.md #20.

Shipped at 2.0 (+0.0272). There is a real interior optimum, but 2.0 and 4.0 are
one session apart, so the top is flat and 2.0 is the lower-variance pick.

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
