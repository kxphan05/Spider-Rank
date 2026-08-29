"""Sweep PRF_WEIGHT -- the pseudo-relevance-feedback expansion leg's RRF weight.

Background and the honest prior are in `starter/prf.py`'s module docstring.
Short version: buying scores 0.688 hit rate against browsing's 0.825 because
the median turn-1 hard constraint is *one very common token* (`cotton` alone
matches 18.8% of the catalog). That is an information problem, and every other
lever in this project redistributes weight among signal that already exists.
PRF is the one classical technique that adds vocabulary.

The prediction to check, stated before the run: **if PRF helps at all, the
gain should be concentrated in `buying`.** Browsing queries are already
verbose, so they have less to gain and more to drift. If a gain shows up
uniformly, or lands in browsing, the mechanism is not the one described above
and the result should be distrusted even if the number is positive.

The known failure mode is query drift, and it is visible by inspection here:
for "Men's Shoes ... leather" the harvested terms are `coats, jackets, faux,
jacket, genuine, collar, zipper`, because leather in this catalog is dominated
by outerwear. `prf_search` anchors the expansion to the original query for
that reason. Expect drift to still cost something.

`PRF_WEIGHT = 0.0` is the exact identity and must reproduce the shipped score
before any other row is believed.

    uv run python3 scripts/sweep_prf_weight.py
"""
from __future__ import annotations

import argparse
import sys

from _common import DEFAULT_CATALOG, DEFAULT_DATASET, isolate_profile_store, score_once  # noqa: F401

from evaluator.local_evaluator import catalog_index, load_jsonl  # noqa: E402
from starter import agent as agent_module  # noqa: E402

DEFAULT_VALUES = (0.0, 0.25, 0.5, 1.0, 2.0)


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
    n = len(samples)

    print(f"{'weight':>7}  {'HitRate':>8} {'hits':>9}  {'MRR':>7}  {'MTTC':>7}"
          f"  {'Technical':>9}  {'delta':>8}  {'buying':>8}  {'browsing':>9}", flush=True)
    baseline = None
    for value in args.values:
        agent_module.PRF_WEIGHT = value
        report, agent = score_once(samples, catalog)
        if agent.dense is None:
            print("  WARNING: dense dark -- not comparable to shipped numbers (#15)",
                  file=sys.stderr)
        technical = report["recommended_technical_score"]
        hits = round(report["hit_rate_at_10"] * n)
        if baseline is None:
            baseline = technical
            delta = "--"
        else:
            delta = f"{technical - baseline:+.4f}"
        scenarios = report["scenario_metrics"]
        buying = scenarios.get("buying", {}).get("hit_rate_at_10", float("nan"))
        browsing = scenarios.get("browsing", {}).get("hit_rate_at_10", float("nan"))
        print(f"{value:7.2f}  {report['hit_rate_at_10']:8.4f} {hits:4d}/{n}"
              f"  {report['mrr']:7.4f}  {report['mttc']:7.3f}  {technical:9.4f}"
              f"  {delta:>8}  {buying:8.3f}  {browsing:9.3f}", flush=True)


if __name__ == "__main__":
    main()
