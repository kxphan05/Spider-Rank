"""A/B the MMR diversity re-rank on the buying track (CLAUDE.md #24).

Browsing has always had the re-rank, buying never has, and the switch has never
been measured. Buying misses sit in coarse categories 2.4x more crowded than its
hits (median 338 vs 138, p = 0.0096) while browsing shows no crowding effect, and
diversity is the only knob that spreads a slate across a category.

Falsification, stated before the run: the gain should land on buying sessions in
crowded categories. A uniform gain is doing something else. Note the two tracks
are not disjoint populations -- the weights key off the classifier's per-turn
label, not scenario_type (#16).

    uv run python3 scripts/ab_buying_diversify.py [--leg identity|treatment]
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics

from _common import DEFAULT_CATALOG, DEFAULT_DATASET, isolate_profile_store, score_once  # noqa: F401

from evaluator.local_evaluator import catalog_index, coarse_category, load_jsonl  # noqa: E402
from starter import agent as agent_module  # noqa: E402

# (label, BUYING_DIVERSIFY)
LEGS = {
    "identity": False,
    "treatment": True,
}

# Shipped score at this HEAD, from `python3 -m evaluator.local_evaluator`.
# Compared to 4dp, which is the precision the number is quoted at everywhere
# else; anything beyond that is below this benchmark's one-session resolution.
SHIPPED_TECHNICAL = 0.697796


def crowding_table(samples, catalog, report):
    """Hit rate by how crowded the target's coarse category is.

    This is the column that decides whether the mechanism claim survives: the
    gain should concentrate where the diagnosis said it would.
    """
    _, categories, _ = catalog
    size = collections.Counter(coarse_category(v) for v in categories.values())
    by_id = {s["sample_id"]: s for s in report["sessions"]}
    rows = []
    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        session = by_id.get(sample["sample_id"])
        if session is None:
            continue
        rows.append((size[coarse_category(categories.get(target, []))],
                     bool(session["hit"]), str(sample["scenario_type"])))
    if not rows:
        return
    crowd = sorted(row[0] for row in rows)
    cuts = [crowd[len(crowd) // 4], crowd[len(crowd) // 2], crowd[3 * len(crowd) // 4]]

    def bucket(value: int) -> int:
        return 0 if value <= cuts[0] else 1 if value <= cuts[1] else 2 if value <= cuts[2] else 3

    labels = ["Q1 least crowded", "Q2", "Q3", "Q4 most crowded"]
    print("    hit rate by category crowding:")
    for index, label in enumerate(labels):
        subset = [row for row in rows if bucket(row[0]) == index]
        if not subset:
            continue
        hits = sum(1 for row in subset if row[1])
        buying = [row for row in subset if row[2] == "buying"]
        buy_hits = sum(1 for row in buying if row[1])
        buy_col = (f"   buying {buy_hits:2d}/{len(buying):<3d} {buy_hits / len(buying):.3f}"
                   if buying else "   buying    n/a")
        print(f"      {label:18s} n={len(subset):3d}  all {hits:3d}/{len(subset):<3d}"
              f" {hits / len(subset):.3f}{buy_col}", flush=True)
    buy = [row for row in rows if row[2] == "buying"]
    hit_crowd = [row[0] for row in buy if row[1]]
    miss_crowd = [row[0] for row in buy if not row[1]]
    if hit_crowd and miss_crowd:
        print(f"    buying median crowding: hits {statistics.median(hit_crowd):.0f}"
              f"  misses {statistics.median(miss_crowd):.0f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--leg", choices=sorted(LEGS), default=None,
                        help="run a single leg (for two concurrent processes); default runs both")
    parser.add_argument("--json-out", default=None, help="write this leg's report to a file")
    args = parser.parse_args()

    isolate_profile_store()
    samples = load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog = catalog_index(args.catalog)
    legs = [args.leg] if args.leg else list(LEGS)

    print(f"{'leg':12s} {'HitRate':>8} {'hits':>9}  {'MRR':>7}  {'MTTC':>7}  {'Technical':>9}  {'delta':>8}")
    baseline = None
    for label in legs:
        agent_module.BUYING_DIVERSIFY = LEGS[label]
        report, agent = score_once(samples, catalog)
        if agent.dense is None or agent.intent_classifier is None:
            print("  WARNING: a component is dark -- not the shipped configuration (#15)")
        technical = report["recommended_technical_score"]
        hits = round(report["hit_rate_at_10"] * len(samples))
        if label == "identity":
            baseline = technical
            drift = technical - SHIPPED_TECHNICAL
            note = "reproduces shipped" if abs(drift) < 5e-5 else f"DRIFT {drift:+.4f} vs shipped -- do not trust the treatment"
            print(f"  identity check: {technical:.4f} vs shipped {SHIPPED_TECHNICAL:.4f} -- {note}", flush=True)
        delta = "--" if baseline is None or label == "identity" else f"{technical - baseline:+.4f}"
        print(f"{label:12s} {report['hit_rate_at_10']:8.4f} {hits:4d}/{len(samples)}"
              f"  {report['mrr']:7.4f}  {report['mttc']:7.3f}  {technical:9.4f}  {delta:>8}", flush=True)
        for name in sorted(report["scenario_metrics"]):
            metrics = report["scenario_metrics"][name]
            print(f"    {name:18s} hit {metrics['hit_rate_at_10']:.3f}  "
                  f"mrr {metrics['mrr']:.3f}  mttc {metrics['mttc']:.3f}", flush=True)
        crowding_table(samples, catalog, report)
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as handle:
                json.dump(report, handle)


if __name__ == "__main__":
    main()
