"""A/B the MMR diversity re-rank on the *buying* track.

Why this experiment exists, and why it is not another weight tune.

`routing_params()` lets the buying/browsing label decide four things: the BM25
weight, the dense weight, the entropy threshold, and whether the MMR diversity
re-rank runs. Three of the four have been measured. The browsing fusion weights
are flat across 0.0-1.5 and must not be re-tuned (#16). The buying weights sit
on a monotone curve with no interior optimum (#16). The entropy threshold
changed no outcome on the tuning subset (#4). **The diversity switch has never
been measured at all** -- browsing has always had it, buying has never had it,
and nobody ran the leg with it on.

The miss census at 0.6978 points straight at it. Splitting every session by how
crowded the target's coarse category is (how many catalog products share it):

    scenario           median crowding, hits   misses    p (permutation, 20k)
    buying                              138      338     0.0096  *
    browsing                            173      212     0.4212
    intent_override                     141      117     0.6080

Buying's misses sit in categories 2.4x more crowded than its hits. Browsing
shows no crowding effect whatsoever. That asymmetry is the thing to explain,
and of the four knobs only diversity changes how much of a *category* one slate
covers -- fusion weights and an entropy threshold reorder or re-ask, they do
not spread. Under EXCLUDE_SHOWN a session surfaces up to 100 distinct products
across ten slates, so covering a crowded category is exactly what a diverse
slate buys, and re-offering ten near-identical items is exactly what it avoids.

Two things this experiment is *not*, both of them mistakes already made here:

  * It is not the turn-annealed lambda of #18, which measured null. That varied
    the diversity strength on a track that already had diversity. This turns it
    on for a track that never had it.
  * It is not a query-side change. #17, #19 and #26 are three separate
    findings that this benchmark punishes touching the query text, because
    89.7% of customer text is verbatim target copy (#14). Nothing here touches
    a query; the same pool is retrieved and only the final ordering changes.

The mechanism is a prediction, so state what would falsify it. If diversity is
the answer, the gain should land on buying sessions in crowded categories and
be near-neutral elsewhere. A leg that gains uniformly across crowding, or that
gains on browsing-scenario samples, is doing something else -- and note the two
tracks are not separable populations (#16): the weights key off the classifier's
per-turn label while `scenario_type` is the sample's ground truth, and the
classifier drifts buying-ward as attributes accumulate.

Identity must reproduce the shipped 0.6978 before the treatment leg is
believed. That check has already caught one real measurement bug in this
project (the gated-branch trap in CLAUDE.md "Blockers").

    uv run python3 scripts/ab_buying_diversify.py
    uv run python3 scripts/ab_buying_diversify.py --leg treatment
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
