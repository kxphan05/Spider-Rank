"""Sweep the cross-encoder rerank stage: fusion weight x pool depth.

Closes the last named gap against spec Pillar I -- until this
stage nothing scored a query and a document *jointly*.

**Depth is the interesting axis and it is easy to get backwards.** At
RERANK_TOP_N == top_k == 10 the reranker can only permute the slate the agent was
already returning: it can move MRR and cannot convert a miss into a hit under any
ordering. Only scoring deeper opens a recall channel.

    depth 10  ->  MRR moves, HitRate pinned to the identity
    depth 20  ->  HitRate can move, in either direction

A HitRate change at depth 10 is a bug signal, not a result.

Backend is MiniLM (~17 ms/pair). Qwen3-Reranker-0.6B judges better but needs ~27
s/pair here -- ~75 hours for one 200-sample eval -- so it is demo-only and cannot
be swept. RERANK_WEIGHT = 0.0 is the exact identity.

    uv run python3 scripts/sweep_rerank.py
"""
from __future__ import annotations

import argparse
import sys

from _common import DEFAULT_CATALOG, DEFAULT_DATASET, isolate_profile_store, score_once  # noqa: F401

from evaluator.local_evaluator import catalog_index, load_jsonl  # noqa: E402
from starter import agent as agent_module  # noqa: E402

# Cut from a 3x2 grid to the four points that answer the actual question.
# Each point is a full 200-sample evaluation at ~15 minutes, and the question
# is not "what is the optimal weight" but "does scoring deeper than the slate
# buy recall". Identity, then weight 1.0 at both depths to isolate depth, then
# 2.0 at depth 20 to show the direction of the weight axis.
DEFAULT_WEIGHTS = (1.0,)
DEFAULT_DEPTHS = (10, 20)
EXTRA_POINTS = ((2.0, 20),)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--weights", type=float, nargs="+", default=list(DEFAULT_WEIGHTS))
    parser.add_argument("--depths", type=int, nargs="+", default=list(DEFAULT_DEPTHS))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    isolate_profile_store()
    samples = load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog = catalog_index(args.catalog)
    n = len(samples)

    print(f"{'weight':>7} {'depth':>6}  {'HitRate':>8} {'hits':>9}  {'MRR':>7}  {'MTTC':>7}"
          f"  {'Technical':>9}  {'delta':>8}", flush=True)

    baseline = None

    def run(weight: float, depth: int) -> float:
        nonlocal baseline
        agent_module.RERANK_WEIGHT = weight
        agent_module.RERANK_TOP_N = depth
        report, agent = score_once(samples, catalog)
        if agent.dense is None:
            print("  WARNING: dense dark -- not comparable to shipped numbers (#15)",
                  file=sys.stderr)
        if weight > 0.0 and agent.reranker is None:
            print("  WARNING: reranker dark -- this row is the identity, not a measurement",
                  file=sys.stderr)
        technical = report["recommended_technical_score"]
        hits = round(report["hit_rate_at_10"] * n)
        delta = "--" if baseline is None else f"{technical - baseline:+.4f}"
        label_depth = "-" if weight == 0.0 else str(depth)
        print(f"{weight:7.2f} {label_depth:>6}  {report['hit_rate_at_10']:8.4f} {hits:4d}/{n}"
              f"  {report['mrr']:7.4f}  {report['mttc']:7.3f}  {technical:9.4f}  {delta:>8}",
              flush=True)
        return technical

    # Identity once: at weight 0 the stage is a no-op, so depth is irrelevant
    # and running both would burn an eval to reproduce the same number.
    baseline = run(0.0, args.depths[0])

    for depth in args.depths:
        for weight in args.weights:
            run(weight, depth)
    for weight, depth in EXTRA_POINTS:
        if weight not in args.weights or depth not in args.depths:
            run(weight, depth)


if __name__ == "__main__":
    main()
