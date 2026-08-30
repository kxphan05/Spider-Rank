"""Sweep SLOT_DECAY -- how fast a stated constraint's influence should fade.

Spec 4.3 lists slot decay as in-scope and the agent has none: a value disclosed on
turn 1 carries full weight on turn 10. The only decay that exists is the override
detector's all-or-nothing wipe, and the blunt version of that idea measured
-0.0580, measured directly.

Only the ratio between turns matters, so this sweeps one scalar. SLOT_DECAY = 1.0
is the identity and must reproduce the shipped score first.

    uv run python3 scripts/sweep_slot_decay.py [--values 1.0 0.9 0.8]
"""
from __future__ import annotations

import argparse
import sys

from _common import DEFAULT_CATALOG, DEFAULT_DATASET, isolate_profile_store, score_once  # noqa: F401

from evaluator.local_evaluator import catalog_index, load_jsonl  # noqa: E402
from starter import agent as agent_module  # noqa: E402

DEFAULT_VALUES = (1.0, 0.95, 0.9, 0.8, 0.7, 0.5)


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

    print(f"{'decay':>7}  {'HitRate':>8} {'hits':>7}  {'MRR':>7}  {'MTTC':>7}  {'Technical':>9}")
    baseline = None
    for value in args.values:
        agent_module.SLOT_DECAY = value
        report, agent = score_once(samples, catalog)
        if agent.dense is None:
            print("  WARNING: dense index dark -- results are not comparable to the shipped"
                  " numbers", file=sys.stderr)
        hits = round(report["hit_rate_at_10"] * len(samples))
        technical = report["recommended_technical_score"]
        if baseline is None:
            baseline = technical
        delta = "" if value == args.values[0] else f"  {technical - baseline:+.4f}"
        print(f"{value:7.2f}  {report['hit_rate_at_10']:8.4f} {hits:4d}/{len(samples)}"
              f"  {report['mrr']:7.4f}  {report['mttc']:7.3f}  {technical:9.4f}{delta}", flush=True)


if __name__ == "__main__":
    main()
