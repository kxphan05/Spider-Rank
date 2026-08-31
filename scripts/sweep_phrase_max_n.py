"""Sweep PHRASE_MAX_N, the longest n-gram size BM25Index.phrase_search tries.

phrase_search (starter/retrieval.py:226) walks n from PHRASE_MAX_N down to
PHRASE_MIN_N, longest first, and only runs the first MAX_PHRASE_QUERIES=24
distinct grams it builds (config.py:201). Raising PHRASE_MAX_N means more of
that fixed budget gets spent on very long, rarer spans before any shorter
gram is tried at all -- worth sweeping explicitly rather than assuming
"longer span = strictly more specific = better" holds once the budget cap
is in the picture.

PHRASE_MAX_N is read at call time from the starter.retrieval module
namespace (not rebound into a closure), so patching
`starter.retrieval.PHRASE_MAX_N` before each run is sufficient -- no index
rebuild needed, matching sweep_phrase_weight.py's pattern for PHRASE_WEIGHT.

    uv run python3 scripts/sweep_phrase_max_n.py [--limit 60] [--track-failures out.json]
"""
from __future__ import annotations

import json
import sys
import argparse
import random
from pathlib import Path

from _common import DEFAULT_CATALOG, DEFAULT_DATASET, isolate_profile_store, score_once  # noqa: F401

from evaluator.local_evaluator import catalog_index, load_jsonl  # noqa: E402
from starter import retrieval as retrieval_module  # noqa: E402

# 5 is the value shipped before this sweep; the rest bracket it on both sides.
DEFAULT_VALUES = (3, 4, 5, 6, 8, 10, 15, 20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--values", type=int, nargs="+", default=list(DEFAULT_VALUES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="shuffle samples before --limit, for a random subset")
    parser.add_argument(
        "--track-failures", metavar="PATH", default=None,
        help="write per-value per-sample hit/miss (sample_id, scenario_type, hit, best_rank) to this JSON file, "
             "and print a diff of which sample_ids flip hit<->miss relative to the first --values entry",
    )
    args = parser.parse_args()

    isolate_profile_store()
    samples = load_jsonl(args.dataset)
    if args.seed is not None:
        random.Random(args.seed).shuffle(samples)
    if args.limit:
        samples = samples[: args.limit]
    catalog = catalog_index(args.catalog)

    original = retrieval_module.PHRASE_MAX_N
    print(f"{'PHRASE_MAX_N':>13}  {'HitRate':>8} {'hits':>9}  {'MRR':>7}  {'MTTC':>7}  {'Technical':>9}  {'delta':>8}")
    baseline = None
    per_value_sessions: dict[int, list[dict]] = {}
    try:
        for value in args.values:
            retrieval_module.PHRASE_MAX_N = value
            report, agent = score_once(samples, catalog)
            if agent.dense is None:
                print("  WARNING: dense index dark -- not comparable to shipped numbers (#15)",
                      file=sys.stderr)
            technical = report["recommended_technical_score"]
            hits = round(report["hit_rate_at_10"] * len(samples))
            if baseline is None:
                baseline = technical
            delta = f"{technical - baseline:+.4f}" if value != args.values[0] else "--"
            print(f"{value:13d}  {report['hit_rate_at_10']:8.4f} {hits:4d}/{len(samples)}"
                  f"  {report['mrr']:7.4f}  {report['mttc']:7.3f}  {technical:9.4f}  {delta:>8}",
                  flush=True)
            if args.track_failures:
                per_value_sessions[value] = [
                    {k: s[k] for k in ("sample_id", "scenario_type", "hit", "first_hit_turn", "best_rank")}
                    for s in report["sessions"]
                ]
    finally:
        retrieval_module.PHRASE_MAX_N = original

    if args.track_failures:
        Path(args.track_failures).write_text(json.dumps(per_value_sessions, indent=2), encoding="utf-8")
        print(f"\nper-sample results written to {args.track_failures}", file=sys.stderr)

        ref_value = args.values[0]
        ref = {s["sample_id"]: s["hit"] for s in per_value_sessions[ref_value]}
        for value in args.values:
            sessions = per_value_sessions[value]
            misses = sorted(s["sample_id"] for s in sessions if not s["hit"])
            print(f"\n--- PHRASE_MAX_N={value}: {len(misses)} misses ---")
            for sid in misses:
                scen = next(s["scenario_type"] for s in sessions if s["sample_id"] == sid)
                flip = ""
                if value != ref_value:
                    if ref.get(sid) is True:
                        flip = "  <- HIT at PHRASE_MAX_N=%d, now MISS" % ref_value
                    elif ref.get(sid) is False:
                        flip = "  (miss at both)"
                print(f"    {sid}  ({scen}){flip}")
            if value != ref_value:
                newly_fixed = sorted(
                    s["sample_id"] for s in sessions
                    if s["hit"] and ref.get(s["sample_id"]) is False
                )
                if newly_fixed:
                    print(f"    newly FIXED vs PHRASE_MAX_N={ref_value}: {newly_fixed}")


if __name__ == "__main__":
    main()
