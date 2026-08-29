"""Sweep the intent-conditioned RRF fusion weights and report score as a curve.

Baseline note: the numbers quoted below were measured when the shipped score
was 0.6070. The shipped score is now 0.7020 (CLAUDE.md #23); re-establish this
sweep's identity point on the HEAD you are sweeping before believing a curve.

Motivation (CLAUDE.md #14): ablating the dense leg entirely *raises*
TechnicalScore 0.6070 -> 0.6216 on the public set, with HitRate identical to
four decimals. We did not drop the leg, because 89.7% of local hard
constraints are verbatim substrings of the target's own catalog text -- the
local set is a pure exact-match benchmark and cannot price the paraphrase
robustness dense exists to provide.

The untried middle is to *retune* rather than remove. The current weights
(buying 2.0/1.0, browsing 1.25/1.5) were set before anyone had checked whether
dense contributes at all.

Only the **ratio** dense:bm25 matters, not the magnitudes: weighted RRF scores
each leg as `weight / (k + rank)`, so scaling both weights by any constant
scales every score identically and leaves the ranking untouched. This script
therefore holds bm25_weight at 1.0 and sweeps the dense weight, which is the
ratio directly. Ratio 0.0 is equivalent to dropping the leg.

The output is deliberately a curve, not a winner. The useful question is not
"which point scores best on this benchmark" -- #14 already establishes that
the benchmark favours no dense at all -- but "how much score is being paid for
paraphrase insurance, and how sharply does that cost change." A flat region
means insurance is nearly free; a steep one means it is not.

Usage:
    uv run python3 scripts/sweep_fusion_weights.py --track buying
    uv run python3 scripts/sweep_fusion_weights.py --track browsing
    uv run python3 scripts/sweep_fusion_weights.py --track both --ratios 0,0.5,1.0
"""
from __future__ import annotations

import argparse
import json
import time

from _common import (DEFAULT_CATALOG, DEFAULT_DATASET, isolate_profile_store,  # noqa: F401
                     score_once)

# _common puts the repo root on sys.path as an import side effect, so the
# starter/evaluator imports below resolve when this script is run directly.
import starter.agent as agent_mod  # noqa: E402
from evaluator.local_evaluator import catalog_index, load_jsonl  # noqa: E402

# Current shipped ratios, for annotating the curve.
CURRENT = {
    "buying": agent_mod.BUYING_DENSE_WEIGHT / agent_mod.BUYING_BM25_WEIGHT,
    "browsing": agent_mod.BROWSING_DENSE_WEIGHT / agent_mod.BROWSING_BM25_WEIGHT,
}
DEFAULT_RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def set_ratio(track: str, ratio: float) -> None:
    """Pin bm25 to 1.0 and dense to `ratio` for the named track(s)."""
    if track in ("buying", "both"):
        agent_mod.BUYING_BM25_WEIGHT = 1.0
        agent_mod.BUYING_DENSE_WEIGHT = ratio
    if track in ("browsing", "both"):
        agent_mod.BROWSING_BM25_WEIGHT = 1.0
        agent_mod.BROWSING_DENSE_WEIGHT = ratio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", choices=["buying", "browsing", "both"], default="buying")
    parser.add_argument("--ratios", default=None,
                        help="comma-separated dense:bm25 ratios (default 0,0.25,...,1.5)")
    parser.add_argument("--limit", type=int, default=None, help="sample cap, for a fast smoke run")
    parser.add_argument("--output", default=None, help="write the full curve as JSON")
    args = parser.parse_args()

    ratios = ([float(r) for r in args.ratios.split(",")] if args.ratios else DEFAULT_RATIOS)

    isolate_profile_store()
    samples = load_jsonl(DEFAULT_DATASET)
    if args.limit:
        samples = samples[: args.limit]
    catalog = catalog_index(DEFAULT_CATALOG)

    print(f"sweeping {args.track} dense:bm25 ratio over {ratios}")
    print(f"currently shipped: buying {CURRENT['buying']:.2f}, browsing {CURRENT['browsing']:.2f}")
    print(f"{len(samples)} samples per point, {len(ratios)} points\n")

    header = f"{'ratio':>7} {'HitRate':>9} {'MRR':>9} {'MTTC':>8} {'Technical':>11} {'vs shipped':>11}  {'mins':>5}"
    print(header)
    print("-" * len(header))

    rows = []
    baseline = None
    for ratio in ratios:
        set_ratio(args.track, ratio)
        start = time.perf_counter()
        report, _ = score_once(samples, catalog)
        elapsed = (time.perf_counter() - start) / 60
        score = report["recommended_technical_score"]
        if abs(ratio - CURRENT[args.track if args.track != "both" else "buying"]) < 1e-9:
            baseline = score
        row = {
            "ratio": ratio,
            "hit_rate_at_10": report["hit_rate_at_10"],
            "mrr": report["mrr"],
            "mttc": report["mttc"],
            "technical_score": score,
            "scenario": {k: v["hit_rate_at_10"] for k, v in report.get("scenario_metrics", {}).items()},
            "minutes": round(elapsed, 1),
        }
        rows.append(row)
        delta = "" if baseline is None else f"{score - baseline:+.4f}"
        print(f"{ratio:7.2f} {row['hit_rate_at_10']:9.4f} {row['mrr']:9.4f} "
              f"{row['mttc']:8.3f} {score:11.4f} {delta:>11}  {elapsed:5.1f}", flush=True)

    print("\ncurve (TechnicalScore vs dense:bm25 ratio):")
    lo = min(r["technical_score"] for r in rows)
    hi = max(r["technical_score"] for r in rows)
    span = (hi - lo) or 1.0
    for r in rows:
        bar = "#" * int(1 + 48 * (r["technical_score"] - lo) / span)
        mark = "  <- shipped" if abs(r["ratio"] - CURRENT.get(args.track, -1)) < 1e-9 else ""
        print(f"  {r['ratio']:5.2f} |{bar:<49} {r['technical_score']:.4f}{mark}")

    if args.output:
        with open(args.output, "w") as fh:
            json.dump({"track": args.track, "samples": len(samples),
                       "shipped_ratios": CURRENT, "curve": rows}, fh, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
