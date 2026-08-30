"""Sweep the dense:bm25 RRF ratio per track.

Only the ratio matters -- weighted RRF is scale-invariant per leg -- so bm25 is
pinned at 1.0. The buying curve is strictly monotone decreasing with no interior
optimum; the browsing curve is flat across 0.0-1.5 and must not be re-tuned.

    uv run python3 scripts/sweep_fusion_weights.py --track buying
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
