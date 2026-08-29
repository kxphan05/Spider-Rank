"""Sweep the disclosed-attribute resort weights (CLAUDE.md #10, NEXT_STEPS #2).

_boost_by_disclosed scores +DISCLOSED_MATCH_BOOST on agreement,
-DISCLOSED_MISMATCH_PENALTY on contradiction, 0 when unextractable. The free
parameter that matters is the **ratio** -- how much a known contradiction is
worth against a known agreement. They are equal by assumption, never by
measurement, and the +-1 scheme was tuned when colour coverage was 58.6% and a
fifth of it was fictional. The word-boundary fix cut coverage to 39.9% and
currently costs 0.0054; retuning against corrected coverage is the indicated
move, not reverting the fix.

Scale caveat: the ratio is the only free parameter **while the masked LM is
dark**. Live, match_score also adds LM_INFERENCE_WEIGHT, so absolute magnitude
sets how a disclosure trades against an inference. `--scale` sweeps that axis;
it is a no-op with the LM absent and the script reports which regime it ran in.

    uv run python3 scripts/sweep_boost_weights.py [--penalties 0,0.5,1,2] [--scale 0.5,1,2]
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

SHIPPED_MATCH = agent_mod.DISCLOSED_MATCH_BOOST
SHIPPED_PENALTY = agent_mod.DISCLOSED_MISMATCH_PENALTY
SHIPPED_RATIO = SHIPPED_PENALTY / SHIPPED_MATCH

DEFAULT_PENALTIES = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]


def set_weights(match: float, penalty: float) -> None:
    agent_mod.DISCLOSED_MATCH_BOOST = match
    agent_mod.DISCLOSED_MISMATCH_PENALTY = penalty


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--penalties", default=None,
                        help="comma-separated mismatch penalties, match boost pinned to --scale "
                             "(default 0,0.25,0.5,1,1.5,2,3)")
    parser.add_argument("--scale", default="1.0",
                        help="comma-separated overall magnitudes for the match boost; only "
                             "distinguishable when the masked LM is live (default 1.0)")
    parser.add_argument("--limit", type=int, default=None, help="sample cap, for a fast smoke run")
    parser.add_argument("--output", default=None, help="write the full curve as JSON")
    args = parser.parse_args()

    penalties = ([float(p) for p in args.penalties.split(",")]
                 if args.penalties else DEFAULT_PENALTIES)
    scales = [float(s) for s in args.scale.split(",")]

    isolate_profile_store()
    samples = load_jsonl(DEFAULT_DATASET)
    if args.limit:
        samples = samples[: args.limit]
    catalog = catalog_index(DEFAULT_CATALOG)

    points = [(s, s * p) for s in scales for p in penalties]
    print(f"sweeping penalty:match over {penalties}, match magnitude over {scales}")
    print(f"currently shipped: match {SHIPPED_MATCH:.2f}, penalty {SHIPPED_PENALTY:.2f} "
          f"(ratio {SHIPPED_RATIO:.2f})")
    print(f"{len(samples)} samples per point, {len(points)} points\n")

    header = (f"{'match':>7} {'penalty':>8} {'ratio':>7} {'HitRate':>9} {'MRR':>9} "
              f"{'MTTC':>8} {'Technical':>11} {'vs shipped':>11}  {'mins':>5}")
    print(header)
    print("-" * len(header))

    rows, baseline, lm_live = [], None, None
    for match, penalty in points:
        set_weights(match, penalty)
        start = time.perf_counter()
        report, agent = score_once(samples, catalog)
        lm_live = getattr(agent, "lm_scorer", None) is not None
        elapsed = (time.perf_counter() - start) / 60
        score = report["recommended_technical_score"]
        shipped_point = (abs(match - SHIPPED_MATCH) < 1e-9
                         and abs(penalty - SHIPPED_PENALTY) < 1e-9)
        if shipped_point:
            baseline = score
        row = {
            "match": match,
            "penalty": penalty,
            "ratio": penalty / match if match else float("inf"),
            "hit_rate_at_10": report["hit_rate_at_10"],
            "hits": round(report["hit_rate_at_10"] * len(samples)),
            "mrr": report["mrr"],
            "mttc": report["mttc"],
            "technical_score": score,
            "scenario": {k: v["hit_rate_at_10"] for k, v in report.get("scenario_metrics", {}).items()},
            "minutes": round(elapsed, 1),
        }
        rows.append(row)
        delta = "" if baseline is None else f"{score - baseline:+.4f}"
        print(f"{match:7.2f} {penalty:8.2f} {row['ratio']:7.2f} {row['hit_rate_at_10']:9.4f} "
              f"{row['mrr']:9.4f} {row['mttc']:8.3f} {score:11.4f} {delta:>11}  "
              f"{elapsed:5.1f}", flush=True)

    print(f"\nmasked LM was {'LIVE -- the --scale axis is real' if lm_live else 'DARK -- only the ratio matters, --scale is a no-op'}")

    print("\ncurve (TechnicalScore vs penalty:match ratio):")
    lo = min(r["technical_score"] for r in rows)
    hi = max(r["technical_score"] for r in rows)
    span = (hi - lo) or 1.0
    for r in rows:
        bar = "#" * int(1 + 48 * (r["technical_score"] - lo) / span)
        shipped = (abs(r["match"] - SHIPPED_MATCH) < 1e-9
                   and abs(r["penalty"] - SHIPPED_PENALTY) < 1e-9)
        mark = "  <- shipped" if shipped else ""
        print(f"  {r['ratio']:5.2f} |{bar:<49} {r['technical_score']:.4f}"
              f"  ({r['hits']}/{len(samples)} hits){mark}")

    if args.output:
        with open(args.output, "w") as fh:
            json.dump({"samples": len(samples), "lm_live": lm_live,
                       "shipped": {"match": SHIPPED_MATCH, "penalty": SHIPPED_PENALTY},
                       "curve": rows}, fh, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
