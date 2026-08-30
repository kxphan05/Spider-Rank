"""Sweep the entropy:answerability ratio in the weighted question picker.

Tests WEIGHTED_QUESTION_SCORE (attributes.select_weighted_attribute) against the
shipped entropy-gate-then-fallback-list structure. Only the ratio matters -- the
combined score is a weighted mean -- so answerability is pinned at 1.0 and the
entropy weight varies. Two points are controls, not candidates: `identity` is the
flag off and must reproduce the shipped score, and entropy weight 0.0 is pure
answerability with no informativeness term at all.

Designed to survive an unattended overnight run with the network off. Every leg
is appended to the log and the JSON as it finishes, so a killed run keeps its
completed points.

    uv run python3 scripts/sweep_question_weights.py --log logs/question_weights.log
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from _common import (DEFAULT_CATALOG, DEFAULT_DATASET, isolate_profile_store,  # noqa: F401
                     score_once)

# _common puts the repo root on sys.path as an import side effect.
import starter.agent as agent_mod  # noqa: E402
from evaluator.local_evaluator import catalog_index, load_jsonl  # noqa: E402

# Shipped score at a9d3999, the number `identity` must reproduce (CLAUDE.md).
SHIPPED = 0.744094
DEFAULT_WEIGHTS = [0.0, 0.5, 1.0, 2.0]


def set_point(entropy_weight: float | None) -> str:
    """Configure one leg. None means the flag off -- the shipped picker."""
    if entropy_weight is None:
        agent_mod.WEIGHTED_QUESTION_SCORE = False
        return "identity"
    agent_mod.WEIGHTED_QUESTION_SCORE = True
    agent_mod.QUESTION_ANSWERABILITY_WEIGHT = 1.0
    agent_mod.QUESTION_ENTROPY_WEIGHT = entropy_weight
    return f"we={entropy_weight:g}"


def live_components(agent) -> dict[str, bool]:
    """Which optional components actually came up (CLAUDE.md #15: silent degrade)."""
    return {
        "dense": getattr(agent, "dense", None) is not None,
        "intent_classifier": getattr(agent, "intent_classifier", None) is not None,
        "override_detector": getattr(agent, "override_detector", None) is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=None,
                        help="comma-separated entropy weights (default 0,0.5,1,2)")
    parser.add_argument("--limit", type=int, default=None, help="sample cap, for a smoke run")
    parser.add_argument("--log", default="logs/question_weights.log",
                        help="append-as-you-go text log")
    parser.add_argument("--output", default=None,
                        help="JSON results (default: the log path with a .json suffix)")
    parser.add_argument("--allow-degraded", action="store_true",
                        help="continue even if the dense leg or a classifier is dark")
    args = parser.parse_args()

    weights = ([float(w) for w in args.weights.split(",")] if args.weights
               else list(DEFAULT_WEIGHTS))
    # The control runs first: a sweep whose identity leg is wrong tells you
    # nothing about the points measured after it.
    points: list[float | None] = [None, *weights]

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(args.output) if args.output else log_path.with_suffix(".json")

    isolate_profile_store()
    samples = load_jsonl(DEFAULT_DATASET)
    if args.limit:
        samples = samples[: args.limit]
    catalog = catalog_index(DEFAULT_CATALOG)
    # Build one agent before any scoring: a missing asset degrades silently
    # (CLAUDE.md #15) and this is the difference between finding out in two
    # minutes and finding out after a two-hour leg has already run.

    rows: list[dict] = []
    baseline: float | None = None
    with log_path.open("a") as log_file:
        for entropy_weight in points:
            label = set_point(entropy_weight)
            start = time.perf_counter()
            report, agent = score_once(samples, catalog)
            elapsed = (time.perf_counter() - start) / 60
            score = report["recommended_technical_score"]

            components = live_components(agent)
            if not all(components.values()) and not args.allow_degraded:
                raise SystemExit(
                    f"degraded components on leg {label!r}: {components} "
                    "-- pass --allow-degraded to continue anyway")

            hits = round(report["hit_rate_at_10"] * len(samples))
            is_identity_leg = baseline is None
            delta = "--" if is_identity_leg else f"{score - baseline:+.4f}"
            if is_identity_leg:
                baseline = score

            rows.append({
                "point": label,
                "entropy_weight": entropy_weight,
                "answerability_weight": None if entropy_weight is None else 1.0,
                "hit_rate_at_10": report["hit_rate_at_10"],
                "hits": hits,
                "mrr": report["mrr"],
                "mttc": report["mttc"],
                "technical_score": score,
                "delta_vs_identity": None if is_identity_leg else score - baseline,
                "scenario_metrics": report.get("scenario_metrics"),
                "minutes": elapsed,
            })
            log_file.write(
                f"{label:12s} hits={hits:3d} mrr={report['mrr']:.4f} "
                f"mttc={report['mttc']:.3f} score={score:.6f} delta={delta} "
                f"({elapsed:.1f} min)\n")
            log_file.flush()
            # Rewritten after every leg so a killed run keeps what it finished.
            json_path.write_text(json.dumps(
                {"shipped": SHIPPED, "sample_count": len(samples), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
