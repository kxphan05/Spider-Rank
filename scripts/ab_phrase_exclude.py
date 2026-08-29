"""Full-set A/B for the phrase leg and shown-item exclusion (CLAUDE.md #20, #23).

Both need the same legs, so they run together. Measured churn before the fix:
5.33 of 10 slots repeated each turn.

    uv run python3 scripts/ab_phrase_exclude.py
"""
from __future__ import annotations

import sys

from _common import DEFAULT_CATALOG, DEFAULT_DATASET, isolate_profile_store, score_once  # noqa: F401

from evaluator.local_evaluator import catalog_index, load_jsonl  # noqa: E402
from starter import agent as agent_module  # noqa: E402

LEGS = (
    ("phrase 0.0 (old ship)", 0.0, False),
    ("phrase 2.0 (new ship)", 2.0, False),
    ("phrase 2.0 + exclude", 2.0, True),
)


def main() -> None:
    isolate_profile_store()
    samples = load_jsonl(str(DEFAULT_DATASET))
    catalog = catalog_index(str(DEFAULT_CATALOG))
    n = len(samples)

    rows = []
    for name, weight, exclude in LEGS:
        agent_module.PHRASE_WEIGHT = weight
        agent_module.EXCLUDE_SHOWN = exclude
        report, agent = score_once(samples, catalog)
        if agent.dense is None:
            print("  WARNING: dense dark -- not comparable (#15)", file=sys.stderr)
        rows.append((name, report))
        print(f"[done] {name}  technical={report['recommended_technical_score']:.4f}",
              file=sys.stderr, flush=True)

    print(f"\n{'leg':<24} {'HitRate':>8} {'hits':>9} {'MRR':>7} {'MTTC':>7} {'Technical':>9} {'delta':>8}")
    base = rows[0][1]["recommended_technical_score"]
    for name, report in rows:
        tech = report["recommended_technical_score"]
        hits = round(report["hit_rate_at_10"] * n)
        delta = "--" if tech == base else f"{tech - base:+.4f}"
        print(f"{name:<24} {report['hit_rate_at_10']:8.4f} {hits:4d}/{n}"
              f" {report['mrr']:7.4f} {report['mttc']:7.3f} {tech:9.4f} {delta:>8}")

    scenarios = sorted(rows[0][1]["scenario_metrics"])
    print(f"\n{'scenario':<18}" + "".join(f"{name:>26}" for name, _ in rows))
    for scenario in scenarios:
        cells = ""
        for _, report in rows:
            m = report["scenario_metrics"][scenario]
            cnt = m["sample_count"]
            cells += f"{m['hit_rate_at_10']:>10.3f} ({round(m['hit_rate_at_10']*cnt):2d}/{cnt:2d}) MRR{m['mrr']:5.3f}"
        print(f"{scenario:<18}{cells}")


if __name__ == "__main__":
    main()
