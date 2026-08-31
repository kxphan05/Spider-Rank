"""Sweep NEGATION_WINDOW_CHARS, the lookback distance _is_negated scans for a
negation marker before a matched cue/vocab word.

Motivated by a test-writing find: the module comment on BUYING_PHRASES claims
"not looking to buy yet" reads browsing via _is_negated, but at the shipped
value (10) the window is too short to bridge "not ... to buy" once any words
sit between them -- classify_intent actually returns "buying" for that exact
phrase (see tests/test_classifier.py's xfail on this). The fix is only a real
fix if the wider window doesn't cost score elsewhere (a wider window also
widens the reach of every other negation check: extract_disclosed_value's
material/color negation, _vocab_hits_positive, etc.) -- so it's measured here
before being shipped, matching this project's own measurement discipline for
every other tunable in this file.

NEGATION_WINDOW_CHARS = 10 is the shipped value and must reproduce the
current score first.

    uv run python3 scripts/sweep_negation_window.py
"""
from __future__ import annotations

import sys

from _common import DEFAULT_CATALOG, DEFAULT_DATASET, isolate_profile_store, score_once  # noqa: F401

from evaluator.local_evaluator import catalog_index, load_jsonl  # noqa: E402
from starter import classifier as classifier_module  # noqa: E402

DEFAULT_VALUES = (10, 15, 20, 25, 30)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--values", type=int, nargs="+", default=list(DEFAULT_VALUES))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    isolate_profile_store()
    samples = load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog = catalog_index(args.catalog)
    n = len(samples)

    print(f"{'window':>7}  {'HitRate':>8} {'hits':>9}  {'MRR':>7}  {'MTTC':>7}"
          f"  {'Technical':>9}  {'delta':>8}  {'buying':>8}  {'browsing':>9}", flush=True)
    baseline = None
    for value in args.values:
        classifier_module.NEGATION_WINDOW_CHARS = value
        report, agent = score_once(samples, catalog)
        if agent.dense is None:
            print("  WARNING: dense dark -- not comparable to shipped numbers", file=sys.stderr)
        technical = report["recommended_technical_score"]
        hits = round(report["hit_rate_at_10"] * n)
        if baseline is None:
            baseline = technical
            delta = "--"
        else:
            delta = f"{technical - baseline:+.4f}"
        scenarios = report["scenario_metrics"]
        buying = scenarios.get("buying", {}).get("hit_rate_at_10", float("nan"))
        browsing = scenarios.get("browsing", {}).get("hit_rate_at_10", float("nan"))
        print(f"{value:7d}  {report['hit_rate_at_10']:8.4f} {hits:4d}/{n}"
              f"  {report['mrr']:7.4f}  {report['mttc']:7.3f}  {technical:9.4f}"
              f"  {delta:>8}  {buying:8.3f}  {browsing:9.3f}", flush=True)


if __name__ == "__main__":
    main()
