"""A/B the two independent defects in the phrase leg's span budget.

`phrase_search` enumerates n-grams longest-first and queries only the first
`MAX_PHRASE_QUERIES` of them. On a 10-turn session that budget is spent before
reaching a span that exists in the catalog -- public_0008's final query builds
135 spans, and the 24 longest are all 5-grams of conversational filler matching
zero products, while "bras everyday bras" (verbatim in the target) is a 3-gram
and never gets queried.

Two separate causes, measured separately so we know which one pays:

  clause   build spans within a clause and never on a stopword edge, so a
           window cannot slide across "requirement is: nylon" and a fragment
           like "i m looking for" cannot consume a lookup. Generalizes: both
           are properties of language, not of this simulator's wording.
  filter   drop contentless replies from the phrase query only, via the
           existing EmbeddingNonAnswerDetector (PHRASE_QUERY_SKIP_NON_ANSWERS).
           Narrower than the whole-query version that measured -0.0410
           (CLAUDE.md #19) -- BM25 and dense keep every character, so the
           exact-match text that loss was made of is never removed.
  budget   simply raise MAX_PHRASE_QUERIES. The blunt control: if this alone
           captures the gain, the classifier is not earning its place.

Identity must reproduce the shipped score before any other leg is believed.

    uv run python3 scripts/ab_phrase_query.py
"""
from __future__ import annotations

import argparse

from _common import DEFAULT_CATALOG, DEFAULT_DATASET, isolate_profile_store, score_once  # noqa: F401

from evaluator.local_evaluator import catalog_index, load_jsonl  # noqa: E402
from starter import agent as agent_module, retrieval  # noqa: E402

# (label, PHRASE_QUERY_SKIP_NON_ANSWERS, PHRASE_CLAUSE_SPANS, MAX_PHRASE_QUERIES)
LEGS = (
    ("identity", False, False, 24),
    ("filter only", True, False, 24),
    ("clause+edge only", False, True, 24),
    ("filter + clause+edge", True, True, 24),
    ("budget 96 (control)", False, False, 96),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    isolate_profile_store()
    samples = load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog = catalog_index(args.catalog)

    print(f"{'leg':22s} {'HitRate':>8} {'hits':>9}  {'MRR':>7}  {'MTTC':>7}  {'Technical':>9}  {'delta':>8}")
    baseline = None
    for label, skip, clause, budget in LEGS:
        agent_module.PHRASE_QUERY_SKIP_NON_ANSWERS = skip
        retrieval.PHRASE_CLAUSE_SPANS = clause
        retrieval.MAX_PHRASE_QUERIES = budget
        report, agent = score_once(samples, catalog)
        if agent.dense is None or agent.non_answer_detector is None:
            print("  WARNING: a component is dark -- not the shipped configuration (#15)")
        technical = report["recommended_technical_score"]
        hits = round(report["hit_rate_at_10"] * len(samples))
        if baseline is None:
            baseline = technical
        delta = "--" if label == LEGS[0][0] else f"{technical - baseline:+.4f}"
        print(f"{label:22s} {report['hit_rate_at_10']:8.4f} {hits:4d}/{len(samples)}"
              f"  {report['mrr']:7.4f}  {report['mttc']:7.3f}  {technical:9.4f}  {delta:>8}",
              flush=True)
        for name in sorted(report["scenario_metrics"]):
            metrics = report["scenario_metrics"][name]
            print(f"    {name:18s} hit {metrics['hit_rate_at_10']:.3f}  "
                  f"mrr {metrics['mrr']:.3f}  mttc {metrics['mttc']:.3f}")


if __name__ == "__main__":
    main()
