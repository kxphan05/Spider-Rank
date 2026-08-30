"""Is the target findable at all, given everything the customer could say?

The simulator's customer has a fixed, small vocabulary -- the whole session can
only ever convey coarse_category(target) + hard_constraints[:2] +
soft_preferences[2:4]. This hands the agent all of it in one turn-1 message.

Result: 173/200, and no target is unreachable -- some leg puts
every one of the 200 inside the top 192. Use it to ask "is this findable", never
"is this the most we could score": it returns one slate of ten where a live
session returns up to ten, and the two conditions differ on 34 samples, 17 in
each direction.

One turn per sample, so far cheaper than a scored run. Writes logs/ceiling.json.

    uv run python3 scripts/eval_ceiling.py [--limit N]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import uuid
from collections import Counter

from _common import DEFAULT_CATALOG, DEFAULT_DATASET, isolate_profile_store  # noqa: F401

from evaluator.local_evaluator import (  # noqa: E402
    TOP_K, catalog_index, coarse_category, load_jsonl, materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent  # noqa: E402

PROBE_N = 2000

# Misses observed at the 0.7020 configuration (logs/failures.log).
# Used only to cross-tabulate ceiling against outcome; not required for the
# ceiling number itself.
KNOWN_MISSES = {
    "public_0008", "public_0012", "public_0016", "public_0018", "public_0028",
    "public_0031", "public_0054", "public_0066", "public_0087", "public_0096",
    "public_0097", "public_0109", "public_0111", "public_0126", "public_0132",
    "public_0142", "public_0144", "public_0146", "public_0151", "public_0159",
    "public_0161", "public_0169", "public_0174", "public_0175", "public_0178",
    "public_0180", "public_0187",
}


def oracle_message(card: dict, category: str) -> str:
    """Everything the customer could ever disclose, in one turn."""
    constraints = [
        *[str(value) for value in card.get("hard_constraints", [])],
        *[str(value) for value in card.get("soft_preferences", [])],
    ]
    unique = list(dict.fromkeys(constraints))
    return f"I'm looking for {category}. " + ". ".join(unique) + "."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="logs/ceiling.json")
    args = parser.parse_args()

    isolate_profile_store()
    samples = load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog)
    if agent.dense is None:
        print("WARNING: dense index dark -- not the shipped configuration (#15)")

    rows = []
    for index, sample in enumerate(samples, 1):
        target = str(sample["ground_truth"]["parent_asin"])
        card, _ = materialize_hidden_fields(sample, products)
        category = coarse_category(categories.get(target, []))
        message = oracle_message(card, category)

        session_id = f"ceil_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        response = agent.respond(session_id, message, 1, TOP_K)
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        oracle_hit = target in ranked

        # How deep is it, if not in the ten? Best rank over the raw legs tells
        # us whether the target is merely mis-ranked or genuinely undescribed.
        legs: dict[str, int | None] = {}
        for name, fn in (("bm25", agent.bm25.search), ("phrase", agent.bm25.phrase_search)):
            try:
                found = fn(message, PROBE_N)
                legs[name] = found.index(target) + 1 if target in found else None
            except Exception:
                legs[name] = None
        legs["dense"] = None
        if agent.dense is not None:
            with contextlib.suppress(Exception):
                found = agent.dense.search(message, PROBE_N)
                legs["dense"] = found.index(target) + 1 if target in found else None
        best = min((r for r in legs.values() if r is not None), default=None)

        rows.append({
            "sample_id": sample["sample_id"],
            "scenario": sample["scenario_type"],
            "oracle_hit": oracle_hit,
            "best_leg_rank": best,
            "legs": legs,
            "n_constraints": len(
                {*card.get("hard_constraints", []), *card.get("soft_preferences", [])}
            ),
            "chars": len(message),
            "message": message,
        })
        print(f"[{index}/{len(samples)}] {sample['sample_id']} {'HIT ' if oracle_hit else 'MISS'}"
              f" best={best} {sample['scenario_type']}", flush=True)

    reachable = sum(r["oracle_hit"] for r in rows)
    n = len(rows)
    print(f"\n=== ceiling: {reachable}/{n} = {reachable / n:.4f} of targets are findable")
    print("    (told every constraint the customer has, on turn 1)\n")

    print(f"{'scenario':<18}{'oracle-reachable':>18}")
    by_scenario: Counter = Counter()
    totals: Counter = Counter()
    for r in rows:
        totals[r["scenario"]] += 1
        by_scenario[r["scenario"]] += int(r["oracle_hit"])
    for scenario in sorted(totals):
        got, tot = by_scenario[scenario], totals[scenario]
        print(f"{scenario:<18}{got:>8}/{tot:<3} {got / tot:>6.3f}")

    unreachable = [r for r in rows if not r["oracle_hit"]]
    print(f"\nOf the {len(unreachable)} unreachable, where does the target sit at all?")
    buckets = Counter()
    for r in unreachable:
        best = r["best_leg_rank"]
        if best is None:
            buckets["no leg finds it in 2000"] += 1
        elif best <= 50:
            buckets["some leg has it in top 50"] += 1
        elif best <= 500:
            buckets["some leg has it in 51-500"] += 1
        else:
            buckets["some leg has it in 501-2000"] += 1
    for label, count in buckets.most_common():
        print(f"  {label:<32}{count:>4}")

    known = [r for r in rows if r["sample_id"] in KNOWN_MISSES]
    if known:
        ours = sum(r["oracle_hit"] for r in known)
        print(f"\nCross-tab against the {len(known)} known misses at 0.7020:")
        print(f"  oracle-reachable but we missed  {ours:>4}   <- ours to win")
        print(f"  not reachable even by oracle    {len(known) - ours:>4}   <- task ceiling")

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
