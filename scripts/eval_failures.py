"""Dump and classify the sessions the shipped agent misses.

Replays the evaluator's loop turn-for-turn, then probes each retrieval leg to
depth PROBE_N with that session's final query:

    unreachable       no leg ranks the target in the top PROBE_N
    lost-in-fusion    a leg finds it deep, the fused pool does not
    ranked-11+        in the pool, outside the ten returned
    excluded          pushed out by EXCLUDE_SHOWN (only if an override is missed)

Don't trust `lost-in-fusion` at face value: it means "not in the pool",
not "fusion dropped it". Takes about an hour. Writes logs/failures.json.

    uv run python3 scripts/eval_failures.py [--limit N] [--out failures.json]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import uuid
from collections import Counter

from _common import DEFAULT_CATALOG, DEFAULT_DATASET, isolate_profile_store  # noqa: F401

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS, TOP_K, catalog_index, coarse_category, customer_reply, initial_message,
    load_jsonl, materialize_hidden_fields, normalize_recommendations,
)
from starter import agent as agent_module  # noqa: E402
from starter.agent import Agent, _build_query  # noqa: E402

PROBE_N = 2000


def probe(agent: Agent, query: str, target: str) -> dict:
    """Rank of `target` in each leg, at depth PROBE_N. None = not found."""
    out: dict[str, int | None] = {}

    def rank_of(ranked: list[str]) -> int | None:
        return ranked.index(target) + 1 if target in ranked else None

    try:
        out["bm25"] = rank_of(agent.bm25.search(query, PROBE_N))
    except Exception:
        out["bm25"] = None
    try:
        out["phrase"] = rank_of(agent.bm25.phrase_search(query, PROBE_N))
    except Exception:
        out["phrase"] = None
    out["dense"] = None
    if agent.dense is not None:
        with contextlib.suppress(Exception):
            out["dense"] = rank_of(agent.dense.search(query, PROBE_N))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="logs/failures.json")
    args = parser.parse_args()

    isolate_profile_store()
    samples = load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog)
    if agent.dense is None:
        print("WARNING: dense index dark -- not the shipped configuration (#15)")

    misses: list[dict] = []
    hits = 0
    for index, sample in enumerate(samples):
        session_id = f"fail_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)
        transcript: list[dict] = []
        hit_turn = None
        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, user_message, turn, TOP_K)
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            transcript.append({
                "turn": turn,
                "customer": user_message,
                "ask": response.get("ask_attribute"),
                "target_in_slate": target in ranked,
            })
            if override_applied and target in ranked:
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break
            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                user_message, boundary_used = customer_reply(
                    effective, response.get("ask_attribute"), disclosed, boundary_used
                )
        if hit_turn is not None:
            hits += 1
            continue

        state = agent._sessions[session_id]
        query = _build_query(state)
        legs = probe(agent, query, target)
        pool_size = max(TOP_K, agent_module.ENTROPY_POOL_SIZE) + (
            len(state.shown) if agent_module.EXCLUDE_SHOWN else 0)
        fused = agent._retrieve(query, TOP_K, pool_size, "buying", state.disclosed, {},
                               state.disclosed_turn, MAX_TURNS, state.belief)
        fused_rank = fused.index(target) + 1 if target in fused else None
        if target in state.shown:
            reason = "excluded-as-shown"
        elif all(value is None for value in legs.values()):
            reason = "unreachable"
        elif fused_rank is None:
            reason = "lost-in-fusion"
        else:
            reason = "ranked-11+"
        misses.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "difficulty": sample.get("difficulty_bucket"),
            "reason": reason,
            "legs": legs,
            "fused_rank": fused_rank,
            "pool_size": pool_size,
            "query": query,
            "disclosed": dict(state.disclosed),
            "asked": list(state.asked_attributes),
            "target_title": str(products[target].get("title", ""))[:160],
            "transcript": transcript,
        })
        print(f"[{index + 1}/{len(samples)}] MISS {sample['sample_id']} "
              f"{sample['scenario_type']:16s} {reason:18s} legs={legs} fused={fused_rank}", flush=True)

    print(f"\n{hits}/{len(samples)} hits, {len(misses)} misses")
    print("\nby reason:")
    for reason, count in Counter(item["reason"] for item in misses).most_common():
        print(f"  {reason:20s} {count:3d}")
    print("\nby scenario:")
    for scenario, count in Counter(item["scenario_type"] for item in misses).most_common():
        print(f"  {scenario:20s} {count:3d}")
    print("\nreachable-by-leg among misses (rank at depth 2000):")
    for leg in ("bm25", "phrase", "dense"):
        found = [item["legs"][leg] for item in misses if item["legs"][leg] is not None]
        print(f"  {leg:8s} found in {len(found):3d}/{len(misses)}"
              + (f", median rank {sorted(found)[len(found) // 2]}" if found else ""))

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(misses, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
