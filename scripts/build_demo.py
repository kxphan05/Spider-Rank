"""Build the session-replay demo page, and optionally re-capture its data.

The demo replays real agent sessions -- `--capture` drives the live Agent through
the evaluator's own customer simulator and records every turn. Nothing on the
page is mocked. Capture needs the catalog, dense index and both models loaded;
rendering is instant, so the captured sessions are committed
(`demo/sessions.json`) and the page rebuilds without them.

UI is out of scope for scoring (TODO.md 4.3). This is for presenting.

    uv run python3 scripts/build_demo.py [--capture] [--ids public_0013]
"""
from __future__ import annotations

import argparse
import json

from _common import DEFAULT_CATALOG, DEFAULT_DATASET, REPO_ROOT, isolate_profile_store  # noqa: F401

DEMO_DIR = REPO_ROOT / "demo"
TEMPLATE = DEMO_DIR / "template.html"
SESSIONS = DEMO_DIR / "sessions.json"
OUT = REPO_ROOT / "dist" / "demo.html"

# One session per scenario. The boundary miss is kept deliberately: it is the
# weakest scenario (0.500 hit rate) and a demo that only shows wins invites the
# question you least want asked live.
SCENARIO_ORDER = ["buying", "browsing", "intent_override", "boundary"]
CANDIDATES_PER_SCENARIO = 6


def product_card(pid: str, products: dict, categories: dict, coarse_category, target: str) -> dict:
    """One recommendation row for the demo page."""
    p = products.get(pid, {})
    price = p.get("price")
    return {"asin": pid, "title": (p.get("title") or "")[:110],
            "rating": p.get("average_rating"), "ratings": p.get("rating_number"),
            "price": price if isinstance(price, (int, float)) else None,
            "category": coarse_category(categories.get(pid, [])),
            "is_target": pid == target}


def capture(ids: list[str] | None) -> list[dict]:
    """Drive the real Agent through the evaluator's own session loop."""
    isolate_profile_store(announce=False)
    from evaluator.local_evaluator import (MAX_TURNS, TOP_K, catalog_index, coarse_category,
                                           customer_reply, initial_message, load_jsonl,
                                           materialize_hidden_fields, normalize_recommendations)
    from starter.agent import Agent
    from starter.classifier import classify_intent

    samples = load_jsonl(DEFAULT_DATASET)
    catalog_ids, categories, products = catalog_index(DEFAULT_CATALOG)

    if ids:
        chosen = [s for s in samples if s["sample_id"] in set(ids)]
    else:
        buckets: dict[str, list] = {k: [] for k in SCENARIO_ORDER}
        for s in samples:
            b = buckets.get(s["scenario_type"])
            if b is not None and len(b) < CANDIDATES_PER_SCENARIO:
                b.append(s)
        chosen = [s for k in SCENARIO_ORDER for s in buckets[k]]

    agent = Agent()
    captured = []
    for sample in chosen:
        sid = f"demo_{sample['sample_id']}"
        agent.reset(sid, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behav = materialize_hidden_fields(sample, products)
        eff = {**sample, "intent_card": card, "behavior": behav}
        disclosed, boundary_used = set(), False
        override_applied = sample["scenario_type"] != "intent_override"
        message = initial_message(eff, coarse_category(categories.get(target, [])), disclosed)
        turns, hit_turn, rank_at_hit = [], None, None

        for turn in range(1, MAX_TURNS + 1):
            state = agent._sessions[sid]
            before = dict(state.disclosed)
            response = agent.respond(sid, message, turn, TOP_K)
            after = dict(state.disclosed)
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)

            def card_for(pid: str, _t: str = target) -> dict:
                return product_card(pid, products, categories, coarse_category, _t)

            turns.append({
                "turn": turn, "user": message, "agent": response.get("message"),
                "ask": response.get("ask_attribute"),
                "intent": classify_intent(message).label,
                "disclosed_before": before, "disclosed_after": after,
                "new_slots": {k: v for k, v in after.items() if before.get(k) != v},
                # Filled in afterwards from the real detector -- inferring it
                # from a shrinking slot dict is wrong, because an override
                # clears the slots and the same turn can immediately refill one.
                "override_fired": False,
                "recs": [card_for(p) for p in ranked[:10]],
                "target_rank": ranked.index(target) + 1 if target in ranked else None,
            })

            if override_applied and target in ranked:
                hit_turn, rank_at_hit = turn, turns[-1]["target_rank"]
                break
            if turn == MAX_TURNS:
                break
            override = eff.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                message, boundary_used = customer_reply(eff, response.get("ask_attribute"), disclosed, boundary_used)

        tp = products.get(target, {})
        captured.append({
            "sample_id": sample["sample_id"], "scenario": sample["scenario_type"],
            "profile": sample["user_profile"],
            "target": {"asin": target, "title": (tp.get("title") or "")[:130],
                       "category": coarse_category(categories.get(target, [])),
                       "rating": tp.get("average_rating"), "ratings": tp.get("rating_number")},
            "turns": turns, "hit_turn": hit_turn, "rank_at_hit": rank_at_hit,
        })
        print(f"  {sample['scenario_type']:16s} {sample['sample_id']} "
              f"turns={len(turns)} hit={hit_turn} rank={rank_at_hit}", flush=True)

    # Ask the real detector which turns are overrides.
    from starter.classifier import EmbeddingOverrideDetector
    from starter.retrieval import load_embedding_model
    detector = EmbeddingOverrideDetector(load_embedding_model())
    for session in captured:
        for i, t in enumerate(session["turns"]):
            # Turn 1 can never be an override; the agent only checks from turn 2.
            t["override_fired"] = bool(i > 0 and detector.is_override(t["user"]))

    if ids:
        return captured
    # Keep the earliest hit per scenario, plus one honest miss.
    keep, seen_hit, seen_miss = [], set(), set()
    for rec in sorted(captured, key=lambda r: (r["hit_turn"] is None, r["hit_turn"] or 99)):
        st = rec["scenario"]
        if rec["hit_turn"] is not None and st not in seen_hit:
            seen_hit.add(st); keep.append(rec)
        elif rec["hit_turn"] is None and st not in seen_miss and not seen_miss:
            seen_miss.add(st); keep.append(rec)
    keep.sort(key=lambda r: SCENARIO_ORDER.index(r["scenario"]))
    return keep


def render() -> None:
    sessions = json.loads(SESSIONS.read_text())
    html = TEMPLATE.read_text().replace("/*__SESSIONS__*/null", json.dumps(sessions, separators=(",", ":")))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT.relative_to(REPO_ROOT)} ({len(html) / 1024:.0f} KB, {len(sessions)} sessions)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", action="store_true", help="re-run the agent and overwrite demo/sessions.json")
    parser.add_argument("--ids", nargs="*", default=None, help="capture these sample_ids instead of auto-selecting")
    args = parser.parse_args()

    if args.capture:
        print("capturing live sessions...")
        sessions = capture(args.ids)
        SESSIONS.write_text(json.dumps(sessions, indent=1))
        print(f"wrote {SESSIONS.relative_to(REPO_ROOT)}")
    render()


if __name__ == "__main__":
    main()
