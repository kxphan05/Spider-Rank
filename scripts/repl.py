"""Interactive REPL for manually testing the Agent against a single session.

Usage:
    uv run python3 scripts/repl.py

Commands (typed instead of a message):
    /reset        start a new session (fresh SessionState)
    /debug        toggle printing internal state (disclosed, asked_attributes) after each turn
    /topk N       change how many recommendations are requested/shown (default 10)
    /quit, /exit  leave the REPL

Anything else is sent as the customer's next message, same as the evaluator
would send one -- turn count auto-increments and mirrors the evaluator's
MAX_TURNS=10 cap (shown as a warning once you cross it; the agent itself
doesn't enforce it, only the grader does).
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from _common import DEFAULT_CATALOG, DEFAULT_DATASET, REPO_ROOT  # noqa: F401

# _common puts the repo root on sys.path as an import side effect, so the
# starter/evaluator imports below resolve when this script is run directly.

from starter.agent import Agent

MAX_TURNS = 10
DEFAULT_TOP_K = 10


def load_catalog_lookup(catalog_path: Path) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            product = json.loads(line)
            lookup[str(product["parent_asin"])] = product
    return lookup


def format_recommendation(pid: str, catalog: dict[str, dict]) -> str:
    product = catalog.get(pid)
    if product is None:
        return f"  {pid}  (not found in catalog)"
    title = str(product.get("title", ""))[:80]
    price = product.get("price")
    price_str = f"${price}" if isinstance(price, (int, float)) else "no price"
    return f"  {pid}  [{price_str}]  {title}"


def main() -> None:
    print("Loading agent (embedding model + dense index)...", file=sys.stderr)
    agent = Agent()
    catalog = load_catalog_lookup(DEFAULT_CATALOG)

    session_id = uuid.uuid4().hex
    agent.reset(session_id, {})
    turn = 0
    top_k = DEFAULT_TOP_K
    debug = False

    print("\nReady. Type a message to start the conversation (empty first message not allowed).")
    print("Commands: /reset  /debug  /topk N  /quit\n")

    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_message:
            continue
        if user_message in ("/quit", "/exit"):
            break
        if user_message == "/reset":
            session_id = uuid.uuid4().hex
            agent.reset(session_id, {})
            turn = 0
            print("-- session reset --\n")
            continue
        if user_message == "/debug":
            debug = not debug
            print(f"-- debug={'on' if debug else 'off'} --\n")
            continue
        if user_message.startswith("/topk"):
            parts = user_message.split()
            if len(parts) == 2 and parts[1].isdigit():
                top_k = int(parts[1])
                print(f"-- top_k={top_k} --\n")
            else:
                print("usage: /topk N\n")
            continue

        turn += 1
        if turn > MAX_TURNS:
            print(f"-- turn {turn} exceeds the evaluator's MAX_TURNS={MAX_TURNS}; "
                  "the real grader would already have scored this session as a miss --")

        response = agent.respond(session_id, user_message, turn, top_k)

        print(f"\nAgent [{response['message']!r}]  ask_attribute={response['ask_attribute']!r}")
        for rec in response["recommendations"]:
            print(format_recommendation(rec["parent_asin"], catalog))

        if debug:
            state = agent._sessions[session_id]
            print(f"  [debug] disclosed={state.disclosed}  profile_hint={state.profile_hint}")
            print(f"  [debug] asked_attributes={state.asked_attributes}")
            print(f"  [debug] profile_key={state.profile_key}  session_index={state.profile_session_index}")
        print()


if __name__ == "__main__":
    main()
