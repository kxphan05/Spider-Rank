"""Streamlit demo UI for the TechJam Track 4 shopping agent.

Not part of the scored submission. `TODO.md` § 4.3 puts UI out of scope, and
nothing here touches `starter/`, so the agent this renders is byte-identical to
the one the evaluator scores. It exists so the pipeline can be *shown* rather
than described: the interesting part of this system is the state it keeps
between turns, and a score line hides all of it.

What it surfaces that a terminal REPL cannot:

  * the slate, as products, updating live
  * which attributes the agent has extracted, and on which turn
  * the buying/browsing route and the fusion weights that route chose
  * whether an intent override fired -- read from the real detector, not
    guessed by watching the slot dict shrink (an override clears the slots and
    the same turn can immediately refill one, so the naive heuristic misses it)

Run it:

    uv run --with streamlit streamlit run demo/streamlit_app.py

First load takes ~30 seconds: it builds the FTS5 index over 50,000 products
and loads the encoder. Both are cached for the session afterwards.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starter.agent import Agent, routing_params  # noqa: E402

# The demo reads `agent._sessions` directly. That is a private attribute, and
# reaching into it is deliberate: adding a public accessor purely for a demo
# would change code that ships into the scored bundle, and this file is
# explicitly out of scope. If the attribute is ever renamed, this panel
# breaks loudly rather than showing stale state.

CATALOG = REPO_ROOT / "data" / "catalog.jsonl"
MAX_TURNS = 10

st.set_page_config(page_title="TechJam Shopping Agent", page_icon="🛍️", layout="wide")


@st.cache_resource(show_spinner="Building the index and loading models…")
def load_agent() -> Agent:
    return Agent(str(CATALOG))


@st.cache_resource(show_spinner="Reading the catalog…")
def load_catalog() -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    with CATALOG.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                product = json.loads(line)
                lookup[str(product["parent_asin"])] = product
    return lookup


def price_of(product: dict) -> str:
    price = product.get("price")
    try:
        return f"${float(price):.2f}"
    except (TypeError, ValueError):
        return "—"


def start_session() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.turn = 0
    st.session_state.history = []
    load_agent().reset(st.session_state.session_id, {})


if "session_id" not in st.session_state:
    start_session()

agent = load_agent()
catalog = load_catalog()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🛍️ Agent state")
    st.caption("Everything here is read from the live agent, not replayed.")

    state = agent._sessions.get(st.session_state.session_id)
    turn = st.session_state.turn

    st.metric("Turn", f"{turn} / {MAX_TURNS}")
    if turn > MAX_TURNS:
        st.warning("Past the 10-turn cap. The grader would have stopped here.")

    if state is not None:
        st.subheader("Extracted constraints")
        if state.disclosed:
            for attribute, value in sorted(state.disclosed.items()):
                on_turn = state.disclosed_turn.get(attribute)
                st.write(f"**{attribute}** = {value}"
                         + (f"  ·  turn {on_turn}" if on_turn else ""))
        else:
            st.write("_Nothing extracted yet._")
            st.caption("Only material, colour and budget have extractors. "
                       "Style, size, use case and feature have none — a known limit.")

        st.subheader("Questions already asked")
        st.write(", ".join(state.asked_attributes) if state.asked_attributes else "_None yet._")

        if st.session_state.history:
            last = st.session_state.history[-1]
            st.subheader("Routing this turn")
            st.write(f"Intent: **{last['intent']}**")
            st.caption(f"BM25 {last['bm25_w']:.2f} · dense {last['dense_w']:.2f} · "
                       f"phrase {last['phrase_w']:.2f}"
                       + ("  ·  diversity re-rank on" if last["diversify"] else ""))
            if last["override"]:
                st.error("Intent override detected — prior constraints cleared.")

    st.divider()
    if st.button("Start a new session", use_container_width=True):
        start_session()
        st.rerun()

# ------------------------------------------------------------------- main
st.title("Conversational product search")
st.caption("Say what you're after. The agent narrows a 50,000-product catalog "
           "and asks one question per turn.")

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["user"])
    with st.chat_message("assistant"):
        st.write(entry["message"])
        if entry["recommendations"]:
            columns = st.columns(5)
            for index, pid in enumerate(entry["recommendations"][:10]):
                product = catalog.get(pid, {})
                with columns[index % 5]:
                    title = str(product.get("title") or pid)
                    st.markdown(f"**{title[:70]}**")
                    st.caption(f"{price_of(product)}  ·  `{pid}`")

prompt = st.chat_input("e.g. I need a leather belt for work")
if prompt:
    st.session_state.turn += 1
    state_before = agent._sessions.get(st.session_state.session_id)
    disclosed_before = dict(state_before.disclosed) if state_before else {}

    response = agent.respond(
        st.session_state.session_id, prompt, st.session_state.turn, 10
    )
    state = agent._sessions[st.session_state.session_id]

    # Read the route from the same function the agent used, so the panel can
    # never disagree with what retrieval actually did.
    signal = (agent.intent_classifier.classify(prompt)
              if agent.intent_classifier is not None else None)
    label = signal.label if signal is not None else "browsing"
    routing = routing_params(label, state.belief)

    override = False
    if agent.override_detector is not None and st.session_state.turn > 1:
        try:
            override = bool(agent.override_detector.is_override(prompt))
        except Exception:
            override = False

    from starter import agent as agent_module

    st.session_state.history.append({
        "user": prompt,
        "message": response["message"],
        "recommendations": [r["parent_asin"] for r in response["recommendations"]],
        "intent": label,
        "bm25_w": routing.bm25_weight,
        "dense_w": routing.dense_weight,
        "phrase_w": agent_module.PHRASE_WEIGHT,
        "diversify": routing.diversify,
        "override": override,
        "disclosed_before": disclosed_before,
    })
    st.rerun()
