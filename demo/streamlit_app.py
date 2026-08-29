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

import gzip
import json
import os
import sys
import tempfile
import urllib.request
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

MAX_TURNS = 10

# Cap on how many sessions the shared Agent keeps. Streamlit caches ONE Agent
# across every visitor, and Agent never evicts from its session dict -- fine
# for the evaluator, which runs 200 sessions and exits, but an unbounded leak
# for a long-lived server that a few people are hammering. Pruned in
# start_session() below.
MAX_TRACKED_SESSIONS = 200

# Point the long-term profile store at a scratch file. It is a write-through
# JSON file keyed by an anonymised profile hash, and the default path lives in
# the repository. On a shared demo that means every visitor writing to one
# file from a pool of worker threads, which is both a concurrency hazard and a
# privacy smell. The store has no effect on ranking anyway (measured dead), so
# a throwaway path costs nothing.
os.environ.setdefault(
    "TECHJAM_PROFILE_STORE",
    str(Path(tempfile.gettempdir()) / "techjam_demo_profiles.json"),
)

st.set_page_config(page_title="TechJam Shopping Agent", page_icon="🛍️", layout="wide")


def resolve_catalog() -> Path | None:
    """Find the catalog, or fetch it if we were given somewhere to fetch from.

    The catalog is 58 MB of organizer-provided Amazon metadata and is
    deliberately not committed, so a cloud deploy clones a repository without
    it. Resolution order:

      1. TECHJAM_CATALOG, or a `catalog_path` secret
      2. data/catalog.jsonl beside the repository
      3. a `catalog_url` secret -- downloaded once into the container

    Option 3 exists so the file never has to be committed. Note that
    DATA_ATTRIBUTION.md requires following the source dataset's terms, so
    point it at storage you control and keep private, not a public mirror.
    """
    explicit = os.environ.get("TECHJAM_CATALOG") or _secret("catalog_path")
    if explicit and Path(explicit).exists():
        return Path(explicit)

    local = REPO_ROOT / "data" / "catalog.jsonl"
    if local.exists():
        return local

    url = os.environ.get("TECHJAM_CATALOG_URL") or _secret("catalog_url")
    if url:
        return download_catalog(url)
    return None


def _secret(name: str) -> str | None:
    try:
        return st.secrets.get(name)  # type: ignore[no-any-return]
    except Exception:
        # No secrets.toml at all -- normal for a local run.
        return None


@st.cache_resource(show_spinner="Downloading the catalog (one time)…")
def download_catalog(url: str) -> Path | None:
    target = Path(tempfile.gettempdir()) / "techjam_catalog.jsonl"
    if target.exists() and target.stat().st_size > 0:
        return target
    try:
        with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310
            raw = response.read()
        if url.endswith(".gz"):
            raw = gzip.decompress(raw)
        target.write_bytes(raw)
        return target
    except Exception as exc:
        st.error(f"Could not download the catalog from the configured URL: {exc}")
        return None


CATALOG = resolve_catalog()

if CATALOG is None:
    st.title("🛍️ TechJam shopping agent")
    st.error("**The product catalog is missing, so the agent cannot start.**")
    st.markdown(
        """
The catalog is 58 MB of organizer-provided data. It is deliberately **not**
committed to this repository, so a fresh clone — including a Streamlit Cloud
deploy — does not have it.

**Running locally:** download `catalog.jsonl.gz` from the competition release,
then:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

**Deploying to Streamlit Cloud:** add a secret pointing at storage you
control, under *Manage app → Settings → Secrets*:

```toml
catalog_url = "https://.../catalog.jsonl.gz"
```

`DATA_ATTRIBUTION.md` requires following the source dataset's terms, so use
private storage rather than a public mirror.
"""
    )
    st.stop()


@st.cache_resource(show_spinner="Building the search index and loading models…")
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
    agent = load_agent()
    # Evict oldest-first before adding another. dicts preserve insertion
    # order, so the first keys are the least recently created.
    tracked = agent._sessions
    while len(tracked) >= MAX_TRACKED_SESSIONS:
        tracked.pop(next(iter(tracked)), None)
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.turn = 0
    st.session_state.history = []
    agent.reset(st.session_state.session_id, {})


if "session_id" not in st.session_state:
    start_session()

agent = load_agent()
catalog = load_catalog()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🛍️ Agent state")
    st.caption("Everything here is read from the live agent, not replayed.")

    # Which components actually came up. This is shown rather than hidden on
    # purpose: the agent's degrade path is silent by design (it keeps serving
    # ten products with a missing model and raises nothing), and that silence
    # has already cost this project real debugging time. A demo that hides it
    # would be misrepresenting what is running.
    with st.expander("Live components", expanded=agent.dense is None):
        components = [
            ("keyword search (BM25)", agent.bm25 is not None),
            ("phrase search", agent.bm25 is not None),
            ("dense search", agent.dense is not None),
            ("intent classifier", agent.intent_classifier is not None),
            ("override detector", agent.override_detector is not None),
            ("non-answer detector", agent.non_answer_detector is not None),
        ]
        for label, live in components:
            st.write(("✅ " if live else "⚪ ") + label)
        if agent.dense is None:
            st.info(
                "Running **lite**: keyword and phrase retrieval only. The model "
                "weights are not present, so the neural components are off.\n\n"
                "This is a fair demo of the ranking, not a crippled one — our "
                "ablation measured the dense leg to add **no** hit-rate on this "
                "benchmark. It only reorders. The phrase leg, which is on, is "
                "the component that actually finds products."
            )

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
