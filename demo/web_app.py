"""FastAPI + hand-built HTML/CSS/JS live demo UI for the shopping agent.

Not part of the scored submission -- same rule as `streamlit_app.py`. This is
a second front end for the identical `starter.agent.Agent`; nothing here
touches `starter/`, so the agent behind it is byte-identical to the one the
evaluator scores. It exists because a chat widget inside a generic Streamlit
theme undersells the parts of this system worth showing live: a slate that
updates in place, attribute chips that light up mid-conversation, and an
intent-override banner that fires the instant a pivot is detected.

Run it:

    uv run --with fastapi --with "uvicorn[standard]" uvicorn demo.web_app:app --reload

First request after boot takes ~30 seconds (lite) or ~2 minutes (full, first
time only -- it downloads model weights): building the FTS5 index over 50,000
products and loading whichever models are present. Cached for the process
after that.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import sys
import tempfile
import urllib.request
import uuid
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starter.agent import Agent, routing_params  # noqa: E402

logger = logging.getLogger(__name__)

MAX_TURNS = 10
MAX_TRACKED_SESSIONS = 200  # see streamlit_app.py -- same unbounded-dict hazard

# Same rationale as streamlit_app.py: point the write-through profile store at
# a scratch file so a shared demo process does not fight itself over one file.
os.environ.setdefault(
    "TECHJAM_PROFILE_STORE",
    str(Path(tempfile.gettempdir()) / "techjam_demo_profiles.json"),
)


def resolve_catalog() -> Path | None:
    """Find the catalog, or fetch it if we were given somewhere to fetch from.

    Mirrors `streamlit_app.resolve_catalog`, minus the `st.secrets` lookup --
    this process has no Streamlit runtime, so only environment variables
    apply. Resolution order:

      1. TECHJAM_CATALOG
      2. data/catalog.jsonl beside the repository
      3. TECHJAM_CATALOG_URL -- downloaded once into the container
    """
    explicit = os.environ.get("TECHJAM_CATALOG")
    if explicit and Path(explicit).exists():
        return Path(explicit)

    local = REPO_ROOT / "data" / "catalog.jsonl"
    if local.exists():
        return local

    url = os.environ.get("TECHJAM_CATALOG_URL")
    if url:
        return _download_catalog(url)
    return None


def _download_catalog(url: str) -> Path | None:
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
    except Exception:
        logger.exception("Could not download the catalog from %s", url)
        return None


class _State:
    """Everything lazily built on first request, guarded by one lock.

    A single worker process is assumed (`uvicorn ... --workers 1`, the
    default): the Agent keeps session state in an in-process dict, so more
    than one worker would silently split sessions across processes.
    """

    def __init__(self) -> None:
        self.lock = Lock()
        self.agent: Agent | None = None
        self.catalog: dict[str, dict] = {}
        self.catalog_path: Path | None = None
        self.turns: dict[str, int] = {}  # session_id -> turns taken so far

    def ready(self) -> tuple[Agent, dict[str, dict]] | None:
        if self.catalog_path is None:
            self.catalog_path = resolve_catalog()
        if self.catalog_path is None:
            return None
        with self.lock:
            if self.agent is None:
                logger.info("Building agent from %s", self.catalog_path)
                self.agent = Agent(str(self.catalog_path))
                with self.catalog_path.open(encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        product = json.loads(line)
                        self.catalog[str(product["parent_asin"])] = product
        return self.agent, self.catalog


_STATE = _State()
app = FastAPI(title="TechJam shopping agent demo")


class MessageIn(BaseModel):
    session_id: str
    message: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "web" / "index.html")


@app.get("/api/status")
def status() -> dict:
    ready = _STATE.ready()
    if ready is None:
        return {
            "ready": False,
            "error": "The product catalog is missing. Set TECHJAM_CATALOG, "
                     "TECHJAM_CATALOG_URL, or place data/catalog.jsonl beside the repo.",
        }
    agent, _ = ready
    return {
        "ready": True,
        "max_turns": MAX_TURNS,
        "components": {
            "keyword search (BM25)": agent.bm25 is not None,
            "phrase search": agent.bm25 is not None,
            "dense search": agent.dense is not None,
            "intent classifier": agent.intent_classifier is not None,
            "override detector": agent.override_detector is not None,
            "non-answer detector": agent.non_answer_detector is not None,
        },
        "lite": agent.dense is None,
    }


@app.post("/api/session")
def new_session() -> dict:
    ready = _STATE.ready()
    if ready is None:
        raise HTTPException(503, "Catalog not available")
    agent, _ = ready
    tracked = agent._sessions
    while len(tracked) >= MAX_TRACKED_SESSIONS:
        tracked.pop(next(iter(tracked)), None)
    session_id = str(uuid.uuid4())
    agent.reset(session_id, {})
    _STATE.turns[session_id] = 0
    while len(_STATE.turns) > MAX_TRACKED_SESSIONS:
        _STATE.turns.pop(next(iter(_STATE.turns)), None)
    return {"session_id": session_id}


def _price(product: dict) -> float | None:
    price = product.get("price")
    try:
        return round(float(price), 2)
    except (TypeError, ValueError):
        return None


def _coarse_category(values: list[str]) -> str:
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned = [part.strip() for value in values for part in value.split(",")
               if part.strip() and part.strip().lower() not in excluded]
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


@app.post("/api/message")
def message(body: MessageIn) -> JSONResponse:
    ready = _STATE.ready()
    if ready is None:
        raise HTTPException(503, "Catalog not available")
    agent, catalog = ready

    state_before = agent._sessions.get(body.session_id)
    if state_before is None or body.session_id not in _STATE.turns:
        raise HTTPException(404, "Unknown session -- start a new one")
    disclosed_before = dict(state_before.disclosed)
    turn = _STATE.turns[body.session_id] + 1
    _STATE.turns[body.session_id] = turn

    response = agent.respond(body.session_id, body.message, turn, 10)
    state = agent._sessions[body.session_id]

    signal = agent.intent_classifier.classify(body.message) if agent.intent_classifier is not None else None
    label = signal.label if signal is not None else "browsing"
    routing = routing_params(label, state.belief)

    override = False
    if agent.override_detector is not None and turn > 1:
        try:
            override = bool(agent.override_detector.is_override(body.message))
        except Exception:
            override = False

    from starter import agent as agent_module

    recs = []
    for pid in [r["parent_asin"] for r in response["recommendations"]]:
        product = catalog.get(pid, {})
        recs.append({
            "asin": pid,
            "title": (product.get("title") or pid)[:120],
            "price": _price(product),
            "rating": product.get("average_rating"),
            "ratings": product.get("rating_number"),
            "category": _coarse_category(product.get("categories") or []),
        })

    new_attrs = {k: v for k, v in state.disclosed.items() if k not in disclosed_before}

    return JSONResponse({
        "message": response["message"],
        "ask_attribute": response.get("ask_attribute"),
        "recommendations": recs,
        "turn": turn,
        "max_turns": MAX_TURNS,
        "intent": label,
        "routing": {
            "bm25": routing.bm25_weight,
            "dense": routing.dense_weight,
            "phrase": agent_module.PHRASE_WEIGHT,
            "diversify": routing.diversify,
        },
        "override": override,
        "disclosed": dict(sorted(state.disclosed.items())),
        "new_attributes": sorted(new_attrs),
        "asked_attributes": list(state.asked_attributes),
    })
