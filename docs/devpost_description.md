# SpiderRank — TechJam Track 4: Conversational E-Commerce Search

## How our solution addresses the problem statement

Track 4 asks for a multi-turn shopping agent that finds one hidden target
product in a frozen 50,000-item catalog in at most 10 turns, by returning 10
recommendations plus one clarifying question each turn. **SpiderRank** is
named for its architecture: five independent retrieval legs — BM25 keyword,
exact-phrase, dense embedding, pseudo-relevance feedback, and a popularity
prior — vote independently and fuse into one ranking by weighted reciprocal
rank fusion, the way a spider's legs move independently but carry it as one.

On top of retrieval, the agent:
- **Routes each session** into one of four tracks (buying / browsing /
  intent-override / boundary) with a zero-shot nearest-centroid classifier
  over frozen embeddings, which re-weights retrieval and reranking per track
  rather than switching pipelines.
- **Boosts, never filters** on disclosed attributes (material, colour,
  budget) — a measured decision, since our own attribute labels disagree
  with the true target 16–37% of the time, so a hard filter would delete the
  answer.
- **Never re-shows a product already ruled out** — the single largest gain
  we measured (+0.084 TechnicalScore), because a further turn is proof every
  item shown so far was wrong.
- **Picks the next question** with one score that trades off how much an
  answer would split the current candidate pool (entropy) against how likely
  the shopper is to actually answer it (a per-shopper belief that Bayes-
  updates across attributes on every reply).
- **Detects a shopper changing their mind mid-session** and clears stale
  state, rather than continuing to search on preferences the shopper already
  abandoned.

Measured on the released 200-sample public set: **HitRate@10 0.945, MRR
0.553, mean turns-to-hit 3.25, TechnicalScore 0.7935** — a 7.6x hit-rate
improvement over the organizer's baseline agent (0.125 HitRate@10). Every
number is measured against the organizer's own unmodified evaluator, not
estimated; ten shipped ideas and five explicitly rejected ones are recorded
with their measured deltas in our team report.

The whole pipeline runs **CPU-only, fully offline at inference time, with
zero LLM API calls and zero token cost** — no hosted model is in the loop
once local weights and the search index are built.

## Development tools used

- **VS Code**, in a Dev Container (`mcr.microsoft.com/devcontainers/python`,
  Python 3.11/3.12) for a reproducible dev environment
- **uv** for dependency resolution, virtual environments, and running
  scripts (`uv sync`, `uv run`)
- **pytest** for the test suite; **ruff** as the lint gate on our own code
- **Git / GitHub** for version control
- **Claude Code** (Anthropic), used heavily for implementation,
  refactoring, and documentation throughout the project — disclosed
  explicitly in our team report, per the competition's model-policy rules

## APIs used

- **None at inference time.** The scored agent calls no hosted LLM or search
  API — every decision is made by local, frozen models running on CPU. This
  was also disclosed and measured: `prompt_tokens = completion_tokens = 0`
  on every turn, reported honestly by the agent.
- **Hugging Face Hub**, used once and only during offline asset setup
  (`scripts/fetch_assets.py`) to download model weights before evaluation —
  not called during scoring, and verified offline afterward with a strict
  preflight check (`HF_HUB_OFFLINE=1`).
- **FastAPI**, for an optional, unscored local demo web app (see below) —
  not an external API, just how the demo's chat UI talks to the agent
  process.

## Libraries and frameworks used

- **sentence-transformers** and **transformers** (Hugging Face) — load and
  run the frozen encoder and cross-encoder
- **PyTorch** (CPU build) — backs the above; no GPU/CUDA used anywhere
- **NumPy** — vector math for fusion, boosting, and similarity scoring
- **SQLite FTS5** (Python standard library) — BM25 keyword and exact-phrase
  retrieval legs over the full 50,000-product catalog
- **pytest**, **ruff** — testing and linting
- **python-pptx** — generates our presentation deck programmatically from
  the measured numbers, so it can't drift from the source of truth
- **Streamlit**, and separately **FastAPI + Uvicorn + Pydantic** — two
  optional, unscored local demo UIs for interacting with the live agent
- **tqdm** — progress bars for long evaluation runs

Models used, all frozen and local:
- `BAAI/bge-small-en-v1.5` (129 MB) — dense retrieval and all three
  embedding-based classifiers (intent, pivot/override, non-answer detection)
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (88 MB) — built and wired for
  re-ranking, shipped at weight 0 because its A/B sweep didn't fit our
  evaluation budget and we don't enable unmeasured switches
- `distilbert-base-uncased` (257 MB, optional, off by default) — masked-LM
  attribute inference, measured net-neutral for its cost
- Nothing is fine-tuned; every model is used strictly as a frozen feature
  extractor or scorer.

## Datasets and assets used

- **Amazon Reviews 2023** (McAuley Lab, UC San Diego) — the competition's
  frozen `Clothing_Shoes_and_Jewelry` catalog (50,000 products), provided by
  the organizer as text and structured metadata only (no images, no reviews,
  no account data)
- The organizer's **200-sample public evaluation set** and unmodified
  **local evaluator/simulator**, used strictly read-only for measurement
- A small set of **hand-written prototype sentences** (12–20 per
  classifier) that we authored ourselves as the nearest-centroid reference
  set for intent, pivot, and non-answer detection — the only "labelled
  data" in the project, and it is our own, not sourced externally
- No external labelled datasets, and no data beyond what the organizer
  provided
