# SpiderRank — TechJam Track 4: Conversational E-Commerce Search

## How our solution addresses the problem statement

Track 4 asks for a multi-turn shopping agent that finds one hidden target
product in a frozen 50,000-item catalog within 10 turns, returning 10
recommendations plus one clarifying question each turn. **SpiderRank** is
named for its architecture: five independent retrieval legs — BM25 keyword,
exact-phrase, dense embedding, pseudo-relevance feedback, and a popularity
prior — vote independently and fuse into one ranking by weighted reciprocal
rank fusion, the way a spider's legs move independently but carry it as one.

On top of retrieval, the agent routes each session into one of four tracks
(buying / browsing / intent-override / boundary) with a zero-shot
nearest-centroid classifier; **boosts rather than filters** on disclosed
attributes, since our own labels disagree with the true target 16–37% of the
time; **never re-shows a ruled-out product** (our single largest gain,
+0.084 TechnicalScore); and picks the next question with one score trading
off pool-splitting entropy against a per-shopper, Bayes-updated
answerability belief.

Measured on the released 200-sample public set: **HitRate@10 0.945, MRR
0.553, mean turns-to-hit 3.25, TechnicalScore 0.7935** — a 7.6x hit-rate
improvement over the organizer's baseline. Every number is measured against
the organizer's unmodified evaluator; ten shipped ideas and five rejected
ones are recorded with their measured deltas in our team report. The whole
pipeline runs **CPU-only and fully offline at inference**, with zero LLM API
calls and zero token cost.

## Development tools used

VS Code in a Dev Container (Python 3.11/3.12), **uv** for dependencies and
scripts, **pytest** and **ruff** for tests/lint, Git/GitHub, and **Claude
Code** for implementation, refactoring, and documentation throughout —
disclosed explicitly per the competition's model-policy rules.

## APIs used

**None at inference time** — the scored agent calls no hosted LLM or search
API; every turn is `prompt_tokens = completion_tokens = 0`. The **Hugging
Face Hub** is used once, offline, to download model weights before
evaluation, then verified dark with a strict offline preflight check.
FastAPI powers an optional, unscored local demo UI only.

## Libraries and frameworks used

**sentence-transformers** and **transformers** (Hugging Face) run the
frozen encoder and cross-encoder on **PyTorch** (CPU-only); **NumPy** for
fusion and scoring math; **SQLite FTS5** (stdlib) for BM25 and phrase
retrieval over the full catalog; **python-pptx** generates our slide deck
from measured numbers so it can't drift; **pytest**/**ruff** for
tests/lint; **Streamlit** and **FastAPI + Uvicorn + Pydantic** for two
optional, unscored demo UIs. Models: `BAAI/bge-small-en-v1.5` (dense
retrieval + all classifiers), `cross-encoder/ms-marco-MiniLM-L-6-v2`
(re-ranking, built and measured), and an optional `distilbert-base-uncased`
for attribute inference. Nothing is fine-tuned — every model is a frozen
feature extractor.

## Datasets and assets used

**Amazon Reviews 2023** (McAuley Lab, UCSD) — the competition's frozen
`Clothing_Shoes_and_Jewelry` catalog (50,000 products, text and structured
metadata only). The organizer's 200-sample public evaluation set and
unmodified local evaluator, used read-only for measurement. The only
hand-labelled data is a small set of prototype sentences (12–20 per
classifier) we wrote ourselves for intent/pivot/non-answer detection — no
external labelled data was used.
