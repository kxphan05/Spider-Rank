# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

Our submission is **SpiderRank** — named for its five independent retrieval legs (BM25, phrase, dense, popularity, and pseudo-relevance feedback) fusing votes into one ranking, the way a spider's legs work independently but move it as one.

See [`TEAM_REPORT.md`](TEAM_REPORT.md) for method, model choice, the
latency/token/cost disclosure, and limitations (full version with ablations
in `docs/team_report.md`).

# How to reproduce our results

## 0. Clone the repo

```bash
git clone git@github.com:kxphan05/tiktok-jam-hackathon-track-4.git
cd tiktok-jam-hackathon-track-4
```

## 1. What you need

| requirement | value |
|---|---|
| Python | 3.12+ |
| Disk | ~420 MB assets + 58 MB catalog |
| Network | Once, in step 3 |
| GPU | None needed, CPU only |
| API keys | None |

---

## 2. Install dependencies

```bash
# Option A -- uv (recommended; same commands on Linux, macOS, and Windows)
uv sync

# Option B -- pip (Linux/macOS)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Option B -- pip (Windows)
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

CPU-only PyTorch, no CUDA. Using pip? Drop the `uv run` prefix below and use `python` (or `python3` on Linux/macOS) directly.

---

## 3. Download the catalog

Not included in the repo.

```bash
uv run python -c "import gzip, shutil; shutil.copyfileobj(gzip.open('catalog.jsonl.gz', 'rb'), open('data/catalog.jsonl', 'wb'))"
```

Works the same on Linux, macOS, and Windows -- no `gzip`/`mv` required. Then confirm the row count:

```bash
uv run python -c "print(sum(1 for _ in open('data/catalog.jsonl', encoding='utf-8')))"   # expect 50000
```

---

## 4. Fetch model weights and build the index

Only step that touches the network.

```bash
uv run python scripts/fetch_assets.py
```

| stage | what | size |
|---|---|---:|
| 1 | `BAAI/bge-small-en-v1.5` — dense search + classifiers | 129 MB |
| 2 | `cross-encoder/ms-marco-MiniLM-L-6-v2` — reranking | 88 MB |
| 3 | masked LM (skipped by default, `--with-lm` to include) | 257 MB |
| 4 | builds `data/dense_index/` from the catalog | 74 MB |

`Qwen/Qwen3-Reranker-0.6B` isn't fetched by default — demo-only, not in the scored pipeline.

---

## 5. Run the evaluator

```bash
uv run python -m evaluator.local_evaluator
```

Writes results to `results.json`, including per-scenario metrics and token usage. The evaluator is organizer-provided and unmodified.

---

## 6. Optional: check the submission bundle

```bash
uv run python scripts/build_submission.py --verify
```

Assembles `dist/submission/`, imports `Agent` from it in a neutral directory, and scores it.

---

## Quick reference

```bash
git clone git@github.com:kxphan05/tiktok-jam-hackathon-track-4.git
cd tiktok-jam-hackathon-track-4
uv sync
#    ... download catalog.jsonl to data/
uv run python scripts/fetch_assets.py
uv run python -m evaluator.local_evaluator
```

---

## Try it interactively

```bash
uv run python scripts/repl.py
```

`/reset`, `/debug`, `/topk N`, `/quit`.

---

## Optional: Streamlit demo

Not scored (`TODO.md` § 4.3 puts UI out of scope).

```bash
uv run --with streamlit streamlit run demo/streamlit_app.py
```

Shows the slate plus a live panel of what the agent extracted, questions asked, routing, and override state — read straight from the running agent.

> Binds all interfaces by default. Add `--server.address 127.0.0.1` to keep it local.

## Optional: SpiderRank live demo (FastAPI + custom UI)

Also not scored. Same agent, a hand-built chat page instead of Streamlit's default theme — chat bubbles, animated attribute chips, an emoji product grid, an intent-override banner.

```bash
uv run --with fastapi --with "uvicorn[standard]" uvicorn demo.web_app:app --port 8000
```

Then open <http://localhost:8000>. See `docs/hosting_the_demo.md` for deploying it beyond your own machine.

---

## Environment variables

All optional, all have working defaults.

| variable | purpose |
|---|---|
| `TECHJAM_CATALOG` | catalog path |
| `TECHJAM_DENSE_INDEX` | dense index directory |
| `TECHJAM_MODEL_DIR` | model weights directory |
| `TECHJAM_PROFILE_STORE` | long-term user-profile store path |
| `TECHJAM_THREADS` | cap CPU threads, for running evals side by side |

---

## Limitations & what we'd improve with more time

- Much of the pipeline leans on hand-written regex and closed vocabularies rather than anything learned,
  so it isn't very dynamic and each new phrasing has to be added by hand.
  Given more time and this dataset, we'd train a purpose-built classifier
  instead of layering regex rules around a frozen off-the-shelf encoder.
- A `ms-marco-MiniLM-L-6-v2` cross-encoder reranker is enabled by default
  (`RERANK_WEIGHT = 3`) but was never swept in isolation — it shipped
  bundled with unrelated changes. It's the one stage that could actually
  convert a miss into a hit rather than just reorder (it scores the top 20
  candidates and can promote a rank-11 item into the shown ten), so an
  isolated sweep of its weight is the highest-value follow-up we didn't get
  to. A larger `Qwen3-Reranker-0.6B` backend is also implemented but stays
  demo-only — ~27s/pair on this hardware makes it too slow to sweep at all.
- Long-term (cross-session) personalization is built but unused: every way
  we tried to consume it regressed the score, and a signal check showed
  same-key sessions' targets agree *below* chance. Given more time we would have done more data analysis on the
  user profiles to see if we can extract any signal from it.
- With more compute, the direction we'd want to explore is replacing the
  current one-pass retrieve-then-fuse pipeline with executor/planner
  agents that drive retrieval iteratively — planning what to search for,
  issuing multiple targeted queries, and deciding when enough evidence has
  been gathered, rather than fusing a single fixed set of retrieval legs
  per turn.

Full ablations and reasoning: [`docs/team_report.md`](docs/team_report.md) § 7,
and the "What we would do with more time" slide in `dist/techjam_track4.pptx`.

## Team

| member | contribution |
|---|---|
| **Phan Kang Xun** | Architecture, retrieval design, experiments and measurement, this report. |
| **Lloyd Wang** | Team Leader, Testing, Quality Assurance, Presentation |

Built with heavy use of Claude Code for implementation and documentation;
permitted per [`docs/submission_rules.md`](docs/submission_rules.md) §
Model Policy.
