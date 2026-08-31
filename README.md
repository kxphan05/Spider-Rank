# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

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
# Option A -- uv
uv sync

# Option B -- pip
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

CPU-only PyTorch, no CUDA. Using pip? Drop the `uv run` prefix below.

---

## 3. Download the catalog

Not included in the repo.

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
wc -l data/catalog.jsonl   # expect 50000
```

---

## 4. Fetch model weights and build the index

Only step that touches the network.

```bash
uv run python3 scripts/fetch_assets.py
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
uv run python3 -m evaluator.local_evaluator
```

Writes results to `results.json`, including per-scenario metrics and token usage. The evaluator is organizer-provided and unmodified.

---

## 6. Optional: check the submission bundle

```bash
uv run python3 scripts/build_submission.py --verify
```

Assembles `dist/submission/`, imports `Agent` from it in a neutral directory, and scores it.

---

## Quick reference

```bash
git clone git@github.com:kxphan05/tiktok-jam-hackathon-track-4.git
cd tiktok-jam-hackathon-track-4
uv sync
#    ... download catalog.jsonl to data/
uv run python3 scripts/fetch_assets.py
uv run python3 -m evaluator.local_evaluator
```

---

## Try it interactively

```bash
uv run python3 scripts/repl.py
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
