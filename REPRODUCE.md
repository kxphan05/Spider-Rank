# How to reproduce our results

Written for judges. Every command below is copy-pasteable. Nothing is
implied and nothing is left to guess.

**Target result** (full 200-sample public set):

```
HitRate@10  0.790    MRR  0.4176    MTTC  4.745    Efficiency  0.6255
TechnicalScore  0.6454
```

---

## 0. What you need

| requirement | value |
|---|---|
| Python | **3.12 or newer** (`requires-python = ">=3.12"`) |
| Disk | ~420 MB of assets, plus a 58 MB catalog |
| Network | Needed **once**, in step 3. Nothing after that. |
| GPU | Not used. Everything runs on CPU. |
| API keys | None. No model is called over the network at any point. |

Our reference timings come from an **Intel i5-8365U (4 cores / 8 threads,
1.6 GHz laptop CPU)**. A faster machine will be quicker; the numbers are there
so you know whether a step has hung or is just slow.

---

## 1. Get the code and install dependencies

Either resolver works. Pick one.

```bash
# Option A -- uv (what we used; installs exact locked versions)
uv sync

# Option B -- pip
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Both install a **CPU-only** build of PyTorch. There is no CUDA dependency.

> If you use Option B, drop the `uv run` prefix from every command below.

---

## 2. Download the catalog

The catalog is organizer-provided and is not in this repository.

```bash
# Download catalog.jsonl.gz from the GitHub Release, then:
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Check it worked:

```bash
wc -l data/catalog.jsonl        # expect 50000
```

---

## 3. Fetch the model weights and build the index

**This is the only step that touches the network.**

```bash
uv run python3 scripts/fetch_assets.py
```

It runs four stages and prints its progress:

| stage | what | size |
|---|---|---:|
| 1 | `BAAI/bge-small-en-v1.5` — dense search and all three classifiers | 129 MB |
| 2 | `cross-encoder/ms-marco-MiniLM-L-6-v2` — re-ranking | 88 MB |
| 3 | masked LM — **skipped by default**, see below | — |
| 4 | builds `data/dense_index/` by embedding all 50,000 products | 74 MB |

Stage 4 is the slow one. It embeds the whole catalog locally and takes
**roughly 10-20 minutes** on our reference laptop. It is CPU-bound, so it
scales with cores.

The masked LM (`distilbert-base-uncased`, 257 MB) is **off by default and you
do not need it**. It is worth +0.001 TechnicalScore, which is inside this
benchmark's noise floor. Add `--with-lm` only if you want to reproduce that
ablation.

You also do not need `Qwen/Qwen3-Reranker-0.6B` (1.2 GB). It appears in our
report but is used for the demo only — it needs ~27 seconds per pair, which is
about 75 hours for one evaluation run, so it is not in the scored pipeline.

---

## 4. Verify the agent came up whole

**Do not skip this.** Our agent degrades *silently* if an asset is missing: it
still starts, and it still returns 10 products per turn, but dense retrieval
and all three classifiers are dead and the score quietly drops.

```bash
uv run python3 scripts/preflight.py --strict
```

This loads the agent with `HF_HUB_OFFLINE=1` set, so it cannot be fooled by a
background download. It **exits non-zero** if any required component is
missing. A clean pass means every component is live.

---

## 5. Run the official evaluator

```bash
uv run python3 -m evaluator.local_evaluator
```

This takes about **15 minutes** on our reference laptop. It prints the
aggregate metrics and writes per-session results to `results.json`.

The evaluator is organizer-provided and **we have never modified it**. You can
confirm that against the upstream copy.

It prints JSON to the terminal and writes the full per-session detail to
`results.json`. The fields that matter:

```json
{
  "sample_count": 200,
  "hit_rate_at_10": 0.79,
  "mrr": 0.4176,
  "mttc": 4.745,
  "efficiency": 0.6255,
  "recommended_technical_score": 0.6454
}
```

The same JSON also carries a `scenario_metrics` block broken down by
`browsing` / `buying` / `intent_override` / `boundary`, and a
`reported_token_usage` block.

### If your numbers differ slightly

The agent is deterministic, so a mismatch means a component is dark rather
than a random seed. Re-run step 4 first — that is almost always the cause.

---

## 6. Optional: check the submission bundle

```bash
uv run python3 scripts/build_submission.py --verify
```

This assembles a self-contained bundle in `dist/submission/`, then imports
`Agent` **from that bundle, in a neutral working directory**, asserts the
models came up live, and scores it on the full public set. It is how we check
that the submitted artifact behaves the same as the source tree.

---

## Quick reference

```bash
uv sync                                          # 1. dependencies
#    ... download catalog.jsonl to data/         # 2. catalog
uv run python3 scripts/fetch_assets.py           # 3. weights + index  (network)
uv run python3 scripts/preflight.py --strict     # 4. verify           (must pass)
uv run python3 -m evaluator.local_evaluator      # 5. score
```

---

## Try it interactively

To talk to the agent yourself rather than read a score:

```bash
uv run python3 scripts/repl.py
```

Commands inside the REPL: `/reset`, `/debug` (shows what the agent has
extracted each turn), `/topk N`, `/quit`.

---

## Optional: the Streamlit demo UI

Not part of the submission and not scored — `TODO.md` § 4.3 puts UI out of
scope. It exists to *show* the pipeline instead of describing it.

```bash
uv run --with streamlit streamlit run demo/streamlit_app.py
```

It renders the slate as products, and a live side panel showing what the agent
extracted and on which turn, which questions it has asked, the buying/browsing
route with the fusion weights that route chose, and whether an intent override
fired. All of it is read from the running agent, not replayed from a recording.

Nothing in `demo/` is imported by `starter/`, so the agent shown here is
byte-identical to the one the evaluator scores.

> By default Streamlit binds all network interfaces. Add
> `--server.address 127.0.0.1` to keep it on your machine.

---

## Environment variables

All optional. All have working defaults. Paths resolve against the working
directory first, then against the package's own directory, so the agent works
whether it is run from the repository root or from an unpacked bundle.

| variable | purpose |
|---|---|
| `TECHJAM_CATALOG` | catalog path, if not `data/catalog.jsonl` |
| `TECHJAM_DENSE_INDEX` | dense index directory, if not `data/dense_index` |
| `TECHJAM_MODEL_DIR` | model weights directory, if not `model/` |
| `TECHJAM_PROFILE_STORE` | long-term user-profile store path |
| `TECHJAM_THREADS` | cap CPU threads per run; useful only when running several evaluations side by side |
