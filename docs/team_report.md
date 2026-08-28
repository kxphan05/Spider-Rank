# Team Report — TechJam Conversational Search

Submission report required by `docs/submission_rules.md` § What Teams Must
Submit: method, model choice, limitations, and a disclosure of latency, token
usage, and estimated model cost.

Every number in this document was measured on the released 200-sample public
set with the scripts named beside it. Nothing here is an estimate unless it
says so.

---

## 1. Headline result

```
HitRate@10  0.745    MRR  0.3876    MTTC  5.09    Efficiency  0.591
TechnicalScore  0.6070
```

Against the shipped weak-BM25 starter (`docs/baseline_results.json`:
HitRate 0.125 / MRR 0.068 / MTTC 9.81), that is a **6.0x improvement in hit
rate** and a halving of turns-to-conversion.

By scenario:

| scenario | n | HitRate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| browsing | 80 | 0.8125 | 0.4313 | 4.38 |
| buying | 80 | 0.6875 | 0.3457 | 5.26 |
| intent_override | 30 | 0.8000 | 0.4662 | 5.93 |
| boundary | 10 | 0.5000 | 0.1375 | 6.90 |

Reproduce with `uv run python3 -m evaluator.local_evaluator` (the evaluator is
used strictly read-only; it has never been modified).

---

## 2. Method

Five stages, all in-memory and all local.

**Intent routing.** A zero-shot nearest-centroid classifier over frozen
bge-small embeddings labels each session buying vs. browsing
(`EmbeddingIntentClassifier`), with a lexical rule as fallback. The label
conditions three downstream choices rather than selecting a separate pipeline.

**Dual-track hybrid retrieval.** BM25 (SQLite FTS5) and dense cosine retrieval
run independently over the full 50k catalog and are combined by weighted
reciprocal rank fusion. The weights are intent-conditioned: buying keeps a
BM25-heavy 2.0/1.0 for precision on hard constraints, browsing shifts to
1.25/1.5 for broader semantic matching and additionally gets an MMR diversity
re-rank, pinned so diversity can never evict the top matches.

**Disclosure extraction and attribute boosting.** Every customer message is
scanned for material, color and budget (negation-aware). Disclosed values
**re-sort** the candidate pool rather than filtering it: +1 where a candidate
agrees, −1 where it is known and disagrees, 0 where unextractable. This
non-eliminating design is deliberate and was measured — an earlier hard filter
eliminated the true target 16.3% of the time on material and 37.0% on color.

**Proactive question selection.** Each turn the agent picks the attribute with
the highest value-diversity in the *current* candidate pool, recomputed fresh
so it stops asking once the pool converges. When nothing structural clears the
entropy threshold it falls back to a fixed order sorted by measured
*answerability* — `feature, style, size, use_case, budget`. That ordering
alone was worth +0.0109 TechnicalScore.

**Intent-override handling.** The agent API exposes no signal for a
mid-session pivot, so it is detected from text: a trimmed nearest-prototype
rule (mean of the 4 closest prototypes) scored over both the full message and
its lead clause. On detection, disclosed constraints and asked-attribute state
are cleared. This replaced a centroid rule that missed terse pivots
("never mind, give me white shoes"); it lifts out-of-distribution probe recall
0.800 → 1.000 at a cost of 2 false positives per 1600 ordinary turns.

---

## 3. Model choice

| component | model | size | why |
|---|---|---:|---|
| dense retrieval + both classifiers | `BAAI/bge-small-en-v1.5` | 129 MB | strong retrieval quality per MB; runs on CPU in-process |
| masked-LM attribute inference | `distilbert-base-uncased` | 257 MB | **optional**, see §6 |
| lexical retrieval | SQLite FTS5 (BM25) | — | stdlib, no model |

**No LLM API is called at inference time.** Both models are local, frozen, and
CPU-only. Nothing is fine-tuned: `TODO.md` § 4.3 places "training or
full-parameter fine-tuning of base foundational LLMs" out of scope, so the
encoder is used strictly as a frozen feature extractor. A logistic head over
those frozen embeddings was built and evaluated as the one permissible learned
variant, and **rejected** — see §6.

---

## 4. Disclosure: latency, tokens, cost

Measured with `scripts/measure_latency.py --limit 20` on an AMD Ryzen 5 PRO
4650U (12 threads, CPU-only, no GPU), Python 3.12.3, torch 2.13.0+cpu.

**Token usage: zero.** `prompt_tokens = completion_tokens = 0`, reported
honestly by the agent because no hosted model is called on any turn.

**Estimated model cost: $0.00.** No API spend at inference or at build time.
The one-time dense index build is local compute only.

**Latency**

| phase | with masked LM | without masked LM |
|---|---:|---:|
| cold start (`Agent()`) | 16.3 s | 17.1 s |
| `reset()` per session | 0.6 ms | 0.4 ms |
| `respond()` mean | 408 ms | 361 ms |
| `respond()` p50 | 335 ms | 309 ms |
| `respond()` p95 | 716 ms | 623 ms |
| `respond()` max | 1389 ms | 1392 ms |
| per session (5.8 turns mean) | 2.36 s | 2.09 s |

Cold start is one-time per process, not per session. The masked LM costs
~47 ms per turn.

**Memory**

| | with masked LM | without |
|---|---:|---:|
| peak RSS after init | 794 MB | 794 MB |
| peak RSS overall | 1286 MB | 934 MB |

Runs comfortably in 2 GB. The catalog and dense index are held in memory as
required by the "must run entirely in-memory" constraint; no external vector
DB is used.

---

## 5. Network access and offline operation

**The agent requires no network access at scoring time**, provided setup has
run. All weights and indexes are read from local disk.

This is load-bearing, because the failure mode is silent. `Agent.__init__`
degrades rather than crashing by design: with a cold HuggingFace cache and no
network, the encoder fails to load and the dense index, both classifiers and
the masked LM all go dark — yet the agent still starts and still returns ten
recommendations. It simply stops being the system that was measured. This was
reproduced directly (`OSError: We couldn't connect to 'https://huggingface.co'`).

Two guards ship as a result:

- `scripts/fetch_assets.py` — the **only** part of this project that needs
  network. Downloads the encoder and builds the dense index.
- `scripts/preflight.py --strict` — loads the agent with `HF_HUB_OFFLINE=1`
  and reports which components are actually live, exiting non-zero if any
  required one is dark. **Run this before any scored run.**

---

## 6. Ablations, and the one result that complicates this submission

Each component was disabled independently against the full public set
(`--limit` omitted; all 200 samples):

| configuration | HitRate | MRR | MTTC | TechnicalScore | Δ |
|---|---:|---:|---:|---:|---:|
| **as shipped** | 0.7450 | 0.3876 | 5.090 | **0.6070** | — |
| − masked LM | 0.7450 | 0.3841 | 5.085 | 0.6060 | −0.0010 |
| − both classifiers | 0.7400 | 0.3728 | 5.070 | 0.6004 | −0.0065 |
| − dense retrieval | 0.7450 | 0.4328 | 5.035 | 0.6216 | **+0.0147** |
| − dense − masked LM | 0.7450 | 0.4351 | 5.035 | **0.6223** | **+0.0154** |
| − everything (offline degrade) | 0.7450 | 0.4289 | 5.025 | 0.6207 | +0.0137 |

Two findings, stated plainly:

**The classifiers earn their place** (+0.0065). **The masked LM barely
registers**: +0.0010 here, and −0.0007 when measured against the no-dense
configuration — a sign flip across configurations is what noise looks like at
this sample size. For 257 MB of weights, 350 MB of peak RSS and 47 ms/turn,
that is not a trade worth making, so the LM is **not fetched by default**
(`fetch_assets.py --with-lm` opts back in).

**Dense retrieval is net-negative on the local set, by 20x the noise floor.**
Removing it *raises* TechnicalScore 0.6070 → 0.6216. Note HitRate is
*identical* to four decimals: the dense leg finds nothing BM25 misses, and its
contribution to the fusion demotes correct items BM25 had already ranked well
(MRR 0.3876 → 0.4328).

**We have not acted on this, and the reason is a measured confound.** The
local simulator builds its customer messages from the target product's own
catalog fields: 359 of 400 turn-1 hard constraints (89.7%) appear as *verbatim
substrings* of the target's own text, and the 41 that don't are only
non-verbatim because the evaluator prefixes them with `"color: "`. Effectively
every local query is an exact-match query, which is the best possible case for
BM25 and the worst possible case for the paraphrase-robustness the dense leg
exists to provide. If the hidden grader phrases customers the same way,
dropping dense is worth +0.015; if it paraphrases at all, dropping it removes
the only defense. We kept dense retrieval because we cannot distinguish those
two worlds from the public set, and the downside is asymmetric.

The honest summary: **the shipped configuration is deliberately not the
highest-scoring one we measured locally.**

---

## 7. Limitations

1. **Only three of ten attributes are structurally extractable.** `material`,
   `color` and `budget` have extractors; `style`, `size`, `use_case`,
   `feature` never populate session state, so they can be asked but never
   filtered on. `category`, `brand` and `other` are unreachable entirely.
2. **Catalog attribute coverage is thin** — material 70.9%, color 39.9%,
   price ~21% after a word-boundary bug fix that cut fictional color labels
   (`"red"` matched inside *embroidered*). The fix is correct and currently
   *costs* 0.005, because the ±1/0 boost weights were tuned against the
   inflated coverage and have not been retuned.
3. **Budget extraction requires a literal `$`** — "fifty dollars" or
   "around 50" will not match.
4. **No cross-encoder or LLM reranking stage.** Spec § 4.2.I names "LLM
   Semantic Ranking" as a pillar; ranking here is hybrid retrieval +
   attribute boost + MMR, with no learned reranker.
5. **Long-term personalization ships inert.** The user-profile store is built
   and correct, but nothing reads its output: three ways of using it each
   regressed the full set, and a signal check showed same-key sessions' targets
   agree *below* chance. There is nothing in the profile to personalize on.
6. **Local-simulator artifacts are load-bearing in two tuned choices.** The
   answerability ordering leans on `feature` being answerable 96% of the time,
   which is partly an artifact of `feature` being the evaluator's catch-all
   bucket; and `budget` is demoted because the local evaluator never buckets
   anything as budget. Both were kept as options rather than hardcoded, but
   both could behave differently on the hidden set.
7. **Boundary scenarios are weakest** (HitRate 0.500, MRR 0.138) — though on
   only 10 samples, so the estimate is coarse.

---

## 8. Setup and reproducibility

**Python 3.12+** (`requires-python = ">=3.12"`). Dependencies: `numpy>=2.5.2`,
`sentence-transformers>=6.0.0`, `torch>=2.9.0` (CPU wheel index pinned in
`pyproject.toml`); `tqdm` is used for progress bars only, imported defensively,
and arrives transitively.

```bash
# 1. install dependencies -- either resolver works
uv sync                              # exact versions, from uv.lock
pip install -r requirements.txt      # equivalent pins, CPU-only torch

# 2. download the catalog -- see README.md "Download the Catalog"
#    -> data/catalog.jsonl

# 3. fetch model weights and build the dense index (the ONLY networked step)
uv run python3 scripts/fetch_assets.py

# 4. verify the pipeline comes up whole with the network disabled
uv run python3 scripts/preflight.py --strict

# 5. run the official harness
uv run python3 -m evaluator.local_evaluator
```

### The submission bundle

`scripts/build_submission.py` assembles a self-contained bundle in the shape
the rules recommend — `agent.py` exporting `Agent`, `src/` for the agent
package, `requirements.txt`, `README.md`, this report as `REPORT.md`, and
`tools/` holding the two setup commands. It is **generated, never
hand-maintained**, so it cannot drift from the source tree:

```bash
uv run python3 scripts/build_submission.py --verify
```

`--verify` imports `Agent` from the built bundle in a neutral working
directory, asserts that dense retrieval and both classifiers actually came up
live rather than silently degrading, and scores the bundle on the full public
set. The bundle is 17 files and ~123 KB; `data/` and `model/` are excluded
(~460 MB together, and the catalog is organizer-provided) and are materialized
by `tools/fetch_assets.py`.

### Environment variables

All optional, all with working defaults. Data and model paths resolve against
the working directory first and then against the package's own directory, so
the harness may be run from either — a submitted bundle is exactly the case
where those differ.

| variable | purpose |
|---|---|
| `TECHJAM_CATALOG` | catalog path, if not `data/catalog.jsonl` |
| `TECHJAM_DENSE_INDEX` | dense index directory, if not `data/dense_index` |
| `TECHJAM_MODEL_DIR` | model weights directory, if not `model/` |
| `TECHJAM_PROFILE_STORE` | long-term user-profile store path |

`preflight.py` sets `HF_HUB_OFFLINE=1` itself, so its check cannot be masked by
a background download.

### Development and diagnostic tooling

| script | purpose |
|---|---|
| `scripts/run_eval.py` | eval wrapper: `--limit`, `--scenario`, `--seed`, progress bar |
| `scripts/repl.py` | interactive single-session REPL |
| `scripts/measure_latency.py` | the §4 disclosure numbers |
| `scripts/preflight.py` | offline asset verification |
| `scripts/fetch_assets.py` | one-command asset setup |
| `scripts/eval_override.py` | override-detector rule sweeps |
| `scripts/eval_intent.py` | intent-classifier rule sweeps |
| `scripts/eval_profile_signal.py` | whether `user_profile` carries signal |
| `scripts/eval_lm_confidence.py` | masked-LM entropy-gate calibration |
| `scripts/train_intent_head.py` | trained-head diagnostic (rejected; `--save` opt-in) |
| `scripts/sweep_fusion_weights.py` | RRF dense:bm25 weight curve |
| `scripts/build_submission.py` | assemble and verify the submission bundle |
