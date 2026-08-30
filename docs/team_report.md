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
HitRate@10  0.945    MRR  0.5534    MTTC  3.250   Efficiency  0.7750
TechnicalScore  0.7935
```

Measured on the full 200-sample public set at the submitted configuration
(commit `1fb1913`, 2026-08-30), with `preflight.py --strict` confirming
beforehand that dense retrieval, both embedding classifiers, and the
cross-encoder reranker all come up live with the network disabled — so this is
the whole pipeline, not a silently degraded one.

Against the shipped weak-BM25 starter (`docs/baseline_results.json`:
HitRate 0.125 / MRR 0.068 / MTTC 9.81), that is a **7.6x improvement in hit
rate** and turns-to-conversion cut to a third.

By scenario:

| scenario | n | HitRate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| browsing | 80 | 1.0000 | 0.6230 | 2.88 |
| intent_override | 30 | 0.9333 | 0.7136 | 4.63 |
| buying | 80 | 0.9125 | 0.4534 | 2.94 |
| boundary | 10 | 0.8000 | 0.3169 | 4.60 |

Re-measured at the current configuration, not carried forward from the
previous one. The ordering is stable and matches the miss census: `boundary`
is hardest and `browsing` easiest, and the two scenarios where the customer
states a hard constraint up front — `buying` and `intent_override` — sit in
between. `boundary` is n=10, so a single session moves it 0.1; read it as
directional only. CLAUDE.md #31 records that the three commits behind this
number (gated attribute asking, a boundary/popularity leg, and query
stripping plus a newly-enabled cross-encoder reranker) were bundled and
measured only as a whole — the individual contribution of each has not been
isolated.

Reproduce with `uv run python3 -m evaluator.local_evaluator` (the evaluator is
used strictly read-only; it has never been modified).

---

## 2. Method

Six stages, all in-memory and all local. No network access is required at
inference time.

**Intent routing.** A zero-shot nearest-centroid classifier over frozen
bge-small embeddings labels each session buying vs. browsing
(`EmbeddingIntentClassifier`), with a lexical rule as fallback. The label
conditions three downstream choices rather than selecting a separate pipeline.

**Dual-track hybrid retrieval, three legs.** BM25 (SQLite FTS5), a
**phrase-match leg**, and dense cosine retrieval run independently over the
full 50k catalog and are combined by weighted reciprocal rank fusion. The
BM25/dense weights are intent-conditioned: buying keeps a BM25-heavy 2.0/0.5
for precision on hard constraints, browsing shifts to 1.25/1.5 for broader
semantic matching and additionally gets an MMR diversity re-rank, pinned so
diversity can never evict the top matches.

The phrase leg is our largest *retrieval-side* measured gain (**+0.0272**) and
the first change in this project that improved *recall* rather than ordering.
(The largest gain overall is the shown-item exclusion below, +0.0837.) It
exists because of a property we measured about the task rather than a general
preference: 89.7% of the simulated customer's turn-1 hard constraints are
verbatim substrings of the target product's own catalog text. Ordinary BM25
dissolves that structure — it ORs the query's unique tokens, so "Buckle
closure" becomes two independent terms against 50k products. The phrase leg
matches the span itself via FTS5 phrase queries, scoring n-grams longest-first
and discarding any span that matches more than 150 products as
non-discriminative. Weight swept at 2.0; the curve has a genuine interior
optimum.

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

**Shown-item exclusion.** The agent never re-recommends a product it has
already shown. This is our largest single gain, **+0.0837**, and it is a
deduction from the scoring rule rather than a retrieval idea: the evaluator
ends a session the moment the target enters the returned slate, so being asked
for another turn at all is proof that every item shown so far is wrong.
Re-offering them spends slots on answers already known to be dead. It is also
the only change here that moves all three score terms the same way at once —
every earlier win traded MRR against MTTC, because converting a late hit into
an earlier, worse-ranked one costs MRR and pays in Efficiency. This one
improves rank *and* speed, because the freed slots refill from deeper in the
same pool. The dependency to state plainly: it assumes the hidden grader also
stops on first hit, which the rules state and `evaluate()` implements, but it
is a stronger reliance on scorer semantics than anything else we ship.

**Cross-encoder re-ranking: built, not enabled.** A `ms-marco-MiniLM-L-6-v2`
stage is implemented (`starter/reranker.py`) behind `RERANK_WEIGHT`, which
ships at **0.0** — the identity, which skips the model entirely. Its sweep
(`scripts/sweep_rerank.py`, four points at ~15 minutes each) did not fit the
remaining evaluation budget, and this project's rule is that an unmeasured
switch ships at identity. So the scored pipeline contains **no** learned
re-ranker; see §7.4. The design intent is recorded because the depth choice is
the non-obvious part: at a scored depth equal to `top_k` the re-ranker can only
permute the ten items already destined for the slate, so it cannot convert a
miss into a hit under any ordering. `RERANK_TOP_N = 20` is what would open the
recall channel, if it were switched on.

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
| dense retrieval + all three classifiers | `BAAI/bge-small-en-v1.5` | 129 MB | strong retrieval quality per MB; runs on CPU in-process |
| cross-encoder re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 87 MB | ~17 ms/pair on CPU, which is what makes the stage measurable at all |
| masked-LM attribute inference | `distilbert-base-uncased` | 257 MB | **optional**, see §6 |
| lexical + phrase retrieval | SQLite FTS5 (BM25) | — | stdlib, no model |

We also evaluated `Qwen/Qwen3-Reranker-0.6B` for the re-ranking stage. It
judges noticeably better, but needs **~27 s per pair** on this hardware
against MiniLM's ~17 ms — roughly 75 hours for a single 200-sample
evaluation, so it cannot be A/B'd, let alone shipped. It is retained for the
demo only. This is a case where the honest engineering constraint was
measurement throughput rather than inference latency: a model we cannot
evaluate is a model we cannot justify.

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

**Stale as of `1fb1913`: the numbers below predate the cross-encoder reranker
being turned on (`RERANK_WEIGHT` 0.0 → 3).** A `respond()` call now runs an
extra scoring pass over `RERANK_TOP_N = 20` candidates through the minilm
cross-encoder, which this table does not account for. Re-run
`scripts/measure_latency.py --limit 20` before citing this table in a final
submission — do not run it alongside another full eval on this box.

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
(`--limit` omitted; all 200 samples). **These rows were measured at the
previous buying fusion weight (dense:bm25 ratio 0.50, TechnicalScore 0.6070),
before the change described at the end of this section.** They are reported at
that baseline rather than re-scaled, because the deltas are only meaningful
against the configuration they were measured in:

| configuration | HitRate | MRR | MTTC | TechnicalScore | Δ |
|---|---:|---:|---:|---:|---:|
| **as measured then** | 0.7450 | 0.3876 | 5.090 | **0.6070** | — |
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

**We acted on this only partly, and the reason is a measured confound.** The
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

**What we did instead was sweep the weight rather than remove the leg**
(`scripts/sweep_fusion_weights.py`, 200 samples per point; only the dense:bm25
ratio matters, since weighted RRF is scale-invariant per leg):

| buying ratio | HitRate | MRR | MTTC | TechnicalScore |
|---:|---:|---:|---:|---:|
| 0.00 (no dense) | 0.7500 | 0.4372 | 5.020 | 0.6258 |
| **0.25 (shipped)** | **0.7500** | **0.4096** | **4.985** | **0.6182** |
| 0.50 (previous) | 0.7450 | 0.3876 | 5.090 | 0.6070 |
| 1.00 | 0.7200 | 0.3615 | 5.225 | 0.5839 |
| 1.50 | 0.6650 | 0.3388 | 5.720 | 0.5397 |

The curve is **strictly monotone decreasing with no interior optimum** — the
hoped-for flat region near the old value does not exist, and 0.50 was already
well down the slope. HitRate falls too, not just MRR: at high dense weight the
dense leg does not merely reorder, it displaces correct BM25 results out of
the top 10 entirely. Halving the weight to 0.25 captures **+0.0112 of the
+0.0188** available from removing dense outright, which keeps a real dense leg
as paraphrase insurance at roughly 40% of its former local cost. The browsing
track was swept too and measured **flat** across 0.0–1.5 (every setting within
one session of every other), so it was left alone.

### The largest win came from re-reading the scorer, not from a model

| leg | HitRate | hits | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| phrase 0.0 (identity) | 0.7550 | 151/200 | 0.3989 | 4.940 | 0.6184 |
| phrase 2.0 | 0.7800 | 156/200 | 0.4103 | 4.825 | 0.6366 |
| **phrase 2.0 + shown-item exclusion** | **0.8550** | **171/200** | **0.4622** | **4.205** | **0.7020** |

That ablation was measured at `3ef2028`. The submitted configuration adds one
further change on top of it — the category-IDF and request-stopword fix — which
was measured separately at **-0.0042, a single session** (171/200 to 170/200),
i.e. flat within this benchmark's resolution. It ships because it fixes two
reproducible query bugs at no measurable cost, not because it scores.

Measured with all three legs on one HEAD (`scripts/ab_phrase_exclude.py`); the
identity leg reproduced the then-shipped 0.6184 first, which is what makes the
other two rows comparable. **+0.0837 for the exclusion — three times the
largest retrieval-side change in this project, and it required no new signal at
all.** It is a rule read straight off the evaluator's stopping condition. We
spent weeks on retrieval mechanisms worth ~+0.01 each while a re-read of
`evaluate()` was worth +0.08. The transferable lesson is the ordering: read the
scorer again before building anything.

The honest summary: **the shipped configuration is deliberately not the
highest-scoring one we measured locally.** We took the part of the dense-leg
finding that is robust to the confound (the weight was too high) and declined
the part that is not (removing the leg entirely).

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
3. **Budget extraction needs an explicit unit.** `$80`, `80 dollars`,
   `45 bucks` and `fifty dollars` all parse, as do ceiling/floor phrasings
   (`under`, `over`, `no more than`); a bare amount with no `$` and no unit
   ("around 50") does not. Note this whole path is **unverifiable on the
   public set**: the evaluator's `classify_constraint()` never buckets any
   sampled constraint as `budget`, so the correct regression check was that
   the fix leaves the score bit-identical, not that it raises it.
4. **No cross-encoder or LLM reranking stage is enabled.** Spec § 4.2.I names
   "LLM Semantic Ranking" as a pillar; the scored pipeline is hybrid retrieval
   + attribute boost + MMR + shown-item exclusion, with no learned reranker.
   A MiniLM cross-encoder stage is implemented and wired, but ships at
   `RERANK_WEIGHT = 0.0` because its sweep did not fit the evaluation budget
   and we do not enable unmeasured switches. Reported as a gap, not as a
   feature.
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

## 8. Team contributions and development process

Required by `docs/competition_specification.md` § Final Deliverables ("a short
report covering architecture, models, cost, limitations, and team
contributions").

### Contributions

| member | contribution to Track 4 |
|---|---|
| **Phan Kang Xun** | All of it: architecture, retrieval design, every experiment and measurement, the evaluation tooling, and this report. |
| **Lloyd Wang** | Registered team member. |

Stated plainly because the specification asks for it: this track's system was
designed, built, measured and written up by one person. The division is not a
reflection of effort withheld — the second member is newer to the stack, and
the work here is dense with retrieval-evaluation methodology that is a poor
first project.

### Development process, and the use of AI assistance

This submission was built with heavy use of an AI coding assistant (Claude
Code) for implementation, refactoring and documentation. `docs/submission_rules.md`
§ Model Policy permits prototyping with any legally accessible model and does
not require this disclosure; we state it because it is materially true.

What the assistant did **not** decide is the part that determined the score.
Every experiment in §6 was specified, run against the frozen 200-sample public
set, and accepted or rejected on its measured number — including the four that
were rejected. The discipline that produced the result is visible in the
repository's own record: a documented rule that an identity setting must
reproduce the shipped score before any swept point is believed, absolute
session counts rather than rates when judging small deltas, and predictions
written down *before* the run that produced them. That rule caught a real bug
in this project — a feature flag whose zero value was not the identity, which
had silently shifted an entire A/B's control leg.

The honest summary is that AI assistance made the implementation fast and the
measurement discipline made it correct, and the second of those is the one
that separated the ideas that worked from the four that did not.

---

## 9. Setup and reproducibility

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
| `scripts/sweep_phrase_weight.py` | phrase-leg RRF weight curve (§6) |
| `scripts/sweep_rerank.py` | cross-encoder weight x scored depth |
| `scripts/sweep_bm25_fields.py` | BM25F per-field weights, coordinate descent |
| `scripts/sweep_boost_weights.py` | disclosed-attribute resort magnitudes |
| `scripts/sweep_prf_weight.py` | pseudo-relevance-feedback leg weight |
| `scripts/build_submission.py` | assemble and verify the submission bundle |
