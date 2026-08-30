# TechJam Conversational Search — Project Notes

## What this is

An AI shopping agent for the TechJam Conversational Search Hackathon. Given a
frozen 50k-product Amazon Clothing/Shoes/Jewelry catalog and a simulated
customer, find the customer's hidden target within 10 turns, asking clarifying
questions along the way.

```
TechnicalScore = 0.50 * HitRate@10 + 0.30 * MRR + 0.20 * Efficiency
Efficiency     = clip((11 - MTTC) / 10, 0, 1)
```

A session ends the moment the target first appears in the top 10. MRR uses the
rank at that first hit, not the best rank ever. A miss after 10 turns scores
MTTC as turn 11. Read `evaluate()` in `evaluator/local_evaluator.py` directly
before changing anything scoring-adjacent — that file, not this summary, is the
authority. Full rules in `README.md`.

## Architecture

1. **Retrieval — four legs, fused by weighted RRF.** BM25 (SQLite FTS5,
   bag-of-words OR), **phrase-match** (FTS5 exact multi-word spans,
   `PHRASE_WEIGHT = 2.0`), dense (bge-small cosine), and **pseudo-relevance
   feedback** (`PRF_WEIGHT = 0.5`, a second BM25 pass seeded from the first
   ranking's top documents, not from the fused pool) — each over the full
   catalog. The phrase and PRF weights are intent-unconditioned; BM25 and dense
   weights are intent-conditioned (buying 2.0/0.5, browsing 1.25/1.5). PRF
   arrived inside #30's unattributed bundle and has never been measured alone.
2. **Shown-item exclusion** (`EXCLUDE_SHOWN = True`). Products already
   recommended are dropped from later slates. Largest win in the project,
   +0.0837 — see Results.
3. **Disclosure extraction.** Every customer message is scanned for
   material/color (vocab match, negation-aware) and budget (`$` amount → price
   bucket) into `SessionState.disclosed`. All three attributes are attempted per
   message, but only one value per attribute is kept, chosen by *vocab order*,
   not text order.
4. **Boost, don't filter** (`Agent._boost_by_disclosed`). The fused pool is
   stably resorted by a categorical match score against `AttributeIndex`: +1 per
   disclosed attribute the candidate agrees on, −1 where known and disagreeing,
   0 where unextractable. Nothing is ever dropped.
5. **Question selection.** Entropy-based pick between `material`/`color`
   (whichever has higher value-diversity in the current pool, recalculated every
   turn). Falls back to `feature, style, size, use_case, budget` — ordered by
   measured answerability, see Results #9. Entropy threshold is
   intent-conditioned (0.10 buying, 0.30 browsing).
6. **Intent-override handling.** A trimmed nearest-prototype embedding
   classifier detects a mid-session "actually, ignore that" pivot; there is no
   structured signal for this in the agent API (`docs/agent_api_contract.json`
   has only `reset_request`, which is a new session). On detection `disclosed`
   and `asked_attributes` are cleared. The query text is cleared **only** when
   the pivot names a product category — see #17/#26.
7. **Browsing-only MMR diversity re-rank** (`Agent._diversify`), pinning the top
   `DIVERSIFY_PIN = 3` and reordering within `DIVERSIFY_WINDOW = 40`.
8. **Long-term user-profile store** (`starter/user_profile.py`). Write-through
   JSON keyed by a hash of the anonymized profile dict. Fully exercised and
   correctly populated; **`SessionState.profile_hint` is deliberately read by
   nothing** — see #5.

**Config lives in `starter/config.py`.** All 64 tunable constants, one file.
Consumers do `from .config import PHRASE_WEIGHT`, which binds the value into the
*consuming* module at import time, so a sweep must patch the module that reads
the knob (`agent_module.PHRASE_WEIGHT = 4.0`), never `config.PHRASE_WEIGHT`.
Vocabularies, regexes and prototype corpora stay in their own modules.

**Intent-conditioned behaviour goes in `routing_params()`** (`agent.py`), which
resolves all four label-dependent knobs in one place. Do not add a fifth
`intent_label == "buying"` branch elsewhere.

Scripts share `scripts/_common.py`, which puts the repo root on `sys.path` as an
import side effect and exports `DEFAULT_CATALOG` / `DEFAULT_DATASET` /
`isolate_profile_store()`. Import it before any `starter`/`evaluator` import.

Data and model paths resolve in three steps (`_resolve_data_path`,
`paths.model_cache_dir`): env var (`TECHJAM_CATALOG` / `TECHJAM_DENSE_INDEX` /
`TECHJAM_MODEL_DIR`), then cwd-relative, then beside the package. They used to
be bare cwd-relative strings, so running from anywhere but the repo root
silently lost the dense index and both classifiers.

## Commands

```
uvx ruff check starter/ scripts/                    lint gate (evaluator/ excluded)
uv run python3 -m evaluator.local_evaluator         official score, 200 samples, ~1h45
uv run python3 scripts/run_eval.py                  dev wrapper: --limit --scenario --seed, progress bar
uv run python3 scripts/fetch_assets.py              the only networked step
uv run python3 scripts/preflight.py --strict        verify the pipeline comes up whole offline
uv run python3 scripts/build_submission.py --verify assemble dist/submission/ and score it
uv run python3 scripts/measure_latency.py --limit 20  numbers for team_report.md §4
uv run python3 scripts/repl.py                      interactive session: /reset /debug /topk /quit
uv run python3 scripts/build_demo.py [--capture]    dist/demo.html replay of demo/sessions.json
```

Diagnostics, all read-only and fast unless noted:

```
eval_override.py         sweeps override-detector rules (#7)
eval_intent.py           sweeps intent-classifier rules (#13)
train_intent_head.py     trained head vs centroid; verdict is don't ship (#12)
eval_profile_signal.py   does user_profile carry any signal (#5) — run before any personalization idea
eval_ceiling.py          is the target reachable at all (#27) — one turn per sample
eval_failures.py         miss taxonomy, probes every leg to depth 2000 (#24) — ~1 hour
cooccurrence.py          P(color|category) etc. over the catalog (#11)
sweep_fusion_weights.py  TechnicalScore vs dense:bm25 ratio (#16)
sweep_prior_leg.py       a third RRF leg from catalog priors or profile — never run
ab_phrase_query.py       the five legs of #25 — five full evals, run alone
ab_buying_diversify.py   MMR on the buying track
```

Design docs: `docs/question_policy_plan.md` (written, not built),
`docs/intent_detection_plan.md`, `docs/user_profile_decision.md` (judge-facing),
`NEXT_STEPS.md`, `REPRODUCE.md`, `docs/team_report.md`.

## Score history

Full public set, 200 samples. **HEAD is `a9d3999`, measured 2026-08-30.**

```
HitRate@10 0.910 (182/200)   MRR 0.4906   MTTC 3.905   Efficiency 0.7095
TechnicalScore 0.7441
```

Per-scenario: browsing 0.950 / 0.5534 / 3.413 (n=80), buying 0.8875 / 0.4143 /
3.763 (80), intent_override 0.9333 / 0.5995 / 5.067 (30), boundary 0.700 /
0.2725 / 5.500 (10). `results.json` is this run.

```
score    what changed                                          entry
0.7441   #30 eight-knob bundle, PRF leg on       +0.0463   #30  (unattributed)
0.6978   #26 category IDF + request stopwords    -0.0042 = 1 session, flat
0.7020   EXCLUDE_SHOWN                           +0.0837   #23
0.6454   phrase leg at PHRASE_WEIGHT 2.0         +0.0272   #20
0.6182   buying dense:bm25 ratio 0.50 -> 0.25    +0.0112   #16
0.6070   override detector TOP_PROTOTYPES = 4    -0.0003   #7  (noise)
0.6073   FALLBACK_ATTRIBUTE_ORDER by answerability +0.0109 #9
0.5964   dual-track buying/browsing routing       ~flat    #4
0.600    boost-not-filter replaces hard filter   +0.017    #1
0.583    (starter BM25-only baseline: 0.125 hit / 0.068 MRR / 9.81 MTTC)
```

**The benchmark's resolution is one session ≈ 0.004 TechnicalScore.** Convert
every rate to an absolute session count before believing a delta. 0.8550 vs
0.8500 is one session, not a trend.

Facts confirmed empirically, not assumed:

- Catalog attribute coverage: material ~70.9%, color ~39.9%, price ~21%
  (post-#10 word-boundary fix; the pre-fix figures 78.1/58.6 were inflated).
- `classify_constraint()` **never** buckets any of 800 sampled disclosed values
  as `budget` — asking about budget locally is a guaranteed wasted turn. Root
  cause: `intent_card()` appends the price candidate last and the
  `hard_constraints[:2]` / `soft_preferences[2:4]` cap truncates it out.
  `budget` is demoted to last-resort, not removed, in case that is a
  local-simulator quirk.
- `category`, `brand`, `other` are structurally unreachable — no code path picks
  them, and `classify_constraint()` never buckets into them either.
- `ALLOWED_ATTRIBUTES` is copied verbatim from the API contract's enum.
- The QSBPS paper behind the entropy design (Zou & Kanoulas, "Learning to Ask")
  is CIKM 2019, not SIGIR. Its mechanism is Generalized Binary Search over
  relevance mass via binary questions; the multi-way categorical entropy here is
  a deliberate adaptation, not a citation match.

## The confound that explains most negative results

**89.7% of local turn-1 hard constraints (359 of 400) are verbatim substrings of
the target product's own catalog text**, because `intent_card()` builds customer
messages out of the target's own fields. The local set is therefore close to a
pure exact-match benchmark: best case for BM25, worst case for anything that
adds semantic tolerance or removes text.

Four separate losses trace to it — #14 (dense weight), #17 (override rewrite),
#19 (query pruning), #26's corpus-driven stopword list. **Any change that
deletes or paraphrases customer text will measure badly here**, and that may or
may not be true of the hidden grader. Weigh the downside asymmetry before acting
on a local gain of this shape.

## Results and verdicts

Numbered entries are historical and referenced across the repo; keep the
numbering stable.

### Shipped and established

**#30 — Eight-knob bundle, +0.0463 to 0.7441. The largest jump since #23, and
none of it is attributed.** Commit `a9d3999` changed eight knobs at once and was
measured once. Every diff against `5889828`:

```
knob                          before                       after
PRF_WEIGHT                    0.0                          0.5
SKIP_NON_ANSWERS_IN_QUERY     False                        True     <- reverses #19
DIVERSIFY_WINDOW              20                           40
DIVERSIFY_PIN                 2                            3
DEFAULT_FIELD_WEIGHTS[5]      1.5                          0
NEGATION_WINDOW_CHARS         20                           10
NON_ANSWER_SPILLOVER          0.6                          0.8
DENSE_QUERY_PREFIX            "...relevant passages: "     "...relevant shopping items: "
SCOPED_OVERRIDE_CLEAR         (new)                        False
```

The gain is real and far above the noise floor: +12 sessions of HitRate
(170 → 182), MTTC down 0.305, and MRR up 0.0339 — **all three score terms move
the same way**, which per #23 is the signature of a mechanism rather than a
reshuffle. HitRate rises in every scenario. So the bundle stays.

**What is not known is which knob did it, and that is a real cost.** Eight
changes, one measurement, zero attribution. Two of them are individually
suspicious: `PRF_WEIGHT = 0.5` turns on a leg that had never been run at all,
and `SKIP_NON_ANSWERS_IN_QUERY = True` re-enables the exact change #19 measured
at −0.0410. Those cannot both be neutral. The plausible reading is that PRF is
carrying the bundle and paying for a knob that still costs, in which case there
is more than +0.0463 available by reverting the losers — but that is a guess
until someone varies them one at a time.

The dense-prefix edit deserves separate suspicion. bge-small's published
retrieval prefix is the literal string `"Represent this sentence for searching
relevant passages: "`; the index was built under that prefix, so querying under
a different one puts query and document embeddings in mismatched conditioning.
Per #14 the dense leg contributes no HitRate locally, so this could measure flat
here and still be wrong on a grader that paraphrases.

**Do not bundle again.** This is the second time the repo has learned that eight
knobs measured once produce a number and no knowledge — and unlike #28's null,
this one is a win nobody can defend or build on.


**#23 — Shown-item exclusion, +0.0837.** The largest win by a factor of three,
and the only change that moves all three score terms the same way at once
(0.7550/0.3989/4.940 → 0.8550/0.4622/4.205 in `ab_phrase_exclude.py`). The
evaluator ends a session at first hit, so being asked for another turn is *proof*
every item shown so far is wrong; re-offering them spends slots on known-dead
answers. It needed no new signal — just a re-read of `evaluate()`. **When stuck,
read the scorer again before building anything.** Caveat: it assumes the hidden
grader also ends on first hit. That is in the rules and in `evaluate()`, but it
is a stronger scorer-semantics dependency than anything else shipped.

**#20 — Phrase-match leg, +0.0272**, and the first retrieval change here that
added *recall* rather than reordering (+8 sessions of HitRate; the dense leg's
HitRate is identical to four decimals whether on or off). N-grams longest-first,
weighted `len(gram)**2`, discarding spans matching more than
`PHRASE_MAX_MATCHES = 150` products.

```
weight   hits      MRR     MTTC  Technical    delta
  0.00  150/200  0.4096    4.985   0.6182       --
  0.50  152/200  0.4278    4.945   0.6294   +0.0112
  1.00  154/200  0.4184    4.870   0.6331   +0.0149
  2.00  158/200  0.4176    4.745   0.6454   +0.0272   <- shipped
  4.00  157/200  0.4254    4.825   0.6436   +0.0254
```

Real interior optimum, but 2.00 and 4.00 are one session apart, so the top is
flat and 2.00 is the lower-variance pick. The reasoning was stated *before* the
sweep and is the reusable part: a phrase leg runs *with* the exact-match
confound instead of against it. Gotcha: `phrase_tokens()` keeps stopwords,
unlike `terms()` — "pull on closure" collapses to "pull closure" under `terms()`
and matches nothing.

**#16 — Buying fusion ratio 0.50 → 0.25, +0.0112.** The buying dense:bm25 curve
is strictly monotone decreasing with no interior optimum; 0.50 was already well
down the slope. HitRate falls too (0.750 → 0.665 across 0.0–1.5), so at high
dense weight the dense leg displaces correct BM25 results out of the top 10, not
merely reorders them. Ratio 0.00 is worth +0.0188; 0.25 captures 60% of that
while keeping a real dense leg as paraphrase insurance.

**Browsing track is measured flat across 0.0–1.5. Do not re-run that sweep.**
The browsing-scenario hit column moves by at most one session over the whole
range. Also: the two tracks are **not separable populations** — the weights key
off the *classifier's* per-turn label while `scenario_type` is ground truth, and
the classifier drifts buying-ward as attributes accumulate. Varying browsing
weights visibly moved buying-scenario metrics.

**#9 — `FALLBACK_ATTRIBUTE_ORDER` sorted by answerability, +0.0109.** Share of
sessions where the customer can still answer after turn 1, derived from the
evaluator's own reply policy:

```
feature 0.960   material 0.725   color 0.255   style 0.085   size 0.045   use_case 0.020
```

The old order asked the three rarest buckets first. Now `feature, style, size,
use_case, budget`. 39 samples reach the target sooner, 5 later, modal delta −3
turns on 24 samples — broad, not lucky. Two things to carry forward. **MRR moved
the wrong way** (−0.0061), because converting a late well-ranked hit into an
early worse-ranked one costs MRR and pays more in Efficiency; MTTC work will keep
showing that drag, so do not judge a change on MRR alone. And **the 96% is partly
an artifact** — `feature` is `classify_constraint()`'s catch-all return.

**#7 — Override detector: trimmed nearest-prototype + lead clause.** Score
against the mean of each class's `TOP_PROTOTYPES = 4` closest prototypes instead
of the centroid, and evaluate both the full message and its lead clause, taking
whichever leans more override.

```
 k   sim recall   sim FP/1600   probe recall   probe FPR
 1        0.933      242              1.000       0.400
 2        1.000       25              1.000       0.200
 3        1.000       12              1.000       0.100
 4        0.933        2              1.000       0.100   <- shipped
 5        0.933        0              0.900       0.100
12        0.900        0              0.900       0.100   (= centroid)
```

Root cause of the centroid's failure: all twelve override prototypes are long
two-clause sentences naming the discarded prior statement, so the mean encodes
that *sentence shape*, and a terse pivot ("never mind, give me white shoes") is
dominated by its imperative half and lands nearer the continuation centroid.
Margins were ~0.03 in a space where both centroids sit at 0.65–0.80 similarity
to almost anything.

End-to-end this is ±0.0005 across k=centroid/3/4 — **below resolution, decide on
the offline table, not the score.** The local simulator cannot measure this at
all: `behavior_for()` emits exactly one override string, near-verbatim
`PROTOTYPE_OVERRIDE[0]`, so simulator recall was 0.900 by construction. The probe
column is the only evidence about phrasings the hidden grader might use, and it
is hand-picked.

**#26 — Category-word IDF and request stopwords. Measured −0.0042 = one session,
flat.** Two user-reported demo bugs, one root cause. Every product hangs off the
root category `"Clothing, Shoes & Jewelry"` (49,990 of 50,000), so those words
sit at df ≈ 1.000 and score ~0 IDF, while request verbs are *rare* in a product
catalog and therefore high-IDF:

```
shoes   df 1.000 (0.235 without the root)     actually df 0.0018
jewelry df 1.000 (0.111 without the root)     forget   df 0.0032
                                              buy      df 0.0261
```

So `"i want to buy shoes"` reduced to `{buy, shoes}`, "shoes" was worth nothing,
and the query was decided by "buy" — which matched the store name "Buy Caps and
Hats". Three fixes: `BM25Index._build` strips the universal root (detected from
the first insert batch, not hardcoded); `_REQUEST_STOPWORDS` in `text_utils.py`;
`SPELLING_VARIANTS` normalizes en-GB to the catalog's en-US via
`normalize_query()` in both query builders, so all four legs see one string
("jewellery" matches 146 products, "jewelry" matches 50,000; "grey" is
deliberately excluded, since it beats "gray" 2017 to 751 here). Separately
`BUYING_PHRASES` had no purchase verb at all, so "i want to buy shoes" scored
zero buying evidence — added, turn-1 intent 0.975 → 0.981.

**The pre-registered risk did not materialize, and that is the finding.**
`_REQUEST_STOPWORDS` strips tokens from every query, the same shape that cost
−0.0410 in #19, and this was flagged before the run. One session, not twelve. It
vindicates the **closed-list discipline**, not query pruning generally.

Override rewriting, narrowly: `respond()` drops query history on a detected
pivot **only when the pivot names a product category**
(`BM25Index.names_category`), built from single-word category nodes with df ≥ 20,
excluding material/colour and demographic words. Any-token fires on 6 of the
evaluator's 30 override turns; single-word-only fires on 2. Tokens from
multi-word nodes are modifiers, not product types ("Water Shoes" → "water"), and
firing on them turns the rewrite back into #17.

**#25 — Phrase-query hygiene: the fixes work, but the dumb control matches
them.** Both flags are **off**.

```
leg                     hits      MRR     MTTC  Technical    delta
identity              171/200   0.4622    4.205   0.7020       --
filter only           174/200   0.4826    4.090   0.7180   +0.0159
clause+edge only      176/200   0.4946    3.915   0.7301   +0.0280
filter + clause+edge  177/200   0.4981    3.900   0.7339   +0.0319
budget 96 (control)   177/200   0.4998    3.995   0.7326   +0.0305
```

Span-budget starvation is real and was costing ~6 sessions (`public_0008` builds
135 spans, all 24 that fit the budget match zero products, while "bras everyday
bras" — verbatim in the target — sits unqueried). But quadrupling
`MAX_PHRASE_QUERIES` to 96 reaches the same 177/200. The fixes win on **cost**,
not score: same result at the original budget of 24 by not building the useless
spans (135 → 38, or 11 with both). Only `clause+edge` is established on its own;
the filter adds one session on top, which cannot be resolved here.

**Do not enable either flag without re-measuring on current HEAD.** The edge
rule reads `STOPWORDS` directly, and #26 added ~24 words to that set, so the
table above was measured against a `STOPWORDS` that no longer exists.

**#21 — `BM25Index` was single-thread-only**, silently disabling two of three
legs under any server front-end. `sqlite3.connect(":memory:")` defaults to
`check_same_thread=True`; Streamlit caches one `Agent` and serves from a worker
pool, so every `search()` raised `ProgrammingError` — and `_retrieve` catches
per-leg exceptions and continues, so the agent returned ten products from the
dense leg alone with nothing surfaced. Fixed with `check_same_thread=False` plus
a `threading.Lock` across every query including the PRF expansion path.

**#15 — The offline degrade is silent.** With a cold HF cache and no network,
`load_embedding_model()` raises, and `Agent.__init__`'s degrade-don't-crash
branches take dense retrieval, both classifiers and the masked LM dark at once
while the agent still starts and still returns 10 recommendations. `model/`
(385MB) and `data/dense_index/` (74MB) are gitignored and the bge-small blob is
over GitHub's 100MB limit. `docs/submission_rules.md` warns network may be
disabled for scoring. **Run `preflight.py --strict` before any scored run.**

**#22 — Intent prototypes were missing the "something for <occasion>" shape**, a
deliberate accepted regression of 2 turn-1 errors in 160 (0.988 → 0.975; #26
later recovered one). "i want something for the summer" classified buying at
+0.0058, a coin flip. Four browsing prototypes of that shape were added. **Do not
"fix" this with a deadzone** — one was tried and swallowed real signal broadly
(buying turn-3 96% → 59%), because the zero-crossing region is genuinely
populated by weakly-buying cases. Accepted because #13 measured the label is
worth ~nothing to score, while a visibly wrong route in a live demo costs
credibility. **Only valid while that stays true.**

**#10 — Catalog attribute extraction was substring-matched, not
word-boundary-matched. Fixed; the fix costs 0.005 and was kept anyway.** Found
while building co-occurrence stats: the strongest "signal" in the catalog was
`P(material=lace | Necklaces)` = 0.867, which is "neck**lace**" containing
"lace". Both the evaluator and this repo's customer-side extractor always used
`\b`; only the catalog side was loose.

```
color:    21.6% of products mislabeled (10,824), 9,312 of them invented where no
          colour word occurs at all ("red" in embroidered, "tan" in titanium)
material:  4.7% mislabeled (2,344), dominated by lace <- necklace
coverage: color 58.6% -> 39.9%,  material 73.7% -> 70.9%
```

This is the root cause of #1's unexplained 16.3%/37.0% disagreement with the true
target. The fix measures −0.0054 (3 samples), reproduced with and without #11.
Best understanding: the buggy extractor labelled 58.6% of the catalog with a
colour, and `_boost_by_disclosed` penalizes a *known* mismatch while ignoring
unknowns, so the wrong labels gave the resort more (noisy) separation.
Correct-but-sparser labels leave more candidates neutral. The indicated next step
is **not** to revert the bug but to retune the boost weights against corrected
coverage — the ±1/0 scheme was tuned when a fifth of colour labels were
fictional.

**#29 — Lexical override cues beside the embedding rule. Shipped; not measured
end-to-end.** `EmbeddingOverrideDetector.is_override` now fires if *either* a
closed list of literal discard phrases matches or the trimmed-prototype
similarity leans override. Union, not a vote, because the two rules fail on
disjoint inputs. A terse pivot is mostly its request half — the same shape
pathology `TOP_PROTOTYPES` was tuned against in #7 — and a literal cue cannot
be outvoted by the rest of the sentence. A paraphrase carrying no cue at all
("My priorities changed…") has no list entry and needs the similarity.

```
pool                                        regex alone   combined
PROTOTYPE_OVERRIDE (12)                         10/12         --
#7 out-of-distribution probes (10)               9/10       11/11 *
continuations + non-answers + templates (35)      0 FP        0 FP
```

\* the combined column was run on the 10 probes plus the one prototype the
regex misses; it is a smoke test on 11 strings, not a measurement.

Two constraints hold the false-positive side down, and both are #26's
**closed-list discipline** rather than #19-style pruning. Bare "actually" and
bare "no" are not cues — "actually I also need it waterproof" is a
continuation, and "no preference there" is the evaluator's own non-answer
shape. And a cue only counts **clause-initially**: matched anywhere, it would
fire on catalog copy carried inside a reply ("For that, what matters is: …").

**What is not measured, and why it matters here.** No full-set run — the box
was saturated by a second session at load ~41. The regex is strictly
recall-adding, so the risk is entirely false positives, and #23 inverted that
error asymmetry: `respond()` now also clears `state.shown` on a detection, so
a spurious clear re-offers slates already proven dead. The 0-FP column above
is partly by construction, since the negative pool overlaps
`PROTOTYPE_CONTINUATION` and `PROTOTYPE_NON_ANSWER`. **The honest FP number is
`eval_override.py`'s harvested simulator turns, and it has not been run against
this rule.** That script sweeps standalone rule classes and never calls
`is_override`, so measuring the union means teaching it about the union first.

### Do not retry — measured and rejected

**#5 — Personalization from `user_profile`. Three implementations, all
regressed; the field carries no signal to act on.** `scripts/eval_profile_signal.py`
shows same-profile-key sessions share a coarse category **0.5%** of the time
against a **1.2% ± 0.5%** random-pair baseline — a key match is not an identity
signal, so there is nothing correct to carry. The store still runs (125 distinct
keys, 30 seen more than once) with zero net effect. Full write-up and the
store-contamination landmine: `.claude/skills/retrieval-experiments/SKILL.md`.

Judge-facing version with two further null tests:
**`docs/user_profile_decision.md`**. The one worth knowing is
`average_prior_rating` → target `average_rating`: **r = 0.1824, permutation
p = 0.0094 over 20,000 shuffles — passes significance and is still rejected.** It
explains 3.3% of variance, runs backwards between its two largest cells (prior
1.0 → 4.393 vs prior 5.0 → 4.413), and dropping one 9-sample cell collapses it to
r = 0.0929, inside the null band. About a dozen tests were run across three
fields, so one pass at 0.05 is what chance produces. `purchase_frequency` is the
constant `"3-4 prior purchases"` on all 200 samples and `summary` is a template
restatement of the other fields, so neither needed testing.

**The live lead that came out of it is profile-independent.** `rating_number` is
the only 100%-covered catalog field and targets come overwhelmingly from the
popular tail: catalog median 12 reviews, target median 7,078, and the top 5.9% of
the catalog by review count holds **83% of all targets, a 14x lift**.
`scripts/sweep_prior_leg.py` was built for exactly this and **has never been
run**. Its `popularity` variant was designed as the *control* for the profile
idea, and the control is the leg with the effect size. Two caveats: it is a
re-ranker inside an existing pool, so it cannot touch the elicitation gap #27
identifies as the real limit; and the concentration is a property of how the
samples were drawn, not of shopping, so it transfers only if the hidden set was
drawn the same way. Run `--weights 0` first and confirm it reproduces the
current HEAD score — that is **0.7441** as of `a9d3999`, not the 0.6978 this
note was written against.

**#19 — Query hygiene on non-answers, −0.0410.** The gap was real: roughly three
asks in five come back empty (mean 2.09 answerable buckets left after turn 1
against MTTC ~5), and every reply was being joined into the BM25 and dense query.
`EmbeddingNonAnswerDetector` detects them well offline (recall 1.000, probe FPR
0.000). Dropping them costs 12 sessions (149 → 137).

**The prediction that was wrong, and why it matters.** All 69 false positives are
the simulator's near-contentless template ("For that, what matters is: Imported;
Pull On closure."), and this was written up mid-run as evidence the FPR column
measured the wrong thing. The A/B says the opposite: per the confound above,
`Pull On closure` is not boilerplate, it is an exact-match key to the target.
**No form of query pruning is safe on this benchmark**, and improving the
classifier cannot fix it — the text is contentless in meaning and load-bearing in
retrieval. The non-answer *observation* is kept and feeds `SessionBelief`; only
the query surgery was reverted.

**Status changed 2026-08-30: `SKIP_NON_ANSWERS_IN_QUERY` is `True` again**, inside
#30's bundle, and the bundle scores +0.0463. That does not overturn the −0.0410
above — the flag was never varied on its own in that run, so its individual sign
is unknown. Either #19 was wrong, or #19 still holds and the other seven knobs are
worth more than +0.0463 between them. **Both readings are live; do not cite #19 as
settled in either direction until the flag is A/B'd alone on `a9d3999`.**

**#17 — Query rewriting on intent override, −0.0580.** The gap was real
(`respond()` cleared slots but never touched `first_message`, so `_build_query`
was byte-identical before and after a pivot). Restarting from the pivot message
is catastrophic: intent_override hit 0.800 → 0.333, MTTC 5.90 → 9.27. Root cause
found by printing a sample rather than reading the template's shape — the
category is stated *once*, in turn 1, and the pivot carries only the changed
attribute:

```
TURN 1: "I'm looking for [... 'Men', 'Accessories', 'Belts']. Buckle closure"
PIVOT : "Actually, ignore my earlier preference. What I need is: leather"
```

Dropping `first_message` throws away "Belts" and searches 50k products for
"leather". This **refutes** a claim NEXT_STEPS §5 once stated as established —
that the override template carries the complete new constraint. Any selective
scheme must preserve the framing and drop only the preference clause; #26 does
the narrow version.

**#14 — The dense leg is net-negative locally, and we shipped it anyway.**

```
configuration          HitRate     MRR     MTTC   Technical      delta
as shipped              0.7450  0.3876    5.090     0.6070          --
- masked LM             0.7450  0.3841    5.085     0.6060     -0.0010
- both classifiers      0.7400  0.3728    5.070     0.6004     -0.0065
- dense retrieval       0.7450  0.4328    5.035     0.6216     +0.0147
- everything (offline)  0.7450  0.4289    5.025     0.6207     +0.0137
```

HitRate is identical to four decimals with and without dense: it finds nothing
BM25 misses, and only demotes correct items BM25 ranked well. That is 20x the
noise floor. **Do not act on it** — see the confound section. If the hidden
grader paraphrases at all, dense is the only defense; the downside is asymmetric.

**#12 — Trained intent head: built, measured, rejected.** Logistic head on
*frozen* bge-small embeddings (the encoder is never touched — TODO.md 4.3 bars
fine-tuning base models, while intent modules are explicitly in scope).

```
pool                        head    centroid
in-distribution 5-fold CV   0.984      0.778   <- memorization, ignore
train turn1 -> test accum   0.521      0.708   <- chance
out-of-distribution probes  0.812      1.000
```

The simulator has exactly two turn-1 templates, so 0.984 is the head learning
`"still exploring"`. Three checks confirm it: a regularization sweep is flat at
chance across C = 0.001…10; OOD accuracy *rises* with C (the more it overfits,
the better it scores OOD — the opposite of a generalization story); and training
on the 20 hand-written prototypes alone, with no simulator text, reaches OOD
1.000 while 640 labelled simulator turns drag it to 0.812. **Do not ship a head
trained on local simulator text, and never read a high in-distribution CV score
as progress.**

**#13 — The trimmed-prototype rule does not transfer to the intent
classifier.** Every variant is worse:

```
rule                    turn-1   accumulated   OOD probe
lexical (fallback)       1.000         0.773       1.000
centroid (current)       0.988         0.708       1.000   <- unchanged
top1-mean                0.781         0.608       0.938
top4-mean                0.769         0.588       0.938
```

Why it transferred to one classifier and not the other: the override prototypes
had a *shape* pathology (#7); the buying/browsing prototypes are varied single
utterances whose mean direction *is* the signal, so trimming throws away the
averaging that makes the centroid robust. **"Centroids are fragile" is not a
general truth about this codebase — it was a fact about one prototype set.**

**There is no headroom to chase here.** The centroid is at 0.988 on turn-1, its
only errors being buying sessions whose hard constraint is contentless ("A key
requirement is: Imported."), and per #4 the thing the label feeds was itself
measured flat. A perfect intent label is worth approximately nothing.

Three measurement caveats. The lexical fallback's turn-1 1.000 is template
memorization — its regexes match the simulator's two templates literally. The OOD
probe pool is partly circular: the lexical rule scores 1.000 by a degenerate path
(browsing probes register 0/0 evidence and fall through to the tie-break), which
is true because the probes were hand-written that way. **Probes can falsify a
rule but must not rank two rules that both pass.** And `accumulated` is not an
accuracy — `scenario_type` is not valid ground truth once clarifying answers
land, since the classifier is built to drift buying-ward.

**#28 — MMR diversity on the buying track: null, and the pre-registered test is
what caught it.** Identity reproduced 0.697796 exactly. The treatment scores
+0.0030, which is under one session, and every part of that is MRR:

```
leg          HitRate     hits      MRR     MTTC   Technical    delta
identity      0.8500  170/200   0.4567    4.210     0.6978       --
treatment     0.8500  170/200   0.4680    4.230     0.7008   +0.0030
```

**HitRate is identical and so is the crowding table, quartile for quartile.** The
diagnosis was that buying misses sit in categories 2.4x more crowded than its
hits, and the falsification stated before the run was that a real gain lands on
crowded buying sessions. Nothing was recovered anywhere — 170 hits before, the
same 170 after.

34 sessions reshuffled, and more got worse than better (19 vs 15). The entire net
comes from five sessions that jumped from rank 7–10 to rank 1; **delete those five
and the delta is −0.0102.** Bootstrap over the 34 deltas gives a 95% CI of
[−0.0091, +0.0335] with 14.8% of resamples at or below zero. That is a lottery on
five near-miss slates, not a mechanism.

Reusable: **an MRR-only move with HitRate flat to four decimals is a reordering of
sessions you already win**, and on a metric where rank 10 → 1 is worth 0.9 while
rank 1 → 3 costs 0.67, a handful of those swamps the other thirty. Check the
per-session deltas before believing any MRR delta under ~0.02. `BUYING_DIVERSIFY`
stays `False`.

**#18 — Turn-annealed slate diversity: null.** Annealing `DIVERSIFY_LAMBDA`
upward as the turn budget depletes measured −0.0038 and −0.0030 on two schedules.
The 0.50 → 0.90 run's browsing metrics are byte-identical to fixed lambda, so the
schedule changed nothing in the track it governs. Note the 0.35 start was worse
because at turn 1 the query is the raw customer message whose top hits are
usually already right — **early-turn diversity is not free here.**

**#4 — Dual-track routing: net flat on the full set** (0.600 → 0.601) despite a
gain on the 60-sample tuning subset. Kept. **Do not retry the hard
filter-then-rerank buying track** (BM25 top-N as a hard filter, dense reranking
within it): measured worse on hit rate and MTTC, because it discards the
full-catalog dense leg that exists to catch differently-phrased targets.

**#1 — Hard-filter self-elimination: fixed.** The old hard-equality filter
dropped candidates disagreeing with a disclosed value, which disagreed with the
*true target* 16.3% of the time on material and 37.0% on color. Replaced by
`_boost_by_disclosed`. **Do not retry reranking by embedding similarity between
the disclosed phrase and catalog embeddings** — swept and monotonically worse as
its weight rose, below even a no-signal baseline.

**#2 — SQL-backed attribute columns.** Raised by a user; the override-clear
semantics argument stands but the motivation (#1) is gone, and at 50k rows with
~30-item pools the speed argument over the Python resort is a wash.

### Where the remaining misses are

**#27 — Every target is reachable. #24's "recall problem" was an artifact of the
starved session query.** `scripts/eval_ceiling.py` hands the agent everything the
customer could ever disclose in one turn-1 message. The customer's entire
vocabulary for a whole session is `coarse_category(target) +
hard_constraints[:2] + soft_preferences[2:4]` — at most four strings, all sliced
from the target's own record. There is nothing else they can say, ever.

```
oracle (all constraints, one turn, one slate)   173/200
actual (10-turn dialogue)                       171/200

of the 27 oracle misses:  top 10  12 | 11-50  10 | 51-500  5 | not found at 2000  0
worst best-leg rank across all 200 samples: 192
```

**No target is unreachable.** #24's "no leg finds the target at all" is true only
of the query the agent actually holds at the end of a live session, which is
starved — `customer_reply()` discloses a constraint only when
`classify_constraint()` buckets it as the attribute just asked, at most two per
turn. Read #24's "only more information can fix these" as being about
**elicitation**, not reach. **The oracle number is not a ceiling** — an earlier
draft wrongly called it one. It returns one slate of ten where a live session
returns up to ten (up to 100 distinct products under `EXCLUDE_SHOWN`), and the
two conditions differ on 34 samples, 17 in each direction.

**A real extractor defect, and the fix it suggested is falsified.**
`AttributeIndex` stores one material and one colour per product chosen by vocab
order; the evaluator picks from the same text by a different rule. Over the 200
targets:

```
                                                  material   colour
agree -> +1                                          58.5%    12.5%
FALSE mismatch: value IS in the product -> -1         9.5%     3.0%
disagree, value outside the fields we read -> -1      6.5%     1.0%
customer never states it                             23.5%    80.0%
```

`public_0008`: product text carries `{nylon, polyester, spandex}`, we store
`polyester`, the customer says `nylon`, the target takes a −1 below every neutral
candidate. 7 of the 27 misses are penalized this way, all buying, including
`public_0111` (the rank-5-to-80 anomaly #24 flagged). **But replaying those 7 at
`DISCLOSED_MISMATCH_PENALTY = 0.0` recovers none of them.** The "all legs rank it
top-3" observation came from the *oracle* query; the live turn-1 query is one
material word and the target is never in the pool for the boost to sink. Same
mistake shape as #24 — a probe run with a richer query than the agent has.

The customer side has the identical limitation: `extract_disclosed_value` returns
the first vocab match, so "polyester and cotton blend" → `cotton`. Measured: 22.6%
of single constraint strings, 44.4% of 2-constraint replies and 63.9% of target
products carry 2+ materials. Both sides collapse a multi-valued field to one
value by different rules, which is why **every** boost mismatch across all 80
buying sessions is a *false* one. The representation is genuinely wrong and its
measured value is currently zero. If anyone builds the set version: put it behind
a `contains()` used **only** by the boost, leave `value_for`/`values_for`
returning the canonical single value so the entropy picker stays bit-identical,
measure on the full set rather than those 7, and do not bundle a weight retune
with it.

**#24 — Miss taxonomy at 0.7020.** Superseded in part by #27; read that first.
27 misses (the run died at sample 187 under memory pressure, so a near-complete
census):

```
by reason                      by scenario
  lost-in-fusion    17           buying           15
  ranked-11+         6           browsing          6
  unreachable        3           intent_override   3
  excluded-as-shown  1           boundary          3
```

**`lost-in-fusion` is a mislabel** — it means "the fused pool did not contain the
target", not "fusion discarded it". 16 of the 17 have no leg ranking the target
better than ~198. The label describes the probe's depth, not a defect. 15 of 27
misses are buying, and the phrase leg reaches the target in only 2 of 27, both
following from the same cause: a buying session's turn-1 signal is usually a
single very common material word ("cotton" → 9,414 products), so there is no
multi-word span to match.

**Buying misses correlate with category crowding; browsing does not.** Median
catalog products sharing the target's coarse category, hits vs misses:

```
scenario           hits   misses   p (permutation, 20k)
buying              138      338    0.0096  *
browsing            173      212    0.4212
intent_override     141      117    0.6080
```

Buying's misses sit in categories 2.4x more crowded than its hits. Boundary is
browsing minus one elicitation turn (`initial_message()` gives it the browsing
opener; `customer_reply()` returns "I don't have a preference" once), and its gap
survives crowd-matching only weakly — 6/10 vs 23/25, p = 0.043, on 4 misses at
n = 10. `ab_buying_diversify.py` tests whether the browsing MMR re-rank, the only
knob that spreads a slate across a category, closes the buying gap.

### Built but not wired / not shipped

**#11 — Attribute inference.** Two attempts at inferring an unstated attribute.

*Item-attribute co-occurrence* (`scripts/cooccurrence.py`): the signal is
enormous once #10 is fixed (`P(material=stainless steel | Watches Wrist
Watches)` = 0.483 vs 0.023 marginal, +79σ). **Built, not wired.** It lives in
`scripts/` because `starter/` ships verbatim into the bundle. Note this is *item*
co-occurrence — the shipped data has no user-item interactions at all, so "users
who bought X also bought Y" is not computable here.

*Local masked-LM belief with an entropy gate* (`starter/lm_confidence.py`, wired
into `_infer_attributes`): distilbert-base-uncased scores the closed vocabularies
in a `[MASK]`ed template. A local model is required — the hosted Claude API
returns no logprobs, and bge-small's published weights carry no LM head.

**The entropy gate works, and that is the solid finding:**

```
material   top-1 0.483 vs 0.322 guess-the-mode baseline
  H < 0.60      n= 61   accuracy 0.787
  H 0.60-0.75   n=105   accuracy 0.371
  H 0.75-0.85   n= 14   accuracy 0.000
```

Monotone and steep, so `MAX_CONFIDENT_ENTROPY = 0.60` is calibrated. **Colour is
deliberately excluded**: top-1 0.189 against a 0.432 always-"black" baseline, its
gate runs *backwards*, and it collapses onto one attractor ("pink" for 14 of 37).
**But the end-to-end payoff is +0.0010 — noise**, reproduced twice. It fires only
on material, which the customer already discloses in 72.5% of sessions, and it
must be boost-only. Cost: 246 ms per inference and 255MB of shipped asset. The
mechanism is validated; the application point is low-leverage. The higher-leverage
use would be predicting *which attribute the customer can answer*.

**#8 — LLM-extracted catalog attributes (offline, one-time).** Designed and
costed, nothing built; working notes in `NEXT_STEPS.md`. A one-time
`claude-haiku-4-5` Batch API pass over the 50k catalog (~$5–12, ~9.4M input
tokens) emitting a static closed-vocabulary JSON sidecar shipped as a local asset,
so the agent still needs no network at scoring time. Motivation: thin coverage,
#27's single-value defect, and the four attributes (`style`, `size`, `use_case`,
`feature`) that have no extractor at all. Caveat: it fixes only the catalog half
of the disagreement.

**#3 — Budget extraction: fixed, and it was three bugs, not one.** (1) The
`\$\s?(\d+)` pattern meant "less than 80 dollars" disclosed nothing. (2) The
comparator was discarded and the amount mapped to the bucket *containing* it,
then compared for equality — so "under $80" resolved to the top bucket `$48+`
and a stated ceiling **boosted the priciest products while penalizing every
cheap one**. (3) `_extract_budget` returned the literal string `"unknown"` for
the 79.2% of the catalog with no price, which is a known value rather than
`None`, so any budget disclosure scored a full mismatch against 39,590 of
50,000 products.

Now `parse_budget_constraint()` returns a bound (`"<=80"` / `">=200"`),
`AttributeIndex.price_for()` exposes the raw price, and `_boost_by_disclosed`
compares numerically with unpriced products neutral. **Unverifiable locally** —
`classify_constraint()` never buckets a disclosure as `budget`, so the correct
expectation is that the public set is bit-identical; that is the regression
check, not a win condition. One judgment call: a bare or approximate amount
("my budget is $80") reads as a *ceiling*, the dominant sense of a stated
shopping budget.

**#6 — No cross-encoder or LLM reranking stage.** `RERANK_WEIGHT = 0.0`, so the
stage is still dark at `a9d3999`; spec 4.2.I mentions "LLM Semantic Ranking".
Current ranking is hybrid retrieval + attribute boost + MMR. `RERANK_BACKEND` is
`"minilm"` with `RERANK_TOP_N = 20`; `RERANK_MODEL_NAME` names
`Qwen/Qwen3-Reranker-0.6B`. **Lloyd is sweeping this weight and `PRF_WEIGHT` as of
2026-08-30** — check with him before starting a third sweep, and remember the box
takes at most two full evals at once.

## Blockers / mistakes already made

- **A feature flag whose zero value is not the identity silently corrupts every
  A/B against it.** The Pillar III branch in `routing_params()` was gated on
  `belief.exhausted` and set `diversify=False` *inside* that branch while its
  magnitude knob was 0.0 — so "all switches off" still disabled the browsing MMR
  re-rank, and the control leg scored 0.6154 instead of 0.6182. Caught only
  because identity was verified against the shipped number first. The code
  comment asserted "the label-only path is unchanged", which was false. **An
  identity setting is a claim to verify with a run, not to assert in a comment**,
  and check that *every* effect in a gated branch is gated, not just the one with
  a number attached.
- **Identity must reproduce the shipped score before any swept point is
  believed.** This has caught two real bugs.
- **Convert rates to absolute session counts before believing a small delta.**
  A +0.0040 delta looks like it clears a ±0.0025 noise estimate and does not.
- **Never more than two full evals at once on this box.** Three drove load past
  17 on 8 cores and got `eval_failures.py` killed after its last miss line but
  before it wrote its JSON — so the run *looked* clean. When chaining
  `long_command; tail log`, the exit code you see belongs to `tail`. Also check
  a queued run isn't redundant: a standalone eval was left holding the box for
  1h46 to confirm a number an A/B running beside it was already producing.
- **Run diagnostics under `uv run`.** A bare `python3` probe reported a config
  flag as a no-op because torch was absent, `Agent.dense` was `None`, and
  `_diversify` returns early. Same silent-degrade class as #15.
- `public_set.jsonl` samples carry **no raw query field**. They have only
  `category_bucket`, `difficulty_bucket`, `ground_truth`, `sample_id`,
  `scenario_type`, `user_profile`; turn-1 queries must be reconstructed via
  `initial_message()` + `materialize_hidden_fields()`.
- The override detector's first prototype set was all *affirmative*
  continuations, with nothing modeling "customer declines to state a preference"
  — which is `customer_reply()`'s standard non-answer template and shares
  negation vocabulary with genuine override cues. 8 of 10 non-answers were
  misclassified as overrides, silently wiping session state on routine turns.
  **Ground the negative class in the evaluator's actual reply vocabulary**;
  semantically adjacent-but-opposite reply types need explicit coverage.
- The old filter's `min_survivors` threshold was compared against the *entropy
  pool size* (~30) rather than `top_k` (10), so the filter fell back to
  unfiltered results almost every time. Moot now, kept for the lesson.
- **Style/size/use_case/feature never populate `SessionState.disclosed`** —
  there is no extractor for them. This looks like a bug in the REPL and is the
  by-design ceiling of three filterable attributes.
- "Use the profile to skip a question instead of biasing ranking" was reasoned to
  have a bounded downside *before* measuring. It doesn't: the exclusion is stored
  in session state and persists, so a wrong guess can block the single most
  informative question for the rest of a 10-turn session. **"This failure mode is
  bounded" is itself a claim to verify empirically**, not to reason about from
  the code's shape.
- **Another Claude session has been active in this repo.** Check `git status` and
  `ps` before assuming the working tree is yours.
