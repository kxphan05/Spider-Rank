# Next steps — queued work not yet started

Working notes for picking up cold in a fresh session. Each item states what
was already verified vs. what is still an untested design argument, since this
project's convention is to measure before trusting either.

`CLAUDE.md` is the standing record: its "Known open problems" list carries the
one-line verdict for everything already tried, and
`.claude/skills/retrieval-experiments/SKILL.md` carries the full write-ups for
closed investigations. **Read both before re-attempting anything here** — hard
attribute filtering, a filter-then-rerank buying track, embedding-similarity
reranking, user-profile personalization, and a trained intent head were each
built, measured, and rejected with numbers recorded.

Current score on the 200-sample public set: **TechnicalScore 0.7020**
(HitRate 0.8550, 171/200 / MRR 0.4622 / MTTC 4.205), measured with the
identity leg reproducing the previous 0.6184 first — which is the check to run
before trusting any new measurement.

Older score lines, for reading the notes below, which were written against
them: **0.6454** (phrase leg, before shown-item exclusion), **0.6182**
(HitRate 0.7500 / MRR 0.4096 / MTTC 4.985, before the phrase leg), and
**0.6070** (HitRate 0.745 / MRR 0.3876 / MTTC 5.09, before the fusion retune).
Deltas quoted below are against whichever baseline that note names; they have
not been re-measured on top of 0.7020.

---

## 1. ~~Retune the RRF fusion weights~~ — **shipped, +0.0112**

**Status:** decided and shipped. Buying runs at dense:bm25 **0.25**; the full
set moved 0.6070 -> **0.6182**, reproducing the sweep's prediction exactly.
Browsing is untouched and must stay that way (its curve was measured flat).
The original framing follows. `scripts/sweep_fusion_weights.py`
exists and both curves are recorded in CLAUDE.md #16. The buying curve is
strictly monotone with no interior optimum (ratio 0.25 captures +0.0112 of the
+0.0188 available from removing dense entirely); the browsing curve is flat and
should not be re-run. **What is still open is whether to ship buying ratio
0.25**, which is a judgment call about the hidden set, not a measurement — the
#14 confound (89.7% of local hard constraints are verbatim substrings of the
target's own catalog text) means the local set systematically under-prices the
dense leg. The original framing follows.

`CLAUDE.md` #14 records the ablation: **removing the dense leg entirely raises
TechnicalScore 0.6070 → 0.6216**, with HitRate identical to four decimals.
Dense finds nothing BM25 misses on this set and only demotes correct items
BM25 already ranked well (MRR 0.3876 → 0.4328).

We did **not** drop dense, and #14 explains why: 89.7% of local hard
constraints are verbatim substrings of the target's own catalog text, so the
local set is a pure exact-match benchmark that cannot price the paraphrase
robustness dense exists to provide. Dropping it would be overfitting to a
simulator artifact.

The unexplored middle: **retune the weights instead of removing the leg.**
`BUYING_BM25_WEIGHT = 2.0 / BUYING_DENSE_WEIGHT = 1.0` and the browsing
`1.25 / 1.5` were tuned before anyone had checked whether dense contributes at
all. Sweeping both tracks further BM25-ward should capture most of the
+0.0147 while keeping paraphrase insurance. A sweep is one script and no API
cost; the constants are `starter/agent.py` `routing_params()`.

Report the result as a curve, not a point — the useful output is how score
varies with the dense weight, since that curve is what tells a later session
how much is being paid for insurance.

## 2. Retune the attribute boost weights — **in progress**

**Status:** indicated by a measured regression; harness built, sweep not yet run.
`starter/agent.py` now exposes `DISCLOSED_MATCH_BOOST` /
`DISCLOSED_MISMATCH_PENALTY` (was a hardcoded +1/-1) and
`scripts/sweep_boost_weights.py` sweeps the penalty:match ratio. Note the ratio
is the only free parameter *while the masked LM is dark*; when it is live,
`LM_INFERENCE_WEIGHT = 0.5` also sets how a real disclosure trades against an
inference, which is the script's `--scale` axis.

`CLAUDE.md` #10 fixed a real bug — catalog attribute extraction was
substring-matched, so `"red"` matched inside *embroidered* and `"lace"` inside
*necklace*, inventing a colour for 21.6% of the catalog. The fix is correct
and **currently costs 0.005**, because `_boost_by_disclosed`'s ±1/0 scheme was
tuned when 58.6% of products carried a colour label and a fifth of those were
fictional. Correct-but-sparser labels (39.9%) leave more candidates neutral.

The indicated move is not to revert the fix but to retune the boost and
penalty magnitudes against the corrected coverage — including asymmetric
weights, since a *known mismatch* and an *unextractable* attribute are
different kinds of evidence and currently score −1 and 0 by assumption rather
than by measurement.

## 3. LLM-extracted catalog attributes (offline, one-time)

**Status:** designed and costed, nothing built. Highest ceiling, highest cost.

### The problem it solves

`starter/attributes.py`'s `AttributeIndex` extracts one value per product by
taking the **first vocab hit** in `title + features`. Consequences, all
measured:

- **Coverage is thin:** material 70.9%, colour 39.9%, price ~21% of the 50k
  catalog (post-#10 figures).
- **Four of the eight allowed attributes have no extractor at all** —
  `style`, `size`, `use_case`, `feature` never populate
  `SessionState.disclosed`, so they can be asked but never filtered on. This
  is the documented ceiling of the entropy-question design, not a bug.

An LLM reads the whole record (title, features, description, details,
categories, store) instead of regex-matching a fixed word list, and can return
`null` rather than guessing.

### Why subagents are the wrong tool

Spawning Claude Code subagents to iterate the catalog was the original
framing — don't. Each spawn is a full cold-start agent session; 50,000 of them
is the most expensive possible way to run a single-shot extraction prompt per
chunk. Use the **Message Batches API** from a one-off script in `scripts/`.

### Cost, measured against this catalog

| Input | tokens |
|---|---|
| `title` + `features` | ~5.9M |
| + `description` | ~9.4M |

With `claude-haiku-4-5`, the Batch API's 50% discount, ~50 products per
request (~1,000 requests, far under the 100k-request cap), and ~60 output
tokens per product: **roughly $5–12 one-time**, well inside the 24h batch SLA.
Re-check pricing at the time of the run rather than trusting this table.

### Rules check (done, all three clear)

- `docs/submission_rules.md` § Model Policy permits prototyping with any
  legally accessible LLM API.
- Output is a **static JSON sidecar** shipped as a local asset, so the agent
  still needs no network at scoring time — which matters, because
  `scripts/preflight.py` exists precisely to enforce that property.
- The read-only-catalog constraint holds: write a new sidecar keyed by
  `parent_asin`, never modify `data/catalog.jsonl`.

### Design constraints (non-negotiable — these are why naive versions fail)

1. **Closed vocabulary, not free-form text.** `AttributeIndex` compares values
   for *equality*; "a rich chocolate-brown leather" is useless to it.
   Constrain the model to a fixed value set per attribute (seed from the
   existing `COLORS` / `MATERIALS` tuples) with structured outputs, so the
   schema is enforced rather than requested.
2. **`null` must be allowed.** An invented attribute is worse than a missing
   one — a confident wrong value sinks the true target.
3. **Determinism.** Store the sidecar in git (50k rows of short categorical
   values is small), don't regenerate per run, and record the model ID and
   prompt version inside the file.

### The honest caveat

This fixes only the **catalog half**. The target-disagreement figure came from
*two independently built extractors* disagreeing: the catalog side and the
customer-text side (`extract_disclosed_value`, which must stay offline and
LLM-free). Upgrading only the catalog side may not move the metric much — the
gain lands fully only if the LLM-derived vocabulary **also becomes the
vocabulary `extract_disclosed_value` matches against**. Plan both halves, or
measure the catalog-only half and expect a smaller delta than the coverage
numbers suggest.

### Suggested first step

Prototype on a few hundred products (direct API, not batch) and check whether
the extracted values agree with `ground_truth` more often than the current
regex does. That agreement rate decides whether the full run is worth it, and
it is computable on the 200-sample public set.

## 4. Smaller, well-defined gaps

- ~~**Budget extraction requires a literal `$`**~~ **Fixed — and it was three
  bugs, not one.** (1) The `\$\s?(\d+)` pattern meant "less than 80 dollars"
  disclosed nothing. (2) The comparator was discarded and the amount mapped to
  the bucket *containing* it, then compared for equality — so "under $80"
  resolved to the top bucket `$48+` and a stated ceiling **boosted the priciest
  products while penalizing every cheap one**. (3) `_extract_budget` returns the
  literal string `"unknown"` for the 79.2% of the catalog with no price, which
  is a known value rather than `None`, so any budget disclosure scored a full
  mismatch against 39,590 of 50,000 products. Now `parse_budget_constraint()`
  returns a bound (`"<=80"` / `">=200"`), `AttributeIndex.price_for()` exposes
  the raw price, and `_boost_by_disclosed` compares numerically with unpriced
  products neutral. **Unverifiable locally** — `classify_constraint()` never
  buckets a disclosure as `budget`, so the correct expectation is that the
  public set is bit-identical; that is the regression check, not a win
  condition. One judgment call recorded: a bare or approximate amount ("my
  budget is $80", "around fifty dollars") reads as a *ceiling*, since that is
  the dominant sense of a stated shopping budget.
- **Co-occurrence priors are built but unwired** (#11). The item-attribute
  signal is real and large once #10's fix is in — the diagnostic reports
  `P(material=cotton | category=Shirts T-Shirts)` = 0.933 against a 0.265
  marginal, and `P(material=sterling silver | Rings Statement)` = 0.756 against
  0.040. **Moved from `starter/` to `scripts/cooccurrence.py`** and given a
  `main()`, since no agent code path reaches it and `starter/` is copied
  verbatim into the submitted bundle. Still the same open choice: wire it into
  `_infer_attributes` alongside the masked LM, or drop it. Move it back into
  the package only once it earns its place by measurement.
- **No cross-encoder or LLM reranking stage** (#6). Spec § 4.2.I names "LLM
  Semantic Ranking" as a core pillar; ranking here is hybrid retrieval +
  attribute boost + MMR, with nothing learned. This is the most visible
  remaining gap against the spec as written, independent of its score impact.

## 5. ~~Rewrite the query on intent override~~ — **built, measured −0.0580, reverted**

**Status:** built and measured. **It fails badly — see CLAUDE.md #17.**

The gap described below is real and the diagnosis of it stands. The fix does
not: restarting the query from the pivot message scored 0.5602 against 0.6182,
with the `intent_override` subset collapsing from hit 0.800 to 0.333.

**And this section previously asserted, as established, the thing that made it
fail.** It claimed the local override template "carries the complete new
constraint, so discarding the history loses nothing locally *by
construction*." That is wrong. Turn 1 is `"I'm looking for [<category>].
<old_value>"` and the pivot is `"...What I need is: leather"` — the **category
appears only in turn 1 and is never restated**, so dropping `first_message`
searches 50k products for one attribute word. The claim came from reading the
template's shape rather than printing a sample; printing one takes a minute and
would have caught it.

Any future attempt must preserve the original framing and drop only the
preference clause. The original write-up follows, for the diagnosis.

`Agent.respond` clears exactly three things when
`EmbeddingOverrideDetector.is_override` fires (`agent.py:364`):

```python
state.disclosed.clear()
state.profile_hint.clear()
state.asked_attributes.clear()
```

`state.first_message` and `state.recent_messages` survive — and `_build_query()`
concatenates precisely those two into the retrieval query, so **the query string
is byte-for-byte identical before and after the clear** (reproduced directly).
The discarded preference stops biasing `_boost_by_disclosed`, but its *words*
keep feeding BM25 and the dense leg: `first_message` for the whole session,
pre-override turns for another `RECENT_WINDOW = 4` messages. Since the boost
only resorts a pool the query already selected, the text channel is the stronger
of the two.

Against the spec, § 4.2.II asks for "Intent Override (slot erasure **and
rewriting**)". Erasure is implemented; rewriting is not.

### The experiment

On override, also reset `first_message` to the override message and drop
`recent_messages`. **Unlike the budget fix (#4), this is locally measurable** —
30 `intent_override` samples, currently hitting at 0.833 (CLAUDE.md #7).

Two things to hold onto when reading the result:

- **30 samples means one session is ±0.033 on that subset.** Per CLAUDE.md #16,
  convert to absolute counts before believing any delta, and report the
  full-set number alongside the subset.
- **The local override template flatters this change.** `behavior_for()` emits
  `"Actually, ignore my earlier preference. What I need is: {X}."`, where `{X}`
  carries the complete new constraint — so discarding the history loses nothing
  locally *by construction*. A terse hidden-set pivot ("never mind, white
  shoes") carries far less, and dropping `first_message` there could leave the
  query almost contentless. If the full-history and rewritten variants measure
  the same, prefer the more conservative one (e.g. drop `recent_messages` but
  keep `first_message`), and consider a middle option: keep the override
  message plus everything after it, discard only what preceded the pivot.

Two things worth verifying while in this code path, both cheap: the clear runs
*before* the disclosure loop (so a constraint stated in the override message is
still captured — deliberate, and why the demo builder can't infer overrides from
the slot dict shrinking), and clearing `asked_attributes` lets the agent re-ask
a pre-pivot question, which is intended but costs a turn.


---

## Not next steps (closed — do not reopen without new evidence)

| idea | verdict | where |
|---|---|---|
| Hard attribute filtering | eliminated the true target 16.3%/37.0% of the time | #1 |
| Embedding-similarity rerank of disclosed values | monotonically worse as weight rose | #1 |
| Filter-then-rerank buying track | worse on hit rate and MTTC | #4 |
| User-profile personalization | 3 variants each regressed; profile carries no signal | #5 |
| Trained intent-classifier head | template memorization; worse out of distribution | #12 |
| Trimmed prototypes for the intent classifier | worse on every pool | #13 |
| Dropping the dense leg | +0.0147 locally, but on an exact-match-only benchmark | #14 |
| Query rewriting on intent override | -0.0580; turn 1 carries the category | #17 |
| Turn-annealed slate diversity | null; browsing metrics byte-identical | #18 |
| Dropping non-answers from the query | -0.0410; contentless text is still an exact-match key | #19 |

---

## 6. Within-session adaptive layer — **built, one stage measured**

Spec Pillar III's short-term half (`starter/session_belief.py`,
`EmbeddingNonAnswerDetector` in `classifier.py`, wiring in `agent.py`). The
agent now observes whether each clarifying question was answered, which it
never did before.

**Every switch ships at its identity value**, so each stage A/Bs independently
against a baseline that reproduces 0.6182 exactly:

| switch | identity | status |
|---|---|---|
| `SKIP_NON_ANSWERS_IN_QUERY` | `False` | **measured -0.0410, rejected (#19)** |
| `SLOT_DECAY` | `1.0` | unmeasured — `scripts/sweep_slot_decay.py` is written |
| `BELIEF_DRIVEN_QUESTIONS` | `False` | unmeasured |
| `BELIEF_REORCHESTRATION` | `False` | unmeasured |
| `EXHAUSTED_BM25_BONUS` | `0.0` | unmeasured |
| `EXPLAIN_RECOMMENDATIONS` | `True` (on) | score-neutral by construction |

Read #19 before touching the first row, and the blockers list before adding a
switch: the re-orchestration branch originally disabled the browsing MMR
re-rank even at its zero setting, so the control leg scored 0.6154 rather than
0.6182 and every stage measured against it would have been wrong.

## 7. Cross-encoder reranking — **downloaded, not wired**

`Qwen/Qwen3-Reranker-0.6B` (1.2 GB, in `model/`) and `starter/reranker.py`
exist; nothing calls them. This is the last named spec gap (#6, Pillar I's
"LLM Semantic Ranking").

Chosen at 0.6B for a measured reason, not a guess. On the target machine
(i5-8365U, no CUDA, ~4.6 GB free RAM) a 2B *generative* model needs ~8.4 s per
41-token scoring call, so an 8B model implies roughly 180 hours for one
200-sample evaluation -- unmeasurable, and this project does not ship unmeasured
changes. A cross-encoder is one forward pass per pair with no generation, which
is the only reason it is tractable: decode here runs under 2 tok/s, prefill at
~60 tok/s.

Wire it behind a `RERANK_WEIGHT` defaulting to 0.0 so identity stays
bit-reproducible, then sweep. **Expect it to measure flat or negative**: it is
the same class of component as the dense leg, on a benchmark where the dense
leg is net-negative for a documented reason (#14). If it does, the defensible
move is #14's -- keep it, report the local cost, and say why the local set
cannot price it.
