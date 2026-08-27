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

Current score on the 200-sample public set: **TechnicalScore 0.6070**
(HitRate 0.745 / MRR 0.3876 / MTTC 5.09).

---

## 1. Retune the RRF fusion weights

**Status:** motivated by a fresh measurement, nothing tried. This is the
highest-value lead on the board, and it is cheap.

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

## 2. Retune the attribute boost weights

**Status:** indicated by a measured regression, nothing tried.

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

- **Budget extraction requires a literal `$`** (`CLAUDE.md` #3). "fifty
  dollars" and "around 50" don't match. Widening is easy; note that the local
  evaluator almost never buckets a disclosure as `budget`, so this cannot be
  validated locally and must be justified on the hidden set's behalf.
- **`starter/cooccurrence.py` is built but unwired** (#11). The item-attribute
  signal is real and large once #10's fix is in
  (`P(material=stainless steel | Watches)` = 0.483 vs 0.023 marginal). Either
  wire it into `_infer_attributes` alongside the masked LM, or delete it — a
  module that no code path reaches is a maintenance cost with no return.
- **No cross-encoder or LLM reranking stage** (#6). Spec § 4.2.I names "LLM
  Semantic Ranking" as a core pillar; ranking here is hybrid retrieval +
  attribute boost + MMR, with nothing learned. This is the most visible
  remaining gap against the spec as written, independent of its score impact.

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
