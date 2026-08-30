# Next steps — queued work not yet started

`CLAUDE.md` is the standing record: it carries the one-line verdict for
everything already tried, and `.claude/skills/retrieval-experiments/SKILL.md`
carries the full write-ups for closed investigations. **Read both before
re-attempting anything here.** This file holds only work that has not been done,
plus the design notes that would otherwise have to be re-derived.

Current score: **TechnicalScore 0.7935** on the 200-sample public set
(HitRate 0.9450, 189/200 / MRR 0.5534 / MTTC 3.250), HEAD `1fb1913`. Deltas
quoted below were measured against whichever baseline the note names and have
not been re-measured on top of 0.7935.

**Two items below shipped without going through the harness this file
describes for them — see CLAUDE.md #31.** The popularity prior leg (item 1)
landed in `9702d62` as `POPULARITY_WEIGHT = 0.5`, and cross-encoder reranking
(the "downloaded, not wired" section below) landed in `1fb1913` as
`RERANK_WEIGHT = 3`. Both are in the 0.7935 number; neither was isolated with
`sweep_prior_leg.py --weights 0` / `sweep_rerank.py` as this file recommends,
so their individual contribution is still unmeasured. That sweep is now the
highest-value remaining item, not a fresh design.

## Ranked by expected value

1. **Isolate the popularity leg and the reranker, one at a time, against
   `a9d3999` (0.7441).** Both shipped inside #31's unattributed bundle.
   `scripts/sweep_prior_leg.py --weights 0` and `scripts/sweep_rerank.py` were
   built for exactly this and neither has been run against current HEAD.
2. **Elicitation** — CLAUDE.md #27's conclusion. 12 of the 27 oracle misses have
   the target in some leg's top 10, so the information is reachable once it
   arrives; what is missing is the constraints themselves.
   `docs/question_policy_plan.md` is written and not built.
3. **Retune the attribute boost weights** (below).
4. **LLM-extracted catalog attributes** (below) — highest ceiling, highest cost.

## Retune the attribute boost weights

Harness built (`scripts/sweep_boost_weights.py`), sweep not run. `agent.py`
exposes `DISCLOSED_MATCH_BOOST` / `DISCLOSED_MISMATCH_PENALTY`, previously a
hardcoded ±1.

The word-boundary fix (CLAUDE.md #10) is correct and currently costs 0.005,
because the ±1/0 scheme was tuned when 58.6% of products carried a colour label
and a fifth of those were fictional. Correct-but-sparser labels (39.9%) leave
more candidates neutral. Retune against the corrected coverage rather than
reverting — including asymmetric weights, since a *known mismatch* and an
*unextractable* attribute are different kinds of evidence and score −1 and 0 by
assumption, not by measurement. The ratio is the only free parameter while the
masked LM is dark; live, `LM_INFERENCE_WEIGHT` also sets how a disclosure trades
against an inference, which is the script's `--scale` axis.

## LLM-extracted catalog attributes (offline, one-time)

Designed and costed, nothing built.

**The problem.** `AttributeIndex` takes the **first vocab hit** in
`title + features`. Coverage is thin (material 70.9%, colour 39.9%, price ~21%),
four of the eight allowed attributes have no extractor at all (`style`, `size`,
`use_case`, `feature` can be asked but never filtered on), and it keeps one value
where 63.9% of target products list several (CLAUDE.md #27). An LLM reads the
whole record and can return `null` rather than guessing.

**Use the Message Batches API from a one-off script, not subagents.** Each
subagent spawn is a full cold-start session; 50,000 of them is the most expensive
possible way to run a single-shot extraction prompt per chunk.

**Cost.** ~5.9M input tokens for `title + features`, ~9.4M with `description`.
With `claude-haiku-4-5`, the Batch API's 50% discount, ~50 products per request
(~1,000 requests, far under the 100k cap) and ~60 output tokens per product:
**roughly $5–12 one-time**, inside the 24h batch SLA. Re-check pricing at run
time rather than trusting this.

**Rules check, all three clear.** Model Policy permits prototyping with any
legally accessible LLM API; the output is a static JSON sidecar shipped as a
local asset, so the agent still needs no network at scoring time (which is what
`preflight.py` enforces); and the read-only-catalog constraint holds as long as
the sidecar is keyed by `parent_asin` and `data/catalog.jsonl` is never modified.

**Design constraints — these are why naive versions fail.**

1. **Closed vocabulary, not free-form text.** `AttributeIndex` compares values
   for *equality*; "a rich chocolate-brown leather" is useless to it. Constrain
   the model to a fixed value set per attribute (seed from `COLORS` /
   `MATERIALS`) with structured outputs, so the schema is enforced rather than
   requested.
2. **`null` must be allowed.** An invented attribute is worse than a missing one
   — a confident wrong value sinks the true target.
3. **Determinism.** Commit the sidecar (50k rows of short categorical values is
   small), don't regenerate per run, and record the model ID and prompt version
   inside the file.

**The honest caveat.** This fixes only the **catalog half**. The disagreement
figure came from two independently built extractors disagreeing, and
`extract_disclosed_value` must stay offline and LLM-free. The gain lands fully
only if the LLM-derived vocabulary also becomes the vocabulary the customer-side
extractor matches against. Plan both halves, or expect a smaller delta than the
coverage numbers suggest.

**First step.** Prototype on a few hundred products (direct API, not batch) and
check whether the extracted values agree with `ground_truth` more often than the
regex does. That agreement rate decides whether the full run is worth it, and it
is computable on the 200-sample public set.

## Within-session adaptive layer — built, one stage measured

Spec Pillar III's short-term half (`starter/session_belief.py`,
`EmbeddingNonAnswerDetector`, wiring in `agent.py`). The agent observes whether
each clarifying question was answered, which it never did before.

**Every switch ships at its identity value**, so each stage A/Bs independently:

| switch | identity | status |
|---|---|---|
| `SKIP_NON_ANSWERS_IN_QUERY` | `False` | **measured −0.0410, rejected (#19)** |
| `SLOT_DECAY` | `1.0` | unmeasured — `sweep_slot_decay.py` is written |
| `BELIEF_DRIVEN_QUESTIONS` | `False` | unmeasured |
| `BELIEF_REORCHESTRATION` | `False` | unmeasured |
| `EXHAUSTED_BM25_BONUS` | `0.0` | unmeasured |
| `EXPLAIN_RECOMMENDATIONS` | `True` (on) | score-neutral by construction |

Read #19 before touching the first row, and the blockers list before adding a
switch: the re-orchestration branch originally disabled the browsing MMR re-rank
even at its zero setting, so its control leg was wrong by an unknown amount.

## Cross-encoder reranking — now wired, weight unswept

`starter/reranker.py`'s minilm backend is live as of `1fb1913`
(`RERANK_WEIGHT = 3`, `RERANK_TOP_N = 20`), folded into #31's unattributed
bundle. `Qwen/Qwen3-Reranker-0.6B` (1.2 GB, in `model/`) is still downloaded
but unused — `RERANK_BACKEND` is `"minilm"`. `scripts/sweep_rerank.py` is
written and has not been run against the weight actually shipped.

0.6B was chosen for a measured reason, in case the Qwen backend is revisited.
On the target machine (i5-8365U, no CUDA, ~4.6 GB free) a 2B *generative*
model needs ~8.4 s per 41-token scoring call, so an 8B model implies ~180
hours for one 200-sample eval. A cross-encoder is one forward pass per pair
with no generation, which is the only reason it is tractable: decode here
runs under 2 tok/s against ~60 tok/s prefill.

**This was expected to measure flat or negative** — it is the same class of
component as the dense leg, on a benchmark where the dense leg is
net-negative for a documented reason (#14). #31's bundle scored well, but
whether the reranker specifically helped, hurt, or was neutral inside it is
unknown until `sweep_rerank.py` runs. If it turns out flat/negative, the
defensible move is #14's: keep it, report the local cost, and say why the
local set cannot price it.

## Co-occurrence priors — built, unwired

`scripts/cooccurrence.py` (CLAUDE.md #11). The signal is real and large
(`P(material=cotton | Shirts T-Shirts)` = 0.933 against a 0.265 marginal). It
lives in `scripts/` because no agent code path reaches it and `starter/` is
copied verbatim into the bundle. The open choice is unchanged: wire it into
`_infer_attributes` alongside the masked LM, or drop it. Move it back into the
package only once it earns its place by measurement.

## Closed — do not reopen without new evidence

| idea | verdict | where |
|---|---|---|
| Hard attribute filtering | eliminated the true target 16.3%/37.0% of the time | #1 |
| Embedding-similarity rerank of disclosed values | monotonically worse as weight rose | #1 |
| Filter-then-rerank buying track | worse on hit rate and MTTC | #4 |
| User-profile personalization | 3 variants each regressed; profile carries no signal | #5 |
| Trained intent-classifier head | template memorization; worse out of distribution | #12 |
| Trimmed prototypes for the intent classifier | worse on every pool | #13 |
| Dropping the dense leg | +0.0147 locally, but on an exact-match-only benchmark | #14 |
| Query rewriting on intent override | −0.0580; turn 1 carries the category | #17 |
| Turn-annealed slate diversity | null; browsing metrics byte-identical | #18 |
| Dropping non-answers from the query | −0.0410; contentless text is still an exact-match key | #19 |
| Corpus-derived stopword list | 41% of the words it drops are exact-match keys | #26 |
| Removing the boost's mismatch penalty | recovers none of the 7 misses it was built for | #27 |
