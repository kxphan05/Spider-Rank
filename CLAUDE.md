# TechJam Conversational Search — Project Notes

## What this is

An AI shopping agent for the TechJam Conversational Search Hackathon. The
task: given a frozen 50k-product Amazon Clothing/Shoes/Jewelry catalog and a
simulated customer, find the customer's hidden target product within 10
conversational turns, asking clarifying questions along the way.

Scoring (`evaluator/local_evaluator.py`, full rules in `README.md`):

```
TechnicalScore = 0.50 * HitRate@10 + 0.30 * MRR + 0.20 * Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

A session ends the moment the target first appears in the top-10
recommendations (MRR uses the rank at that first hit, not the best rank ever
achieved). A miss after 10 turns scores MTTC as turn 11. All of this is
implemented in `evaluate()` in `evaluator/local_evaluator.py` — read that file
directly before changing scoring-adjacent behavior, don't rely on summaries
of it going stale.

All agent logic lives in `starter/`:
- `starter/agent.py` — the `Agent` class: session state, retrieval
  orchestration, question selection, the response contract.
- `starter/retrieval.py` — `BM25Index` (SQLite FTS5), `DenseIndex`
  (bge-small cosine similarity), `reciprocal_rank_fusion`.
- `starter/attributes.py` — `AttributeIndex` (per-product material/color/
  price extraction from catalog text), entropy-based dynamic question
  selection, disclosed-value extraction from customer text.
- `starter/classifier.py` — buying-vs-browsing intent classification and
  mid-session intent-override detection, both via nearest-centroid cosine
  similarity against hand-written prototype utterances (shares the same
  loaded bge-small model as retrieval).

## Current architecture (as of last commit)

1. **Retrieval**: BM25 (SQLite FTS5) + dense (bge-small cosine) run
   independently over the full catalog, combined via weighted reciprocal
   rank fusion (BM25 weighted 2.0, dense 1.0 — empirically tuned; equal
   weights dropped MRR 0.450→0.385 on this catalog).
2. **Disclosure extraction**: every customer message (including the first)
   is scanned for material/color (vocab match, negation-aware) and budget
   (`$` amount → price bucket) and stored in `SessionState.disclosed`.
3. **Filter + rerank**: if anything's disclosed, the fused candidate pool is
   filtered (drop only *confident* mismatches — a candidate whose extracted
   attribute is known and disagrees; unknowns are kept, not penalized), then
   survivors are re-ranked by pure dense cosine similarity to the
   accumulated query text. A safety fallback discards the filter entirely if
   too few candidates survive (protects against over-filtering wiping out
   the true target).
4. **Question selection**: entropy-based dynamic pick between `material`/
   `color` (whichever has higher value-diversity in the *current*,
   already-filtered candidate pool — recalculated fresh every turn, so it
   naturally stops asking about an attribute once the pool has converged on
   it). Falls back to a fixed order (`style, size, use_case, feature,
   budget`) once nothing structural clears the entropy threshold.
5. **Intent-override handling**: a nearest-centroid embedding classifier
   detects a mid-session "actually, ignore what I said" pivot (no
   structured signal for this exists in the agent API — confirmed against
   `docs/agent_api_contract.json`, only `reset_request` exists and that's a
   new session, not a mid-session change). On detection, `disclosed` and
   `asked_attributes` are cleared so the agent re-asks and re-filters from
   scratch instead of carrying stale constraints forward.

Manual testing: `uv run python3 scripts/repl.py` — interactive single-session
REPL against the live agent. Commands: `/reset`, `/debug` (prints
`disclosed`/`asked_attributes` each turn), `/topk N`, `/quit`.

## Progress / what's been measured

Full-public-set (`uv run python3 -m evaluator.local_evaluator`, 200 samples)
results as of the last run:

```
HitRate@10: 0.735   MRR: 0.359   MTTC: 5.62   Efficiency: 0.538
TechnicalScore: 0.583
```

(Starter BM25-only baseline was HitRate 0.125 / MRR 0.068 / MTTC 9.81 per
`docs/baseline_results.json`.)

Notable things confirmed empirically along the way, not just assumed:
- The QSBPS paper the entropy-question design is inspired by (Zou &
  Kanoulas, "Learning to Ask") is CIKM 2019, not SIGIR 2019 as an earlier
  docstring claimed — corrected. Its actual mechanism is Generalized Binary
  Search over candidate relevance mass via binary questions, not the
  multi-way categorical entropy used here — the multi-way version is a
  deliberate adaptation, not a citation match, and there's no
  pairwise-comparison phase in that paper (a related but different idea
  from critique-based recommender literature).
- `ALLOWED_ATTRIBUTES` in `agent.py` is copied verbatim from
  `docs/agent_api_contract.json`'s enum — not invented.
- Catalog attribute coverage (measured, not assumed): material ~78.1%,
  color ~58.6%, price ~21% of products. Budget/price is missing on ~79% of
  the catalog.
- The local evaluator's `classify_constraint()` (which decides whether an
  agent's `ask_attribute` gets a real answer) **never** buckets any of the
  800 sampled disclosed constraint values across the full public set as
  `budget` — asking about budget locally is a guaranteed wasted turn. Root
  cause: `intent_card()` appends the price candidate last, and it almost
  always gets truncated out by the `hard_constraints[:2]` /
  `soft_preferences[2:4]` cap before the customer can ever disclose it.
  `budget` was demoted to last-resort-only in the fallback question order
  as a result (not dropped outright — this may be a local-simulator-only
  quirk that doesn't generalize to the hidden grader, so it's kept as an
  option rather than removed).
- `category`, `brand`, `other` are structurally unreachable in the current
  code (no code path ever picks them as `ask_attribute`), and separately,
  `classify_constraint()` never buckets any disclosed value into `category`
  or `brand` either, so local self-play can't validate those even if wired
  up.

## Known open problems / left to do

1. **Hard-filter self-elimination (biggest open bug).** The filter step
   (#3 above) hard-drops a candidate when its `AttributeIndex`-extracted
   value disagrees with a disclosed value. Measured directly: 16.3% of
   material disclosures and 37.0% of color disclosures disagree with
   `AttributeIndex`'s own value for the *true* target product — meaning the
   filter actively eliminates the correct answer in a meaningful fraction
   of sessions. Root cause: `AttributeIndex` extracts one value per product
   (first vocab hit in `title+features`) while disclosed text is scraped
   from broader/different fields — two independently-built single-value
   extractors over overlapping-but-different text don't always agree, even
   about the true target. This is why the filter's introduction left
   `TechnicalScore` roughly flat overall despite cutting MTTC (efficiency
   gains were offset by MRR losses), and why `intent_override` regressed
   hardest even before the override-clearing fix. **Proposed fix, not yet
   built**: switch from hard elimination to a non-eliminating boost
   (reorder matches to the front, never drop mismatches), and/or extract a
   *set* of plausible values per product from all relevant text fields and
   match on set-membership instead of equality.
2. A user raised storing the catalog in a real SQL database with indexed
   attribute columns, using disclosed answers as `WHERE` clauses, and
   clearing those clauses on intent override. Worth doing for the
   override-clear semantics and (optionally) moving the filter earlier in
   the pipeline as a real pre-query — but confirmed with the user that this
   does **not** by itself fix problem #1: `WHERE material = 'nylon'` is the
   same hard-equality operation as the Python filter, and inherits the same
   disagreement rate unless the underlying matching semantics are fixed
   first. At current catalog/pool scale (50k rows, ~30-item pools) the raw
   speed argument for SQL over a Python list filter is close to a wash;
   the value of that proposal is the override-clear semantics, not latency.
3. Budget disclosure extraction (`extract_disclosed_value`) requires a
   literal `$` in the customer's text (`r"\$\s?(\d+...)"`)) — phrasings
   like "fifty dollars" or "around 50" without a dollar sign won't match.
   Not yet widened.
4. From the original spec gap analysis (`TODO.md` has the full competition
   spec; this list is what's still missing against it): explicit
   Buying-vs-Browsing dual-track routing isn't actually wired to different
   retrieval behavior yet (the classifier exists and is computed every
   turn, but nothing branches on its output); no cross-encoder or LLM
   reranking stage (`4.2.I` mentions "LLM Semantic Ranking" — current
   ranking is pure hybrid retrieval + attribute filter, no learned/LLM
   reranker); `user_profile` passed into `reset()` is accepted but
   completely unused (no personalization); no long-term user-profile
   persistence across sessions (`4.2.III`'s "long-term user profiles").

## Blockers / mistakes already made (so they aren't repeated)

- Diagnostic scripts assumed `public_set.jsonl` samples carried a raw query
  field (`first_message`, `query`, etc.) — they don't. Samples only have
  `category_bucket`, `difficulty_bucket`, `ground_truth`, `sample_id`,
  `scenario_type`, `user_profile`; turn-1 queries must be reconstructed via
  `evaluator.local_evaluator.initial_message()` + `materialize_hidden_fields()`.
- First cut of the override detector's prototype set was all *affirmative*
  continuations ("Yes, that sounds good") with nothing modeling "customer
  declines to state a preference" — which is an extremely common reply
  shape (`customer_reply()`'s standard non-answer template) and shares
  negation vocabulary with genuine override cues ("don't"/"no"). Result:
  8 of 10 non-answer replies misclassified as overrides, silently wiping
  `disclosed`/`asked_attributes` on routine turns. Fixed by adding explicit
  "no preference" prototypes to the continuation class. Lesson: when
  building a prototype-based classifier against this evaluator, ground the
  negative class in the evaluator's actual reply vocabulary, not just
  generic hand-written examples — semantically adjacent-but-opposite reply
  types need explicit coverage, they won't fall out for free.
- The `_filter_by_disclosed` fallback threshold (`min_survivors`) was
  originally compared against the *entropy pool size* (~30) rather than the
  actual number of recommendations needed (`top_k`, e.g. 10) — meaning the
  filter fell back to unfiltered results almost every time, since removing
  a meaningful chunk of a 30-item pool is the whole point of filtering.
  Fixed by threading the real `top_k` through as `min_survivors`
  separately from the padded pool-size argument.
- Style/size/use_case/feature disclosures never populate `SessionState.disclosed`
  — there's no structural vocab/extractor for them (same limitation the
  original entropy design already had for question *selection*; it also
  applies to filtering). This can look like a bug during manual REPL
  testing ("disclosed isn't filling up") when it's actually just the
  by-design ceiling of only 3 filterable attributes (material, color,
  budget).
