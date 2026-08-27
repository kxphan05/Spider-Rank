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

1. **Retrieval, dual-track by buying/browsing intent**: BM25 (SQLite FTS5) +
   dense (bge-small cosine) run independently over the full catalog,
   combined via weighted reciprocal rank fusion — both tracks fuse the same
   two legs, but the weights are intent-conditioned: buying keeps the
   original BM25-heavy weighting (2.0/1.0 — empirically tuned; equal
   weights dropped MRR 0.450→0.385 on this catalog) for precision on stated
   hard constraints, browsing shifts toward dense (1.25/1.5) for broader
   semantic/cross-category matches, then gets an MMR diversity re-rank
   (`Agent._diversify`) on top of the fused+boosted pool before truncating
   to `top_k` — pinning the top `DIVERSIFY_PIN=2` items and only
   reordering within the top `DIVERSIFY_WINDOW=20`, so diversity can never
   evict the best match(es). See "Known open problems" #4 for what was
   tried and measured to get here (a hard filter-then-rerank buying track
   was tried first and measured worse).
2. **Disclosure extraction**: every customer message (including the first)
   is scanned for material/color (vocab match, negation-aware) and budget
   (`$` amount → price bucket) and stored in `SessionState.disclosed`.
3. **Boost, don't filter**: if anything's disclosed, the fused candidate pool
   is stably resorted by a categorical match score against
   `AttributeIndex` — +1 per disclosed attribute where the candidate's
   extracted value agrees, −1 where it's known and disagrees, 0 where it's
   unextractable — so confident matches float to the front and confident
   mismatches sink to the back, but nothing is ever dropped from the pool
   (`Agent._boost_by_disclosed`). This replaced an earlier hard-equality
   filter that eliminated candidates outright on disagreement; see
   "Known open problems" below for why and what was measured.
4. **Question selection**: entropy-based dynamic pick between `material`/
   `color` (whichever has higher value-diversity in the *current*,
   already-filtered candidate pool — recalculated fresh every turn, so it
   naturally stops asking about an attribute once the pool has converged on
   it). Falls back to a fixed order (`style, size, use_case, feature,
   budget`) once nothing structural clears the entropy threshold. The
   entropy threshold itself is also intent-conditioned: 0.10 for buying
   (ask eagerly — locking a hard constraint is high-value), 0.30 for
   browsing (ask less eagerly, bias toward recommending sooner). Measured
   to have no effect on the 60-sample tuning subset's outcome (didn't
   change which samples hit or when) — kept anyway since it's a direct,
   costless implementation of the spec's "Proactive Guidance" bullet and
   may matter on other slices/the hidden set.
5. **Intent-override handling**: a nearest-centroid embedding classifier
   detects a mid-session "actually, ignore what I said" pivot (no
   structured signal for this exists in the agent API — confirmed against
   `docs/agent_api_contract.json`, only `reset_request` exists and that's a
   new session, not a mid-session change). On detection, `disclosed` and
   `asked_attributes` are cleared so the agent re-asks and re-filters from
   scratch instead of carrying stale constraints forward.
6. **Long-term user-profile store** (`starter/user_profile.py`,
   `UserProfileStore`): write-through JSON file (`data/user_profiles.json`,
   gitignored — runtime state, not source data) that persists across process
   restarts, not just within one `Agent` instance's lifetime. The API
   contract's `reset_request.user_profile` (`docs/agent_api_contract.json`)
   has no customer/user id, so there's nothing to key long-term state on
   except the anonymized profile dict's own content — `profile_key()` hashes
   it (sorted-key JSON → sha256, first 16 hex chars) and two sessions
   presenting an identical profile are treated as the same returning
   "shopper profile." `Agent.reset()` calls `start_session()`, which bumps a
   per-key session counter and returns any *corroborated* (recurred ≥2x,
   `MIN_CORROBORATION`) historical material/color/budget disclosure as
   `SessionState.profile_hint`; `respond()` calls `record_disclosure()`
   whenever a genuine this-session disclosure happens, appending to that
   key's history. **`profile_hint` is currently populated but not consulted
   anywhere in retrieval or question logic** — see "Known open problems" #5
   for three different uses of it that were each tried and measured to
   regress the full public set, and why. The store still runs (fully
   exercised, correctly populated — 125 distinct keys / 30 seen more than
   once across the 200-sample public set) with zero net effect on scoring
   while a viable way to use it is worked out.

Manual testing: `uv run python3 scripts/repl.py` — interactive single-session
REPL against the live agent. Commands: `/reset`, `/debug` (prints
`disclosed`/`asked_attributes` each turn), `/topk N`, `/quit`.

## Progress / what's been measured

Full-public-set (`uv run python3 -m evaluator.local_evaluator`, 200 samples)
results as of the last run:

```
HitRate@10: 0.755   MRR: 0.384   MTTC: 5.60   Efficiency: 0.540
TechnicalScore: 0.601
```

(Previous run, before the dual-track buying/browsing routing described in
"Known open problems" #4: HitRate 0.755 / MRR 0.373 / MTTC 5.49 /
Efficiency 0.551 / TechnicalScore 0.600 — net roughly flat on the full set
despite a clearer gain on the 60-sample tuning subset, see #4 for the
honest breakdown.)

(Before that, before the boost-not-eliminate fix described in "Known open
problems" #1: HitRate 0.735 / MRR 0.359 / MTTC 5.62 / TechnicalScore 0.583 —
every metric improved.)

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

1. ~~**Hard-filter self-elimination.**~~ **Fixed.** The old filter step
   hard-dropped a candidate when its `AttributeIndex`-extracted value
   disagreed with a disclosed value. Measured directly: 16.3% of material
   disclosures and 37.0% of color disclosures disagreed with
   `AttributeIndex`'s own value for the *true* target product — the filter
   was eliminating the correct answer in a meaningful fraction of sessions.
   Root cause: `AttributeIndex` extracts one value per product (first vocab
   hit in `title+features`) while disclosed text is scraped from a
   different, noisier source (the customer's own phrasing) — two
   independently-built single-value extractors over overlapping-but-
   different text don't always agree, even about the true target.
   Replaced with `Agent._boost_by_disclosed` (see architecture #3 above):
   a stable resort by categorical match score, never eliminating anything.
   Full-set result: HitRate 0.735→0.755, MRR 0.359→0.373, MTTC 5.62→5.49,
   TechnicalScore 0.583→0.600 — every metric improved.
   **A tempting alternative was tried and measured worse, kept here so it
   isn't retried blindly**: reranking by cosine similarity between the raw
   disclosed phrase (e.g. `"leather blue"`) and the same dense catalog
   embeddings retrieval already uses, instead of the categorical match.
   Swept on a fixed 60-sample subset (seed 7): TechnicalScore went 0.604
   (boost weight 0, i.e. no filtering signal at all) → 0.546 (weight 1.5),
   *monotonically worse* as the boost weight increased, and even at weight
   0 it scored below the old hard-equality filter's 0.618 on the same
   subset. A short disclosed phrase dotted against a full-product embedding
   is too diluted a signal for this; exact vocab agreement, imperfect as it
   is, is the stronger one on this catalog. The categorical
   `_boost_by_disclosed` above scored 0.624 on that same subset.
2. A user raised storing the catalog in a real SQL database with indexed
   attribute columns, using disclosed answers as `WHERE` clauses, and
   clearing those clauses on intent override. The override-clear-semantics
   argument still stands, but it's no longer motivated by problem #1 (that's
   fixed above without SQL) — at current catalog/pool scale (50k rows,
   ~30-item pools) the raw speed argument for SQL over the Python resort is
   close to a wash either way.
3. Budget disclosure extraction (`extract_disclosed_value`) requires a
   literal `$` in the customer's text (`r"\$\s?(\d+...)"`)) — phrasings
   like "fifty dollars" or "around 50" without a dollar sign won't match.
   Not yet widened.
4. ~~**Buying-vs-Browsing dual-track routing not wired up.**~~ **Fixed.**
   The classifier (`starter/classifier.py`) was already computed every turn
   but only ever logged (`agent.py:174` in the pre-fix version) — confirmed
   via `grep` that no other call site consumed its output. Now `signal.label`
   is computed before retrieval and threads through to: (a) intent-
   conditioned RRF fusion weights in `Agent._retrieve` (buying 2.0/1.0
   BM25-heavy, browsing 1.25/1.5 dense-heavy), (b) an MMR diversity re-rank
   (`Agent._diversify`, browsing-only) on top of the fused+boosted pool, and
   (c) the entropy threshold in `_next_attribute` (architecture #4 above).
   Full-set: HitRate 0.755→0.755 (flat), MRR 0.373→0.384 (up), MTTC
   5.49→5.60 (slightly worse), TechnicalScore 0.600→0.601 (~flat) — net a
   wash on the full 200-sample set despite a real gain on the 60-sample
   seed-7 tuning subset (TechnicalScore 0.624→0.637, every metric up
   there). Being honest about this rather than overselling it: the
   subset-level gain didn't fully generalize, likely because the weights
   were tuned on that subset. Kept anyway since it's a real, spec-required
   architectural piece (TODO.md's "Dual-Track Routing" and "heterogeneous
   retrieval routing (weights...)" bullets) with no full-set regression,
   and is a more promising base for further tuning than the unwired
   classifier it replaced.
   **A more literal reading of the spec was tried first and measured worse,
   kept here so it isn't retried blindly**: a hard filter-then-rerank
   buying track (BM25 top-50 as a hard filter, `DenseIndex.rank_subset`
   reranking only within that filtered set, dropping the full-catalog dense
   leg entirely) — this is what "high-precision filter track... to lock
   hard constraints" most literally suggests. Isolated on the 60-sample
   subset: buying-only TechnicalScore contribution dropped from the
   baseline's 0.6818/0.3315/6.09 (hit/MRR/MTTC) to 0.6364/0.3468/6.45 —
   worse hit rate and MTTC. Root cause: BM25's top-`CANDIDATE_N` keyword
   filter is lossy — it drops the true target whenever the target's
   catalog text uses different words than the customer's phrasing, exactly
   the case dense retrieval exists to catch, and discarding the dense leg
   entirely for buying threw that recall away. It also weight-inverted
   `browsing` to 1.0/2.0 with the same test batch, which alone dropped
   browsing hit rate 0.870→0.826 and MTTC 3.96→4.52 (MRR did improve,
   0.404→0.420) — kept as a data point on how far dense-heavy weighting can
   be pushed before it costs more than it gives; 1.25/1.5 was chosen
   instead as a smaller step in the same direction.
   Also caught during this work, not before landing: the first `_diversify`
   implementation was silently a no-op in every test above, because it was
   being asked to pick `top_n` items from a pool of exactly `top_n` size
   (both were the padded `ENTROPY_POOL_SIZE`-or-`top_k` pool length, not the
   real per-turn recommendation count) — its own "nothing to trim" early
   return fired every time. Fixed by threading the real `top_k` through
   `Agent._retrieve` separately from the padded `pool_size`, so `_diversify`
   reorders only the front `top_k` of the larger pool and appends the rest
   unchanged (order doesn't matter for the entropy scoring downstream,
   only the value distribution does). All the "browsing" numbers above are
   post-fix, with MMR actually running.
5. ~~**No long-term user-profile persistence across sessions.**~~
   **Store built (architecture #6 above); personalization use still open.**
   `starter/user_profile.py`'s `UserProfileStore` now persists a per-profile-
   key disclosure history to disk across process restarts, keyed by a
   content hash of the anonymized `user_profile` dict (no customer id exists
   in the API contract to key on instead). Three different ways of actually
   *using* the carried-forward value (`SessionState.profile_hint`) were
   tried and each measured to regress the full 200-sample public set
   (baseline HitRate 0.755 / MRR 0.384 / MTTC 5.60 / TechnicalScore 0.601 —
   see "Progress" above, unchanged by the store itself since it's currently
   inert):
   - **Boost candidate ranking at full weight**, merged directly into
     `disclosed` and run through the existing `_boost_by_disclosed` (same
     ±1 categorical match/mismatch scoring as a genuine disclosure): on the
     60-sample seed-7 subset, TechnicalScore 0.637→0.622, entirely from one
     `intent_override` sample whose carried material/color disagreed with
     its true target and got demoted rank 1→7.
   - **Boost at half weight** (`PROFILE_HINT_WEIGHT`, in a separate
     `profile_hint` dict from genuine `disclosed` so a real this-session
     answer always overrides a stale hint): 60-sample subset only recovered
     to 0.633, not back to baseline. Root cause found by inspection, not
     guessing: *any* nonzero mismatch penalty sinks a candidate below every
     neutral (unknown-attribute) candidate in the pool regardless of
     magnitude, so the cost is structural, not a weight-tuning problem —
     confirmed by also trying weight 0.25 with no further improvement on
     the one affected sample. Gating the carry on corroboration (same value
     recurring ≥2x in history, `MIN_CORROBORATION`) before trusting it
     didn't fix the full-set number either (TechnicalScore 0.585→0.588,
     still well below the 0.601 baseline) — collisions on this catalog are
     template-level (many genuinely different customers share the same
     coarse `preference_tags` combination), so a value recurring twice
     doesn't actually make it more likely correct for a *third*,
     unrelated session sharing that key.
   - **Use the corroborated hint to skip a likely-redundant *question*
     instead of biasing ranking** (add it to `_next_attribute`'s `excluded`
     set), reasoning the downside should be bounded to "waste one turn if
     wrong" rather than the ranking approach's unbounded downside: measured
     worse than expected on the full set (HitRate 0.755→0.740, TechnicalScore
     0.601→0.589), concentrated entirely in `browsing` (0.8125→0.775).
     Root cause: the exclusion isn't actually bounded to one wasted turn —
     it persists for the rest of the session exactly like a genuine
     disclosure would, so a wrong corroborated guess can permanently block
     ever asking about that attribute again, even on a later turn where the
     remaining candidate pool would make it the single most informative
     question available. Confirmed directly: several `browsing` sessions
     that used to hit flipped to a full miss, with the material/color
     question silently never asked for the rest of the session.
   Conclusion: on this competition's public-set profile generation (a small
   template set produces the `preference_tags` combinations — 125/200
   profiles are unique, i.e. collisions are common and don't correlate with
   the *current* session's actual target), a profile-key match is a weak
   enough identity signal that using it for either ranking or question
   selection costs more than it gives, however the risk is hedged. Left as
   explicitly inert (`profile_hint` populated, never read) rather than
   shipped with a measured regression — this may be worth revisiting if the
   hidden grader's profile generation turns out less collision-prone, or if
   a differently-shaped signal (e.g. nudging `FALLBACK_ATTRIBUTE_ORDER` from
   this *session's own* freshly-given `preference_tags`, which needs no
   cross-session identity assumption at all) is tried instead.
6. Remaining gaps from the original spec analysis (`TODO.md` has the full
   competition spec): no cross-encoder or LLM reranking stage (`4.2.I`
   mentions "LLM Semantic Ranking" — current ranking is hybrid retrieval +
   attribute boost + MMR diversity, no learned/LLM reranker).

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
- (Historical — `_filter_by_disclosed`/`min_survivors` no longer exist,
  replaced by `_boost_by_disclosed`; kept for the general lesson.) The old
  filter's fallback threshold (`min_survivors`) was originally compared
  against the *entropy pool size* (~30) rather than the actual number of
  recommendations needed (`top_k`, e.g. 10) — meaning the filter fell back
  to unfiltered results almost every time, since removing a meaningful
  chunk of a 30-item pool is the whole point of filtering. Fixed at the
  time by threading the real `top_k` through separately from the padded
  pool-size argument; the whole fallback-threshold mechanism became moot
  once elimination itself was replaced with a non-eliminating resort.
- Style/size/use_case/feature disclosures never populate `SessionState.disclosed`
  — there's no structural vocab/extractor for them (same limitation the
  original entropy design already had for question *selection*; it also
  applies to filtering). This can look like a bug during manual REPL
  testing ("disclosed isn't filling up") when it's actually just the
  by-design ceiling of only 3 filterable attributes (material, color,
  budget).
- When building the long-term user-profile store (see "Known open
  problems" #5), "use it to skip a question instead of biasing ranking"
  was reasoned to have a strictly bounded downside ("waste one turn if the
  guess is wrong") *before* measuring — it doesn't. Excluding an attribute
  from being asked is not the same kind of action as a one-off skipped
  question: the exclusion is stored in session state and persists for
  every remaining turn, so a wrong guess can silently block the single
  most informative question for the rest of a 10-turn session. Lesson:
  "this failure mode is bounded" is itself a claim to verify empirically
  on the full set, not reason about from the code's shape alone — this
  project's own convention (measure before trusting a design argument)
  applies to safety arguments about a design, not just its expected
  benefit.
