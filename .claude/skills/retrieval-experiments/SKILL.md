---
name: retrieval-experiments
description: Full write-ups of retrieval, routing, and personalization approaches already tried and measured on this project. Read before re-attempting hard attribute filtering, a filter-then-rerank buying track, embedding-similarity reranking of disclosed values, or any user-profile personalization — each was built, measured, and rejected with numbers recorded here.
---

# Closed investigations

Archive of resolved retrieval/routing/personalization problems on this
project. Each was built and measured on the 200-sample public set, then
either fixed or abandoned. Read this before re-attempting hard attribute
filtering, a filter-then-rerank buying track, embedding-similarity reranking
of disclosed values, or any user-profile personalization — the numbers below
are why each was rejected.

Entry numbers are stable identifiers for this file, not references to
anything external.

---

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

---

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

---

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
   shipped with a measured regression.

   **Why all of the above failed, measured directly rather than inferred
   from the A/B results** (`scripts/eval_profile_signal.py`, read-only, runs
   in seconds — re-run it first if the hidden set's profiles look
   different). The three experiments above each asked "does *this* way of
   using the profile help?"; the script asks the prior question none of them
   did — *is there any signal in `user_profile` to use?* Three independent
   checks, all null on the 200-sample public set:
   - **A shared `profile_key` does not mean a shared shopper.** This is the
     load-bearing assumption behind the entire cross-session store. Of 125
     distinct keys, 30 recur (105 sessions). Across the 409 within-key
     target pairs, only **0.5%** share a `coarse_category` — against a
     **1.2% ± 0.5%** random-pair baseline. Two sessions with an identical
     profile hash are, if anything, *less* alike than two sessions picked at
     random. That single number explains all three regressions at once: the
     store was faithfully carrying forward values from sessions whose
     targets are uncorrelated with the current one, so every carry is a coin
     flip that can only cost.
   - **A session's own `preference_tags` predict nothing about what it can
     answer** — this was the one remaining idea logged here as promising,
     because it needs no cross-session identity assumption. It has no
     signal either. Comparing each (tag, bucket) cell against the null
     Binomial(tag support, bucket marginal): across the 5 tags with n≥30,
     **nothing exceeds 2σ and the largest departure is 1.8σ** over 30 cells,
     i.e. exactly what noise produces. (Judge these by σ, not by lift — the
     rare buckets have tiny marginals, so `use_case` shows lift 4.17 on a
     single sample.) Root cause: the profile is generated from the
     *reviewer's* history while the answerable buckets come from the
     *target product's* feature text via `intent_card()` — two independent
     sources, so there is no mechanism by which they could correlate.
   - **No profile field predicts `scenario_type`** either (which would have
     been a cheap, robust replacement for the thin-margin embedding intent
     classifier in #7): every field's per-value distribution sits on the
     0.40/0.40/0.15/0.05 marginal. `purchase_frequency` is the clearest
     case — it is the literal string `"3-4 prior purchases"` for all 200
     samples, carrying zero bits by construction.
   So the profile store is inert because there is nothing in the profile to
   act on, not because the three consumption strategies were badly chosen. A
   fourth strategy is not worth writing against this data. Revisit only if
   the collision check above comes back positive on the hidden set.

   **Methodology landmine found and fixed while measuring this.** The store
   is write-through and outlives the process, so re-running the local eval
   fed each run's history into the next: counting sessions whose `reset()`
   received a carried hint, three consecutive runs over the same 200 samples
   gave **45/200 → 105/200 → 200/200**. Any A/B of a `profile_hint`-consuming
   agent was therefore confounded by how many times the eval had been run
   before, drifting toward "every session is influenced" the longer you
   iterate — which is very likely why the numbers recorded above for the
   three experiments are erratic (e.g. the corroboration-gated run scoring
   0.585→0.588 against a 0.601 baseline). Treat those specific figures as
   indicative, not reproducible; the *direction* (all three regress) is
   corroborated independently by the null signal checks. Fixed by making the
   store path overridable (`$TECHJAM_PROFILE_STORE`, `starter/user_profile.py`);
   `scripts/run_eval.py` now points each run at a fresh temp store by
   default, with `--profile-store` to opt back into a persistent one. Isolate
   the official CLI the same way:
   `TECHJAM_PROFILE_STORE=/tmp/store.json uv run python3 -m evaluator.local_evaluator`.
   Cross-session persistence remains a real feature for the graded single
   pass — this only removes the re-scoring artifact. Score-neutral today
   since `profile_hint` is unread.

---

6. ~~**Prototype sentence length in `EmbeddingIntentClassifier`.**~~
   **Hypothesis rejected: no accuracy gain from shorter prototypes.**
   Hypothesis was that `PROTOTYPE_BUYING`/`PROTOTYPE_BROWSING`
   (`starter/classifier.py`) are long, multi-clause sentences that pack
   several attributes into one embedding (e.g. `"I want a wool sweater,
   crew neck, for under $50"` carries material + style + price at once),
   diluting the buying/browsing signal in the resulting vector and hurting
   nearest-centroid accuracy. Tested with `scripts/ab_prototype_length.py`
   (kept, read-only): held the eval set fixed (160 harvested turn-1
   utterances + 16 hand-written OOD probes, same pools `eval_intent.py`
   uses) and varied only the prototype set feeding the same centroid rule
   — baseline (current, full sentences, avg 8.8 words) against each of the
   13/14 originals rewritten as a short fragment carrying the *same*
   attributes (avg 4.0 words, content held constant, e.g. `"I need a pair
   of black leather boots in size 9."` → `"black leather boots, size 9"`;
   not truncation — every rewrite is a complete, natural utterance).
       variant                 avg words   turn-1   OOD probe
       baseline (full)               8.8    0.981       1.000
       short (content-matched)       4.0    0.969       1.000
   Turn-1 accuracy moved 0.981→0.969, i.e. ~2 of 160 examples — the wrong
   direction for the hypothesis, and not distinguishable from noise at
   this n, but zero evidence it helps either.
   Conclusion: on this eval set, prototype length is not the accuracy
   lever — full sentences are at least as good as, and by a small margin
   better than, shorter content-matched ones. This is consistent with the
   existing top-*k*-trimmed-prototype sweep in `eval_intent.py`'s own
   docstring (every trimmed variant scored worse than the untrimmed
   centroid) — two independent ways of "shrinking" the prototype signal
   both came back flat-to-negative. Do not re-shorten
   `PROTOTYPE_BUYING`/`PROTOTYPE_BROWSING` on this theory without new
   evidence.
   (`ab_prototype_length.py` also has two mechanical-truncation variants,
   @4 and @6 words, scoring markedly worse — 0.869 and 0.912. That's not a
   length result: truncation frequently cuts off the purchase-verb/
   attribute cue itself, e.g. `"I want to buy a pair of shoes"` →
   `"I want to buy a"`, so the drop is cue-loss, not dilution. Kept in the
   script as a documented confound, not as a second data point for this
   entry.)

---

7. ~~**PHRASE_MAX_N (verbatim-span cap in `BM25Index.phrase_search`).**~~
   **Fixed: 5 → 6.** `phrase_search` (`starter/retrieval.py:226`) walks
   n-gram size from `PHRASE_MAX_N` down to `PHRASE_MIN_N`, longest first,
   and only runs the first `MAX_PHRASE_QUERIES=24` distinct grams it
   builds — so raising `PHRASE_MAX_N` isn't free: it spends more of that
   fixed budget on long, rare spans before any shorter gram is tried at
   all. Swept with `scripts/sweep_phrase_max_n.py` (kept, read-only,
   supports `--track-failures` to dump per-sample hit/rank per value).
   60-sample seed-7 subset first (values 3–20), then the full 200-sample
   set at the shortlist it pointed to:
       PHRASE_MAX_N   HitRate    hits      MRR    MTTC  Technical    delta
       5 (was shipped)  0.945  189/200  0.5564   3.250     0.7944       --
       6                0.945  189/200  0.5799   3.290     0.8007  +0.0062
       8                0.940  188/200  0.5549   3.390     0.7887  -0.0058
       10               0.940  188/200  0.5575   3.400     0.7893  -0.0052
   5 and 6 produce the **identical 11-sample miss set** — 6's entire gain
   is from re-ranking the 189 shared hits (32 samples changed rank, 20
   improved / 12 worsened, a real net gain not one lucky sample). 8 and 10
   are identical to each other in outcome and add exactly one new miss on
   top of those 11 (`public_0159`, a `buying` sample, hit at 5/6, miss at
   both 8 and 10) — the budget-crowding mechanism costing a real hit, not
   noise. 10, 15, and 20 scored byte-for-byte identical on the subset
   sweep too: once `PHRASE_MAX_N` exceeds the longest query actually seen,
   no new grams get generated, so headroom past ~10 is dead weight, and
   the 6→8 range is where the crowding cost already exceeds the
   longer-span benefit. Shipped at **`PHRASE_MAX_N = 6`**
   (`starter/config.py`). Do not raise it back toward 8–10+ on the "longer
   verbatim span = more specific = better" intuition without re-running
   this sweep — it does not hold once the 24-query budget is in the
   picture.

---

8. ~~**`EmbeddingNonAnswerDetector` misclassifying terse catalog-jargon
   answers as non-answers.**~~ **Fixed.** Found while root-causing the 11
   full-set misses left after entry 7 (all `lost-in-fusion`: the target
   never reaches BM25/dense's own top pages, let alone the fused pool —
   see `scripts/eval_failures.py --sample-ids`, which now supports a
   `--sample-ids` filter for exactly this kind of targeted re-probe
   instead of the full ~hour run). 8 of the 11 shared one shape: the agent
   asks the `feature` question, the customer *does* answer with real
   content straight from the catalog (the evaluator's `customer_reply`
   template is `"For that, what matters is: {value}."`, and `value` is
   only ever emitted when non-empty — informative by construction), but
   the value is often a bare spec fragment (`"Imported."`, `"Button
   closure."`, `"Pull On closure."`) rather than a natural sentence.
   `PROTOTYPE_INFORMATIVE` (`starter/classifier.py`) was all full
   sentences, so these fragments embedded closer to
   `PROTOTYPE_NON_ANSWER` and got scored a non-answer — silently dropped
   from the query at `agent.py`'s `SKIP_NON_ANSWERS_IN_QUERY` gate, so the
   session's query never grew past "category + material" for the rest of
   the session. Confirmed directly, not inferred: measured
   `EmbeddingNonAnswerDetector.is_non_answer` on the exact transcript
   strings before touching anything —
       text                                          sim_na  sim_inf  verdict
       "For that, what matters is: Imported."          .619    .599  non_answer (wrong)
       "...Button closure; Hand Wash Only."             .627    .610  non_answer (wrong)
       "...Pull On closure."                            .599    .582  non_answer (wrong)
       "...color: grey."                                .651    .648  answer (correct — carries an explicit label the informative prototypes resemble)
   `classify_reply_lexically` (the non-embedding fallback) got all of
   these right already — the bug was specific to the embedding path.
   Fix: added ~12 terse catalog-jargon examples to `PROTOTYPE_INFORMATIVE`
   (`"Imported."`, `"Button closure."`, `"Hand Wash Only."`, etc.) rather
   than hard-coding the evaluator's literal template string, so the fix is
   a real embedding-similarity improvement (should transfer to the hidden
   grading simulator's own phrasing) and not local-template memorization.
   Re-verified against the same transcript strings post-fix: every one
   flips to `answer`, and the two genuine declines tested alongside them
   (`"I don't have an additional preference for feature."`, the boundary
   hand-back phrasing) still correctly read `non_answer`.
   Full 200-sample set, before → after (before = entry 7's shipped state,
   `PHRASE_MAX_N=6`, no other change):
       metric             before    after     delta
       HitRate            0.9450   0.9650   +0.0200  (189/200 -> 193/200)
       MRR                0.5799   0.5980   +0.0181
       MTTC                3.290    2.985   -0.305 (faster)
       TechnicalScore      0.8007   0.8222   +0.0215
   Exactly 4 samples flipped miss→hit (`public_0052, 0083, 0095, 0174`),
   **zero regressions** anywhere in the 200-sample set (checked
   sample-by-sample, not just the aggregate). The other 4 diagnosed
   samples (`public_0020, 0096, 0111, 0161`) did not flip — probed again
   post-fix and confirmed the fix still helped (e.g. `public_0111` moved
   from completely unreachable to a real fused rank of 129), but those
   customers never disclose anything beyond material + one generic
   catalog term for the rest of the session, so "category + material +
   Imported" still isn't enough to outrank hundreds of similar products.
   That is a genuine information ceiling, not a further instance of this
   bug — do not re-chase it by further editing `PROTOTYPE_INFORMATIVE`.
