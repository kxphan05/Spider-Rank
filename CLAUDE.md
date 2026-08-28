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
   once across the 200-sample public set) with zero net effect on scoring.
   Its inertness is now a measured conclusion rather than a pending TODO:
   `scripts/eval_profile_signal.py` shows the `user_profile` dict carries no
   signal to act on in the first place (same-key sessions' targets agree
   *below* chance), so there is no fourth strategy worth trying — see
   "Known open problems" #5.

Code layout note: the buying/browsing label decides four things (BM25 weight,
dense weight, entropy threshold, whether the MMR diversity re-rank runs). All
four are resolved in one place, `routing_params()` in `agent.py` — previously
the `intent_label == "buying"` conditional was written out separately in
`_retrieve`, `_next_attribute` and `respond`'s debug log, which meant the log
could report weights retrieval wasn't using. Add new intent-conditioned
behaviour to `RoutingParams`, not as a fourth branch.

Scripts share `scripts/_common.py`, which puts the repo root on `sys.path` as
an import side effect and exports `DEFAULT_CATALOG` / `DEFAULT_DATASET` (both
anchored to the repo root, so scripts run from any directory) and
`isolate_profile_store()`. Import it before any `starter`/`evaluator` import.

Lint gate (config in `pyproject.toml`, `evaluator/` excluded as
organizer-provided): `uvx ruff check starter/ scripts/`.

Eval runs: `uv run python3 scripts/run_eval.py` (dev wrapper — `--limit`,
`--scenario`, `--seed`, and a tqdm progress bar on stderr, `--no-progress` to
suppress). tqdm is imported defensively and comes in transitively with
sentence-transformers, so the submitted dependency set is unchanged.

Submission bundle: `uv run python3 scripts/build_submission.py --verify`
assembles `dist/submission/` in the rules' recommended shape (`agent.py`
exporting `Agent`, `src/`, `requirements.txt`, `README.md`, `REPORT.md`,
`tools/`) and then imports that bundle's `Agent` from a neutral working
directory, asserts dense retrieval and both classifiers came up live, and
scores it on the full public set. The bundle is **generated, never
hand-edited** — change `starter/`, then rebuild.

Data and model paths resolve in three steps (`starter/agent.py`
`_resolve_data_path`, `starter/paths.py` `model_cache_dir`): the
`TECHJAM_CATALOG` / `TECHJAM_DENSE_INDEX` / `TECHJAM_MODEL_DIR` env var, then
the path relative to the working directory, then the path beside the package.
Before this, all three were bare cwd-relative strings (`"data/catalog.jsonl"`,
`"./model"`), so running the agent from anywhere but the repo root silently
lost the dense index and both classifiers — the same silent-degrade class as
#15, and the exact condition a submitted bundle runs under.

Offline-asset setup: `uv run python3 scripts/fetch_assets.py` (the only step
that needs network) then `uv run python3 scripts/preflight.py --strict` to
verify the pipeline comes up whole with the network disabled. See "Known open
problems" #15 -- the degrade is silent, so this check is not optional.

Runtime disclosure: `uv run python3 scripts/measure_latency.py --limit 20`
produces the latency/memory/token numbers in `docs/team_report.md` § 4.

Manual testing: `uv run python3 scripts/repl.py` — interactive single-session
REPL against the live agent. Commands: `/reset`, `/debug` (prints
`disclosed`/`asked_attributes` each turn), `/topk N`, `/quit`.

Override diagnostics: `uv run python3 scripts/eval_override.py` — sweeps
scoring rules for `EmbeddingOverrideDetector` against the simulator's own
harvested override/continuation turns plus hand-written out-of-distribution
pivots. Read-only and Agent-free. Run this before touching the detector; the
measured table is in "Known open problems" #7.

Intent diagnostics: `uv run python3 scripts/eval_intent.py` — sweeps scoring
rules for `EmbeddingIntentClassifier` on turn-1, accumulated, and
out-of-distribution pools, with the lexical fallback as a floor. Companion to
`eval_override.py`; the measured table is in "Known open problems" #13.

Intent-head diagnostics: `uv run python3 scripts/train_intent_head.py` — trains
a logistic head on frozen bge-small embeddings and scores it against the
centroid classifier on in-distribution, held-out-template, and
out-of-distribution pools. Saving is opt-in; the measured verdict is "don't
ship it" ("Known open problems" #12).

Presentation demo: `uv run python3 scripts/build_demo.py` renders
`dist/demo.html`, a turn-by-turn replay of four real sessions (committed in
`demo/sessions.json`). `--capture` re-runs the live agent to regenerate them.
UI is out of scope for scoring (`TODO.md` 4.3) — this is for presenting only.
Note the override flag comes from the real detector, not from watching the slot
dict shrink: an override clears the slots and the same turn can immediately
refill one, so the naive heuristic misses it.

Fusion-weight sweep: `uv run python3 scripts/sweep_fusion_weights.py --track buying`
— reports TechnicalScore as a curve against the dense:bm25 ratio (only the ratio
matters; weighted RRF is scale-invariant per leg). Companion:
`scripts/sweep_prior_leg.py`, which sweeps a *third* RRF leg built from catalog
priors (`popularity`, `rating`) or from the user profile (`profile_rating`).
Note the two profile-independent variants are the controls — a gain from
`profile_rating` means nothing unless it beats `rating`.

Question-policy design: `docs/question_policy_plan.md` — replaces the
two-stage entropy-then-fixed-order picker with one expected-value objective
(answerability x gain). Written, not built. Records two corrections found by
reading the evaluator: there is no per-turn cost to trade off (the agent cannot
end a session), and `ask_attribute: null` still draws a reply, so it is a minor
refinement rather than a win.

Intent-detection research + staged plan: `docs/intent_detection_plan.md`
(industry practice, where this pipeline is strong vs thin, and why slot filling
rather than the intent classifier is the bottleneck).

Profile diagnostics: `uv run python3 scripts/eval_profile_signal.py` — asks
whether `user_profile` carries any usable signal at all (tag→answerable-bucket
departure from the null, profile field→scenario_type, profile-key
collision→target similarity, and cross-run store contamination). Read-only and
Agent-free, so it runs in seconds. Run this *before* wiring any new
personalization idea up; on the public set every check is null, and the
measured numbers are in "Known open problems" #5.

## Progress / what's been measured

Full-public-set (`uv run python3 -m evaluator.local_evaluator`, 200 samples)
results as of the last run:

```
HitRate@10: 0.750   MRR: 0.4096   MTTC: 4.985   Efficiency: 0.6015
TechnicalScore: 0.6182
```

(That line is with the buying dense:bm25 ratio at **0.25**, shipped after the
sweep in #16 — the previous 0.50 scored 0.6070 on this same HEAD, so the
change is worth +0.0112 and reproduced the sweep's prediction exactly.)

(That line is with the `TOP_PROTOTYPES = 4` override detector from "Known open
problems" #7, re-run on this HEAD. The centroid it replaced scored 0.6073 —
the two are indistinguishable at this sample size, and #7 explains why the
change was made on offline detector quality rather than on this number. An
earlier revision of this block quoted the *centroid's* 0.388/0.6073 while
labelling it k=4; the k=4 figures are the ones above.)

(Previous run, before the `FALLBACK_ATTRIBUTE_ORDER` answerability reorder
described in "Known open problems" #9: HitRate 0.740 / MRR 0.395 /
MTTC 5.605 / Efficiency 0.540 / TechnicalScore 0.596 — the reorder is
almost purely an MTTC win, +0.0109 TechnicalScore. Note this pair of runs
brackets the #10 word-boundary fix and the #11 LM inference, both already
landed, so it is *not* comparable to the 0.601 line below.)

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

1. ~~**Hard-filter self-elimination.**~~ **Fixed.** The old hard-equality
   filter dropped candidates whose extracted value disagreed with a disclosed
   one — which disagreed with the *true target* 16.3% of the time on material
   and 37.0% on color, so it eliminated the right answer that often. Replaced
   by the non-eliminating `Agent._boost_by_disclosed` resort (architecture
   #3); every metric improved. **Do not retry reranking by embedding
   similarity between the disclosed phrase and catalog embeddings** — swept
   and measured *monotonically worse* as its weight rose, below even a
   no-signal baseline. (Root cause of the disagreement itself was later found
   to be the substring-matching bug in #10.) Full write-up and the sweep
   numbers: `.claude/skills/retrieval-experiments/SKILL.md`.
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
   `signal.label` now threads into intent-conditioned RRF fusion weights, a
   browsing-only MMR diversity re-rank, and the entropy threshold. Net a wash
   on the full set (TechnicalScore 0.600→0.601) despite a real gain on the
   60-sample tuning subset. **Do not retry the hard filter-then-rerank buying
   track** (BM25 top-`CANDIDATE_N` as a hard filter, dense reranking only
   within it): measured worse on hit rate and MTTC, because it discards the
   full-catalog dense leg that exists to catch targets phrased differently
   from their catalog text. Full write-up and weight sweeps:
   `.claude/skills/retrieval-experiments/SKILL.md`.
5. ~~**No long-term user-profile persistence across sessions.**~~ **Store
   built; personalization measured dead.** `UserProfileStore` persists
   per-profile-key disclosure history across restarts, and
   `SessionState.profile_hint` carries the corroborated value — but nothing
   reads it, deliberately. **Do not wire it into ranking or question
   selection.** Three different ways were each measured to regress the full
   200-sample set, and `scripts/eval_profile_signal.py` shows why they had
   to: same-key sessions' targets share a coarse category 0.5% of the time
   against a 1.2% ± 0.5% random-pair baseline, so a profile-key match is not
   an identity signal and there is nothing correct to carry. Re-run that
   script first if the hidden set's profiles look different. Full write-up,
   including all three experiments and the store-contamination landmine:
   `.claude/skills/retrieval-experiments/SKILL.md`.
6. Remaining gaps from the original spec analysis (`TODO.md` has the full
   competition spec): no cross-encoder or LLM reranking stage (`4.2.I`
   mentions "LLM Semantic Ranking" — current ranking is hybrid retrieval +
   attribute boost + MMR diversity, no learned/LLM reranker).

7. **Override detector: centroid replaced by a trimmed nearest-prototype
   rule + lead-clause scoring. Detector measurably better, end-to-end score
   flat (-0.0005).** The diagnosis below was confirmed and acted on; the
   analysis is kept because it explains the fix.

   New rule (`EmbeddingOverrideDetector`, `classifier.py`): score against the
   mean of each class's `TOP_PROTOTYPES = 3` closest prototypes instead of the
   class centroid, and evaluate both the full message and its lead clause
   (`lead_clause()`), taking whichever leans more override. Swept in the new
   `scripts/eval_override.py`, which harvests the simulator's own override and
   continuation turns (30 positives / 1600 negatives over the 200 public
   samples) plus 20 hand-written out-of-distribution probes:

   ```
   rule                    sim recall  sim FPR   probe recall  probe FPR
   centroid (previous)          0.900    0.000          0.800      0.100
   nearest prototype (k=1)      0.933    0.151          1.000      0.300
   top3-mean                    0.933    0.007          1.000      0.100
   top3-mean + lead clause      1.000    0.007          1.000      0.100   <- shipped
   top4-mean + lead clause      0.933    0.001          1.000      0.100
   top5-mean + lead clause      0.933    0.000          0.900      0.100
   ```

   Confirms the CLAUDE.md #7 prediction exactly: **k=1 (max) does fix every
   terse pivot but raises the false-positive rate 20x** (0.000 -> 0.151, i.e.
   241 of 1600 ordinary turns silently wiping session state). A small trimmed
   mean gets max's shape-robustness without its fragility.

   **Full 200-sample A/B, both legs on this HEAD:**

   ```
                 HitRate    MRR      MTTC    Technical
   before         0.7450   0.3888   5.0900    0.6073
   after          0.7450   0.3875   5.0950    0.6068   (-0.0005)
   ```

   Only **4 of 200 sessions changed**: one `intent_override` improved
   (rank 9->6), and three regressed slightly (two `buying`, one `browsing`)
   from false positives firing on the simulator's disclosure template
   ("For that, what matters is: ..."), which is where all 12 remaining FPs
   live. Kept anyway, on the same reasoning as #10: **the local simulator
   cannot measure this at all.** `behavior_for()` emits exactly one override
   string, `"Actually, ignore my earlier preference. What I need is: {X}."`,
   which is near-verbatim `PROTOTYPE_OVERRIDE[0]` — so simulator recall was
   already 0.900 by construction and there is no local headroom to win. The
   probe column (0.800 -> 1.000) is the only evidence about phrasings the
   hidden grader might use, and it is hand-picked. Treat -0.0005 as noise, not
   as a cost.

   **Resolved: `TOP_PROTOTYPES = 4` shipped.** Swept against the real class
   (not a reimplementation) and A/B'd on the full set:

   ```
    k   sim recall   sim FPR      sim FP   probe recall   probe FPR
    1        0.933    0.1512    242/1600          1.000       0.400
    2        1.000    0.0156     25/1600          1.000       0.200
    3        1.000    0.0075     12/1600          1.000       0.100
    4        0.933    0.0013      2/1600          1.000       0.100   <- shipped
    5        0.933    0.0000      0/1600          0.900       0.100
   12        0.900    0.0000      0/1600          0.900       0.100   (= centroid)

                 HitRate    MRR      MTTC    Technical
   centroid       0.7450   0.3888   5.0900    0.6073
   k=3            0.7450   0.3875   5.0950    0.6068
   k=4            0.7450   0.3876   5.0900    0.6070
   ```

   k=4 keeps the entire out-of-distribution gain (probe recall 1.000, vs the
   centroid's 0.800) while cutting false positives 12 -> 2 of 1600. Per-sample
   it changes **2 of 200 sessions** against the centroid (k=3 changed 4): one
   `intent_override` improves rank 9->6, one `buying` regresses rank 2->5. The
   two overrides k=4 loses are both the pathological template case where
   `new_value` is 180 characters of verbatim catalog copy.

   All three legs are within ±0.0005 TechnicalScore, which is **below this
   benchmark's resolution** — 200 samples cannot separate them, and the
   end-to-end numbers should not be read as evidence either way. The decision
   rests on the offline table, where k=4 dominates the centroid on both recall
   columns at a cost of 2 false positives in 1600 turns. The centroid baseline
   was re-run and reproduced 0.6073 exactly, so the legs are deterministic and
   comparable.

   Applied to `EmbeddingIntentClassifier` too? **No — measured and rejected,
   see #13.** Two corrections to what this section used to claim about that
   classifier. Its output is *not* "a continuous score feeding fusion
   weights": `signal.score` is referenced nowhere but a log line
   (`agent.py:310`), and every consumer — fusion weights, the MMR re-rank, the
   entropy threshold, `_next_attribute` — branches on `signal.label`. It is a
   binary decision, same as this one. And it does not in fact share the
   thin-margin problem in any way that produces errors: it scores 0.988 on
   turn-1 intent. The *trained* route for it was also tried and rejected — #12.

   Original analysis follows.

   Reported from manual REPL use and
   reproduced directly: `"never mind, give me white shoes"` was **not**
   detected as an intent override — `EmbeddingOverrideDetector.is_override`
   returns `False` (override centroid 0.6836 vs continuation centroid
   0.7171, margin −0.034), so `disclosed`/`asked_attributes` are never
   cleared and the stale earlier constraints keep biasing the pool for the
   rest of the session (`agent.py:247`). Measured margins on hand-written
   probes around it:

   ```
   never mind, give me white shoes            ovr 0.6836  cont 0.7171  -> MISS
   scratch that, white shoes                  ovr 0.6521  cont 0.6555  -> MISS
   never mind what I said, give me white shoes ovr 0.7122 cont 0.6989  -> hit
   actually, give me white shoes instead      ovr 0.7537  cont 0.6834  -> hit
   I changed my mind, white shoes please      ovr 0.7767  cont 0.6682  -> hit
   ```

   Root cause is the *centroid*, not the prototypes: every entry in
   `PROTOTYPE_OVERRIDE` is a long two-clause sentence that explicitly names
   the discarded prior statement ("what I said before", "my earlier
   preference"), so the mean vector encodes that whole sentence *shape*.
   A terse pivot is dominated by its imperative-request half
   ("give me white shoes"), which sits closer to the continuation
   centroid. Mean-pooling also throws away the fact that a single
   prototype matches strongly. Decision margins are ~0.03 in a space where
   both centroids sit at ~0.65–0.80 similarity to almost anything — the
   boundary is inside the noise floor, which is the same fragility already
   documented for the buying/browsing deadzone experiment
   (`classifier.py:322`).

   Ordered by cost, before jumping straight to fine-tuning:
   - **Nearest-prototype (max) instead of centroid** — one-line change,
     no training. On the 8-example hand-written probe above it got 8/8
     where the centroid got 6/8, including both terse pivots. *This is an
     8-example hand-picked probe, not a measurement* — it has to be run
     through `scripts/eval_classifier.py` and the full 200-sample set
     before being believed, and it will likely raise false positives
     (max-similarity is less robust to one badly-placed prototype).
   - **Add terse-pivot prototypes** ("never mind, X", "scratch that, X",
     "forget it, X") — also free, but treats the symptom; the shape
     mismatch will recur for whatever phrasing the hidden simulator uses.
   - **Train a real classifier** (logistic regression / small MLP head on
     the frozen bge-small embeddings — not fine-tuning the encoder, which
     `TODO.md`'s out-of-scope list arguably bars and which needs far more
     data). A learned boundary can weight the discard-cue dimensions
     instead of averaging them away with the request half of the sentence.
     Blocker: **there is no labeled override data yet.** The label source
     would be the evaluator's own reply generator — `customer_reply()` and
     `initial_message()` in `evaluator/local_evaluator.py` already produce
     override turns for `intent_override` samples with known ground truth,
     so a labeled set can be harvested from the 200 public samples the same
     way `scripts/eval_classifier.py` already reconstructs turns. Caveat
     that applies to any of these: training on the local simulator's
     vocabulary risks overfitting to it, which is exactly the failure the
     zero-shot design was chosen to avoid (`classifier.py:6`) — hold out
     the hand-written probes above as an out-of-distribution check.

   The same argument applies to `EmbeddingIntentClassifier` (buying vs
   browsing), which shares the mechanism and the thin-margin problem, but
   the override detector is the higher-value target: a missed override is a
   whole-session failure, a wrong buying/browsing label only shifts fusion
   weights.

8. **LLM-extracted catalog attributes (offline, one-time).** Designed and
   costed, nothing built — full working notes in `NEXT_STEPS.md`. Replaces
   `AttributeIndex`'s first-vocab-hit extraction with a one-time
   `claude-haiku-4-5` Batch API pass over the 50k catalog (~$5–12, ~9.4M
   input tokens), emitting a static closed-vocabulary JSON sidecar shipped
   as a local asset so the agent still needs no network at scoring time.
   Motivation: thin coverage (78.1/58.6/21% material/color/price), the
   16.3%/37.0% disagreement with the true target from #1, and the four
   attributes (`style`, `size`, `use_case`, `feature`) that have no
   extractor at all. Caveat recorded there: this fixes only the catalog
   half of the two-extractor disagreement.

9. ~~**`FALLBACK_ATTRIBUTE_ORDER` asks the three least-answerable
   attributes first.**~~ **Fixed — the single largest remaining MTTC win.**
   The order is now sorted by *answerability*: share of the 200 public
   samples where the customer can still answer a question about each
   attribute after turn 1, derived from the evaluator's own reply policy (a
   constraint is disclosed only when `classify_constraint()` buckets it as
   the asked attribute):

   ```
   feature   0.960      style     0.085
   material  0.725      size      0.045
   color     0.255      use_case  0.020
   ```

   The old order (`style, size, use_case, feature, budget`) made the agent
   ask the three *rarest* buckets (8.5%/4.5%/2.0%) before the one answerable
   in 96% of sessions, once the entropy picker ran out of material/color.
   Now `feature, style, size, use_case, budget`.

   Full 200-sample A/B, both legs run on this HEAD:

   ```
                 HitRate    MRR      MTTC    Technical
   before         0.7400   0.3949   5.6050    0.5964
   after          0.7450   0.3888   5.0900    0.6073   (+0.0109)
   ```

   Per-sample: **39 samples reach the target sooner, 5 later**, 156
   unchanged; 3 new hits against 2 lost. The modal delta is **−3 turns on 24
   samples**, which is exactly the predicted mechanism (skip style/size/
   use_case, ask the one bucket the customer can answer). So the win is
   broad, not a handful of lucky samples.

   Two things worth keeping on the record. **MRR moved the wrong way**
   (−0.0061): the score's MRR uses the rank at the *first* hit, not the best
   rank ever, so converting a late-and-well-ranked hit into an early-and-
   worse-ranked one costs MRR while paying more in Efficiency. That trade is
   inherent to the scoring and it nets strongly positive here — but it means
   MTTC work will keep showing a small MRR drag, and a future change should
   not be judged on MRR alone. **The 96% is partly an artifact**: `feature`
   is `classify_constraint()`'s catch-all `return`, the bucket every
   unmatched value falls into, so this may be a local-simulator quirk that
   doesn't hold for the hidden grader — the same shape as the `budget`
   finding in "Progress" above. `budget` stays last-resort rather than
   dropped for the same reason.

10. **Catalog attribute extraction was substring-matched, not word-boundary
    matched — fixed, and it is a real bug whose fix currently *costs* 0.005.**
    Found while building co-occurrence statistics (#11): the strongest
    "signal" in the catalog was `P(material=lace | category=Necklaces
    Pendant Necklaces) = 0.867`, which is not a fact about jewelry — it is
    "neck**lace**" containing "lace". `_extract_material`/`_extract_color`
    tested `word in text` with no boundaries, while both the evaluator
    (`MATERIAL_RE`/`COLOR_RE`) and this repo's own customer-side
    `extract_disclosed_value` have always used `\b`. Only the catalog side
    was loose. Measured over the 50k catalog:

    ```
    color:    21.6% of products mislabeled (10,824), of which 9,312 are a
              value invented where no color word occurs at all
              ("red" in embroidered, "tan" in titanium/instant)
    material:  4.7% mislabeled (2,344), dominated by lace <- necklace
    coverage: color 58.6% -> 39.9%, material 73.7% -> 70.9%
    ```

    **This is the root cause of the previously-unexplained disagreement in
    open problem #1** (16.3% material / 37.0% color conflict with the true
    target), which had been worked around by weakening the hard filter to
    `_boost_by_disclosed`. Fixed in `attributes.py` (`_vocab_matcher`, one
    word-boundary alternation, longest-first, re-ranked by vocab order to
    preserve the original selection rule).
    **Honest result: the fix measures -0.0054 on the full public set**
    (HitRate 0.755→0.740, MRR 0.384→0.390, MTTC 5.600→5.585,
    TechnicalScore 0.6008→0.5954), reproduced both with and without the LM
    work in #11. Best current understanding of why: the buggy extractor
    assigned a color to 58.6% of the catalog instead of 39.9%, and
    `_boost_by_disclosed` penalizes a *known* mismatch by −1 while ignoring
    unknowns, so the wrong labels gave the resort far more (noisy)
    separation to work with. Correct-but-sparser labels leave more
    candidates neutral. The indicated next step is therefore **not** to
    revert the bug but to retune the boost/penalty weights against the
    corrected coverage — the current ±1/0 scheme was tuned when 58.6% of
    color labels existed and a fifth of them were fictional. Note the
    HitRate delta is 3 samples out of 200; treat it as directional.

11. **Co-occurrence and LLM-perplexity attribute inference.** Two attempts at
    the same goal — infer an attribute the customer has not stated — after
    the profile-based route in #5 measured dead.
    - **Item-attribute co-occurrence** (`scripts/cooccurrence.py`): counts
      `P(color | category)`, `P(color | material)` etc. over the catalog.
      Runnable as a diagnostic (`uv run python3 scripts/cooccurrence.py`); it
      lives in `scripts/` rather than `starter/` precisely because no agent
      code path reaches it, and `starter/` ships verbatim into the bundle.
      The signal is enormous and real once #10 is fixed
      (`P(material=stainless steel | category=Watches Wrist Watches)` = 0.483
      vs 0.023 marginal, +79σ; `P(material=mesh | Running Road Running)` =
      0.425 vs 0.032). **Built but not wired into the agent.** Note clearly:
      this is *item* co-occurrence, not user co-occurrence — the shipped data
      has no user-item interactions at all (catalog fields are
      `parent_asin, title, features, description, price, categories, details,
      average_rating, rating_number, store`; `average_rating` is a per-item
      aggregate). "Users who bought X also bought Y" is not computable here.
    - **Local masked-LM belief with an entropy gate**
      (`starter/lm_confidence.py`, wired into `Agent._infer_attributes`):
      distilbert-base-uncased (67M, ~255MB, cached in `model/` beside
      bge-small) scores the closed `MATERIALS`/`COLORS` vocabularies in a
      `[MASK]`ed template and returns a normalized-entropy confidence.
      A local model is required because the hosted Claude Messages API
      returns no token logprobs, and because `docs/submission_rules.md`
      § Model Policy warns network access may be disabled for official
      scoring; bge-small itself cannot do it (its published weights carry no
      LM head — 200 tensors, no `cls.predictions.*`).

      **The entropy gate works — this is the solid finding**
      (`scripts/eval_lm_confidence.py`, predicting the true target's
      extracted material from the turn-1 message):

      ```
      material   top-1 0.483 vs 0.322 guess-the-mode baseline   (+0.161)
        H < 0.60      n= 61   accuracy 0.787
        H 0.60-0.75   n=105   accuracy 0.371
        H 0.75-0.85   n= 14   accuracy 0.000
      ```

      Monotonic and steep, so `MAX_CONFIDENT_ENTROPY = 0.60` is calibrated
      rather than guessed. **Color is deliberately excluded**: top-1 0.189
      against a 0.432 always-"black" baseline — worse than doing nothing —
      and its gate ran *backwards* (confident 0.172 vs unsure 0.250), while
      collapsing onto a single attractor value ("pink" for 14 of 37
      targets). Template phrasing mattered more than expected: three colour
      templates scored 0/3 on probes whose answer was stated in the text and
      predicted "purple" for every input.

      **But the end-to-end payoff is ~noise: +0.0010 TechnicalScore**
      (MRR +0.0045, HitRate flat, MTTC 0.02 worse), reproduced twice —
      0.6008→0.6018 on the original extractor and 0.5954→0.5964 on the
      fixed one. Why the calibrated predictor buys so little: it fires only
      on material; the customer already discloses material in 72.5% of
      sessions, so the inference mostly duplicates information that arrives
      anyway; and it must be boost-only (agreement lifts, disagreement is
      ignored) because #5 measured that any mismatch penalty sinks a
      candidate below every neutral one. Cost: **246 ms per inference**
      (~2.5 s per 10-turn session) and ~255MB of shipped asset.
      Conclusion: the mechanism is validated, the application point is
      low-leverage. If it is kept, the higher-leverage use is predicting
      *which attribute the customer can answer* (see #9's answerability
      marginals) rather than nudging the candidate resort.

12. **Trained intent-classifier head: built, measured, rejected.** The
    "train a real classifier" option floated in #7 was implemented for the
    buying-vs-browsing classifier — a logistic-regression head on *frozen*
    bge-small embeddings (`scripts/train_intent_head.py`). Note the encoder is
    never touched: TODO.md 4.3 puts "training or full-parameter fine-tuning of
    base foundational LLMs" out of scope, while "designing highly sensitive
    intent-detection modules to split traffic into Buying and Browsing tracks"
    is explicitly *in* scope, so a head over frozen embeddings is the only
    version of this idea that is allowed at all.

    **It is worse than the zero-shot centroid it would replace:**

    ```
    pool                        head    centroid
    in-distribution 5-fold CV   0.984      0.778   <- memorization, ignore
    train turn1 -> test accum   0.521      0.708   <- chance
    out-of-distribution probes  0.812      1.000
    ```

    The in-distribution number is the trap: the simulator has exactly two
    turn-1 templates (`"...A key requirement is: {c}."` vs `"...but I'm still
    exploring."`), so 0.984 is the head learning `"still exploring"`. Three
    checks confirm that reading. **A regularization sweep is flat at chance**
    on the held-out surface form across C = 0.001 … 10 (0.537 / 0.521 / 0.519
    / 0.521 / 0.523), so the transfer failure is the data, not a
    hyperparameter. **OOD accuracy *rises* with C** (0.688 -> 0.812), i.e. the
    more it overfits the templates the better it scores out of distribution —
    the opposite of a generalization story, and a sign the 16-probe trend is
    noise. And **the control settles it**: training on the 20 hand-written
    `PROTOTYPE_*` utterances alone, with no simulator text at all, reaches OOD
    1.000 while 640 labelled simulator turns drag it to 0.812.

    So the labels are real but the *surface forms* are two templates, and
    supervised training on them destroys exactly the zero-shot generalization
    the prototype design was chosen for (`classifier.py:6`). **Do not ship a
    head trained on local simulator text**, and do not read a high
    in-distribution CV score as progress. The script is kept as a diagnostic
    with saving opt-in (`--save`); re-run it only if the hidden set's phrasing
    turns out more varied than the local templates, which is the one condition
    that would change the answer.

    The unsupervised alternative (the trimmed-prototype rule from #7) was then
    tried as well, and also rejected — #13.

13. **Trimmed-prototype rule does *not* transfer to the intent classifier.**
    #7's fix for the override detector was swept for `EmbeddingIntentClassifier`
    in the new `scripts/eval_intent.py`. Every variant is worse, and not
    marginally:

    ```
    rule                    turn-1   accumulated   OOD probe
    lexical (fallback)       1.000         0.773       1.000
    centroid (current)       0.988         0.708       1.000   <- unchanged
    top1-mean                0.781         0.608       0.938
    top4-mean                0.769         0.588       0.938
    top10-mean (= centroid)  0.831         0.704       0.938
    ```

    Why it transferred to one classifier and not the other: the override
    prototypes had a specific *shape* pathology — all twelve are long
    two-clause sentences that name the discarded prior statement, so the mean
    encoded that sentence form and terse pivots fell outside it. The
    buying/browsing prototypes have no such common form; they are varied
    single utterances whose mean direction *is* the signal, so trimming to the
    top-k throws away the averaging that makes the centroid robust. **The
    lesson is that "centroids are fragile" is not a general truth about this
    codebase's classifiers — it was a fact about one prototype set.** Check the
    prototype set's shape before assuming the fix generalizes.

    **There is also no headroom to chase here.** The centroid is at 0.988 on
    turn-1, its only two errors being buying sessions whose hard constraint is
    contentless ("A key requirement is: Imported."). And per #4, the thing the
    label feeds — dual-track routing — was itself measured net flat on the
    full set (0.600 -> 0.601). A perfect intent label is worth approximately
    nothing; do not spend more on this classifier.

    Two measurement caveats worth carrying forward, both instances of the #12
    trap. **The lexical fallback's 1.000 on turn-1 is template memorization**:
    `\bkey\s+requirement\b` and `\bstill\s+(?:exploring|...)\b` in
    `BUYING_PHRASES`/`BROWSING_PHRASES` match the simulator's two templates
    literally, so that number means nothing, exactly like #12's 0.984. **And
    the OOD probe pool is partly circular** — the lexical rule scores 1.000
    there by a degenerate path (every browsing probe registers 0/0 evidence
    and falls through to the browsing tie-break, while every buying probe hits
    attribute vocabulary), which is true because the probes were hand-written
    to have exactly that property. The probes can falsify a rule but should
    not be used to rank two rules that both pass.

    The `accumulated` column is reported for shape only and is **not** an
    accuracy: `scenario_type` is not valid ground truth once clarifying
    answers land, because the classifier is deliberately built to drift
    buying-ward as concrete attributes accumulate (`classifier.py:14`).

14. **Component ablation: the dense leg is net-negative locally, and we
    shipped it anyway.** Each embedding-dependent component was disabled
    independently against the full 200-sample public set (scratch script;
    reproduce by nulling `agent.dense` / `.intent_classifier` /
    `.override_detector` / `.lm_scorer` after `Agent()` and calling
    `evaluate()`):

    ```
    configuration            HitRate     MRR     MTTC   Technical      delta
    as shipped                0.7450  0.3876    5.090     0.6070          --
    - masked LM               0.7450  0.3841    5.085     0.6060     -0.0010
    - both classifiers        0.7400  0.3728    5.070     0.6004     -0.0065
    - dense retrieval         0.7450  0.4328    5.035     0.6216     +0.0147
    - dense - masked LM       0.7450  0.4351    5.035     0.6223     +0.0154
    - everything (offline)    0.7450  0.4289    5.025     0.6207     +0.0137
    ```

    **HitRate is identical to four decimals with and without dense.** The
    dense leg finds nothing BM25 misses; its contribution to the fusion only
    demotes correct items BM25 already ranked well (MRR 0.3876 -> 0.4328).
    This is 20x the +-0.0005 noise floor, so it is not a sampling artifact.

    **Do not act on this without reading the confound.** The local simulator
    builds customer messages out of the target's own catalog fields:
    **359 of 400 turn-1 hard constraints (89.7%) are verbatim substrings of
    the target product's own text**, and the 41 that aren't are only
    non-verbatim because `intent_card()` prefixes them with `"color: "`. So
    effectively every local query is an exact-match query -- the best case
    for BM25 and the worst case for the paraphrase robustness the dense leg
    exists to provide (#4). If the hidden grader phrases customers the same
    way, dropping dense is worth +0.015; if it paraphrases at all, dropping
    it removes the only defense. The downside is asymmetric, so dense stays.
    The measured-but-unshipped alternative is recorded here so a later
    session doesn't rediscover it and assume it's free.

    The intermediate option nobody has tried yet: **retune the RRF fusion
    weights** rather than removing the leg. The current 2.0/1.0 and 1.25/1.5
    were tuned when nobody had checked whether dense contributes at all;
    pushing both tracks further BM25-ward should capture most of the +0.0143
    while keeping paraphrase insurance.

15. **The offline degrade is silent, and was never verified until now.**
    With a cold HF cache and no network, `load_embedding_model()` raises
    `OSError: We couldn't connect to 'https://huggingface.co'`, and
    `Agent.__init__`'s degrade-don't-crash branches take dense retrieval,
    *both* embedding classifiers and the masked LM dark at once -- while the
    agent still starts and still returns 10 recommendations. `model/` (385MB)
    and `data/dense_index/` (74MB) are both gitignored, and the bge-small
    weight blob is 128MB, over GitHub's 100MB per-file limit, so they cannot
    simply be committed. `docs/submission_rules.md` warns network access may
    be disabled for official scoring. Two guards now ship:
    `scripts/fetch_assets.py` (the only networked step in the project) and
    `scripts/preflight.py --strict`, which loads the agent under
    `HF_HUB_OFFLINE=1` and exits non-zero if any required component is dark.
    **Run preflight before any scored run.**

16. **Fusion-weight sweep: the buying curve is strictly monotone, and there is
    a usable middle at ratio 0.25.** #14's open question was whether retuning
    the RRF weights could capture the dense-ablation gain without dropping the
    leg. Swept with `scripts/sweep_fusion_weights.py` (200 samples per point;
    only the dense:bm25 *ratio* matters, since weighted RRF is scale-invariant
    per leg, so bm25 is pinned to 1.0 and dense is the ratio):

    ```
    buying track          HitRate     MRR     MTTC   Technical   vs shipped
      ratio 0.00           0.7500  0.4372    5.020     0.6258      +0.0188
      ratio 0.25           0.7500  0.4096    4.985     0.6182      +0.0112
      ratio 0.50           0.7450  0.3876    5.090     0.6070   <- shipped
      ratio 0.75           0.7400  0.3727    5.125     0.5993      -0.0077
      ratio 1.00           0.7200  0.3615    5.225     0.5839      -0.0230
      ratio 1.25           0.6900  0.3494    5.505     0.5597      -0.0473
      ratio 1.50           0.6650  0.3388    5.720     0.5397      -0.0672
    ```

    The harness reproduces the shipped 0.6070 exactly at ratio 0.50, so the
    other points are comparable.

    Three findings. **There is no interior optimum** — the curve decreases
    monotonically across the whole swept range and the decline accelerates, so
    the hoped-for flat region near the shipped value does not exist; 0.50 is
    already well down the slope. **HitRate falls too**, 0.750 -> 0.665, not just
    MRR: at high dense weight the dense leg does not merely reorder, it
    displaces correct BM25 results out of the top 10 entirely. That is a
    stronger claim than #14's ablation made. **But halving the weight to 0.25
    captures +0.0112 of the +0.0188 available from full removal** — roughly 60%
    of the gain while keeping a real dense leg as paraphrase insurance. That is
    the compromise #14 predicted; it just sits lower than anyone guessed.

    Not yet decided whether to ship 0.25. The #14 confound still applies in
    full: the local set is a near-pure exact-match benchmark (89.7% of hard
    constraints are verbatim substrings of the target's own text), so it
    systematically under-prices the dense leg.

    **Browsing track: measured flat. Do not tune it.**

    ```
    browsing track        Technical   overall hits   browsing-scenario hits
      ratio 0.00             0.6036        148/200            66/80
      ratio 0.25             0.6011        147/200            65/80
      ratio 0.50             0.6054        149/200            65/80
      ratio 0.75             0.6063        149/200            65/80
      ratio 1.00             0.6110        150/200            65/80
      ratio 1.25             0.6068        149/200            65/80
      ratio 1.50             0.6004        147/200            63/80
      (ships at 1.20 -> 0.6070)
    ```

    The apparent +0.0040 peak at ratio 1.00 is **one session** — 150/200 against
    149/200 — and the browsing-scenario column moves by at most a single session
    across the entire sweep. There is no optimum here; the measurement cannot
    distinguish any setting from any other in 0.0–1.5. Leave browsing at 1.20
    and do not re-run this.

    **Two methodological notes, both of which cost a wrong conclusion first.**
    A +0.0040 delta on 200 samples *looks* like it clears a ±0.0025 noise
    estimate and does not: convert rates to absolute session counts before
    believing any small delta, because 0.7500 vs 0.7450 is one session. And the
    two tracks are **not separable** — varying the *browsing* weights visibly
    moved *buying-scenario* metrics (0.675 -> 0.700), because `scenario_type` is
    the sample's ground-truth label while the weights key off the *classifier's*
    per-turn label, and the classifier deliberately drifts buying-ward as
    attributes accumulate (`classifier.py:14`). Per-track sweeps therefore
    measure overlapping, not disjoint, populations.

    **Sharper read of where the buying gain comes from.** Buying-scenario hit
    rate is 0.688 at both ratio 0.00 and the shipped 0.50 — removing dense finds
    *no additional targets*. The whole gain is MRR (0.4372 vs 0.3876) plus
    intent_override hit rate (0.833 vs 0.800). That is #14's mechanism confirmed
    at finer grain: the dense leg does not add recall on this benchmark, it
    demotes items BM25 had already ranked well.

17. **Query rewriting on intent override: measured −0.0580, reverted.** The
    gap was real — `Agent.respond` clears `disclosed`/`profile_hint`/
    `asked_attributes` on a detected pivot but never touches `first_message`
    or `recent_messages`, so `_build_query`'s output is byte-for-byte
    identical before and after the clear and the discarded preference keeps
    feeding BM25. Spec § 4.2.II asks for "slot erasure *and rewriting*"; only
    erasure existed. The obvious fix — restart the query from the pivot
    message — is catastrophic:

    ```
                            HitRate     MRR     MTTC   Technical
    ratio 0.25 only          0.7500  0.4096    4.985     0.6182
    + override rewrite       0.6800  0.3668    5.490     0.5602   -0.0580
      intent_override subset:  hit 0.800 -> 0.333,  MTTC 5.90 -> 9.27
    ```

    **Root cause, found by printing a sample instead of reading the
    template's shape.** The category is stated *once*, in turn 1, and never
    restated; the pivot message carries only the changed attribute:

    ```
    TURN 1 : "I'm looking for ['Clothing, Shoes & Jewelry', 'Men',
              'Accessories', 'Belts']. Buckle closure"
    PIVOT  : "Actually, ignore my earlier preference. What I need is: leather"
    ```

    Dropping `first_message` therefore throws away "Belts" and searches 50k
    products for "leather". **This refutes a claim an earlier revision of
    NEXT_STEPS §5 stated as established** — that the local override template
    "carries the complete new constraint, so discarding the history loses
    nothing locally by construction." It carries the changed *attribute*, not
    the request.

    Any selective-history scheme here must preserve the original framing (the
    category) and drop only the preference clause. The aggressive version is
    not merely suboptimal, it is structurally wrong for this task shape. Not
    retried; the surgical version requires knowing which clause of turn 1 is
    the preference, which on this simulator means parsing its template.

18. **Turn-annealed slate diversity: measured null, not shipped.** The idea:
    `DIVERSIFY_LAMBDA` is fixed at 0.5 and conditioned on intent but not on
    the turn, which leaves the 10-turn cap unexploited — a diverse slate is
    nearly free early (a miss costs one turn and buys information) and pure
    downside late (no turn left to recover in). So anneal lambda upward as the
    budget depletes. Two schedules, both against the shipped 0.6182:

    ```
                            HitRate     MRR     MTTC   Technical
    fixed 0.5 (shipped)      0.7500  0.4096    4.985     0.6182
    anneal 0.35 -> 0.90      0.7450  0.4052    4.980     0.6144   -0.0038
    anneal 0.50 -> 0.90      0.7450  0.4087    4.995     0.6152   -0.0030
    ```

    Both are one session below on HitRate (150 -> 149) and MTTC does not move,
    which was the entire point. The 0.50 -> 0.90 run is the informative one:
    its **browsing metrics are byte-identical** to the fixed-lambda run (hit
    0.825, MRR 0.470, MTTC 4.29), so the schedule changed nothing in the track
    it governs, and the whole delta is a single `intent_override` session that
    the classifier routed browsing-ward. No signal in either direction.

    Worth noting *why* the first schedule was worse: `lambda_early = 0.35` made
    turn 1 more diverse than the previous fixed 0.5, and at turn 1 the query is
    the raw customer message whose top hits are usually already right, so
    hedging displaced good candidates. If anyone revisits this, the lesson is
    that early-turn diversity is not free on this benchmark — but the honest
    conclusion is that there is nothing here to revisit.

19. **Query hygiene on non-answers: measured -0.0410, reverted. Contentless
    customer text is still retrieval signal on this benchmark.** The gap was
    real and is now quantified: the agent never observed whether its question
    was answered, and `respond()` appended every reply to `recent_messages`,
    which `_build_query` joins into the BM25 and dense query. Derived from the
    evaluator's own reply policy, a session has a mean of **2.09** distinct
    answerable attribute buckets left after turn 1 against an MTTC of ~5, so
    roughly three asks in five come back empty
    (`scripts/eval_dialogue_efficiency.py`).

    `EmbeddingNonAnswerDetector` (classifier.py) detects those replies, reusing
    the trimmed-prototype rule from #7. Offline quality:

    ```
    rule                  sim recall  sim FPR   probe recall  probe FPR
    lexical floor              1.000    0.000          0.100      0.000
    trimmed-prototype          1.000    0.168          1.000      0.000
    ```

    The lexical floor's perfect simulator score is #12/#13's trap again -- its
    regex contains "don't have", which matches the template literally, and
    probe recall of 0.100 confirms it learned nothing transferable.

    **Full 200-sample A/B, both legs on one HEAD:**

    ```
                          HitRate     MRR     MTTC   Technical
    identity               0.7450  0.4089   4.9900     0.6154
    + skip non-answers     0.6850  0.3970   5.3600     0.5744   -0.0410
    ```

    HitRate falls 149 -> 137 sessions. **The prediction that was wrong, and
    why it matters:** all 69 of the detector's false positives are the
    simulator's near-contentless constraint template ("For that, what matters
    is: Imported; Pull On closure."), and this was written up mid-session as
    evidence that the FPR column was measuring the wrong thing -- that
    `Imported` is catalog boilerplate with no retrieval value, so dropping it
    was plausibly a gain. The A/B says the opposite, for a reason already in
    this file: per #14, **89.7% of customer text is a verbatim substring of the
    target's own catalog record**, so `Pull On closure` is not boilerplate here,
    it is an exact-match key to the target. Deleting it deletes exactly what
    BM25 wins with.

    Generalize past the experiment: **no form of query pruning is safe on this
    benchmark**, and improving the classifier cannot fix it, because the text
    being pruned is contentless in meaning and load-bearing in retrieval. Same
    shape as #17 -- an intervention that is semantically sensible and
    structurally wrong for this task. The non-answer *observation* is kept and
    feeds `SessionBelief`; only the query surgery is reverted.

## Blockers / mistakes already made (so they aren't repeated)

- **A feature flag whose zero value is not the identity silently corrupts
  every A/B run against it.** The Pillar III re-orchestration branch in
  `routing_params()` was gated on `belief.exhausted` and set
  `diversify=False` *inside* that branch, while its magnitude knob
  (`EXHAUSTED_BM25_BONUS`) was 0.0. So "all switches off" still disabled the
  browsing MMR re-rank, and the control leg scored 0.6154 instead of the
  shipped 0.6182 -- caught only because verification step 8 (confirm identity
  reproduces the shipped number *before* trusting any swept point) was run.
  Every stage measured against that control would have been off by an unknown
  amount. The code comment at the time asserted "the label-only path is
  unchanged", which was simply false. Lesson: an identity setting is a claim to
  verify with a run, not to assert in a comment -- and check that *every*
  effect in a gated branch is gated, not just the one with a numeric knob.
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
