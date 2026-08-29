# Handoff — 2026-08-29, ~21:30

Written for a fresh session picking this up on the same machine.

## HEAD is measured. #26 is flat.

The previous handoff's lead item — "no number in this repo describes the code
in it" — is resolved. Full public set at `5889828`:

```
HitRate@10: 0.850  (170/200)   MRR: 0.4567   MTTC: 4.210
Efficiency: 0.6790             TechnicalScore: 0.6978
```

`results.json` is this run, and is no longer stale.

Against the 0.7020 measured at `3ef2028`, that is **-0.0042, which is one
session** (171 hits to 170). Every feature flag was re-checked at its identity
value first (`PHRASE_CLAUSE_SPANS = False`, `PHRASE_QUERY_SKIP_NON_ANSWERS =
False`, `MAX_PHRASE_QUERIES = 24`, `RERANK_WEIGHT = 0.0`, `EXCLUDE_SHOWN =
True`, `PHRASE_WEIGHT = 2.0`), and #26 is the only `starter/` diff in the
range, so this is a clean before/after for #26 and nothing else.

**Treat #26 as flat, not as a regression.** The specific worry going in was
that `_REQUEST_STOPWORDS` removes tokens from every query, the same shape that
cost -0.0410 in #19. It did not repeat: one session, not twelve. The
closed-list discipline held. That does *not* reopen query pruning generally —
the corpus-driven version of the same list is still measured unsafe, at the end
of CLAUDE.md #26.

Per-scenario, for anyone hunting the remaining 30 misses:

```
scenario           n    hit     MRR    MTTC
browsing          80  0.925  0.5235   3.613
intent_override   30  0.833  0.4084   5.267
buying            80  0.813  0.4204   4.213
boundary          10  0.600  0.3569   5.800
```

Buying and boundary are where the misses concentrate, which is what #24 and #27
already said. `preflight.py --strict` passed on this HEAD before the run: all
four embedding components come up live offline, so this is the full pipeline
and not a silent degrade.

## `user_profile` is closed, with a judge-facing writeup

Asked afresh this session, tested afresh, still no — but on two *new* tests
that #5's original evidence did not cover. `docs/user_profile_decision.md` is
the defensible version, and it explains every statistic it uses.

The one worth knowing: `average_prior_rating` -> target `average_rating` gives
r = 0.1824 at permutation p = 0.0094, which **passes significance and is still
rejected**. It explains 3.3% of the variance, it runs backwards between its two
largest cells (prior 1.0 -> 4.393 vs prior 5.0 -> 4.413), and dropping the one
9-sample cell collapses it to r = 0.0929, inside the null band. About a dozen
tests were run across three fields, so one pass at 0.05 is what chance
produces. `preference_tags` -> target category is flatly null: every conditional
sits on the 0.590 marginal, and the top four tags appear on 50-82% of samples,
so they cannot separate customers even in principle.

**The live direction that came out of it is not the profile.** `rating_number`
is the only 100%-covered catalog field, and targets come overwhelmingly from
the popular tail — catalog median 12 reviews, target median 7,078; the top 5.9%
of the catalog by review count holds 83% of all targets, a 14x lift.
`scripts/sweep_prior_leg.py` was built for exactly this and **has never been
run** — no result in CLAUDE.md, no log in `logs/`. Its `popularity` variant was
designed as the *control* for the profile idea, and the control is the leg with
the effect size.

Two things to hold it to. It is a re-ranker inside an existing pool, so it
cannot touch the elicitation gap #27 identifies as the real remaining limit.
And the concentration is a property of how the samples were drawn, not a fact
about shopping, so it transfers only if the hidden set was drawn the same way —
a stronger scorer-semantics dependency than anything shipped except #23. Run
`--weights 0` first and confirm it reproduces 0.6978.

## The phrase-query A/B finished, and the answer is "no"

```
leg                     HitRate      hits      MRR     MTTC  Technical    delta
identity                 0.8550  171/200   0.4622    4.205     0.7020       --
filter only              0.8700  174/200   0.4826    4.090     0.7180   +0.0159
clause+edge only         0.8800  176/200   0.4946    3.915     0.7301   +0.0280
filter + clause+edge     0.8850  177/200   0.4981    3.900     0.7339   +0.0319
budget 96 (control)      0.8850  177/200   0.4998    3.995     0.7326   +0.0305
```

Identity reproduced 0.7020 exactly, so the legs are comparable, and that also
proves the run was on clean HEAD rather than on #26.

**The control matches the best fix** — 177/200 both, 0.0013 apart, inside one
session of resolution. By the rule stated when the control was designed, the
fixes did not beat it. They win on *cost* instead: the same score at the
original budget of 24 phrase lookups instead of 96. Only `clause+edge` is
established on its own (+0.0280, five sessions); the filter adds one session on
top of it, which this benchmark cannot resolve. Full write-up in CLAUDE.md #25.

**Both flags are still off, deliberately.** `PHRASE_CLAUSE_SPANS` reads
`STOPWORDS`, and #26 adds ~24 words to that set, so this table was measured
against a `STOPWORDS` that no longer exists. Re-measure the two together, with
identity re-established on that HEAD, before flipping anything.

## What the remaining misses actually are

CLAUDE.md #27 replaces the picture #24 painted. New diagnostic,
`scripts/eval_ceiling.py`, hands the agent every constraint the customer could
ever state in one turn-1 message:

```
oracle (all constraints, one turn)   173/200
actual (10-turn dialogue)            171/200

of the 27 oracle misses, where is the target?
  in some leg's top 10   12        11-50    10
  51-500                  5        not found at 2000   0
```

**No target is unreachable.** Some leg puts every one of the 200 inside the top
192. #24's "no retrieval leg finds the target at all" describes the starved
*live* query, not the catalog or the legs — the customer only discloses a
constraint when the agent asks the attribute it buckets into. Read #24's advice
as being about elicitation, not reach; it now carries a pointer saying so.

Two traps recorded in #27 that cost real time today:

- **The oracle number is not a ceiling.** It returns one slate of ten where a
  live session returns up to ten. The two conditions differ on 34 samples, 17
  in each direction, so neither dominates.
- **The attribute-boost fix it suggested is falsified.** `AttributeIndex`
  really does store one material per product while the evaluator picks
  differently from the same text, and 9.5% of targets take a -1 for a material
  printed in their own record — `public_0111`'s rank-5-to-80 anomaly included.
  But replaying the 7 affected misses at `DISCLOSED_MISMATCH_PENALTY = 0.0`
  recovers **none** of them. The "all three legs rank it top-3" observation came
  from the oracle query; the live query is one material word and the target is
  never in the pool for the boost to sink. If you build the set-membership
  version anyway, #27 has the design constraints — the short one is: put it
  behind a `contains()` used only by the boost, leave `value_for`/`values_for`
  alone so the entropy question-picker stays bit-identical, and don't bundle a
  weight retune with it.

The direction the evidence supports is **elicitation**: 12 of the 27 oracle
misses have the target in some leg's top 10, so the information is reachable
once it arrives. What is missing in a live session is the constraints
themselves.

## Deliverables still outstanding

Ordered. Steps 1 and 2 are **done on this HEAD**; the rest are not.

1. ~~`uv run python3 -m evaluator.local_evaluator`~~ — **done**, 0.6978, top of
   this file. `results.json` holds it.
2. ~~`uv run python3 scripts/preflight.py --strict`~~ — **done**, passed on this
   HEAD: dense retrieval, both classifiers and the non-answer detector all come
   up live with the network disabled. The one warning is the expected
   `RERANK_WEIGHT = 0.0` cross-encoder note.
3. `uv run python3 scripts/build_submission.py --verify` — itself a full eval,
   so do not run it beside step 1.
4. `uv run python3 scripts/measure_latency.py --limit 20` — the numbers in
   `docs/team_report.md` §4 predate the cross-encoder and #26.
5. Demo re-capture (`scripts/build_demo.py --capture`).

## Documentation state

Refreshed today and current: `docs/team_report.md` (headline, per-scenario
table, ablations, limitations), `REPRODUCE.md`, `NEXT_STEPS.md`,
`docs/question_policy_plan.md`, both sweep docstrings, CLAUDE.md #24–#27.

Two corrections went into the report worth knowing about, because they were
wrong rather than merely stale. It described **cross-encoder re-ranking as part
of the pipeline** when `RERANK_WEIGHT = 0.0` and the sweep never ran — the
scored pipeline has no learned re-ranker, which its own limitations section had
said all along. And it **omitted shown-item exclusion entirely**, the largest
win in the project at +0.0837.

Still to do: `scripts/build_deck.py` has the right score but no slide for #26
or #27, and §4's latency table waits on step 4 above.

## Box rules, learned the hard way

- **Never more than two full evals at once.** Three drove load past 17 on 8
  cores and got `eval_failures.py` killed mid-run.
- When chaining `long_command; tail log`, the exit code you see belongs to
  `tail`. A run that died looks like it succeeded.
- **Another Claude session was active in this repo today.** Check
  `git status` and `ps` before assuming the working tree is yours; #26 appeared
  in it mid-session with no warning.
- Identity must reproduce the shipped number before any swept point is
  believed. This has already caught one real bug (first bullet of CLAUDE.md
  "Blockers") and it caught the clean-HEAD question today.
- Convert rates to absolute session counts before believing a small delta.
  0.8550 vs 0.8500 is one session.
