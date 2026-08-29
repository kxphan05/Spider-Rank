# Handoff — 2026-08-29, ~17:15

Written for a fresh session picking this up on the same machine. Everything is
pushed; `git status` is clean and `main` is level with `origin/main` at
`5403717`.

## Read this first: HEAD has no measured score

**0.7020 is not HEAD's number.** It was measured at `3ef2028`
(HitRate 0.8550, 171/200; MRR 0.4622; MTTC 4.205), with identity reproducing
0.6184 first, so that figure is sound *for that commit*.

`56f9df0` landed afterwards. It is CLAUDE.md #26 — a root-category IDF fix in
`BM25Index._build`, request-verb stopwords, en-GB spelling normalization,
purchase verbs in the intent classifier, and a narrow category-naming override
rewrite. It was found by using the demo and verified in the REPL, but it has
**never been scored on the 200-sample set**. It changes how the catalog is
indexed and how every query is tokenized, which are the two things BM25 is made
of, on a benchmark BM25 carries.

**So the first job is a plain full eval of HEAD:**

```
uv run python3 -m evaluator.local_evaluator
```

Roughly 30 minutes on this box. Until it finishes, no number in this repo
describes the code in it. Watch one thing in particular: `_REQUEST_STOPWORDS`
removes tokens from **every** query, which is the shape that cost -0.041 in
#19. The argument that this list is different — ways of *asking*, never things
to ask about — is a prediction, and #19 is the standing reminder that
predictions of exactly that shape have been wrong here before.

`results.json` in the repo is stale (Aug 28, 0.615212). The eval above
overwrites it, which is the tidiest way to fix it.

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

Ordered. None has been run against current HEAD.

1. `uv run python3 -m evaluator.local_evaluator` — see above; everything else
   is meaningless until this exists.
2. `uv run python3 scripts/preflight.py --strict` — passed at `3ef2028`, not
   re-run since #26. The offline degrade is silent (#15), so this is not
   optional before a scored run.
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
