# Handoff — 2026-08-29, updated ~16:45

Written for a fresh session picking this up on the same machine.

## Where the score is

**TechnicalScore 0.7020** (HitRate 0.8550, 171/200; MRR 0.4622; MTTC 4.205),
committed at `3ef2028`. Measured with identity reproducing 0.6184 first, so it
is believable. `CLAUDE.md` "Progress" and entry #23 now say so.

`results.json` in the repo is **stale** (Aug 28, 0.615212). Ignore it.

## The phrase-query A/B has finished

```
leg                     HitRate      hits      MRR     MTTC  Technical    delta
identity                 0.8550  171/200   0.4622    4.205     0.7020       --
filter only              0.8700  174/200   0.4826    4.090     0.7180   +0.0159
clause+edge only         0.8800  176/200   0.4946    3.915     0.7301   +0.0280
filter + clause+edge     0.8850  177/200   0.4981    3.900     0.7339   +0.0319
budget 96 (control)      0.8850  177/200   0.4998    3.995     0.7326   +0.0305
```

**Identity reproduced 0.7020 exactly**, so the legs are comparable and this
also proves the run was on clean HEAD, not on the uncommitted #26 work that
appeared in the tree after it started.

The verdict is not the one the flags were hoping for, and CLAUDE.md #25 now
carries it in full. Short version: **the control matches the best fix** —
177/200 both, 0.0013 apart, inside one session of resolution — so the fixes did
not beat it on score. They win on *cost* instead: same result at the original
budget of 24 lookups instead of 96. Of the two, only `clause+edge` is
established (+0.0280, five sessions); the filter adds one session on top of it,
which this benchmark cannot resolve.

**Neither flag has been flipped, deliberately.** The edge rule reads
`STOPWORDS`, and the uncommitted #26 work adds ~24 words to that set, so the
two changes interact and +0.0280 was measured against a `STOPWORDS` that no
longer exists once #26 lands. The next measurement should be #26 and
`PHRASE_CLAUSE_SPANS` together, with identity re-established on that HEAD.

## Another session is working in this tree

`starter/agent.py`, `classifier.py`, `retrieval.py` and `text_utils.py` carry
uncommitted changes that are **not** from this handoff — a root-category IDF
fix, request-verb stopwords, en-GB spelling normalization, purchase verbs in
the intent classifier, and a narrow category-naming override rewrite. They are
written up as CLAUDE.md #26 and were verified in the REPL, but **never measured
on the 200-sample set**. `_REQUEST_STOPWORDS` removes tokens from every query,
which is the shape that cost -0.041 in #19. Measure before shipping.

## Do these before any more tuning

They protect the score that already exists, and neither has been run against
current HEAD.

1. `uv run python3 scripts/preflight.py --strict` — loads the agent with the
   network disabled. CLAUDE.md #15: the offline degrade is **silent**, and it
   takes dense retrieval plus both classifiers dark while the agent still
   returns 10 products.
2. `uv run python3 scripts/build_submission.py --verify`.

Then: `measure_latency.py` (report numbers are stale), demo re-capture, and
push — `3ef2028` and this commit are both unpushed.

## Stale documentation

These still quote 0.6182 / 0.6070 / 0.6454. CLAUDE.md is done; the rest are not:

```
docs/team_report.md      (including its per-scenario table)
REPRODUCE.md
NEXT_STEPS.md
docs/question_policy_plan.md
scripts/build_deck.py    (deck needs the new number)
scripts/sweep_phrase_weight.py, scripts/sweep_fusion_weights.py  (docstrings)
```

## Where the remaining score is, and is not

CLAUDE.md #24 has the miss census. The short version: 15 of 27 misses are
`buying`, and in almost all of them **no retrieval leg finds the target at
all** — the turn-1 signal is one very common material word. That is a recall
problem, not a ranking one, so fusion tuning cannot fix it.

Two things not to chase:

- `lost-in-fusion` in the failure log does **not** mean fusion dropped a
  candidate it had. 16 of 17 had no leg ranking the target better than ~198.
- The plan file's next sweeps (BM25 field weights, then attribute boost
  weights) are still the best-value directions, but with ~±0.01 in them.
  Deliverables outrank them now.

One genuine loose end worth ten minutes: `public_0111` has BM25 rank 5 and a
final rank of 80. One session, so it is a curiosity, not a project.
