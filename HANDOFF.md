# Handoff — 2026-08-29, ~14:15

Written for a fresh session picking this up on the same machine.

## Where the score is

**TechnicalScore 0.7020** (HitRate 0.8550, 171/200; MRR 0.4622; MTTC 4.205),
committed at `3ef2028`. Measured with identity reproducing 0.6184 first, so it
is believable. `CLAUDE.md` "Progress" and entry #23 now say so.

`results.json` in the repo is **stale** (Aug 28, 0.615212). Ignore it.

## What is running right now

```
scripts/ab_phrase_query.py   PID 126962   log: logs/ab_phrase_query.log
```

Five legs, in order: `identity`, `filter only`, `clause+edge only`,
`filter + clause+edge`, `budget 96 (control)`. It had printed only its header
when this session ended, so **no leg has landed yet**. Roughly five full evals.

It was restarted at 14:07 with `TECHJAM_THREADS=6` after the other two runs
were stopped — the first attempt ran 1h20 pinned to 3 threads on an 8-core box
it no longer had to share, and had not finished leg 1. That first log is kept
as `logs/ab_phrase_query.3thread.log`; it contains nothing but a header.

Read it with:

```
grep -av "it/s" logs/ab_phrase_query.log
```

**The identity leg must print 0.7020 before any other leg is believed.** If it
does not, stop and find out why — every other number in that table is then
meaningless. This rule has already caught one real bug in this project (see the
first bullet of CLAUDE.md "Blockers").

If it died, rerun it alone: `TECHJAM_THREADS=6 uv run python3 scripts/ab_phrase_query.py`.
Do not start a second full eval beside it.

## What that A/B decides

Both changes are committed but **flagged off**, so HEAD behaves exactly as the
0.7020 measurement. Full rationale in CLAUDE.md #25.

| flag | file | default |
|---|---|---|
| `PHRASE_CLAUSE_SPANS` | `starter/retrieval.py` | `False` |
| `PHRASE_QUERY_SKIP_NON_ANSWERS` | `starter/agent.py` | `False` |

Ship a flag only if its leg beats identity **and** beats the `budget 96`
control. The control exists because raising the budget is the dumb fix; if it
matches the clever fixes, the clever fixes have not earned their complexity.
Convert rates to session counts before believing anything — 0.8550 vs 0.8500 is
one session.

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
