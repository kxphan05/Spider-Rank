# Handoff — 2026-08-29

For a fresh session picking this up on the same machine. `CLAUDE.md` is the
standing record and `NEXT_STEPS.md` is the queue; this file holds only what is
true *right now* and would not be obvious from either.

## State

HEAD is `5889828`, measured, and `results.json` is that run:

```
HitRate@10 0.850 (170/200)   MRR 0.4567   MTTC 4.210   TechnicalScore 0.6978
```

`preflight.py --strict` passed on this HEAD before the run, so all four
embedding components come up live offline — this is the full pipeline, not a
silent degrade. Against 0.7020 at `3ef2028` that is −0.0042, one session, with
#26 the only `starter/` diff in the range. **Treat #26 as flat.**

Per-scenario, for anyone hunting the remaining 30 misses:

```
browsing         80  0.925  0.5235  3.613
intent_override  30  0.833  0.4084  5.267
buying           80  0.813  0.4204  4.213
boundary         10  0.600  0.3569  5.800
```

Buying and boundary are where the misses concentrate. CLAUDE.md's "Where the
remaining misses are" section has the taxonomy; the short version is that the
limit is **elicitation**, not reach — every one of the 200 targets is findable
given the full constraint set.

## Deliverables outstanding

Ordered. Steps 1 and 2 are done on this HEAD.

1. ~~`uv run python3 -m evaluator.local_evaluator`~~ — done, 0.6978.
2. ~~`uv run python3 scripts/preflight.py --strict`~~ — done, passed. The one
   warning is the expected `RERANK_WEIGHT = 0.0` cross-encoder note.
3. `uv run python3 scripts/build_submission.py --verify` — itself a full eval, so
   do not run it beside another.
4. `uv run python3 scripts/measure_latency.py --limit 20` — the numbers in
   `docs/team_report.md` §4 predate the cross-encoder and #26.
5. Demo re-capture (`scripts/build_demo.py --capture`).
6. `scripts/build_deck.py` has the right score but no slide for #26 or #27.

## In flight

Nothing. `scripts/ab_buying_diversify.py` finished: identity reproduced 0.697796
exactly, treatment scored 0.7008 with **the same 170 hits and an unchanged
crowding table**. Null, written up as #28. `BUYING_DIVERSIFY` stays `False`.
Reports kept in `logs/bd_*.json`.

## Box rules, learned the hard way

- **Never more than two full evals at once.** Three drove load past 17 on 8 cores
  and got a run killed mid-write, looking clean.
- When chaining `long_command; tail log`, the exit code belongs to `tail`.
- Run diagnostics under `uv run`, never bare `python3` — without torch the agent
  degrades silently and a probe will report a live flag as a no-op.
- **Another Claude session has been active in this repo.** Check `git status` and
  `ps` before assuming the working tree is yours.
- Identity must reproduce the shipped number before any swept point is believed.
- Convert rates to absolute session counts before believing a small delta. One
  session is ~0.004 TechnicalScore.
