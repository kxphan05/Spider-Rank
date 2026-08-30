# Handoff — 2026-08-30

For a fresh session picking this up on the same machine. `CLAUDE.md` is the
standing record and `NEXT_STEPS.md` is the queue; this file holds only what is
true *right now* and would not be obvious from either.

## State

HEAD is `1fb1913`, measured, and `results.json` is that run:

```
HitRate@10 0.945 (189/200)   MRR 0.5534   MTTC 3.250   TechnicalScore 0.7935
```

`preflight.py --strict` passed on this HEAD, including the newly-live
minilm cross-encoder reranker (`RERANK_WEIGHT` was `0.0`, is now `3`) — this is
the full pipeline, not a silent degrade. Against `a9d3999` (0.7441) that is
+0.0494 over three commits measured only as a bundle. See CLAUDE.md #31: **the
individual contribution of the gated-question logic, the boundary/popularity
leg, and the reranker is unknown** — don't cite any one of them as the cause.

Per-scenario:

```
browsing         80  1.000  0.6230  2.875
intent_override  30  0.933  0.7136  4.633
buying           80  0.9125 0.4534  2.938
boundary         10  0.800  0.3169  4.600
```

Boundary and buying are still where the misses concentrate, though boundary
moved from 6/10 to 8/10 hit. CLAUDE.md's "Where the remaining misses are"
section has the taxonomy predating this run; it has not been re-derived
against current HEAD.

## Deliverables outstanding

1. ~~`uv run python3 -m evaluator.local_evaluator`~~ — done, 0.7935
   (`results.json` on disk).
2. ~~`uv run python3 scripts/preflight.py --strict`~~ — done, passed, reranker
   included.
3. `uv run python3 scripts/build_submission.py --verify` — bundle assembled at
   `dist/submission/`; a `--limit 20` verify was run during cleanup and took
   several minutes (the reranker adds real per-turn cost now). The full-set
   `--verify` has not been run since the reranker went live — it is itself a
   full eval, so do not run it beside another.
4. `uv run python3 scripts/measure_latency.py --limit 20` — the numbers in
   `docs/team_report.md` §4 predate the reranker and are flagged stale there.
   Needs a re-run; expect `respond()` latency to rise materially.
5. Isolate `POPULARITY_WEIGHT` and `RERANK_WEIGHT` one at a time against
   `a9d3999` — `scripts/sweep_prior_leg.py --weights 0` and
   `scripts/sweep_rerank.py`, neither run against current HEAD. This is
   NEXT_STEPS.md's #1 item now.
6. Demo re-capture (`scripts/build_demo.py --capture`) — stale since before
   the three unrecorded commits.
7. `scripts/build_deck.py` — check it reflects the current score and #31;
   `dist/techjam_track4.pptx` predates this handoff.
8. `manual_qa_notes.md` (2026-08-29) lists live-demo bugs found before the
   gated-attribute-asking and boundary/popularity changes. Worth re-checking
   whether any are still reproducible before treating them as open.

## In flight

Nothing. The `--limit 20` submission verify launched during this cleanup
session completed; see `dist/submission/` for the assembled bundle.

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
- The reranker being live means diagnostics that call `respond()` in a loop
  (sweeps, `eval_failures.py`, `ab_*` scripts) will run noticeably slower than
  the numbers in their own docstrings suggest. Budget accordingly before
  kicking off an "overnight" run.
