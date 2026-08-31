# Team Report — SpiderRank

Submitted to the TechJam Conversational E-Commerce Search Challenge, Track 4.
Required by `docs/submission_rules.md`: method, model choice, limitations,
and a disclosure of latency, token usage, and estimated model cost. Full
version with ablations: `docs/team_report.md`.

## Result

`TechnicalScore 0.7935` (HitRate@10 0.945, MRR 0.5534, MTTC 3.250) on the
200-sample public set, vs. 0.125 HitRate for the shipped BM25 starter.
Reproduce: `uv run python3 -m evaluator.local_evaluator`.

## Method

Hybrid retrieval (BM25 + phrase-match + dense cosine, fused by
intent-conditioned RRF) → disclosed-attribute re-sorting (material, color,
budget) → proactive question selection → shown-item exclusion (never
re-recommend a shown product — the single largest gain, +0.0837) → text-rule
intent-override detection. All local, in-memory, no network at inference
time. A cross-encoder reranker is implemented but ships disabled
(`RERANK_WEIGHT = 0.0`) because its sweep wasn't measured.

## Model choice

| component | model | size |
|---|---|---:|
| dense retrieval + classifiers | `BAAI/bge-small-en-v1.5` | 129 MB |
| reranker (built, not enabled) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 87 MB |
| attribute inference (optional, off by default) | `distilbert-base-uncased` | 257 MB |

No LLM API called at inference time. All models local, frozen, CPU-only;
nothing fine-tuned.

## Disclosure: latency, tokens, cost

- **Tokens: 0.** `prompt_tokens = completion_tokens = 0` — no hosted model is
  ever called.
- **Cost: $0.00.** No API spend at inference or build time.
- **Latency** (AMD Ryzen 5 PRO 4650U, CPU-only): `respond()` mean ~360-410 ms,
  p95 ~620-720 ms, ~2.1-2.4 s per full session. Cold start ~16-17 s, one-time
  per process. Peak RSS under 1.3 GB.

Requires no network at scoring time once setup has run
(`scripts/preflight.py --strict` verifies this offline).

## Limitations

- Only material, color, and budget are extractable; four other askable
  attributes are never filtered on.
- Reranking stage is built but not enabled in the scored pipeline.
- Long-term personalization is built but unused — every use of it regressed
  the score.
- Boundary scenarios are weakest (n=10, coarse estimate).

## Team

**Phan Kang Xun** — architecture, retrieval design, all experiments and
measurement, this report. **Lloyd Wang** — registered team member. Built
with heavy use of Claude Code for implementation and documentation;
permitted per `docs/submission_rules.md` § Model Policy.
