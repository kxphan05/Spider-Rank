# Team Report — SpiderRank

Submitted to the TechJam Conversational E-Commerce Search Challenge, Track 4.
Required by `docs/submission_rules.md`: method, model choice, limitations,
and a disclosure of latency, token usage, and estimated model cost. Full
version with ablations: `docs/team_report.md`.

## Result

`TechnicalScore 0.8228` (HitRate@10 0.965, MRR 0.5995, MTTC 2.98) on the
200-sample public set, vs. 0.125 HitRate for the shipped BM25 starter.
Reproduce: `uv run python3 -m evaluator.local_evaluator`.

## Method

Hybrid retrieval (BM25 + phrase-match + dense cosine, fused by
intent-conditioned RRF) → cross-encoder re-ranking of the pool head →
disclosed-attribute re-sorting (material, color, budget) → proactive
question selection → shown-item exclusion (never re-recommend a shown
product — the single largest gain, +0.0837) → text-rule intent-override
detection. All local, in-memory, no network at inference time.

Contentless clarifying replies ("no preference", "up to you") are excluded
from the retrieval query by a one-class embedding threshold against a small,
closed set of decline-phrasing prototypes, rather than a nearest-class contest
against an "informative" prototype set that would otherwise need to be grown
indefinitely to cover arbitrary catalog vocabulary — see
`docs/team_report.md` § 6 for the measured comparison.

## Model choice

| component | model | size |
|---|---|---:|
| dense retrieval + classifiers | `BAAI/bge-small-en-v1.5` | 129 MB |
| reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 87 MB |
| attribute inference (optional, off by default in fetch_assets.py) | `distilbert-base-uncased` | 257 MB |

No LLM API called at inference time. All models local, frozen, CPU-only;
nothing fine-tuned.

## Disclosure: latency, tokens, cost

- **Tokens: 0.** `prompt_tokens = completion_tokens = 0` — no hosted model is
  ever called.
- **Cost: $0.00.** No API spend at inference or build time.
- **Latency** (AMD Ryzen 5 PRO 4650U, CPU-only, reranker + masked LM both
  live): `respond()` mean ~1.0 s, p95 ~1.3 s, ~3.4 s per full session. Cold
  start ~16-17 s, one-time per process. Peak RSS under 1.5 GB.

Requires no network at scoring time once setup has run
(`scripts/preflight.py --strict` verifies this offline).

## Limitations

- Only material, color, and budget are extractable; four other askable
  attributes are never filtered on.
- Long-term personalization is built but unused — every use of it regressed
  the score.
- Boundary scenarios are weakest (n=10, coarse estimate).

## Team

**Phan Kang Xun** — architecture, retrieval design, all experiments and
measurement, this report. 

**Lloyd Wang** — Team Leader, Testing, Quality Assurance, Presentation.

Built
with heavy use of Claude Code for implementation and documentation;
permitted per `docs/submission_rules.md` § Model Policy.
