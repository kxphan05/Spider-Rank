# Intent detection: industry practice, and a plan for this project

Research note + implementation plan. Written August 2026.

---

## 1. What production systems actually do

Four patterns recur across the current literature and vendor practice.

**A cheap classifier first, an LLM only on uncertainty.** This is the dominant
production shape. Queries go to the smallest model that can answer them; the
system escalates to an LLM only when the cheap model's confidence falls below a
threshold. Reported savings are large — 45–85% cost reduction at ~95% of
quality, and on a production NER workload 31% cheaper at equal micro-F1 versus
large-model-only inference. The reason is bluntly economic: LLM latency and
serving cost are prohibitive at scale for a per-query classification.

**Thresholds are calibrated, never guessed.** The consistent warning: *"any
system that sets confidence thresholds by intuition will be miscalibrated."*
The prescribed method is to measure what fraction of queries the cheap model
gets right at each confidence level on your own workload, then set the
threshold from an acceptable error rate.

**Few-shot beats hand-built rules, and needs less data than people expect.**
SetFit — contrastive fine-tuning of a sentence transformer plus a lightweight
classification head, no prompts — reaches competitive accuracy with orders of
magnitude fewer parameters than the alternatives. The measured floor for
embedding/similarity methods is *"5–10 training utterances per intent for
getting accuracy above 70%."* LLM-generated data augmentation is listed as one
of the four standard approaches to getting those utterances.

**Sentence encoders beat general-purpose embeddings here.** SentenceBERT-family
models outperform large general embeddings on intent tasks, because intents are
short utterances and those models are trained for short-context similarity.

**Multi-turn is not intent classification.** The task-oriented dialogue pipeline
is NLU (intent **+ slot filling**) → dialogue state tracking → policy → response
generation. Intent detection is one component of the first stage. The multi-turn
literature puts its weight on slot filling, state tracking across turns, and
digression/correction handling — letting users supply information as they think
of it, and correct it later.

---

## 2. Where this project actually sits

Mapping our components onto that pipeline:

| Pipeline stage | This project | State |
|---|---|---|
| NLU — intent | `EmbeddingIntentClassifier`, nearest-centroid, buying/browsing | **0.988** on turn-1 |
| NLU — slot filling | `extract_disclosed_value` — material, colour, budget | **3 of 8** attributes |
| DST | `SessionState.disclosed`, wiped on override | minimal but real |
| Digression / correction | `EmbeddingOverrideDetector`, trimmed-prototype k=4 | probe recall 1.000 |
| Policy | entropy-based question selection, answerability-ordered fallback | strongest component |
| NLG | fixed templates | adequate; not scored |

**The intent classifier is not the weak link, and improving it is near-worthless.**
Two measurements already on file say so. It scores 0.988 on turn-1, its only two
errors being sessions whose stated constraint is contentless (`"A key
requirement is: Imported."`). And the thing the label feeds — dual-track routing —
was measured net flat end-to-end (0.600 → 0.601). A perfect label is worth
approximately nothing.

**By industry standard, the real gap is slot filling.** Four of the eight allowed
attributes — `style`, `size`, `use_case`, `feature` — have **no extractor at
all**. They can be asked but never recorded, so the answer cannot narrow the
candidate pool. This is the documented ceiling of the entire question-asking
design, and it is exactly the stage the multi-turn literature weights most.

**One constraint reshapes every borrowed pattern:** `docs/submission_rules.md`
warns that network access may be disabled for scoring, and `preflight.py` now
enforces offline operation. **There is no runtime LLM to escalate to.** Any
cascade must be distilled offline into local weights.

---

## 3. Plan

Ordered by value-per-effort. Each stage names what to measure, because a stage
that cannot be measured on the public set should not be built.

### Stage 0 — Use the confidence we already compute *(hours, no new assets)*

The single clearest gap against industry practice, and it is nearly free.

`EmbeddingIntentClassifier.classify()` returns a `signal.score`. **That score is
referenced in exactly one place in the codebase — a `logger.debug` line
(`agent.py:310`).** Every functional consumer branches on `signal.label` alone.
We compute a confidence and throw it away, which is precisely the input the
standard cascade uses to decide whether to trust the cheap model.

The tract:

1. Plot accuracy against `|score|` on turn-1 messages — the calibration curve
   the literature insists on. `scripts/eval_intent.py` already harvests the pool.
2. Where the margin is thin, **abstain instead of committing**: interpolate the
   buying and browsing fusion weights rather than snapping to one track. Routing
   is already resolved in one place (`routing_params()`), so this is a small,
   contained change.
3. Measure end-to-end. Expect little — routing is worth ~0.001 — but this is the
   correct shape and it costs almost nothing.

Precedent: the masked-LM entropy gate (`MAX_CONFIDENT_ENTROPY = 0.60`) was
calibrated exactly this way, against measured accuracy-by-entropy buckets. We
have done this properly once already, just not here.

### Stage 1 — Diverse utterances + a SetFit-style head *(1–2 days)*

This is the textbook fix for a previously-measured failure: a trained head hit
0.984 in-distribution and 0.521 on a held-out surface form. The diagnosis was
**not** bad labels — it was that the simulator has only
two turn-1 templates, so the head memorised `"still exploring"`. The control
proved it: training on the 20 hand-written prototypes *alone* reached OOD 1.000,
while adding 640 simulator turns dragged it to 0.812.

Industry practice says the remedy is LLM-generated augmentation, and that 5–10
utterances per intent suffice.

1. Generate a few hundred diverse buying/browsing utterances offline with
   `claude-haiku-4-5` — varied register, length, category, ellipsis, typos,
   multi-intent — and commit them as a static JSONL. Same offline-asset pattern
   as the #8 catalog plan; no network at scoring time.
2. Train SetFit-style: contrastive pairs, then a logistic head.
3. **Hold out both the simulator templates and the hand-written probes.** Report
   the held-out-surface-form number, never the in-distribution CV — that number
   is the trap #12 fell into.

**Scope ruling needed before building.** `TODO.md` § 4.3 bars *"training or
full-parameter fine-tuning of base foundational LLMs."* A head over frozen
embeddings is unambiguously fine (#12 established that). SetFit's contrastive
step tunes the 33M-parameter *sentence encoder*, which is arguably not a "base
foundational LLM" — but it is a judgment call, and the safe fallback is
head-only on frozen embeddings with the new diverse data.

**Guard against circularity.** Training on Claude-generated utterances and
testing on Claude-generated probes proves nothing. The held-out set must come
from a different generator or be hand-written. We already have this bug in
miniature: `eval_intent.py`'s OOD probe pool is partly circular, and the
lexical rule's 1.000 on turn-1 is pure template memorisation.

### Stage 2 — Close the slot-filling gap *(the actual bottleneck)*

Where industry practice says the weight belongs, and where ours is thinnest:
half the allowed attributes have no extractor. This is already designed and
costed — a one-time `claude-haiku-4-5` Batch API
pass over the 50k catalog (~$5–12) emitting a closed-vocabulary JSON sidecar,
shipped as a local asset.

Two additions from this research:

- **Extract the customer side with the same vocabulary.** The recorded caveat
  on #8 is that it fixes only the catalog half. The dialogue literature treats
  slot filling as one component with one ontology; two independently-built
  extractors that disagree is the actual defect.
- **Support correction, not just accumulation.** Users supply information as
  they think of it and revise it later. We handle the total-wipe case (override
  detection) and the append case, but not per-slot revision — "actually, make it
  blue" should overwrite the colour slot without discarding the material slot.
  This is a genuine, cheap capability gap.

### Not in the plan, deliberately

- **Runtime LLM escalation.** No network at scoring time. Distil offline instead.
- **More work on the buying/browsing classifier itself.** 0.988 already, feeding
  a mechanism measured flat. Stage 0 is the last thing worth doing to it.
- **Fine-tuning bge-small end-to-end.** Out of scope per § 4.3, and #12's control
  shows the zero-shot prototype design already generalises better than
  simulator-trained supervision.

---

## 4. Honest expected value

Stage 0 is correct-by-standard and nearly free, but the mechanism it improves is
worth ~0.001, so expect no score movement. Stage 1 buys robustness against a
hidden set phrased unlike the local simulator — real insurance, unmeasurable
locally by construction. **Stage 2 is the only one with a plausible path to a
material score gain**, because it lifts a documented structural ceiling rather
than tuning a component that is already at 0.988.

If only one stage gets built, build Stage 2.

---

## Sources

- [Intent Detection in the Age of LLMs](https://arxiv.org/pdf/2410.01627)
- [Exploring Zero and Few-shot Techniques for Intent Classification](https://arxiv.org/pdf/2305.07157)
- [SetFit / ModernBERT few-shot text classification results](https://moshewasserblat.medium.com/new-results-on-setfit-modernbert-for-text-classification-with-few-shot-training-53c154df7c0e)
- [Pre-training Tasks for User Intent Detection and Embedding Retrieval in E-commerce Search (CIKM)](https://dl.acm.org/doi/10.1145/3511808.3557670)
- [UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing](https://arxiv.org/pdf/2605.18796)
- [Is Escalation Worth It? A Decision-Theoretic Characterization of LLM Cascades](https://arxiv.org/pdf/2605.06350)
- [LLM Routing and Model Cascades — cost/quality tradeoffs](https://tianpan.co/blog/2025-11-03-llm-routing-model-cascades)
- [A Survey on Recent Advances in LLM-Based Multi-turn Dialogue Systems](https://arxiv.org/html/2402.18013v1)
- [Conversational Language Understanding: multi-turn entity slot filling (Microsoft)](https://learn.microsoft.com/en-us/azure/ai-services/language-service/conversational-language-understanding/concepts/multi-turn-conversations)
