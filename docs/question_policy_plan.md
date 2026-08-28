# Value-of-information question selection

A design for replacing the entropy question picker with a single expected-value
objective. Written before implementation; every number cited is already measured
in this repo, and the places where a number would have to be *assumed* are
called out as such.

## 1. What the code does today

`Agent._next_attribute` runs two disjoint mechanisms in sequence:

1. `select_dynamic_attribute` scores `STRUCTURAL_ATTRIBUTES` — only `material`
   and `color` — by normalized entropy over the candidate pool, and returns the
   best if it clears an intent-conditioned `min_entropy` (0.10 buying / 0.30
   browsing).
2. If nothing clears, it walks `FALLBACK_ATTRIBUTE_ORDER`
   (`feature, style, size, use_case, budget`) and returns the first
   not-yet-asked entry.

The two stages optimize different things and cannot be compared to each other.
Stage 1 maximizes pool-narrowing; stage 2 is a fixed list. There is no
representation of the one variable that turned out to matter most.

## 2. The correction that motivates this

CLAUDE.md #9 measured the *answerability* of each attribute — the share of the
200 public samples where the customer can still answer a question about it after
turn 1, derived from the evaluator's own reply policy:

```
feature   0.960      style     0.085
material  0.725      size      0.045
color     0.255      use_case  0.020
```

A **48x spread**, invisible to both stages above. Reordering stage 2's fixed
list by it was worth +0.0109 TechnicalScore — the largest single remaining win,
with 39 of 200 sessions reaching the target sooner. That was a hardcoded list.
This document is the principled version of the same idea.

## 3. Two corrections to the obvious formulation

Both were found by reading `evaluator/local_evaluator.py` rather than reasoning
from the code's shape, and both change the design.

### 3.1 There is no per-turn cost to trade off

The tempting objective is `expected gain − turn cost`, with the turn cost
derived from the scoring formula (Efficiency is `(11-MTTC)/10` weighted 0.20, so
one turn is worth 0.02 TechnicalScore). **That term does not belong.** The agent
cannot end a session — `evaluate()` runs until the target enters the top 10 or
turns are exhausted, and the agent returns 10 recommendations every turn
regardless of what it asks. Turn count is not an action the agent chooses, so
the cost is constant across all actions and drops out of the argmax.

What the agent actually controls is only *how fast the pool converges*. The
objective is therefore expected information gain, full stop.

### 3.2 "Ask nothing" is not free, and is a minor refinement

`ask_attribute: null` is legal, and I previously suggested it as a significant
win. It is smaller than that. `customer_reply()` still returns a message when
the agent asks nothing:

```python
if not attribute:
    return "Those options are not quite right yet. Ask me about one specific attribute.", boundary_used
```

So a null ask does not avoid a turn or avoid query noise — it swaps one
contentless string for another. Its only advantage is that the string is
*constant across sessions*, whereas a failed ask returns
`"I don't have an additional preference for {attribute}."`, which injects the
attribute word into the query. Real, but second-order. Keep it in the design as
a threshold behaviour; do not expect it to carry the result.

## 4. The objective

For each attribute `a` not already asked or already evidenced in the query:

```
EV(a) = P(answerable | a, session state) x Gain(a)
```

### 4.1 Gain has two distinct components

This is the part the current code conflates, and it explains why `feature` won
in #9 despite having no extractor at all.

- **Structural narrowing.** For `material` and `color`, a disclosed value feeds
  `_boost_by_disclosed` and resorts the pool. Measured by the existing
  `normalized_entropy` over the pool's values.
- **Query enrichment.** For *every* attribute, the disclosed text is appended to
  the query and improves BM25 and dense retrieval directly. `style`, `size`,
  `use_case` and `feature` have no extractor, so their structural narrowing is
  exactly zero — their entire value is enrichment.

So:

```
Gain(a) = ALPHA * entropy(a, pool)   +  BETA
                (0 for the four unextractable attributes)
```

`BETA` is a constant because an enriching answer is worth roughly the same
whichever attribute produced it. With `entropy = 0` for the non-structural
attributes, `EV` collapses to `P(answerable) * BETA` — i.e. rank by
answerability, which is exactly #9's measured win. **The framework reproduces
the known-good behaviour as a special case**, which is the main reason to
believe it.

`ALPHA/BETA` is the one genuinely free parameter and must be swept, not guessed.
Sweep it the way `sweep_fusion_weights.py` does: only the ratio matters.

### 4.2 P(answerable) has a prior and a within-session update

**Prior:** the #9 table. Flag honestly: those marginals were derived from the
local simulator's reply policy, so hardcoding them is fitting to the simulator —
the same trap as #12 and #13. Two mitigations. The ordering is *plausible on
general grounds* (a shopper can describe a feature more readily than a "use
case"), so it is not pure memorization. And the update below is
simulator-agnostic.

**Update:** `customer_reply()` consumes constraints as they are disclosed
(`matches[:2]`, then `disclosed.update(matches)`), so once a session's constraint
list is exhausted **every** further question returns a non-answer. The agent can
observe this directly: a reply of the "I don't have an additional preference"
shape is evidence that the well is dry. A simple multiplicative decay on
`P(answerable)` after each observed non-answer captures it, needs no simulator
knowledge, and is exactly the "runtime workflow re-orchestration" the spec's
§ 4.2.III asks for.

## 5. Integration

Replace the two-stage body of `_next_attribute` with one ranking over all
candidate attributes. Structure it so the intent-conditioned `min_entropy`
becomes the `EV` threshold below which the agent asks nothing, keeping
`routing_params()` as the single place intent-conditioned behaviour is resolved
(per CLAUDE.md's code-layout note — add to `RoutingParams`, never a fourth
branch).

`select_dynamic_attribute` stays as the entropy primitive; it just stops being
the decision-maker.

## 6. How to measure it

Full 200-sample A/B against the current 0.6070 baseline, both legs on the same
HEAD. Then:

- **Report absolute session counts, not just rates.** Per CLAUDE.md #16, 0.7500
  vs 0.7450 is one session; the benchmark's resolution is about +-0.0025.
- **Expect MRR to move against us.** #9 recorded this: the score's MRR uses the
  rank at the *first* hit, so converting a late-well-ranked hit into an
  early-worse-ranked one costs MRR while paying more in Efficiency. That trade
  is inherent. Judge on TechnicalScore, never on MRR alone.
- **Report the per-sample delta distribution.** #9's win was credible because 39
  samples improved and the modal delta was -3 turns on 24 of them — broad, not a
  few lucky sessions. Demand the same evidence here.
- **Ablate the two terms separately.** Run `BETA`-only (pure answerability
  ranking) as a control. If the full objective does not beat that control, the
  entropy term is not earning its complexity and should be dropped.

## 7. Honest expected value

The `BETA`-only control is already known to be worth +0.0109, and it is already
shipped as the reordered fallback list. **So the incremental gain here is
whatever the entropy term and the within-session decay add on top of a win we
have already banked** — likely small. The reasons to build it anyway are that it
replaces two incomparable mechanisms with one objective, it makes the
answerability insight explicit and tunable rather than frozen into a list, and
the decay directly implements a named spec pillar that currently has nothing
behind it.

Do not expect a large number. Expect a defensible design and a small gain.
