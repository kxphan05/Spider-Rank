# Manual QA Findings — Demo Testing

**Date:** 2026-08-29
**Testing time:** ~2:30 PM
**Scope:** Live Streamlit demo, manual testing

These bugs were identified through manual testing of the live Streamlit demo based on observed screenshots.

The automated 200-sample evaluator would **not catch these issues**, because it only checks whether the true target appears in the **top 10 results**. It does not evaluate:

* Whether the reasoning shown to the user is internally consistent
* Whether an override actually changes the recommendation results
* Whether the returned products are relevant to the user's explicit intent
* Whether the system asks for information that the user has already provided

> **Note:** The code locations and root-cause explanations below are hypotheses based on outward behavior. They have **not yet been confirmed by inspecting the live internal state**.

---

# Bug 1 — Retrieval/Recommendation Slate Does Not Properly Reflect Current Intent

This appears to manifest in **two related ways**:

1. Explicit product/category intent can produce highly irrelevant results.
2. An intent override can be detected correctly, while the resulting recommendations remain essentially unchanged.

Together, these suggest a possible issue in the **downstream retrieval/ranking pipeline**, rather than purely in intent detection.

---

## 1A. Explicit "Shoes" Query Returns Irrelevant Products

### Input

> I'm looking for shoes. A key requirement is: blue

### Observed behavior

The visible top results include:

* A dupatta/scarf
* A leather wallet
* A luggage spinner
* Two T-shirts

There are **no shoes** among the visible results.

### Why this looks broken

`"shoes"` is a plain and explicit product-category query.

Even though the user additionally specified:

> `blue`

the recommendation slate should still strongly prioritize **shoes**, with blue acting as an additional constraint/preference.

Instead, the returned results appear to prioritize other properties or historical/semantic similarity over the user's explicit category.

This raises the possibility that:

* Category relevance is not being weighted strongly enough
* The `"blue"` constraint is dominating retrieval
* The category term `"shoes"` is not being represented correctly in the retrieval query
* The retrieval/indexing pipeline is failing to preserve explicit product intent
* The ranking stage is overriding strong category relevance

### Confidence

**Medium / lower confidence.**

This could potentially be a catalog-content or ranking issue rather than a logic bug.

To determine whether this is systematic, test additional simple category queries:

```text
I need a dress
I need a watch
I need a shirt
I need shoes
```

If unrelated products consistently appear for explicit category queries, that would provide stronger evidence of a retrieval/ranking problem.

---

## 1B. Intent Override Is Detected, But Recommendations Do Not Change

### Input

**Turn 1:**

> I'm looking for shoes. A key requirement is: blue

**Turn 2:**

> I'm looking for shirts. shoes

### Observed behavior

In Turn 2, the system successfully detects an **intent override** and clears the internal constraints:

```text
disclosed = {}
```

However, the recommended product slate remains virtually identical to Turn 1's results, including the same `"summer"` items.

The system also classifies Turn 2 as:

> **Intent: buying**

while the agent message states:

> "no hard constraints stated yet; buying track"

### Why this looks broken

The state machine appears to correctly register the override and reset the disclosed dictionary.

However, the **actual recommendation slate does not materially change**.

This is important because it suggests that successfully updating the conversational state may not be enough to affect downstream retrieval.

A possible flow is:

```text
User override
      ↓
Override detector ✓
      ↓
State constraints cleared ✓
      ↓
_build_query(state)
      ↓
Stale context still included?
      ↓
Retrieval
      ↓
Same / similar recommendations ✗
```

For example, if `state.recent_messages` or `state.history` continues to contain information from Turn 1, the retrieval query may still be influenced by previous tokens such as `"summer"`.

### Additional UI/state inconsistency

There is also a potential inconsistency between:

> **Router:** `buying`

and

> **Agent:** `"no hard constraints stated yet; buying track"`

This may be expected if the new turn contains no scalar constraints, but it should be verified against the actual internal state.

### Where to investigate

#### `starter/agent.py` — `respond()` and `_build_query(state)`

Check how:

```python
state.recent_messages
state.history
```

are handled when:

```python
self.override_detector.is_override(user_message)
```

returns `True`.

Specifically, investigate whether:

* `recent_messages` is cleared or pruned after an override
* Historical messages are still passed into `_build_query()`
* `_build_query()` incorporates stale conversational context
* Retrieval is still influenced by the previous intent after the state reset

---

## Why 1A and 1B May Be the Same Bug

These two observations point toward the same broader question:

> **Does the downstream retrieval/ranking pipeline actually prioritize the user's current explicit intent?**

### In 1A

The system knows the user said:

```text
shoes
```

but the returned products are not shoes.

### In 1B

The system detects that the user has **changed intent**, but the returned products still look like they belong to the previous context.

So the problem may not be:

> "The intent router doesn't understand the user."

Instead, it may be:

> **"The intent is being detected, but that intent is not being properly reflected in the retrieval query or final ranking."**

This distinction is important because it shifts the investigation downstream from the intent classifier toward:

```text
Intent detection
       ↓
State update
       ↓
_build_query()
       ↓
Retrieval
       ↓
Ranking
       ↓
Recommendation slate
```

The exact failure point still needs to be confirmed.

---

## Suggested Resolution / Debugging Strategy

Use the interactive REPL debugger:

```bash
uv run python3 scripts/repl.py --debug
```

### Test 1 — Override behavior

Run:

```text
Turn 1: I want something for the summer
Turn 2: actually forget that, show me jewellery
```

After Turn 2, inspect:

```text
state.recent_messages
state.history
state.disclosed
_build_query(state)
```

The key question is whether the generated retrieval query still contains information from Turn 1.

---

### Test 2 — Explicit category retrieval

Run clean single-category queries:

```text
I need shoes
I need a dress
I need a watch
I need a shirt
```

Inspect:

```text
_build_query(state)
```

and the resulting retrieval slate.

The goal is to determine whether the category itself is reaching the search engine and whether it receives sufficient ranking weight.

---

### Expected behavior

For an explicit category query such as:

> "I need shoes"

the retrieval/ranking pipeline should strongly prioritize shoe products.

For an override such as:

> "Actually forget that, show me jewellery"

the recommendation slate should materially shift toward jewellery rather than remaining dominated by the previous context.

---

# Bug 2 — Asks About an Attribute It Just Said It Matched On

### Input

> i need shoes. blue, rubber, under 100, us size 10, water-resistant

### Observed behavior

The response states:

> "matching on color=blue, material=rubber; buying track; asking about material because it still splits these 10 options. What material are you looking for?"

### Why this looks broken

The system simultaneously:

1. Identifies **material = rubber**
2. Says it is **matching on material = rubber**
3. Immediately asks:

   > "What material are you looking for?"

If `material=rubber` was already disclosed and used for retrieval/boosting, the system should generally recognize that attribute as already provided and avoid asking for it again.

This suggests that the two attribute-extraction paths may disagree.

---

## Where to investigate

### `starter/agent.py`

Investigate:

```python
_next_attribute()
```

Pay particular attention to the `excluded` set and how already-disclosed attributes are excluded from follow-up questions.

### `classifier.py`

Investigate:

```python
detected_attributes()
```

Compare its handling of:

```text
rubber
```

against the extraction logic used by:

```python
extract_disclosed_value()
```

---

## Hypothesis

A likely explanation is that there are **two separate extraction paths**:

```text
extract_disclosed_value()
        ↓
used for scoring / boosting


detected_attributes()
        ↓
used to determine which attributes
have already been provided
        ↓
_next_attribute()
```

These two paths may disagree about whether `"rubber"` is a valid material.

For example:

```text
extract_disclosed_value()
    → material = rubber ✓

detected_attributes()
    → material not detected ✗

_next_attribute()
    → thinks material is missing
    → asks for material again
```

### Suggested next step

**Confirm the vocabulary mismatch theory before changing any logic.**

Grep for `"rubber"` across:

```text
starter/attributes.py
starter/classifier.py
```

If `"rubber"` exists in one vocabulary but not the other, that would explain the mismatch.

Only after confirming this should the extraction logic be changed.

---

# Recommended Next Step

Before handing these over to the technical teammate as **confirmed bugs**, re-run both issues through:

```bash
uv run python3 scripts/repl.py --debug
```

The debugger should allow inspection of the agent's actual internal state, including:

```text
disclosed
asked_attributes
profile_key
recent_messages
history
_build_query(state)
```

This is important because the screenshots only show the **external behavior**. They do not show what the agent internally believes after each turn.

## Current assessment

| Finding                                                                       | Confidence  | Main thing to verify                                                                        |
| ----------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------- |
| **Bug 1 — Retrieval/recommendation does not properly reflect current intent** | Medium      | Whether explicit intent is being lost/overridden in `_build_query()`, retrieval, or ranking |
| **Bug 2 — Re-asks for material**                                              | Medium–High | Whether `rubber` is detected consistently by both attribute-extraction paths                |

### Important distinction

These should currently be treated as **manual QA findings / suspected bugs**, rather than confirmed implementation bugs.

The REPL `/debug` run should establish whether the observed behavior comes from:

1. Incorrect state management
2. Stale conversational context
3. Incorrect query construction
4. Retrieval/ranking weights
5. Attribute vocabulary/extraction mismatch
6. Expected behavior that only appears incorrect from the UI
7. A malformed test input
