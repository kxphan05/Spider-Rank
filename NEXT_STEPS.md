# Next steps — queued work not yet started

Working notes for picking up cold in a fresh session. Each item states what
was already verified vs. what is still an untested design argument, since
this project's convention is to measure before trusting either.

Cross-references into `CLAUDE.md`:
- Open problem **#7** (nearest-centroid classifiers are the weak link) is
  already written up there in full, with the reproduced failure and the
  cheap-before-expensive option list. Not duplicated here.
- Open problems **#1–#6** are the standing list; read them before touching
  retrieval or the profile store, they record what was already tried and
  measured worse.

---

## 1. LLM-extracted catalog attributes (offline, one-time)

**Status:** designed and costed, nothing built. This is the highest-value
unstarted item.

### The problem it solves

`starter/attributes.py`'s `AttributeIndex` extracts one value per product by
taking the **first vocab hit** in `title + features`. Two consequences, both
already measured and recorded in `CLAUDE.md`:

- **Coverage is thin:** material ~78.1%, color ~58.6%, price ~21% of the
  50k catalog.
- **It disagrees with the true target** 16.3% of the time on material and
  37.0% on color (measured in the open-problem-#1 investigation). That
  disagreement is what forced the switch from hard filtering to the
  non-eliminating `_boost_by_disclosed` resort.
- **Four of the eight allowed attributes have no extractor at all** —
  `style`, `size`, `use_case`, `feature` never populate
  `SessionState.disclosed` and can never be filtered on or converged. This
  is the documented ceiling of the whole entropy-question design, not a bug.

An LLM reads the entire record (title, features, description, details,
categories, store) instead of regex-matching a fixed word list, and can
return `null` rather than guessing.

### Why subagents are the wrong tool

Spawning Claude Code subagents to iterate the catalog was the original
framing — don't. Each spawn is a full cold-start agent session; 50,000 of
them, or even 1,000 batched ones, is the most expensive possible way to run
what is a single-shot extraction prompt per chunk. Use the **Message
Batches API** from a one-off script in `scripts/`.

### Cost, measured against this catalog

| Input | tokens |
|---|---|
| `title` + `features` | ~5.9M |
| + `description` | ~9.4M |

With `claude-haiku-4-5` ($1.00 / $5.00 per MTok), the Batch API's 50%
discount, ~50 products per request (~1,000 requests — far under the
100k-request / 256MB per-batch cap), and ~60 output tokens per product:

**roughly $5–12 one-time, completing in well under the 24h batch SLA.**

Re-check pricing at the time of the run rather than trusting this table.

### Rules check (done, all three clear)

- `docs/submission_rules.md` § Model Policy explicitly permits prototyping
  with any legally accessible LLM API.
- Output is a **static JSON sidecar** shipped as a "lightweight local
  asset" (an allowed submission content type), so the agent needs **no
  network access at scoring time** — this matters, since organizer policy
  may disable network access for official scoring.
- The read-only-catalog constraint holds: write a new sidecar file keyed by
  `parent_asin`, never modify `data/catalog.jsonl`.

### Design constraints (non-negotiable, these are why naive versions fail)

1. **Closed vocabulary, not free-form natural language.** `AttributeIndex`
   compares values for *equality*; "a rich chocolate-brown leather" is
   useless to it. Constrain the model to a fixed value set per attribute
   (extend the existing `COLORS` / `MATERIALS` tuples in
   `starter/classifier.py` as the seed) and use structured outputs so the
   schema is enforced rather than requested.
2. **`null` must be an allowed value.** An invented attribute is worse than
   a missing one — the whole point of open problem #1 is that a *confident
   wrong* value sinks the true target.
3. **Determinism / reproducibility.** Store the sidecar in git (it's small
   — 50k rows of short categorical values), don't regenerate it per run.
   Record the model ID and prompt version inside the file so a later
   session can tell what produced it.

### The honest caveat

This fixes only the **catalog half** of the disagreement. The 16%/37%
target-disagreement figure comes from *two independently-built extractors*
disagreeing: the catalog side and the customer-text side
(`extract_disclosed_value`, which runs at inference time and must stay
offline/LLM-free). Upgrading only the catalog side may not move the metric
much on its own — the gain lands fully only if the LLM-derived closed
vocabulary **also becomes the vocabulary `extract_disclosed_value` matches
against**, so both sides speak the same value set. Plan both halves, or
measure the catalog-only half honestly and expect a smaller delta than the
coverage numbers suggest.

### Suggested first step

Prototype on a few hundred products (non-batch, direct API) before
committing to the full run — enough to inspect the schema, eyeball quality,
and sanity-check that the extracted values actually agree with the true
targets more often than the current regex does. That agreement rate against
`ground_truth` is the number that decides whether the full run is worth it,
and it can be computed on the 200-sample public set.

---

## 2. User-profile personalization (the branch this was queued behind)

See `CLAUDE.md` open problem **#5**. The store itself is built and correct;
three ways of *using* `profile_hint` were each measured to regress the full
200-sample public set, so it currently ships inert. The last paragraph there
records the one untried idea that needs no cross-session identity
assumption: nudging `FALLBACK_ATTRIBUTE_ORDER` from the *current session's
own* freshly-given `preference_tags`.
