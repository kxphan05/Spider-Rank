"""Every tunable knob in the agent, in one file.

This module holds the values you would *tune, sweep, or check*: fusion
weights, pool sizes, thresholds, and feature flags. It deliberately does not
hold vocabularies, regexes, or classifier prototype corpora -- those are the
data a module's logic is made of, not configuration, and they stay next to the
code that reads them.

Each constant keeps the comment block that records how its value was arrived
at. On this project that history is the most valuable thing about a number:
almost every one below is the output of a measured A/B, and several are
identity settings whose whole purpose is to make a baseline reproducible.

HOW TO OVERRIDE ONE IN AN EXPERIMENT -- read this before writing a sweep.

Consumers import these by name (`from .config import PHRASE_WEIGHT`), which
binds the value into the *consuming* module's namespace at import time. So a
sweep must patch the module that reads the knob, not this one:

    import starter.agent as agent_module
    agent_module.PHRASE_WEIGHT = 4.0        # correct -- agent reads this name

    from starter import config
    config.PHRASE_WEIGHT = 4.0              # WRONG -- silently does nothing
                                            # to an already-imported agent

That is not an accident of style, it is what keeps every existing sweep in
`scripts/` working unchanged. Getting it backwards produces a sweep whose legs
are all secretly identical -- which is exactly the failure mode recorded in
CLAUDE.md's "Blockers" list, where a knob whose zero value was not the identity
made every measured point wrong by an unknown amount. If you add a knob, give
it an identity default, and verify identity reproduces the shipped score with
a real run before trusting any other leg.
"""
from __future__ import annotations

from pathlib import Path



# ==========================================================================
# Agent: routing, fusion weights, candidate pools, feature flags
# ==========================================================================

# Data-asset location. The evaluator constructs `Agent()` with no arguments,
# so these defaults decide where a *submitted* bundle looks for the catalog
# and dense index -- and a wrong guess degrades silently (a missing dense
# index costs the whole dense leg without raising). Resolution order:
#
#   1. the TECHJAM_CATALOG / TECHJAM_DENSE_INDEX environment variables
#   2. the path relative to the current working directory (repo-root runs)
#   3. the same path relative to this package's parent directory
#
# Step 3 is what makes the bundle work when the harness runs from somewhere
# other than the directory holding `data/`. Step 2 is listed first so
# behaviour from the repo root is exactly what it has always been.
CATALOG_PATH_ENV = "TECHJAM_CATALOG"
DENSE_INDEX_PATH_ENV = "TECHJAM_DENSE_INDEX"

# material/color are chosen dynamically (see attributes.py:
# select_dynamic_attribute) based on value-diversity in the current
# candidate pool. style/size/use_case/feature have no cheap structural
# extractor from catalog text, so they stay a fixed fallback order, used
# once the dynamic pick has nothing left worth asking about.
#
# The order is by *answerability*, measured against the local evaluator's
# own reply policy (scripts/eval_profile_signal.py --check tags): share of
# the 200 public samples where the customer can still answer a question
# about the attribute after turn 1 --
#
#   feature 0.960   style 0.085   size 0.045   use_case 0.020
#
# A question the customer can't answer burns a whole turn, and MTTC is 20%
# of the score, so the most-answerable bucket goes first. Caveat kept on
# the record: `feature` is `classify_constraint()`'s catch-all return, so
# its 96% is partly an artifact of being the bucket everything unmatched
# falls into -- this may be a local-simulator quirk, same shape as the
# `budget` finding below. `budget` is last-resort only: measured against
# the local evaluator, disclosed answers almost never bucket as budget
# (see attributes.py STRUCTURAL_ATTRIBUTES comment), so it's demoted below
# every other question rather than dropped outright.
FALLBACK_ATTRIBUTE_ORDER = ["feature", "style", "size", "use_case", "budget"]

RECENT_WINDOW = 4  # messages after the first, kept for query context
CANDIDATE_N = 50  # per-leg retrieval depth before fusion
ENTROPY_POOL_SIZE = 30  # fused-candidate pool size used for attribute-entropy scoring

# Dual-track routing (TODO.md "Core Architecture: Intent Routing"), both
# tracks built on the same full-catalog RRF fusion (a filter-then-rerank
# buying track was tried first and measured worse -- see CLAUDE.md, it lost
# recall whenever BM25's top-CANDIDATE_N keyword filter missed a
# paraphrased target that dense would have caught): "buying" keeps the
# already-tuned BM25-heavy weighting for "lock hard constraints" precision;
# "browsing" shifts weight toward dense for "unlock cross-category scenario
# matching" breadth, then gets an MMR diversity re-rank (Agent._diversify)
# on top.
# Buying dense weight halved to a 0.25 dense:bm25 ratio, from the sweep in
# CLAUDE.md #16: the buying curve is strictly monotone decreasing with no
# interior optimum, and 0.25 captures +0.0112 of the +0.0188 available from
# removing the dense leg outright. We keep a real dense leg rather than take
# the full gain because the local set is a near-pure exact-match benchmark
# (89.7% of hard constraints are verbatim substrings of the target's own
# catalog text, #14), so it systematically under-prices the paraphrase
# robustness dense exists to provide. Browsing is untouched: its curve was
# measured flat across 0.0-1.5 and must not be re-tuned (#16).
BUYING_BM25_WEIGHT = 2.0
BUYING_DENSE_WEIGHT = 0.5
BROWSING_BM25_WEIGHT = 1.25
BROWSING_DENSE_WEIGHT = 1.5

# Entropy threshold for select_dynamic_attribute (attributes.py), intent-
# conditioned: buying sessions ask more eagerly (locking a hard constraint
# is high-value), browsing sessions ask less eagerly (per CLAUDE.md's
# documented non-answer-misclassification lesson, vague-intent customers
# are more likely to burn a turn on a non-answer) and bias toward
# recommending sooner instead.
MIN_ENTROPY_BUYING = 0.10
MIN_ENTROPY_BROWSING = 0.30

# Browsing-track diversity re-rank (Agent._diversify): MMR over the top
# DIVERSIFY_WINDOW of the pool, with the top DIVERSIFY_PIN items pinned
# unconditionally so the best match(es) can never be traded away for
# variety. DIVERSIFY_LAMBDA is a fixed constant, not swept -- the pin +
# narrow window already bound how much diversity can cost relevance, and a
# third tunable knob isn't worth the added complexity here.
DIVERSIFY_WINDOW = 20
DIVERSIFY_PIN = 2

# The buying track has never run the MMR diversity re-rank; browsing always
# has. That asymmetry is the only one of the four intent-conditioned knobs that
# changes how much of a *category* a slate covers, and the miss census points
# straight at it: buying's misses sit in coarse categories holding a median 338
# products against 138 for its hits (permutation p = 0.0096 over 20k shuffles),
# while browsing shows no crowding effect at all (p = 0.42). Under
# EXCLUDE_SHOWN a session can surface up to 100 distinct products, so covering
# a crowded category is exactly what a diverse slate buys. Flag defaults to
# False, which is the shipped behaviour, so identity is bit-reproducible.
BUYING_DIVERSIFY = False

DIVERSIFY_LAMBDA = 0.5

# Drop a contentless clarifying reply from the query history instead of
# searching the catalog with it. A session has a mean of only 2.09 answerable
# attribute buckets left after turn 1 against an MTTC of ~5, so roughly three
# asks in five come back empty and each one contributes its wording
# ("preference", "judgment", the attribute name) to BM25 as if the customer
# had said something. See session_belief.py for the derivation.
#
# MEASURED AND REJECTED -- do not re-enable without reading CLAUDE.md #19.
# This costs -0.0410 TechnicalScore on the full public set (HitRate 0.7450 ->
# 0.6850, i.e. 149 -> 137 of 200 sessions). The detector is not the problem:
# every one of its false positives is the simulator's near-contentless
# constraint template ("For that, what matters is: Imported; Pull On
# closure."), which *looks* like boilerplate and is in fact an exact-match key
# to the target, because 89.7% of customer text on this benchmark is a
# verbatim substring of the target's own catalog record (CLAUDE.md #14). Here,
# semantically contentless customer text is still retrieval signal, so query
# pruning cannot be made safe by improving the classifier.
#
# The non-answer *observation* is kept and still feeds SessionBelief -- it is
# the query surgery that fails, not the detection.
SKIP_NON_ANSWERS_IN_QUERY = False

# Drop contentless replies from the *phrase leg's* query only, leaving the
# BM25 and dense query untouched.
#
# This is deliberately narrower than the flag above, which measured -0.0410 and
# was reverted (CLAUDE.md #19). That loss had a specific mechanism: 89.7% of
# customer text is a verbatim substring of the target's own catalog record
# (#14), so text the detector called contentless was still an exact-match key
# for BM25, and deleting it deleted what BM25 wins with. The phrase leg has the
# opposite problem -- it is span-budget-limited (MAX_PHRASE_QUERIES), and
# filler spans match nothing while still consuming the budget that the
# informative spans needed. Same classifier, opposite cost structure.
PHRASE_QUERY_SKIP_NON_ANSWERS = False

# Geometric decay applied to a disclosed slot's boost per turn since it was
# stated (_boost_by_disclosed). 1.0 == no decay == the shipped behaviour, and
# is the identity setting the A/B baseline must reproduce exactly.
#
# Motivation: override handling is all-or-nothing -- respond() wipes
# `disclosed` outright on a detected pivot -- and CLAUDE.md #17 measured that
# the blunt form of that idea is structurally wrong (restarting the query from
# the pivot message scored -0.0580, because turn 1 carries the category and
# the pivot carries only the changed attribute). Decay is the graceful middle:
# a stale early constraint fades rather than being kept at full strength or
# discarded entirely.
SLOT_DECAY = 1.0

# Pick the next question by the session's own answerability belief rather than
# the static FALLBACK_ATTRIBUTE_ORDER. The belief is *initialized* to that
# order's marginals, so turn 1 behaviour is identical either way -- this
# strictly generalizes the #9 reorder rather than replacing it.
BELIEF_DRIVEN_QUESTIONS = False

# Added to the BM25 leg's weight once the belief reports the card exhausted.
# 0.0 disables the re-orchestration and is the identity setting.
EXHAUSTED_BM25_BONUS = 0.0
BELIEF_REORCHESTRATION = False

# Third RRF leg: verbatim-span matching (BM25Index.phrase_search). 0.0 is the
# identity setting and disables the leg entirely, including its lookup cost.
#
# Rationale is the measured shape of this task rather than a general IR
# preference. Every retrieval-side change that has *lost* score here lost the
# same way -- by adding semantic tolerance to a benchmark where 89.7% of
# customer text is a verbatim substring of the target's own catalog record
# (#14 dense weight, #17 override rewrite, #19 query pruning). A phrase leg
# runs the other direction: it rewards exact spans, which is the signal that
# measurement says actually carries here. Sweep before shipping; the fusion is
# scale-invariant per leg, so only this weight's ratio to the other two
# matters.
#
# Swept (scripts/sweep_phrase_weight.py, 200 samples/point, identity
# reproduced 0.6182 exactly). Unlike every other retrieval knob here, this
# curve has a real interior optimum and the gain is *recall*, not reordering:
#
#   weight   HitRate      hits      MRR     MTTC  Technical     delta
#     0.00    0.7500  150/200   0.4096    4.985     0.6182        --
#     0.50    0.7600  152/200   0.4278    4.945     0.6294   +0.0112
#     1.00    0.7700  154/200   0.4184    4.870     0.6331   +0.0149
#     2.00    0.7900  158/200   0.4176    4.745     0.6454   +0.0272  <- shipped
#     4.00    0.7850  157/200   0.4254    4.825     0.6436   +0.0254
#
# +8 sessions of hit rate. Contrast the dense leg (#14), which adds no recall
# at all. 2.00 and 4.00 are one session apart, so the top is flat and the
# choice between them is arbitrary; 2.00 is the lower-variance pick.
PHRASE_WEIGHT = 2.0

# Pseudo-relevance feedback (prf.py): a fourth RRF leg that re-retrieves with
# terms harvested from the top FEEDBACK_DOCS of the BM25 ranking.
#
# Rationale is the buying diagnostic, not a general preference for PRF. Buying
# scores 0.688 hit rate against browsing's 0.825, and the cause is measured:
# the median turn-1 hard constraint is *one token*, and the modal values are
# `cotton` (18.8% of the catalog), `polyester` (13.8%) and `leather` (12.6%).
# That is a low-entropy query, not a contentless one, and every other lever
# here redistributes weight among existing signal rather than adding any. PRF
# is the one classical technique that adds vocabulary: it replaces "cotton"
# with the words that actually co-occur in the products "cotton" already ranks
# well.
#
# Shipped at 0.0 (exact identity) until swept. The known failure mode is query
# drift when the feedback documents are wrong, which is a live risk on a
# benchmark where the target is frequently already in the top handful -- there
# is more to lose on the sessions that already work than to gain on the ones
# that don't. That asymmetry is why this is a weighted leg rather than a
# rewrite of the BM25 query: a leg turns down continuously, a rewrite does not
# (#17 is the cautionary case -- an aggressive query rewrite measured -0.0580).
PRF_WEIGHT = 0.0

# Cross-encoder reranking (reranker.py, Qwen3-Reranker-0.6B). Closes the one
# remaining named gap against spec Pillar I, "Multi-Route Retrieval -> LLM
# Semantic Ranking" (CLAUDE.md #6): until now nothing learned or generative
# ever scored a (query, product) pair jointly.
#
# 0.0 is the identity and skips the model entirely, including its load cost.
# The reranked order is fused with the incoming order by RRF rather than
# replacing it, so the leg can be weighted rather than trusted outright --
# same treatment as the dense leg, and for the same reason: this is a semantic
# component on a benchmark that measurement says favours exact matching (#14),
# so it should be able to lose gracefully.
RERANK_WEIGHT = 0.0

# Which backend scores the pairs. "minilm" is the shipped, measurable stage;
# "qwen" is the LLM-scale variant kept for the report and the demo. See
# reranker.py -- Qwen judges well but needs ~27 s per pair on this hardware,
# which is ~75 hours for one 200-sample evaluation and therefore cannot be
# A/B'd at all.
RERANK_BACKEND = "minilm"

# Depth of the pool handed to the cross-encoder. Cost is linear in this.
#
# This must be strictly greater than `top_k` to be worth running at all, and
# an earlier version of this comment had the reasoning backwards. It claimed
# reranking beyond the returned slate "cannot change HitRate, only which of
# the top items lead" -- but at RERANK_TOP_N == top_k == 10 the reranker only
# ever permutes the slate the agent was already going to return, so it cannot
# add a hit under *any* ordering; it can move MRR and nothing else. Scoring
# deeper than the slate is precisely what opens the recall channel: an item
# the retrieval fusion left at rank 11-20 can be promoted into the returned
# ten. That is the only way this stage can convert a miss into a hit.
#
# Swept against RERANK_WEIGHT in scripts/sweep_rerank.py, which is why both
# are constants rather than one.
RERANK_TOP_N = 20

# Do not re-recommend a product that has already been shown and rejected.
#
# This is a deduction from the scoring rule, not a heuristic. `evaluate()` in
# the local evaluator breaks the session the moment the target appears in the
# returned slate, so being *asked for another turn at all* is proof that none
# of the items shown so far was the target. Re-offering them spends slots that
# are known-dead: with ~3 replies in 5 carrying no new constraint
# (session_belief.py), the query barely moves between turns and the next slate
# largely repeats the one just disproven. Across a 10-turn session the
# evaluator affords up to 100 distinct guesses; without this the agent spends
# them on far fewer distinct products.
#
# THE EXCEPTION THAT MAKES THIS SUBTLE, and why it is wired to the override
# detector: the evaluator suppresses the hit check until a pivot lands --
#
#     override_applied = sample["scenario_type"] != "intent_override"
#     if override_applied and target in ranked: ... break
#
# -- so in an intent_override session (15% of the public set) an item shown
# *before* the pivot may still be the target, and excluding it would be
# actively wrong. `respond()` therefore clears the shown set on exactly the
# signal that clears `disclosed`, and the pre-pivot turns become admissible
# again. This also inverts the override detector's error asymmetry: a missed
# override now risks permanently excluding the true target, where before it
# only left stale constraints (see CLAUDE.md #7, whose rule was tuned toward
# precision under the old cost model).
#
# Assumption worth stating in the report: this relies on first-hit-ends-the-
# session, which the spec implies by defining MTTC as turns-to-conversion but
# which is only *verified* against the local evaluator.
#
# Measured on the full 200-sample public set (scripts/ab_phrase_exclude.py),
# all three legs on one HEAD:
#
#     leg                      HitRate      hits     MRR    MTTC  Technical
#     phrase 0.0 (identity)     0.7550  151/200  0.3989   4.940     0.6184
#     phrase 2.0                0.7800  156/200  0.4103   4.825     0.6366
#     phrase 2.0 + exclude      0.8550  171/200  0.4622   4.205     0.7020
#
# +0.0837, and unlike every reordering change here it moves all three terms at
# once: +15 sessions of HitRate, MRR up, MTTC down 0.62 turns. Every scenario
# improves (buying 0.738 -> 0.812, browsing 0.825 -> 0.925, boundary flat).
EXCLUDE_SHOWN = True

# Prepend a plain-language rationale to `turn_response.message` (spec's
# "transparent recommendation explanations"). Score-neutral by construction --
# the evaluator never reads the message back -- so this is on by default and
# judged on the demo, not the benchmark.
EXPLAIN_RECOMMENDATIONS = True

# Local masked-LM attribute inference (lm_confidence.py). Fills in an
# attribute the customer has NOT stated, gated on the model's own entropy --
# measured accuracy 0.787 below MAX_CONFIDENT_ENTROPY vs a 0.322
# guess-the-mode baseline, falling to 0.000 above it. Applied boost-only and
# at a fraction of a real disclosure's weight: an inference can lift a
# candidate but never sink one, so the ~21% of confident predictions that are
# wrong cost nothing beyond a missed lift. That asymmetry is not a style
# choice -- CLAUDE.md #5 measured that *any* nonzero mismatch penalty drops a
# candidate below every neutral (unknown-attribute) candidate regardless of
# weight, which is what made every previous profile-hint experiment regress.
LM_INFERENCE_WEIGHT = 0.5

# Disclosed-attribute resort weights (_boost_by_disclosed). A *match* between
# a disclosed value and the catalog's extracted value is rewarded, a known
# *mismatch* is penalized, and an unextractable attribute scores 0. These were
# +1/-1 by assumption, never by measurement -- and they were set when
# attributes.py's extractor was substring-matching (CLAUDE.md #10), so 58.6%
# of the catalog carried a colour label and roughly a fifth of those were
# fictional. The word-boundary fix cut real colour coverage to 39.9% and cost
# 0.005 TechnicalScore, because correct-but-sparser labels leave far more
# candidates at a neutral 0. Retuning against the corrected coverage is the
# indicated follow-up (NEXT_STEPS #2); sweep with
# scripts/sweep_boost_weights.py. Kept asymmetric-capable on purpose: a known
# mismatch and an unextractable attribute are different kinds of evidence.
DISCLOSED_MATCH_BOOST = 1.0
DISCLOSED_MISMATCH_PENALTY = 1.0

MAX_QUERY_CHARS = 2000


# ==========================================================================
# Retrieval: dense model, phrase leg, BM25 index build
# ==========================================================================

DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DENSE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Verbatim-span matching parameters (BM25Index.phrase_search).
PHRASE_MIN_N = 2
PHRASE_MAX_N = 5

# Per-query budget on phrase lookups. Each is an indexed FTS5 phrase scan, so
# they are cheap, but an accumulated 10-turn query has a long token tail.
MAX_PHRASE_QUERIES = 24

# A span appearing in more than this many products carries no identifying
# information, so it is dropped rather than scored.
PHRASE_MAX_MATCHES = 150

# Build spans within a clause instead of across the whole query, and require
# both edge tokens to carry content.
#
# The budget above is spent longest-first, on the assumption that a longer span
# is a more specific one. That holds for catalog copy and fails for
# conversation: the longest spans of a 10-turn query are sentences of filler
# that match nothing. Measured on public_0008's final query -- 135 spans built,
# and all 24 that fit the budget matched zero products, while "bras everyday
# bras" (verbatim in the target) sat unqueried at 3-gram depth.
#
# Two rules, both properties of language rather than of this simulator's
# wording -- a template blacklist would score well locally and transfer nothing
# (CLAUDE.md #12/#13):
#   1. no span crosses a clause boundary, since catalog text never does
#   2. no span begins or ends on a stopword, which is the standard
#      phrase-extraction heuristic for a span that is a fragment of one
PHRASE_CLAUSE_SPANS = False

# BM25F field weights, in the FTS5 column order declared above:
#
#   parent_asin, title, categories, features, details, store, description
#
# parent_asin is UNINDEXED so its weight is inert and pinned at 0.0. The rest
# were hand-picked when the index was first built and have never been swept --
# the IR literature is consistent that BM25F field weights are collection- and
# query-dependent and need a grid search, so hand-picked values on a 50k
# clothing catalog are unlikely to be right. `scripts/sweep_bm25_fields.py`
# sweeps them; DEFAULT_FIELD_WEIGHTS is the identity and must reproduce the
# shipped score exactly before any swept point is believed.
DEFAULT_FIELD_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)

# Every product in this catalog hangs off one root category
# ("Clothing, Shoes & Jewelry", 49,990 of 50,000), so the three words in it
# are present on essentially every document. BM25 gives a term appearing in
# ~100% of documents an IDF of ~0, which means the customer's own category
# word was worth nothing:
#
#     term      df with root      df with root dropped
#     shoes           1.000                     0.235
#     jewelry         1.000                     0.111
#     clothing        1.000                     0.433
#
# So "show me shoes" could not rank a shoe above a t-shirt -- the only query
# word that mattered was whatever conversational filler came with it (see
# _REQUEST_STOPWORDS in text_utils.py). Dropping the root restores the IDF of
# the leaf category words, which are the ones that actually discriminate.
#
# Detected rather than hardcoded: the root is whatever value leads
# `categories` on at least CATALOG_ROOT_MIN_SHARE of the first
# CATALOG_ROOT_SAMPLE products. Below that threshold nothing is stripped, so a
# catalog without a universal root is left exactly as it was. The decision is
# made from the first insert batch, which is still in memory, so this costs no
# extra pass over the catalog file.
CATALOG_ROOT_SAMPLE = 1000
CATALOG_ROOT_MIN_SHARE = 0.95

# What counts as naming a product category, for names_category() below.
#
# Only category nodes that are a SINGLE word qualify ("Shoes", "Jewelry",
# "Dresses", "Watches"). Tokens drawn out of multi-word nodes do not, because
# they are modifiers rather than product types: "Water Shoes" contributes
# "water", "Hand Wash Only" contributes "only", and treating either as a
# category pivot fires on ordinary attribute talk. Measured against the
# evaluator's own 30 override turns, the loose token rule fired on 6 of them
# and the single-word-node rule fires on 1.
#
# The df floor drops store names and one-off merchandising nodes ("Westlake",
# "Toddler Test"), which are category-shaped strings naming no product type.
CATEGORY_TERM_MIN_DF = 20


# ==========================================================================
# Classifiers: prototype scoring
# ==========================================================================

NEGATION_WINDOW_CHARS = 20

# How many best-matching prototypes each class averages over. A plain
# centroid (k = len(prototypes)) is dominated by the prototypes' shared
# sentence *shape* -- every PROTOTYPE_OVERRIDE entry is a long two-clause
# sentence that names the discarded prior statement, so a terse pivot
# ("never mind, give me white shoes") is judged mostly on its
# imperative-request half and lands nearer the continuation centroid. The
# opposite extreme, k=1 (nearest prototype), fixes terse pivots but is at the
# mercy of a single badly-placed prototype: measured 0.151 false-positive
# rate on the simulator's own continuation turns, versus 0.007 at k=3. A
# small trimmed mean gets the shape-robustness without that fragility.
# Swept in scripts/eval_override.py; see CLAUDE.md open problem #7.
TOP_PROTOTYPES = 4


# ==========================================================================
# Session belief: answerability priors
# ==========================================================================

# Share of the 200 public samples where the customer can still answer a
# question about each attribute after turn 1, computed from the evaluator's
# reply policy: a constraint is disclosable only when classify_constraint()
# buckets it as the asked attribute. Measured, not chosen --
# scripts/eval_dialogue_efficiency.py recomputes this table, and it reproduces
# the independently-derived figures in CLAUDE.md #9 exactly.
#
# Two caveats carried from #9. `feature` is classify_constraint()'s catch-all
# return -- the bucket every unmatched value falls into -- so its 0.960 is
# partly a local-simulator artifact and may not hold for the hidden grader.
# And `budget` is never bucketed at all locally (intent_card() truncates the
# price candidate out), which is why it sits last rather than being dropped:
# it may be a local quirk, so it stays available as a last resort.
ANSWERABILITY_PRIOR: dict[str, float] = {
    "feature": 0.960,
    "material": 0.725,
    "color": 0.255,
    "style": 0.085,
    "size": 0.045,
    "use_case": 0.020,
    "budget": 0.001,
}

# Multiplier applied to every *other* remaining attribute when one comes back
# empty. A non-answer is evidence about this card as a whole, not only about
# the attribute asked: the card holds at most four constraints
# (hard_constraints[:2] + soft_preferences[2:4]), so each empty reply makes it
# likelier the rest are exhausted too. 1.0 disables the cross-attribute update
# and is the identity setting for A/B purposes.
NON_ANSWER_SPILLOVER = 0.6

# Below this, an attribute is treated as not worth asking about. Set so the
# prior alone never suppresses anything (the smallest prior, budget, is 0.001)
# -- only *observed* non-answers can push an attribute under it.
EXHAUSTED_THRESHOLD = 0.0005


# ==========================================================================
# Masked-LM attribute inference
# ==========================================================================

LM_MODEL_NAME = "distilbert-base-uncased"

# Beliefs at or above this normalized entropy are treated as "the model does
# not know". Calibrated, not chosen by feel: measured against the true
# target's extracted material over the 200-sample public set
# (scripts/eval_lm_confidence.py), top-1 accuracy by entropy band was
#
#     H < 0.60      n= 61   0.787
#     0.60-0.75     n=105   0.371
#     0.75-0.85     n= 14   0.000
#
# -- monotonic, and steep. 0.60 is where the belief is still worth acting on;
# above it the prediction is at or below the always-guess-the-mode baseline
# of 0.322, so acting on it would be worse than doing nothing.
MAX_CONFIDENT_ENTROPY = 0.60


# ==========================================================================
# Pseudo-relevance feedback (disabled by default)
# ==========================================================================

# How many top-ranked products are assumed relevant. Classic PRF uses 10-20;
# smaller is safer here because a wrong assumption is what causes drift.
FEEDBACK_DOCS = 10

# How many expansion terms survive into the second query.
EXPANSION_TERMS = 8

# A term must appear in at least this many feedback documents to be
# considered. A term in one document out of ten is that document's
# idiosyncrasy, not a property of the result set.
MIN_FEEDBACK_DF = 3

# Terms shorter than this are dropped -- FTS5 tokenises aggressively and short
# fragments are almost always noise.
MIN_TERM_LENGTH = 3


# ==========================================================================
# Cross-encoder re-ranker (disabled by default)
# ==========================================================================

RERANK_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
MINILM_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MINILM_MAX_LENGTH = 256

# Characters of product text shown to the reranker. Cost here is dominated by
# prefill, which is linear in prompt length, so this is the main runtime knob.
# Long enough to carry the title plus the first feature or two, which is where
# material/colour/closure information actually lives in this catalog.
MAX_DOC_CHARS = 400

# Pairs scored per forward pass. Kept small: peak RSS matters more than
# throughput on a 4.6 GB-free machine, and padding waste grows with batch size
# when document lengths are uneven.
BATCH_SIZE = 4


# ==========================================================================
# Long-term profile store
# ==========================================================================

# Overridable so a *local benchmark* run can be isolated from the persistent
# store. This matters more than it looks: the store is write-through and
# survives process exit, so repeated local eval runs accumulate history and
# feed it back into subsequent runs. Measured on the 200-sample public set,
# counting sessions whose reset() received a non-empty carried hint:
# run 1 -> 45/200, run 2 -> 105/200, run 3 -> 200/200. Any A/B of a
# hint-consuming agent is therefore confounded by how many times the eval had
# been run before, and drifts toward "every session is influenced" as you
# iterate. Cross-session persistence is a real, intended feature for the
# graded run (one pass, genuine session history); it is purely an artifact
# when re-scoring the same 200 samples over and over. scripts/run_eval.py
# sets this to a per-run temp path by default; set it explicitly to isolate
# `python3 -m evaluator.local_evaluator` the same way:
#     TECHJAM_PROFILE_STORE=/tmp/store.json uv run python3 -m evaluator.local_evaluator
STORE_PATH_ENV = "TECHJAM_PROFILE_STORE"
DEFAULT_STORE_PATH = Path("data/user_profiles.json")

# A profile key is a content hash with no customer id behind it (see module
# docstring) -- most repeat-key sessions on this catalog turn out to be a
# coincidental template collision, not a genuine returning shopper. Measured
# directly: carrying forward *any* single historical disclosure as a hint
# regressed the full 200-sample public set (HitRate 0.755->0.745, MRR
# 0.384->0.355, TechnicalScore 0.601->0.585) because a single wrong guess
# sinks the true target below every neutral (unknown-attribute) candidate,
# regardless of how small its boost weight is -- lowering the weight alone
# can't fix that (the experiment's PROFILE_HINT_WEIGHT constant is long gone
# from agent.py; the finding is written up in CLAUDE.md #5). Requiring the
# *same* value to recur at least twice in history before it's trusted as a
# hint filters out one-off coincidental collisions while still catching a
# shopper who has genuinely stated the same preference more than once.
#
# That last sentence is the hypothesis, and it was tested and did not hold:
# gating on corroboration still scored below baseline, because a value
# recurring twice under one key does not make it likelier to be right for a
# third, unrelated session sharing that key. Kept as the gate on `carried`
# anyway -- it is the conservative choice for whatever reads it next, and it
# costs nothing while nothing does.
MIN_CORROBORATION = 2


# ==========================================================================
# Model cache location
# ==========================================================================

MODEL_DIR_ENV = "TECHJAM_MODEL_DIR"
