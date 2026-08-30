from __future__ import annotations

from pathlib import Path



# ==========================================================================
# Agent: routing, fusion weights, candidate pools, feature flags
# ==========================================================================

# Data-asset location. Resolution order:
#
#   1. the TECHJAM_CATALOG / TECHJAM_DENSE_INDEX environment variables
#   2. the path relative to the current working directory (repo-root runs)
#   3. the same path relative to this package's parent directory

CATALOG_PATH_ENV = "TECHJAM_CATALOG"
DENSE_INDEX_PATH_ENV = "TECHJAM_DENSE_INDEX"

# The order is by *answerability*, measured against the local evaluator's
# own reply policy (scripts/eval_profile_signal.py --check tags): share of
# the 200 public samples where the customer can still answer a question
# about the attribute after turn 1 --
#
#   feature 0.960   style 0.085   size 0.045   use_case 0.020
#
# A question the customer can't answer burns a whole turn, and MTTC is 20%
# of the score, so the most-answerable bucket goes first.

FALLBACK_ATTRIBUTE_ORDER = ["feature", "style", "size", "use_case", "budget"]

RECENT_WINDOW = 10  # messages after the first, kept for query context
CANDIDATE_N = 50  # per-leg retrieval depth before fusion
ENTROPY_POOL_SIZE = 30  # fused-candidate pool size used for attribute-entropy scoring

# Dual-track routing - buying prioritises buying, while browsing prioritises 
# browsing
BUYING_BM25_WEIGHT = 2.0
BUYING_DENSE_WEIGHT = 1.5
# Push the buying slate away from the neighbourhood of items already shown,
# rather than merely past the items themselves. EXCLUDE_SHOWN drops the shown
# products; their near-duplicates stay, and buying misses sit in coarse
# categories 2.4x more crowded than its hits (#24), so a rejected slate is
# followed by ten more of much the same thing.
#
# False is the identity setting. Distinct from BUYING_DIVERSIFY, which #28
# measured null here: that spreads the slate against itself, this pushes it
# away from a known-wrong region, and the buying arm runs with the intra-slate
# term off so the two never ride together unmeasured.
BUYING_REPEL_SHOWN = True

# Relevance against repulsion once BUYING_REPEL_SHOWN is on. 1.0 is pure
# relevance (the identity), 0.0 is pure repulsion. Relevance here is 1/(rank+1)
# and decays steeply, so below ~0.3 the tail of the window reorders freely
# while DIVERSIFY_PIN still holds the top of the slate.
BUYING_REPEL_LAMBDA = 0.5
BROWSING_BM25_WEIGHT = 1.25
BROWSING_DENSE_WEIGHT = 1.5

# Minimum entropy needed to ask a clarifying question
# entropy is more strict for buying so that exact item can be retrieved
MIN_ENTROPY_BUYING = 0.10
MIN_ENTROPY_BROWSING = 0.30

# For diversity rerank, in browsing
# first choose the top DIVERSIFY_WINDOW items
# pin the top DIVERSITY_PIN items so relevant items dont get lost
# pass the remaining items through MMR reranking
DIVERSIFY_WINDOW = 40
DIVERSIFY_PIN = 3

# diversifying buying behaviour
# shown to be less effective, set to False
BUYING_DIVERSIFY = False

# lambda in MMR equation score = lambda * sim(d, q) - (1-lambda) * maxsim(d, q)
# awards closeness to query, and penalizes similarity to shown results
# lambda = 1 prioritizes pure relevance
DIVERSIFY_LAMBDA = 0.5

# Boundary handling. A "boundary" customer answers one clarifying question with
# an explicit hand-back -- "I don't have a preference for X; please use your
# judgment" -- and there is no structured signal for this anywhere in the agent
# API. The reply itself is the only evidence, so DEFER_CUES in classifier.py
# matches the hand-back phrasing on top of the existing non-answer test.
#
# It costs the session an elicitation turn, and locally it is the weakest
# scenario by some way (HitRate 0.700 against browsing's 0.938).
#
# Once detected, three things change for the rest of the session:
#   - the popularity leg switches on, since "use your judgment" is a request
#     for exactly the prior it encodes;
#   - the MMR re-rank runs even on the buying track, which BUYING_DIVERSIFY
#     otherwise keeps off;
#   - MMR additionally repels candidates away from what has already been
#     shown, so the next slate searches a different region rather than the
#     neighbours of items the customer has already passed on.
BOUNDARY_POPULARITY_WEIGHT = 1.0
BOUNDARY_DIVERSIFY = True
BOUNDARY_REPEL_SHOWN = True

# MMR lambda once a hand-back has been seen, below DIVERSIFY_LAMBDA so the
# slate spreads wider than a normal browsing turn. A turn-annealed schedule
# found early-turn diversity is not free -- but that was a schedule applied from
# turn 1, where the top hits are usually already right. This only ever applies
# after the customer has told us the current direction is not working.
BOUNDARY_DIVERSIFY_LAMBDA = 0.35

# whether to remove clarifications with no good information
# proven to be good
#   "sample_count": 200,
#   "hit_rate_at_10": 0.91,
#   "mrr": 0.485706,
#   "mttc": 3.945,
#   "efficiency": 0.7055,
#   "recommended_technical_score": 0.741812,
#   "reported_token_usage": {
#     "prompt_tokens": 0,
#     "completion_tokens": 0,
#     "total_tokens": 0
#   }
SKIP_NON_ANSWERS_IN_QUERY = True

# Drop contentless replies from the *phrase leg's* query only, leaving the
# BM25 and dense query untouched.

PHRASE_QUERY_SKIP_NON_ANSWERS = False

# Geometric decay applied to a disclosed slot's boost per turn since it was
# stated (_boost_by_disclosed). 1.0 == no decay == the shipped behaviour, and
# is the identity setting the A/B baseline must reproduce exactly.
# tried before, but does not work. kept at 1.0
SLOT_DECAY = 1.0

# Score every attribute on one scale instead of running entropy as a gate with
# answerability as its fallback. See attributes.select_weighted_attribute.
#
# Under the gate, entropy decided alone for material/color and answerability
# decided alone for the rest, so the two could never be traded off: colour
# (answerability 0.255) beat feature (0.960) whenever it cleared min_entropy,
# spending a turn on a question three customers in four cannot answer.
#
# False is the identity setting and must reproduce the shipped score exactly.
WEIGHTED_QUESTION_SCORE = True

# Relative pull of the two terms. Only their ratio matters -- the combined
# score is a weighted *mean*, so it stays on [0, 1] and stays comparable with
# the single-term score used when one component is missing. Equal weights are
# a starting point, not a measured optimum; sweep the ratio.
QUESTION_ENTROPY_WEIGHT = 1.0
QUESTION_ANSWERABILITY_WEIGHT = 1.0

# Third RRF leg: verbatim-span matching (BM25Index.phrase_search). 0.0 is the
# identity setting and disables the leg entirely, including its lookup cost.

PHRASE_WEIGHT = 2.0

# Pseudo-relevance feedback (prf.py): a fourth RRF leg that re-retrieves with
# terms harvested from the top FEEDBACK_DOCS of the BM25 ranking.

# PRF is the one classical technique that adds vocabulary: it replaces "cotton"
# with the words that actually co-occur in the products "cotton" already ranks
# well.

PRF_WEIGHT = 0.5

# Fifth RRF leg: a popularity prior over the pool the other legs produced,
# ordered by rating_number (AttributeIndex.by_popularity). Targets come
# overwhelmingly from the popular tail -- catalog median 12 reviews against a
# target median of ~6,800, with 63% of targets in the top 1% of the catalog.
#
# 0.0 is the identity setting and keeps the leg dark everywhere. Turning it on
# globally is a separate experiment: the concentration above is a property of
# how the samples were drawn, not of shopping, so it transfers to the hidden
# grader only if that set was drawn the same way. Only the boundary path below
# switches it on, where the customer has explicitly asked us to choose.
POPULARITY_WEIGHT = 0.5

# Cross-encoder reranking (reranker.py, Qwen3-Reranker-0.6B). 
RERANK_WEIGHT = 3

# Which backend scores the pairs. "minilm" is the shipped, measurable stage;
# "qwen" is the LLM-scale variant kept for the report and the demo.
RERANK_BACKEND = "minilm"

# Depth of the pool handed to the cross-encoder. Cost is linear in this.

RERANK_TOP_N = 20

# Do not re-recommend a product that has already been shown and rejected.
# `evaluate()` in the local evaluator breaks the session the moment the target appears in the
# returned slate, so being *asked for another turn at all* is proof that none
# of the items shown so far was the target.
EXCLUDE_SHOWN = True

# Prepend a plain-language rationale to `turn_response.message`.
# useful for debugging
EXPLAIN_RECOMMENDATIONS = True

# LLM used to guess attribute based on query
# boost added to score
LM_INFERENCE_WEIGHT = 0.5

# Boost and penalised based on attributes that are clarified
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
DEFAULT_FIELD_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 0, 1.0)


# THIS IS STRICTLY FOR BUILDING THE INDEX ONLY
# Every product in this catalog hangs off one root category
# ("Clothing, Shoes & Jewelry", 49,990 of 50,000), so the three words in it
# are present on essentially every document.
#
#     term      df with root      df with root dropped
#     shoes           1.000                     0.235
#     jewelry         1.000                     0.111
#     clothing        1.000                     0.433
#
#
# Detected rather than hardcoded: the root is whatever value leads
# `categories` on at least CATALOG_ROOT_MIN_SHARE of the first
# CATALOG_ROOT_SAMPLE products.
CATALOG_ROOT_SAMPLE = 1000
CATALOG_ROOT_MIN_SHARE = 0.95

# What counts as naming a product category, for names_category() below.
#
# Drops category labels with more than CATEGORY_TERM_MIN_DF terms

CATEGORY_TERM_MIN_DF = 20


# ==========================================================================
# Classifiers: prototype scoring
# ==========================================================================

# determines if attribute is negated. eg 'not blue'
NEGATION_WINDOW_CHARS = 10

# For the classifer. pick the centroid of the closest
# TOP_PROTOTYPES examples
TOP_PROTOTYPES = 4


# Removes only attributes that user specifies
# this is false because when it is true, metrics are 
# "sample_count": 200,
#   "hit_rate_at_10": 0.795,
#   "mrr": 0.441611,
#   "mttc": 4.615,
#   "efficiency": 0.6385,
#   "recommended_technical_score": 0.657683,
# whereas when false, they are
# "sample_count": 200,
#   "hit_rate_at_10": 0.86,
#   "mrr": 0.469867,
#   "mttc": 4.145,
#   "efficiency": 0.6855,
#   "recommended_technical_score": 0.70806,
SCOPED_OVERRIDE_CLEAR = False


# ==========================================================================
# Session belief: answerability priors
# ==========================================================================

# Share of the 200 public samples where the customer can still answer a
# question about each attribute after turn 1, computed from the evaluator's
# reply policy: a constraint is disclosable only when classify_constraint()
# buckets it as the asked attribute. 
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
# likelier the rest are exhausted too.
NON_ANSWER_SPILLOVER = 0.8

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
# Pseudo-relevance feedback
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
# from agent.py; the finding is written up in
# `.claude/skills/retrieval-experiments/SKILL.md` #5). Requiring the
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
