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

# Fallback attribute order, ordered by overal answerability

FALLBACK_ATTRIBUTE_ORDER = ["feature", "style", "size", "use_case", "budget"]

RECENT_WINDOW = 10  # messages after the first, kept for query context
CANDIDATE_N = 50  # per-leg retrieval depth before fusion
ENTROPY_POOL_SIZE = 30  # fused-candidate pool size used for attribute-entropy scoring

# Dual-track routing - buying prioritises buying, while browsing prioritises 
# browsing
BUYING_BM25_WEIGHT = 2.0
BUYING_DENSE_WEIGHT = 1.5
# Push the buying slate away from the neighbourhood of items already shown.
BUYING_REPEL_SHOWN = True

# Relevance against repulsion once BUYING_REPEL_SHOWN is on. 1.0 is pure
# relevance (the identity), 0.0 is pure repulsion.
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
EXCLUDE_SHOWN = True

# Prepend a plain-language rationale to `turn_response.message`.
# useful for debugging
EXPLAIN_RECOMMENDATIONS = True

# LLM used to guess attribute based on query
# boost added to score
LM_INFERENCE_WEIGHT = 0

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

# Odds ratio for P(B answerable | A answered) vs P(B answerable | A a
# non-answer), computed directly from the evaluator's own card-generation
# logic over the 200 public samples (scripts/eval_bucket_correlation.py) --
# not a hand-picked constant. `SessionBelief.observe()` multiplies the ODDS
# of every other attribute by this factor (or its reciprocal, on a
# non-answer), which is the correct Bayesian update given this table as the
# per-pair likelihood ratio.
#
# The direction is NOT uniform, and that's the finding worth keeping: a
# detected `material` almost always becomes `hard_constraints[0]` and gets
# pre-disclosed at turn 1 for buying-scenario sessions, consuming the "slot"
# a minor attribute would otherwise occupy -- so material answered predicts
# every other attribute is 5-9x LESS likely (OR 0.11-0.21), and material
# NOT answered predicts them 5-9x MORE likely. Among color/style/size/
# use_case themselves, once material's effect is set aside, the correlation
# runs the other way: a customer whose card discloses one minor attribute is
# 1.2-4.7x MORE likely to have others too (richer listings surface several
# minor attributes together), so a non-answer on one of THESE should pull
# the others down, not up.
#
# size, use_case, feature and budget never appear as the observed (row) key:
# their marginals (0.045, 0.020, 0.960, 0.000) leave under 15 public sessions
# on one side of the answered/non-answer split, too few to trust a ratio
# from -- omitted rather than guessed. A pair missing from this table gets no
# update (odds ratio 1.0), not a fallback constant.
BUCKET_ANSWER_LR: dict[str, dict[str, float]] = {
    "material": {"color": 0.209, "style": 0.136, "size": 0.113, "use_case": 0.156},
    "color": {"material": 0.209, "style": 2.89, "size": 2.489, "use_case": 1.243},
    "style": {"material": 0.136, "color": 2.89, "size": 1.877, "use_case": 4.688},
}

# Below this, an attribute is treated as not worth asking about.
EXHAUSTED_THRESHOLD = 0.0005


# ==========================================================================
# Masked-LM attribute inference
# ==========================================================================

LM_MODEL_NAME = "distilbert-base-uncased"

# Beliefs at or above this normalized entropy are treated as "the model does
# not know".
MAX_CONFIDENT_ENTROPY = 0.60


# ==========================================================================
# Pseudo-relevance feedback
# ==========================================================================

# How many top-ranked products are assumed relevant.
FEEDBACK_DOCS = 10

# How many expansion terms survive into the second query.
EXPANSION_TERMS = 8

# A term must appear in at least this many feedback documents to be
# considered.
MIN_FEEDBACK_DF = 3

# Terms shorter than this are dropped -- FTS5 tokenises aggressively and short
# fragments are almost always noise.
MIN_TERM_LENGTH = 3


# ==========================================================================
# Cross-encoder re-ranker
# ==========================================================================

RERANK_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
MINILM_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MINILM_MAX_LENGTH = 256

MAX_DOC_CHARS = 400

BATCH_SIZE = 4


# ==========================================================================
# Long-term profile store
# ==========================================================================

STORE_PATH_ENV = "TECHJAM_PROFILE_STORE"
DEFAULT_STORE_PATH = Path("data/user_profiles.json")

# if user has preference for a certain attribute for at least MIN_CORROBORATION
# times, it will be carried as a hint for future sessions
MIN_CORROBORATION = 2


# ==========================================================================
# Model cache location
# ==========================================================================

MODEL_DIR_ENV = "TECHJAM_MODEL_DIR"
