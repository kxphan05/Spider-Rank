from __future__ import annotations

from pathlib import Path



# ==========================================================================
# Agent: routing, fusion weights, candidate pools, feature flags
# ==========================================================================

# Data-asset location: TECHJAM_CATALOG / TECHJAM_DENSE_INDEX env vars, else
# path relative to cwd (repo-root runs), else relative to this package's
# parent directory.

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

# Diversity rerank (browsing): take top DIVERSIFY_WINDOW items, pin the top
# DIVERSITY_PIN so relevant items don't get lost, MMR the rest.
DIVERSIFY_WINDOW = 40
DIVERSIFY_PIN = 3

# diversifying buying behaviour
# shown to be less effective, set to False
BUYING_DIVERSIFY = False

# lambda in MMR equation score = lambda * sim(d, q) - (1-lambda) * maxsim(d, q)
# awards closeness to query, and penalizes similarity to shown results
# lambda = 1 prioritizes pure relevance
DIVERSIFY_LAMBDA = 0.5

# Boundary handling: a customer answers a clarifying question with an
# explicit hand-back ("use your judgment"), detected via DEFER_CUES in
# classifier.py. Once detected: popularity leg turns on, MMR re-rank runs
# even on the buying track, and MMR repels away from already-shown items.
BOUNDARY_POPULARITY_WEIGHT = 1.0
BOUNDARY_DIVERSIFY = True
BOUNDARY_REPEL_SHOWN = True

# MMR lambda once a hand-back has been seen, below DIVERSIFY_LAMBDA so the
# slate spreads wider than a normal browsing turn.
BOUNDARY_DIVERSIFY_LAMBDA = 0.35

# Whether to remove non-answer clarifications from the query.
# Measured to help: technical score 0.7418 vs. False's baseline.
SKIP_NON_ANSWERS_IN_QUERY = True

# Drop contentless replies from the *phrase leg's* query only, leaving the
# BM25 and dense query untouched.

PHRASE_QUERY_SKIP_NON_ANSWERS = False

# Geometric decay applied to a disclosed slot's boost per turn since it was
# stated (_boost_by_disclosed). 1.0 == no decay == the shipped behaviour, and
# is the identity setting the A/B baseline must reproduce exactly.
# tried before, but does not work. kept at 1.0
SLOT_DECAY = 1.0

# Score every attribute on one scale instead of running entropy as a gate
# with answerability as its fallback. See attributes.select_weighted_attribute.
# False is the identity setting and must reproduce the shipped score exactly.
WEIGHTED_QUESTION_SCORE = True

# Relative pull of the two terms (only their ratio matters -- it's a
# weighted mean, kept on [0, 1]). Equal weights are a starting point, not a measured optimum.
QUESTION_ENTROPY_WEIGHT = 1.0
QUESTION_ANSWERABILITY_WEIGHT = 1.0

# Third RRF leg: verbatim-span matching (BM25Index.phrase_search). 0.0 is the
# identity setting and disables the leg entirely, including its lookup cost.

PHRASE_WEIGHT = 2.0

# Pseudo-relevance feedback (prf.py): a fourth RRF leg that re-retrieves with
# terms harvested from the top FEEDBACK_DOCS of the BM25 ranking -- adds
# vocabulary by replacing "cotton" with words that co-occur in its top hits.
PRF_WEIGHT = 0.5

# Fifth RRF leg: a popularity prior over the pool the other legs produced,
# ordered by rating_number. Targets skew heavily popular, but 0.0 keeps this
# off everywhere except the boundary path, where the customer asked us to choose.
POPULARITY_WEIGHT = 0.5

# Cross-encoder reranking
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

# Verbatim-span matching parameters (BM25Index.phrase_search). Swept in
# scripts/sweep_phrase_max_n.py -- 6 measured +0.0062 Technical over 5 on
# the full 200-sample set. See retrieval-experiments skill entry 7.
PHRASE_MIN_N = 2
PHRASE_MAX_N = 6

# Per-query budget on phrase lookups. Each is an indexed FTS5 phrase scan, so
# they are cheap, but an accumulated 10-turn query has a long token tail.
MAX_PHRASE_QUERIES = 24

# A span appearing in more than this many products carries no identifying
# information, so it is dropped rather than scored.
PHRASE_MAX_MATCHES = 150


# BM25F field weights: parent_asin (UNINDEXED, inert), title, categories,
# features, details, store, description. Hand-picked, never swept --
# `scripts/sweep_bm25_fields.py` sweeps them; this is the identity baseline.
DEFAULT_FIELD_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 0, 1.0)


# Index-build only. Every product hangs off one root category ("Clothing,
# Shoes & Jewelry" on 49,990/50,000), so it's stripped when it leads
# `categories` on at least CATALOG_ROOT_MIN_SHARE of sampled products.
CATALOG_ROOT_SAMPLE = 1000
CATALOG_ROOT_MIN_SHARE = 0.95

# What counts as naming a product category, for names_category() below --
# drops category labels below CATEGORY_TERM_MIN_DF document frequency.
CATEGORY_TERM_MIN_DF = 20


# ==========================================================================
# Classifiers: prototype scoring
# ==========================================================================

# Negation window (chars), e.g. detecting 'not blue'. 10 was too short to
# bridge "not ... to buy" once words sit between negation and verb; 20 fixes
# that and measured bit-identical TechnicalScore elsewhere.
NEGATION_WINDOW_CHARS = 20

# For the classifer. pick the centroid of the closest
# TOP_PROTOTYPES examples
TOP_PROTOTYPES = 4

# EmbeddingNonAnswerDetector: one-class threshold on similarity to
# PROTOTYPE_NON_ANSWER (a decline is a small, closed class; "informative" is
# not, so it's classified by an absolute floor rather than a nearest-class
# contest -- see scripts/sweep_nonanswer_threshold.py). Chosen as the lowest
# threshold with 100% recall on the deterministic simulator templates
# (including the "ask something more specific" filler line, which needed its
# own prototype coverage -- see PROTOTYPE_NON_ANSWER), 100% on held-out OOD
# decline paraphrases, and zero false positives on the terse-catalog-fragment
# regression case, at 99.95% precision against ~2k diverse catalog-derived
# answer strings.
NON_ANSWER_THRESHOLD = 0.68


# Removes only attributes the user specifies (vs. clearing all on override).
# False measured better: technical score 0.708 vs. True's 0.658.
SCOPED_OVERRIDE_CLEAR = False


# ==========================================================================
# Session belief: answerability priors
# ==========================================================================

# Share of the 200 public samples where the customer can still answer a
# question about each attribute after turn 1 (from the evaluator's reply policy).
ANSWERABILITY_PRIOR: dict[str, float] = {
    "feature": 0.960,
    "material": 0.725,
    "color": 0.255,
    "style": 0.085,
    "size": 0.045,
    "use_case": 0.020,
    "budget": 0.001,
}

# Odds ratio for P(B answerable | A answered) vs P(B a non-answer), measured
# over the 200 public samples. Direction isn't uniform: material answered
# predicts others 5-9x LESS likely; among the minor attributes it's the reverse.
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
