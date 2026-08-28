from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from .attributes import (AttributeIndex, COLORS, FILTERABLE_ATTRIBUTES, MATERIALS,
                         budget_constraint_satisfied, extract_disclosed_value,
                         select_dynamic_attribute)
from .classifier import (
    EmbeddingIntentClassifier,
    EmbeddingNonAnswerDetector,
    EmbeddingOverrideDetector,
    classify_intent,
    classify_reply_lexically,
    detected_attributes,
)
from .session_belief import SessionBelief
from .lm_confidence import ATTRIBUTE_TEMPLATES, MaskedLMScorer, belief_for_attribute
from .retrieval import BM25Index, DenseIndex, load_embedding_model, reciprocal_rank_fusion
from .user_profile import UserProfileStore

logger = logging.getLogger(__name__)

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
_PACKAGE_PARENT = Path(__file__).resolve().parent.parent


def _resolve_data_path(relative: str, env_var: str) -> Path:
    override = os.environ.get(env_var)
    if override:
        return Path(override)
    candidate = Path(relative)
    if candidate.exists():
        return candidate
    fallback = _PACKAGE_PARENT / relative
    if fallback.exists():
        logger.debug("%s not found at %s; using %s", relative, candidate, fallback)
        return fallback
    return candidate  # let the caller raise/degrade with the conventional path

ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}

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

ATTRIBUTE_QUESTIONS = {
    "material": "What material are you looking for?",
    "color": "Do you have a color preference?",
    "style": "Any particular style or fit you'd like?",
    "size": "What size do you need?",
    "use_case": "What will you be using this for?",
    "budget": "Do you have a budget in mind?",
    "feature": "Is there a specific feature that matters most to you?",
}

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

@dataclass(frozen=True)
class RoutingParams:
    """Everything the buying/browsing label decides, resolved in one place.

    The label used to be branched on separately in `_retrieve` (fusion
    weights), `_next_attribute` (entropy threshold) and `respond`'s debug
    log, which meant the log could report weights the retrieval step wasn't
    actually using. Resolving once removes that drift by construction.
    """

    bm25_weight: float
    dense_weight: float
    min_entropy: float
    diversify: bool


def routing_params(intent_label: str, belief: SessionBelief | None = None) -> RoutingParams:
    if intent_label == "buying":
        params = RoutingParams(BUYING_BM25_WEIGHT, BUYING_DENSE_WEIGHT, MIN_ENTROPY_BUYING, diversify=False)
    else:
        params = RoutingParams(BROWSING_BM25_WEIGHT, BROWSING_DENSE_WEIGHT, MIN_ENTROPY_BROWSING, diversify=True)
    # Runtime re-orchestration (spec Pillar III, "Adaptive Orchestration"):
    # once the customer has stopped producing new constraints, no further
    # information is coming, so there is nothing left for the exploratory
    # (dense/diverse) track to earn -- commit to precision on what was
    # actually stated.
    #
    # Gated by an explicit flag, NOT by EXHAUSTED_BM25_BONUS == 0.0. An
    # earlier revision keyed only off `belief.exhausted` and set diversify
    # False unconditionally inside this branch, which meant the "all switches
    # at identity" baseline silently lost the browsing MMR re-rank and scored
    # 0.6154 instead of the shipped 0.6182. A knob whose zero value is not
    # actually the identity is a measurement trap, not just a bug.
    if BELIEF_REORCHESTRATION and belief is not None and belief.exhausted:
        params = replace(params, bm25_weight=params.bm25_weight + EXHAUSTED_BM25_BONUS,
                         diversify=False)
    return params

DIVERSIFY_LAMBDA = 0.5

# --- Within-session adaptive layer (spec Pillar III, "Self-Evolution:
# Dynamic Context Programming"; and 4.3's in-scope "slot decay over time").
# All three switches below default to the pre-existing behaviour so each can
# be A/B'd independently against a bit-reproducible baseline.

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

# Prepend a plain-language rationale to `turn_response.message` (spec's
# "transparent recommendation explanations"). Score-neutral by construction --
# the evaluator never reads the message back -- so this is on by default and
# judged on the demo, not the benchmark.
EXPLAIN_RECOMMENDATIONS = True

# Long-term profile carry-over (user_profile.py): a profile_key is a content
# hash of an anonymized profile dict with no customer id, so two sessions
# sharing a key can be a coincidental template collision, not an actual
# returning shopper (125/200 public-set profiles are unique -- collisions do
# happen). SessionState.profile_hint carries the corroborated cross-session
# value (see user_profile.py's MIN_CORROBORATION) but is intentionally NOT
# consulted anywhere in retrieval/question logic below -- three different
# uses of it were each tried and measured to regress the full 200-sample
# public set: biasing candidate ranking (HitRate 0.755->0.745, TechnicalScore
# 0.601->0.585, weight-tuning down to 0.25 didn't help -- a wrong guess sinks
# the true target below every neutral candidate outright); and using it to
# skip a "likely-redundant" question (0.755->0.740, 0.601->0.589) -- that
# skip isn't actually bounded to "waste one turn" as it first appears: the
# exclusion persists for the whole session like a genuine disclosure would,
# so a wrong corroborated guess can permanently block ever asking about that
# attribute, even when the remaining candidate pool would make it the most
# informative possible question. The root cause under all three is that a
# shared profile_key simply isn't an identity signal on this data: same-key
# sessions' targets agree on coarse category 0.5% of the time vs a 1.2%
# random baseline (scripts/eval_profile_signal.py --check collision), so
# there is nothing correct to carry. See CLAUDE.md #5 for the full write-up.


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
ATTRIBUTE_VOCAB = {"material": MATERIALS, "color": COLORS}


@dataclass
class SessionState:
    first_message: str | None = None
    recent_messages: list[str] = field(default_factory=list)
    asked_attributes: list[str] = field(default_factory=list)
    # attribute -> extracted value, from a message THIS session (asked or
    # volunteered). Used to filter/rerank the candidate pool in _retrieve,
    # not just as extra query text. Never seeded from cross-session history --
    # see profile_hint below for that -- so a fresh (never-before-seen)
    # profile session behaves identically to before user_profile.py existed.
    disclosed: dict[str, str] = field(default_factory=dict)
    # attribute -> corroborated value (recurred >=2x, see user_profile.py's
    # MIN_CORROBORATION) seen for this profile_key in *past* sessions.
    # Populated on every reset(), read by nothing -- deliberately. A
    # profile-key match turns out not to be an identity signal at all: over
    # the 409 public-set session pairs sharing a key, 0.5% want a target in
    # the same coarse category against a 1.2% +- 0.5% random baseline
    # (scripts/eval_profile_signal.py). Every way of consuming this that was
    # tried regressed the full set; see _next_attribute below and CLAUDE.md
    # #5. Kept populated so the store stays exercised and the diagnostic has
    # something to check if the hidden set's profiles behave differently.
    profile_hint: dict[str, str] = field(default_factory=dict)
    # identity for the long-term profile store; set once in reset().
    profile_key: str | None = None
    profile_session_index: int = 0
    # Turn each disclosed attribute was stated on, for SLOT_DECAY. Parallel to
    # `disclosed` rather than folded into it: `disclosed` is passed straight
    # through to _retrieve and _infer_attributes, and keeping it a plain
    # attribute->value mapping keeps those signatures unchanged.
    disclosed_turn: dict[str, int] = field(default_factory=dict)
    # Running within-session estimate of what this customer can still answer.
    belief: SessionBelief = field(default_factory=SessionBelief)
    # The attribute asked last turn, whose outcome this turn's reply reveals.
    pending_ask: str | None = None


MAX_QUERY_CHARS = 2000


def _build_query(state: SessionState) -> str:
    # Recent messages carry the freshest signal (and matter most for intent
    # override); if the budget is tight, trim the first message first, and
    # only cut into `recent` (keeping its tail -- the most recent text) if
    # recent alone still exceeds the cap.
    recent = " ".join(state.recent_messages)[-MAX_QUERY_CHARS:]
    first = state.first_message or ""
    budget_for_first = max(0, MAX_QUERY_CHARS - len(recent) - 1)
    parts = [first[:budget_for_first], recent]
    return " ".join(part for part in parts if part)


def _next_attribute(
    state: SessionState,
    query: str,
    candidate_pool: list[str],
    attribute_index: AttributeIndex,
    intent_label: str,
    belief: SessionBelief | None = None,
) -> str | None:
    # Skip attributes already evidenced in the accumulated text (e.g. a
    # buying session's turn-1 message names a material) -- asking about it
    # again just wastes a turn, since the evaluator's customer policy only
    # reveals values not already marked "disclosed". Deliberately NOT
    # excluding (or boosting ranking with) state.profile_hint -- see its
    # comment on SessionState and user_profile.py's module docstring: three
    # different ways of using a cross-session profile_hint were each tried
    # and measured to regress the full 200-sample public set, because a
    # profile_key match is frequently a coincidental template collision
    # rather than a genuine returning shopper, and a wrong guess is costly
    # however it's used (sinks true-target ranking, or -- worse than
    # expected -- permanently blinds the session to ever asking about that
    # attribute, since exclusion here would persist for the rest of the
    # session same as a genuine disclosure).
    excluded = set(state.asked_attributes) | detected_attributes(query)

    dynamic_pick = select_dynamic_attribute(
        candidate_pool, attribute_index, excluded, min_entropy=routing_params(intent_label).min_entropy
    )
    if dynamic_pick is not None:
        return dynamic_pick

    # Nothing structural is worth asking about (either excluded, or the
    # current pool already agrees on material/color/budget) -- fall back to
    # attributes we can't score entropy for, most-answerable first.
    #
    # The belief is initialized to exactly FALLBACK_ATTRIBUTE_ORDER's measured
    # marginals (session_belief.ANSWERABILITY_PRIOR), so on turn 1 the two
    # branches below agree by construction -- this generalizes the #9 reorder
    # to react to what this customer actually does, it does not replace it.
    if BELIEF_DRIVEN_QUESTIONS and belief is not None:
        return belief.best(excluded)
    for attribute in FALLBACK_ATTRIBUTE_ORDER:
        if attribute not in excluded:
            return attribute
    return None


class Agent:
    """Dual-track BM25 + dense (bge-small) retrieval, routed by buying/browsing
    intent: both tracks fuse the same two legs via reciprocal rank fusion,
    but buying weights BM25 heavier (precision) and browsing weights dense
    heavier (semantic breadth) plus an MMR diversity re-rank on top of the
    fused pool (see _retrieve/_diversify)."""

    def __init__(
        self,
        catalog_path: str | Path | None = None,
        dense_index_dir: str | Path | None = None,
        user_profile_store: UserProfileStore | None = None,
    ) -> None:
        catalog_path = (Path(catalog_path) if catalog_path is not None
                        else _resolve_data_path("data/catalog.jsonl", CATALOG_PATH_ENV))
        dense_index_dir = (dense_index_dir if dense_index_dir is not None
                           else _resolve_data_path("data/dense_index", DENSE_INDEX_PATH_ENV))
        self.profile_store = user_profile_store if user_profile_store is not None else UserProfileStore()
        self.bm25 = BM25Index(catalog_path)
        self.attribute_index = AttributeIndex.build(catalog_path)

        embedding_model = None
        try:
            embedding_model = load_embedding_model()
        except Exception:
            logger.exception("failed to load embedding model; dense retrieval and the "
                              "embedding intent classifier will both fall back")

        self.dense: DenseIndex | None = None
        if embedding_model is not None:
            try:
                self.dense = DenseIndex(Path(dense_index_dir), model=embedding_model, catalog_path=catalog_path)
            except Exception as exc:
                # Broad on purpose: a corrupt/partial embeddings.npy or
                # meta.json can raise ValueError/OSError/json.JSONDecodeError,
                # not just the RuntimeError this code used to only catch --
                # any of them should degrade to BM25-only, not crash Agent init.
                logger.error("dense index unavailable, falling back to BM25-only: %s", exc)

        self.lm_scorer: MaskedLMScorer | None = None
        try:
            self.lm_scorer = MaskedLMScorer()
        except Exception as exc:
            # Same degrade-don't-crash policy as the dense index: a missing or
            # corrupt local LM must cost the inference boost, not the agent.
            logger.error("masked LM unavailable, attribute inference disabled: %s", exc)

        self.intent_classifier: EmbeddingIntentClassifier | None = None
        self.override_detector: EmbeddingOverrideDetector | None = None
        self.non_answer_detector: EmbeddingNonAnswerDetector | None = None
        if embedding_model is not None:
            self.intent_classifier = EmbeddingIntentClassifier(embedding_model)
            self.override_detector = EmbeddingOverrideDetector(embedding_model)
            self.non_answer_detector = EmbeddingNonAnswerDetector(embedding_model)

        self._sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(user_profile, dict):
            logger.warning("reset(%s): user_profile is not a dict (%r); ignoring", session_id, type(user_profile))
            user_profile = {}
        # user_profile is anonymized aggregate data (purchase_frequency, rating
        # style, preference_tags, summary) with no customer id -- see
        # user_profile.py for how this is turned into a long-term profile key.
        key, session_index, carried = self.profile_store.start_session(user_profile)
        state = SessionState(profile_key=key, profile_session_index=session_index, profile_hint=carried)
        self._sessions[session_id] = state
        logger.debug(
            "reset(%s): profile_key=%s session_index=%d carried=%s",
            session_id, key, session_index, carried,
        )

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        if not isinstance(user_message, str):
            logger.warning("respond(%s, turn=%r): user_message is not a str (%r); coercing", session_id, turn, type(user_message))
            user_message = "" if user_message is None else str(user_message)
        if not isinstance(top_k, int) or top_k <= 0:
            logger.warning("respond(%s): invalid top_k=%r; defaulting to 10", session_id, top_k)
            top_k = 10

        state = self._sessions[session_id]
        if state.first_message is None:
            state.first_message = user_message
        else:
            if self.override_detector is not None and self.override_detector.is_override(user_message):
                logger.debug("respond(%s, turn=%s): override detected, clearing disclosed/asked state", session_id, turn)
                state.disclosed.clear()
                state.disclosed_turn.clear()
                state.profile_hint.clear()
                state.asked_attributes.clear()
            # Observe what last turn's question actually bought us -- the
            # feedback signal the agent has never had. `asked_attributes`
            # records that a question was asked; nothing recorded whether it
            # was answered, and extract_disclosed_value covers only
            # material/color/budget, so an answer about style or feature is
            # invisible to it either way.
            contentless = self._is_non_answer(user_message)
            state.belief.observe(state.pending_ask, was_answered=not contentless)
            # A contentless reply carries no constraint but real query noise:
            # _build_query joins recent_messages into the BM25 and dense query,
            # so "I don't have an additional preference for color." would be
            # searched against the catalog as if the customer had said it.
            if not (contentless and SKIP_NON_ANSWERS_IN_QUERY):
                state.recent_messages.append(user_message)
                del state.recent_messages[:-RECENT_WINDOW]

        for attribute in FILTERABLE_ATTRIBUTES:
            value = extract_disclosed_value(attribute, user_message)
            if value is not None:
                state.disclosed[attribute] = value
                state.disclosed_turn[attribute] = turn if isinstance(turn, int) else 1
                self.profile_store.record_disclosure(state.profile_key, state.profile_session_index, attribute, value)

        query = _build_query(state)

        signal = self.intent_classifier.classify(query) if self.intent_classifier is not None else classify_intent(query)
        routing = routing_params(signal.label, state.belief)
        logger.debug(
            "respond(%s, turn=%s): intent=%s score=%+.2f bm25_w=%.2f dense_w=%.2f mmr=%s min_entropy=%.2f asked=%s",
            session_id, turn, signal.label, signal.score,
            routing.bm25_weight, routing.dense_weight, routing.diversify, routing.min_entropy,
            state.asked_attributes,
        )

        inferred = self._infer_attributes(query, state.disclosed)
        if inferred:
            logger.debug("respond(%s, turn=%s): LM-inferred %s", session_id, turn, inferred)
        candidate_pool = self._retrieve(
            query, top_k, max(top_k, ENTROPY_POOL_SIZE), signal.label, state.disclosed, inferred,
            state.disclosed_turn, turn if isinstance(turn, int) else 1, state.belief,
        )
        recommendations = candidate_pool[:top_k]

        attribute = _next_attribute(state, query, candidate_pool, self.attribute_index,
                                    signal.label, state.belief)
        if attribute is not None:
            state.asked_attributes.append(attribute)
            message = ATTRIBUTE_QUESTIONS[attribute]
        else:
            message = "Here are the closest matches I found so far."
        state.pending_ask = attribute

        if EXPLAIN_RECOMMENDATIONS:
            try:
                rationale = self._explain(state, signal, routing, attribute, len(recommendations))
            except Exception:
                # Never let the rationale cost a turn: evaluate() blanks the
                # whole response if `message` is not a str.
                logger.exception("explanation failed; falling back to the plain message")
            else:
                message = f"{rationale} {message}" if attribute is not None else rationale

        if attribute is not None and attribute not in ALLOWED_ATTRIBUTES:
            # docs/agent_api_contract.json constrains ask_attribute to an
            # enum; an out-of-enum value would invalidate the turn. An
            # assert would be stripped under `python -O`, so this is a real
            # check: drop the question rather than emit an invalid field.
            logger.error("respond(%s, turn=%s): dropping out-of-enum ask_attribute %r",
                         session_id, turn, attribute)
            attribute = None
            message = "Here are the closest matches I found so far."
        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": [{"parent_asin": pid} for pid in recommendations],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def _infer_attributes(self, query: str, disclosed: dict[str, str]) -> dict[str, str]:
        """Attributes the customer has not stated, that the LM is confident about.

        Only ever fills gaps: an attribute already in `disclosed` is skipped
        outright, so a genuine answer is never second-guessed by a guess.
        """
        if self.lm_scorer is None or not query.strip():
            return {}
        inferred: dict[str, str] = {}
        for attribute in ATTRIBUTE_TEMPLATES:
            if attribute in disclosed:
                continue
            vocab = ATTRIBUTE_VOCAB.get(attribute)
            if not vocab:
                continue
            try:
                belief = belief_for_attribute(self.lm_scorer, attribute, query, vocab)
            except Exception:
                logger.exception("LM inference failed for %s", attribute)
                continue
            if belief is not None and belief.confident:
                inferred[attribute] = belief.value
        return inferred

    def _retrieve(
        self,
        query: str,
        top_k: int,
        pool_size: int,
        intent_label: str,
        disclosed: dict[str, str] | None = None,
        inferred: dict[str, str] | None = None,
        disclosed_turn: dict[str, int] | None = None,
        turn: int = 1,
        belief: SessionBelief | None = None,
    ) -> list[str]:
        if not query.strip():
            return []
        try:
            bm25_ranked = self.bm25.search(query, CANDIDATE_N)
        except Exception:
            logger.exception("BM25 search failed for query %r", query)
            bm25_ranked = []
        dense_ranked: list[str] = []
        if self.dense is not None:
            try:
                dense_ranked = self.dense.search(query, CANDIDATE_N)
            except Exception:
                logger.exception("dense search failed for query %r", query)

        routing = routing_params(intent_label, belief)
        legs = [(bm25_ranked, routing.bm25_weight), (dense_ranked, routing.dense_weight)]
        rank_lists = [ranked for ranked, _ in legs if ranked]
        weights = [weight for ranked, weight in legs if ranked]
        if not rank_lists:
            return []
        fused = reciprocal_rank_fusion(rank_lists, top_n=pool_size, weights=weights)

        candidates = (self._boost_by_disclosed(fused, disclosed, inferred, disclosed_turn, turn)
                      if (disclosed or inferred) else fused)

        if routing.diversify:
            return self._diversify(candidates, top_k)
        return candidates

    def _diversify(self, candidate_ids: list[str], head_n: int) -> list[str]:
        """Greedy MMR re-rank of the pool's front, browsing-track only.

        Only the front `head_n` (what respond() truncates the pool to for
        actual recommendations) gets reordered for diversity; the full pool
        is returned at its original length since _next_attribute's entropy
        scoring downstream needs the whole pool, not just the head, and
        entropy doesn't care about ordering. Two fixed safeguards keep
        diversity from ever beating relevance outright: the top
        DIVERSIFY_PIN candidates are kept unconditionally, and MMR only ever
        chooses among the top DIVERSIFY_WINDOW of the pool (never trading a
        top-3 item for something ranked #45 just for variety).
        """
        if self.dense is None or len(candidate_ids) <= head_n:
            return candidate_ids

        window = candidate_ids[:DIVERSIFY_WINDOW]
        known_ids, vectors = self.dense.vectors_for(window)
        if len(known_ids) <= DIVERSIFY_PIN:
            return candidate_ids

        # known_ids preserves window order, so its index doubles as a
        # relevance-rank proxy (earlier = more relevant).
        relevance = [1.0 / (rank + 1) for rank in range(len(known_ids))]
        sim = vectors @ vectors.T  # bge embeddings are L2-normalized: dot product == cosine

        selected = list(range(DIVERSIFY_PIN))
        remaining = [i for i in range(len(known_ids)) if i not in selected]
        while remaining and len(selected) < head_n:
            best_idx, best_score = None, -float("inf")
            for idx in remaining:
                max_sim = max((sim[idx, s] for s in selected), default=0.0)
                score = DIVERSIFY_LAMBDA * relevance[idx] - (1 - DIVERSIFY_LAMBDA) * max_sim
                if score > best_score:
                    best_score, best_idx = score, idx
            selected.append(best_idx)
            remaining.remove(best_idx)

        picked_ids = [known_ids[i] for i in selected]
        picked_set = set(picked_ids)
        remainder = [cid for cid in candidate_ids if cid not in picked_set]
        return picked_ids + remainder

    def _explain(self, state: SessionState, signal, routing: RoutingParams,
                 attribute: str | None, n_recs: int) -> str:
        """A one-line rationale for this turn's slate.

        The spec lists "transparent recommendation explanations" as an
        Innovation Direction and the contract leaves exactly one place to put
        one: `turn_response.message` is a free string, while the response
        object is additionalProperties:false and each recommendation admits
        only parent_asin and score.

        Score-neutral by construction -- the evaluator validates that
        `message` is a str and then never reads it; `customer_reply()` keys
        only off `ask_attribute`. So this is judged on the demo and the
        report, not on TechnicalScore, and it must never raise: a non-str
        `message` makes evaluate() blank the entire turn.
        """
        parts = []
        if state.disclosed:
            parts.append("matching on " + ", ".join(
                f"{a}={v}" for a, v in sorted(state.disclosed.items())))
        else:
            parts.append("no hard constraints stated yet")
        parts.append(f"{signal.label} track")
        if routing.diversify:
            parts.append("showing a deliberately varied slate")
        if state.belief.exhausted:
            parts.append("you've told me all you need to, so this is my best match on what I have")
        elif attribute is not None:
            parts.append(f"asking about {attribute} because it still splits these {n_recs} options")
        return "Here are the closest matches. (" + "; ".join(parts) + ".)"

    def _is_non_answer(self, text: str) -> bool:
        """Did this reply decline to state a preference?

        Embedding rule when available, lexical floor otherwise -- the offline
        degrade is silent (CLAUDE.md #15), so a component with no fallback
        just disappears. The error asymmetry runs opposite to the override
        detector's: a false positive here discards a *real* disclosure, while
        a false negative only leaves the previous behaviour in place, so both
        rules are tuned toward precision and concrete attribute vocabulary
        vetoes a non-answer verdict outright.
        """
        if self.non_answer_detector is not None:
            return self.non_answer_detector.is_non_answer(text)
        return classify_reply_lexically(text) == "non_answer"

    def _boost_by_disclosed(
        self,
        candidate_ids: list[str],
        disclosed: dict[str, str],
        inferred: dict[str, str] | None = None,
        disclosed_turn: dict[str, int] | None = None,
        turn: int = 1,
    ) -> list[str]:
        # Non-eliminating by construction: a stable resort, so nothing is
        # ever dropped -- unlike the hard-equality filter this replaced,
        # which compared two independent single-value, first-vocab-hit
        # extractions (customer text vs. catalog text) for string equality
        # and dropped the candidate on any disagreement. Measured directly:
        # that disagreed with the *true target* product 16.3% of the time on
        # material and 37.0% on color, so the hard veto was silently
        # eliminating the right answer that often.
        #
        # (An embedding-similarity boost -- scoring the disclosed phrase
        # against catalog embeddings instead of this categorical match --
        # was tried and measured worse across the board on a 60-sample
        # subset, monotonically so as its weight increased: 0.604 at
        # weight=0 down to 0.546 at weight=1.5, both below this categorical
        # approach's 0.618 at the old hard filter and matching its ~0.60+
        # here. A short disclosed phrase dotted against a full-product
        # embedding is too diluted a signal; exact vocab agreement, even
        # imperfect, is the stronger one here.)
        def slot_weight(attribute: str) -> float:
            # Geometric decay in turns since the value was stated (spec 4.3,
            # "slot decay over time"). SLOT_DECAY == 1.0 makes this the
            # constant 1.0 and the whole resort bit-identical to before.
            if SLOT_DECAY >= 1.0 or not disclosed_turn:
                return 1.0
            stated = disclosed_turn.get(attribute)
            if stated is None:
                return 1.0
            return SLOT_DECAY ** max(0, turn - stated)

        def match_score(pid: str) -> float:
            score = 0.0
            for attribute, value in disclosed.items():
                weight = slot_weight(attribute)
                if attribute == "budget":
                    # A budget is a *bound*, not a value to match: "under $80"
                    # is satisfied by every product at or below $80, not only
                    # by those in $80's price bucket. Bucket equality here
                    # used to boost the priciest bucket for a stated ceiling
                    # and penalize everything cheaper. An unpriced product
                    # (79.2% of the catalog) is neutral, exactly like an
                    # unextractable material or colour.
                    price = self.attribute_index.price_for(pid)
                    if price is None:
                        continue
                    satisfied = budget_constraint_satisfied(value, price)
                    if satisfied is None:
                        continue
                    score += weight * (DISCLOSED_MATCH_BOOST if satisfied
                                       else -DISCLOSED_MISMATCH_PENALTY)
                    continue
                actual = self.attribute_index.value_for(attribute, pid)
                if actual is None:
                    continue
                score += weight * (DISCLOSED_MATCH_BOOST if actual == value
                                   else -DISCLOSED_MISMATCH_PENALTY)
            for attribute, value in (inferred or {}).items():
                # Boost-only: agreement lifts, disagreement is ignored rather
                # than penalized. See LM_INFERENCE_WEIGHT above for why the
                # asymmetry is load-bearing and not just caution.
                if self.attribute_index.value_for(attribute, pid) == value:
                    score += LM_INFERENCE_WEIGHT
            return score

        return sorted(candidate_ids, key=match_score, reverse=True)
