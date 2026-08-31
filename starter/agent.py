from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from .attributes import (AttributeIndex, COLORS, FILTERABLE_ATTRIBUTES, MATERIALS,
                         STRUCTURAL_ATTRIBUTES, budget_constraint_satisfied,
                         extract_disclosed_value, select_dynamic_attribute,
                         select_weighted_attribute)
from .classifier import (
    EmbeddingIntentClassifier,
    EmbeddingNonAnswerDetector,
    EmbeddingOverrideDetector,
    classify_intent,
    classify_reply_lexically,
    detected_attributes,
    matches_defer_cue,
)
from .reranker import CrossEncoderReranker, QwenReranker
from .session_belief import SessionBelief
from .lm_confidence import ATTRIBUTE_TEMPLATES, MaskedLMScorer, belief_for_attribute
from .retrieval import BM25Index, DenseIndex, load_embedding_model, reciprocal_rank_fusion
from .user_profile import UserProfileStore
from .text_utils import normalize_query
from .config import (ANSWERABILITY_PRIOR, BOUNDARY_DIVERSIFY, BOUNDARY_DIVERSIFY_LAMBDA,
                     BOUNDARY_POPULARITY_WEIGHT, BOUNDARY_REPEL_SHOWN, BUYING_REPEL_LAMBDA, BUYING_REPEL_SHOWN,
                     POPULARITY_WEIGHT, BROWSING_BM25_WEIGHT,
                     BROWSING_DENSE_WEIGHT, BUYING_BM25_WEIGHT, BUYING_DENSE_WEIGHT, BUYING_DIVERSIFY,
                     CANDIDATE_N, CATALOG_PATH_ENV, DENSE_INDEX_PATH_ENV, DISCLOSED_MATCH_BOOST,
                     DISCLOSED_MISMATCH_PENALTY, DIVERSIFY_LAMBDA, DIVERSIFY_PIN, DIVERSIFY_WINDOW,
                     ENTROPY_POOL_SIZE, EXCLUDE_SHOWN, EXPLAIN_RECOMMENDATIONS,
                     FALLBACK_ATTRIBUTE_ORDER, LM_INFERENCE_WEIGHT, MAX_QUERY_CHARS, MIN_ENTROPY_BROWSING,
                     MIN_ENTROPY_BUYING, PHRASE_QUERY_SKIP_NON_ANSWERS, PHRASE_WEIGHT, PRF_WEIGHT,
                     QUESTION_ANSWERABILITY_WEIGHT, QUESTION_ENTROPY_WEIGHT, RECENT_WINDOW,
                     RERANK_BACKEND, RERANK_TOP_N, RERANK_WEIGHT, SCOPED_OVERRIDE_CLEAR,
                     SKIP_NON_ANSWERS_IN_QUERY, SLOT_DECAY, WEIGHTED_QUESTION_SCORE)

logger = logging.getLogger(__name__)

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


ATTRIBUTE_QUESTIONS = {
    "material": "What material are you looking for?",
    "color": "Do you have a color preference?",
    "style": "Any particular style or fit you'd like?",
    "size": "What size do you need?",
    "use_case": "What will you be using this for?",
    "budget": "Do you have a budget in mind?",
    "feature": "Is there a specific feature that matters most to you?",
}


@dataclass(frozen=True)
class RoutingParams:
    """Resolved routing knobs for one turn."""

    bm25_weight: float
    dense_weight: float
    min_entropy: float
    diversify: bool
    popularity_weight: float = POPULARITY_WEIGHT
    diversify_lambda: float = DIVERSIFY_LAMBDA
    repel_shown: bool = False


def routing_params(intent_label: str, belief: SessionBelief | None = None,
                   boundary: bool = False) -> RoutingParams:
    if intent_label == "buying":
        params = RoutingParams(BUYING_BM25_WEIGHT, BUYING_DENSE_WEIGHT, MIN_ENTROPY_BUYING,
                               diversify=BUYING_DIVERSIFY)
        if BUYING_REPEL_SHOWN:
            params = replace(params, repel_shown=True,
                             diversify_lambda=BUYING_REPEL_LAMBDA)
    else:
        params = RoutingParams(BROWSING_BM25_WEIGHT, BROWSING_DENSE_WEIGHT, MIN_ENTROPY_BROWSING, diversify=True)
    if boundary:
        # No preference stated: favor variety and popular items.
        params = replace(
            params,
            popularity_weight=BOUNDARY_POPULARITY_WEIGHT,
            diversify=params.diversify or BOUNDARY_DIVERSIFY,
            diversify_lambda=BOUNDARY_DIVERSIFY_LAMBDA,
            repel_shown=BOUNDARY_REPEL_SHOWN,
        )
    return params


ATTRIBUTE_VOCAB = {"material": MATERIALS, "color": COLORS}


@dataclass
class SessionState:
    first_message: str | None = None
    recent_messages: list[str] = field(default_factory=list)
    asked_attributes: list[str] = field(default_factory=list)
    # Values the customer stated then retracted.
    retracted_values: set[str] = field(default_factory=set)
    disclosed: dict[str, str] = field(default_factory=dict)
    # Carried-over preference guesses from past sessions.
    profile_hint: dict[str, str] = field(default_factory=dict)
    profile_key: str | None = None
    profile_session_index: int = 0
    # Turn each disclosed attribute was stated on, for SLOT_DECAY.
    disclosed_turn: dict[str, int] = field(default_factory=dict)
    # Running within-session estimate of what this customer can still answer.
    belief: SessionBelief = field(default_factory=SessionBelief)
    # The attribute asked last turn, whose outcome this turn's reply reveals.
    pending_ask: str | None = None
    # Products already offered this session.
    shown: set[str] = field(default_factory=set)
    informative_messages: list[str] = field(default_factory=list)
    boundary: bool = False


def _strip_retracted(text: str, retracted: set[str]) -> str:
    """Remove withdrawn attribute values from query text, word-bounded.

    Word-bounded: a substring match would strip "lace" out of "necklace".
    """
    if not retracted:
        return text
    for value in retracted:
        text = re.sub(rf"\b{re.escape(value)}\b", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip()


def _build_query(state: SessionState) -> str:
    # Prioritise first query over clarifications
    recent = _strip_retracted(" ".join(state.recent_messages), state.retracted_values)[-MAX_QUERY_CHARS:]
    first = _strip_retracted(state.first_message or "", state.retracted_values)
    budget_for_recent = max(0, MAX_QUERY_CHARS - len(first) - 1)
    parts = [first, recent[:budget_for_recent]]
    return normalize_query(" ".join(part for part in parts if part))


def _build_phrase_query(state: SessionState) -> str:
    """Query for the phrase leg, optionally with contentless replies removed.

    Kept separate from `_build_query`: conflating the two previously cost 0.041.
    """
    if not PHRASE_QUERY_SKIP_NON_ANSWERS:
        return _build_query(state)
    recent = _strip_retracted(" ".join(state.informative_messages), state.retracted_values)[-MAX_QUERY_CHARS:]
    first = _strip_retracted(state.first_message or "", state.retracted_values)
    budget_for_recent = max(0, MAX_QUERY_CHARS - len(first) - 1)
    parts = [first, recent[:budget_for_recent]]
    return normalize_query(" ".join(part for part in parts if part))


def _next_attribute(
    state: SessionState,
    query: str,
    candidate_pool: list[str],
    attribute_index: AttributeIndex,
    intent_label: str,
    belief: SessionBelief | None = None,
) -> str | None:
    excluded = set(state.asked_attributes) | detected_attributes(query)

    # Single score trading informativeness against answerability; belief
    # decays answerability as non-answers land, else population marginals apply.
    if WEIGHTED_QUESTION_SCORE:
        answerability = dict(belief.answerable) if belief is not None else dict(ANSWERABILITY_PRIOR)
        return select_weighted_attribute(
            candidate_pool,
            attribute_index,
            excluded,
            answerability,
            entropy_weight=QUESTION_ENTROPY_WEIGHT,
            answerability_weight=QUESTION_ANSWERABILITY_WEIGHT,
            min_entropy=routing_params(intent_label).min_entropy,
        )

    # Fallback: highest-entropy attribute, or None once entropy drops below threshold.
    dynamic_pick = select_dynamic_attribute(
        candidate_pool, attribute_index, excluded, min_entropy=routing_params(intent_label).min_entropy
    )
    if dynamic_pick is not None:
        return dynamic_pick

    for attribute in FALLBACK_ATTRIBUTE_ORDER:
        if attribute not in excluded:
            return attribute
    return None


class Agent:
    """Dual-track BM25 + dense (bge-small) retrieval, routed by buying/browsing
    intent. Both tracks fuse the same legs via reciprocal rank fusion, but
    buying weights BM25 heavier and browsing weights dense heavier, plus an
    MMR diversity re-rank on the fused pool (see _retrieve/_diversify)."""

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
                logger.error("dense index unavailable, falling back to BM25-only: %s", exc)

        self.reranker: CrossEncoderReranker | QwenReranker | None = None
        if RERANK_WEIGHT > 0.0:
            try:
                self.reranker = (QwenReranker() if RERANK_BACKEND == "qwen"
                                 else CrossEncoderReranker())
            except Exception as exc:
                logger.error("cross-encoder reranker unavailable, skipping: %s", exc)

        self.lm_scorer: MaskedLMScorer | None = None
        try:
            self.lm_scorer = MaskedLMScorer()
        except Exception as exc:
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
        # Anonymized aggregate data, no customer id; see user_profile.py for
        # how it becomes a long-term profile key.
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
                # A repeated category word isn't a new category, so "never mind,
                # white shoes" after "shoes" is a scoped color change, not a
                # full restart. Without this, restating the product name (common
                # in typed input) would wrongly trigger a full reset.
                prior_text = " ".join(
                    [state.first_message or ""] + list(state.recent_messages)
                )
                prior_categories = self.bm25.names_category(prior_text)
                names_category = self.bm25.names_category(user_message) - prior_categories
                pivoted_attributes = [
                    attribute for attribute in FILTERABLE_ATTRIBUTES
                    if extract_disclosed_value(attribute, user_message) is not None
                ]
                scoped = bool(SCOPED_OVERRIDE_CLEAR and pivoted_attributes and not names_category)
                if scoped:
                    # Scoped override: erase only the named slot(s)
                    logger.debug(
                        "respond(%s, turn=%s): scoped override, clearing %s only",
                        session_id, turn, pivoted_attributes,
                    )
                    for attribute in pivoted_attributes:
                        # Record the withdrawn value first so _build_query stops
                        # searching for it.
                        superseded = state.disclosed.get(attribute)
                        replacement = extract_disclosed_value(attribute, user_message)
                        if (
                            attribute in STRUCTURAL_ATTRIBUTES
                            and superseded
                            and superseded != replacement
                        ):
                            state.retracted_values.add(superseded)
                        state.disclosed.pop(attribute, None)
                        state.disclosed_turn.pop(attribute, None)
                        state.profile_hint.pop(attribute, None)
                else:
                    logger.debug("respond(%s, turn=%s): override detected, clearing disclosed/asked state", session_id, turn)
                    state.disclosed.clear()
                    state.disclosed_turn.clear()
                    state.shown.clear()
                    state.informative_messages.clear()
                    state.profile_hint.clear()
                    state.asked_attributes.clear()
                    state.retracted_values.clear()
                    state.boundary = False
                if names_category:
                    # Full pivot to a new category: drop prior query context.
                    logger.debug(
                        "respond(%s, turn=%s): pivot names a new category %s; dropping prior query text",
                        session_id, turn, names_category,
                    )
                    state.first_message = user_message
                    state.recent_messages.clear()
            contentless = self._is_non_answer(user_message)
            # A hand-back is a non-answer that also defers to us. Gated on
            # `contentless` so "no preference, but black is fine" still counts
            # as an answer.
            if contentless and matches_defer_cue(user_message):
                logger.debug("respond(%s, turn=%s): hand-back detected, boundary mode on",
                             session_id, turn)
                state.boundary = True
            state.belief.observe(state.pending_ask, was_answered=not contentless)
            # Remove contentless query from retrieval
            if not (contentless and SKIP_NON_ANSWERS_IN_QUERY):
                state.recent_messages.append(user_message)
                del state.recent_messages[:-RECENT_WINDOW]
            if not contentless:
                state.informative_messages.append(user_message)
                del state.informative_messages[:-RECENT_WINDOW]

        for attribute in FILTERABLE_ATTRIBUTES:
            value = extract_disclosed_value(attribute, user_message)
            if value is not None:
                # Coming back to a value un-retracts it.
                state.retracted_values.discard(value)
                state.disclosed[attribute] = value
                state.disclosed_turn[attribute] = turn if isinstance(turn, int) else 1
                self.profile_store.record_disclosure(state.profile_key, state.profile_session_index, attribute, value)

        query = _build_query(state)
        phrase_query = _build_phrase_query(state)

        signal = self.intent_classifier.classify(query) if self.intent_classifier is not None else classify_intent(query)
        routing = routing_params(signal.label, state.belief, state.boundary)
        logger.debug(
            "respond(%s, turn=%s): intent=%s score=%+.2f bm25_w=%.2f dense_w=%.2f pop_w=%.2f "
            "mmr=%s lambda=%.2f repel=%s boundary=%s min_entropy=%.2f asked=%s",
            session_id, turn, signal.label, signal.score,
            routing.bm25_weight, routing.dense_weight, routing.popularity_weight,
            routing.diversify, routing.diversify_lambda, routing.repel_shown, state.boundary,
            routing.min_entropy, state.asked_attributes,
        )

        inferred = self._infer_attributes(query, state.disclosed)
        if inferred:
            logger.debug("respond(%s, turn=%s): LM-inferred %s", session_id, turn, inferred)
        # Grow the pool by len(shown) so excluding already-shown items still
        # leaves ~ENTROPY_POOL_SIZE candidates for the entropy question picker.
        pool_size = max(top_k, ENTROPY_POOL_SIZE)
        if EXCLUDE_SHOWN:
            pool_size += len(state.shown)
        candidate_pool = self._retrieve(
            query, top_k, pool_size, signal.label, state.disclosed, inferred,
            state.disclosed_turn, turn if isinstance(turn, int) else 1, state.belief,
            phrase_query, state.boundary, state.shown,
        )
        if EXCLUDE_SHOWN and state.shown:
            fresh = [pid for pid in candidate_pool if pid not in state.shown]
            # Never return a short slate -- pad with known-dead (already-shown)
            # items rather than under-fill. Only happens when the pool is exhausted.
            if len(fresh) < top_k:
                fresh += [pid for pid in candidate_pool if pid in state.shown]
            candidate_pool = fresh
        recommendations = candidate_pool[:top_k]
        state.shown.update(recommendations)

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
                # Never let the rationale cost a turn.
                logger.exception("explanation failed; falling back to the plain message")
            else:
                message = f"{rationale} {message}" if attribute is not None else rationale

        if attribute is not None and attribute not in ALLOWED_ATTRIBUTES:
            # ask_attribute must stay in-enum per docs/agent_api_contract.json.
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
        """Attributes the customer hasn't stated that the LM is confident about.

        An attribute already in `disclosed` is skipped, so a
        real answer is never second-guessed by a guess.
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
        phrase_query: str | None = None,
        boundary: bool = False,
        shown: set[str] | None = None,
    ) -> list[str]:
        if not query.strip():
            return []
        # Each leg must retrieve at least pool_size deep, or fusion can't fill
        # it -- with EXCLUDE_SHOWN the pool grows every turn.
        leg_n = max(CANDIDATE_N, pool_size * 2)
        try:
            bm25_ranked = self.bm25.search(query, leg_n)
        except Exception:
            logger.exception("BM25 search failed for query %r", query)
            bm25_ranked = []
        dense_ranked: list[str] = []
        if self.dense is not None:
            try:
                dense_ranked = self.dense.search(query, leg_n)
            except Exception:
                logger.exception("dense search failed for query %r", query)
        phrase_ranked: list[str] = []
        if PHRASE_WEIGHT > 0.0:
            try:
                phrase_ranked = self.bm25.phrase_search(
                    phrase_query if phrase_query is not None else query, leg_n)
            except Exception:
                logger.exception("phrase search failed for query %r", query)
        prf_ranked: list[str] = []
        if PRF_WEIGHT > 0.0 and bm25_ranked:
            # Seeded from BM25, not the fused pool: PRF assumes its feedback
            # docs are relevant, and BM25 is the leg that carries this
            # benchmark -- seeding from fusion would feed it dense's own picks.
            try:
                prf_ranked = self.bm25.prf_search(query, leg_n, bm25_ranked)
            except Exception:
                logger.exception("PRF search failed for query %r", query)

        routing = routing_params(intent_label, belief, boundary)
        legs = [
            (bm25_ranked, routing.bm25_weight),
            (dense_ranked, routing.dense_weight),
            (phrase_ranked, PHRASE_WEIGHT),
            (prf_ranked, PRF_WEIGHT),
        ]
        if routing.popularity_weight > 0.0:
            # Ranked over the union other legs found, never the full catalog --
            # ordering all 50k by review count would bias every session the
            # same way. Weighted RRF means an item this leg never sees just
            # scores zero, rather than being penalized.
            union = list(dict.fromkeys([*bm25_ranked, *dense_ranked, *phrase_ranked, *prf_ranked]))
            if union:
                legs.append((self.attribute_index.by_popularity(union), routing.popularity_weight))
        rank_lists = [ranked for ranked, _ in legs if ranked]
        weights = [weight for ranked, weight in legs if ranked]
        if not rank_lists:
            return []
        fused = reciprocal_rank_fusion(rank_lists, top_n=pool_size, weights=weights)

        candidates = (self._boost_by_disclosed(fused, disclosed, inferred, disclosed_turn, turn)
                      if (disclosed or inferred) else fused)

        candidates = self._rerank(query, candidates)

        # `repel` alone triggers the re-rank pass (diversify is off on buying).
        # `intra_diversity=routing.diversify` keeps behavior unchanged when
        # diversify is already on.
        repel = shown if (routing.repel_shown and shown) else None
        if routing.diversify or repel:
            return self._diversify(candidates, top_k, routing.diversify_lambda, repel,
                                   intra_diversity=routing.diversify)
        return candidates

    def _rerank(self, query: str, candidate_ids: list[str]) -> list[str]:
        """Fuse a cross-encoder ordering of the pool head into the current one.

        Runs after the attribute boost and before diversity -- reranking is a
        relevance judgement, diversity is presentation on top of it. Only the
        head is scored, deliberately deeper than the returned slate, since
        scoring just the slate could only permute it and never turn a miss
        into a hit.
        """
        if self.reranker is None or RERANK_WEIGHT <= 0.0 or not candidate_ids:
            return candidate_ids
        head, tail = candidate_ids[:RERANK_TOP_N], candidate_ids[RERANK_TOP_N:]
        try:
            texts = self.bm25.document_text(head)
            scored = self.reranker.scores(query, [texts.get(pid, "") for pid in head])
        except Exception:
            logger.exception("reranking failed; keeping the retrieval order")
            return candidate_ids
        reranked = [pid for _, pid in sorted(zip(scored, head, strict=True), key=lambda pair: -pair[0])]
        fused = reciprocal_rank_fusion([head, reranked], top_n=len(head),
                                       weights=[1.0, RERANK_WEIGHT])
        return fused + tail

    def _diversify(self, candidate_ids: list[str], head_n: int,
                   lambda_: float = DIVERSIFY_LAMBDA,
                   repel: set[str] | None = None,
                   intra_diversity: bool = True) -> list[str]:
        """Greedy MMR re-rank of the pool's front, browsing-track only.

        Reorders the top n items for diversity, while keeping relevance
        to query.
        """
        if self.dense is None or len(candidate_ids) <= head_n:
            return candidate_ids

        window = candidate_ids[:DIVERSIFY_WINDOW]
        known_ids, vectors = self.dense.vectors_for(window)
        if len(known_ids) <= DIVERSIFY_PIN:
            return candidate_ids

        # repel items that are close to shown items
        repel_sim = None
        if repel:
            _, repel_vectors = self.dense.vectors_for(list(repel))
            if len(repel_vectors):
                repel_sim = (vectors @ repel_vectors.T).max(axis=1)

        relevance = [1.0 / (rank + 1) for rank in range(len(known_ids))]
        sim = vectors @ vectors.T

        selected = list(range(DIVERSIFY_PIN))
        remaining = [i for i in range(len(known_ids)) if i not in selected]
        while remaining and len(selected) < head_n:
            best_idx, best_score = None, -float("inf")
            for idx in remaining:
                max_sim = (max((sim[idx, s] for s in selected), default=0.0)
                           if intra_diversity else 0.0)
                if repel_sim is not None:
                    max_sim = max(max_sim, float(repel_sim[idx]))
                score = lambda_ * relevance[idx] - (1 - lambda_) * max_sim
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
        """One-line rationale for this turn's slate.

        Purely for debugging purposes.
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

        Embedding rule when available, regex as a fallback.
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
        def slot_weight(attribute: str) -> float:
            # Geometric decay in turns since the value was stated. SLOT_DECAY
            # == 1.0 makes this a constant 1.0 (no decay).
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
                # Boost only if the attribute wasn't already disclosed.
                if attribute not in disclosed and self.attribute_index.value_for(attribute, pid) == value:
                    score += LM_INFERENCE_WEIGHT
            return score

        return sorted(candidate_ids, key=match_score, reverse=True)
