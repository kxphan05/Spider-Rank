from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .attributes import AttributeIndex, FILTERABLE_ATTRIBUTES, extract_disclosed_value, select_dynamic_attribute
from .classifier import EmbeddingIntentClassifier, EmbeddingOverrideDetector, classify_intent, detected_attributes
from .retrieval import BM25Index, DenseIndex, load_embedding_model, reciprocal_rank_fusion

logger = logging.getLogger(__name__)

ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}

# material/color are chosen dynamically (see attributes.py:
# select_dynamic_attribute) based on value-diversity in the current
# candidate pool. style/size/use_case/feature have no cheap structural
# extractor from catalog text, so they stay a fixed fallback order, used
# once the dynamic pick has nothing left worth asking about. `budget` is
# last-resort only: measured against the local evaluator, disclosed answers
# almost never bucket as budget (see attributes.py STRUCTURAL_ATTRIBUTES
# comment), so it's demoted below every other question rather than dropped
# outright.
FALLBACK_ATTRIBUTE_ORDER = ["style", "size", "use_case", "feature", "budget"]

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
# Empirically (full public-set run): equal-weight RRF matched BM25-only hit
# rate but dropped MRR 0.450->0.385 -- BM25 is the higher-precision leg on
# these near-exact-match queries, so it's weighted higher in the fusion.
BM25_FUSION_WEIGHT = 2.0
DENSE_FUSION_WEIGHT = 1.0


@dataclass
class SessionState:
    first_message: str | None = None
    recent_messages: list[str] = field(default_factory=list)
    asked_attributes: list[str] = field(default_factory=list)
    # attribute -> extracted value, from any message (asked or volunteered).
    # Used to filter/rerank the candidate pool in _retrieve, not just as
    # extra query text.
    disclosed: dict[str, str] = field(default_factory=dict)


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
    state: SessionState, query: str, candidate_pool: list[str], attribute_index: AttributeIndex
) -> str | None:
    # Skip attributes already evidenced in the accumulated text (e.g. a
    # buying session's turn-1 message names a material) -- asking about it
    # again just wastes a turn, since the evaluator's customer policy only
    # reveals values not already marked "disclosed".
    excluded = set(state.asked_attributes) | detected_attributes(query)

    dynamic_pick = select_dynamic_attribute(candidate_pool, attribute_index, excluded)
    if dynamic_pick is not None:
        return dynamic_pick

    # Nothing structural is worth asking about (either excluded, or the
    # current pool already agrees on material/color/budget) -- fall back to
    # the fixed order for attributes we can't score entropy for.
    for attribute in FALLBACK_ATTRIBUTE_ORDER:
        if attribute not in excluded:
            return attribute
    return None


class Agent:
    """Hybrid BM25 + dense (bge-small) retrieval with reciprocal rank fusion."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        dense_index_dir: str | Path = "data/dense_index",
    ) -> None:
        catalog_path = Path(catalog_path)
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

        self.intent_classifier: EmbeddingIntentClassifier | None = None
        self.override_detector: EmbeddingOverrideDetector | None = None
        if embedding_model is not None:
            self.intent_classifier = EmbeddingIntentClassifier(embedding_model)
            self.override_detector = EmbeddingOverrideDetector(embedding_model)

        self._sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(user_profile, dict):
            logger.warning("reset(%s): user_profile is not a dict (%r); ignoring", session_id, type(user_profile))
        # user_profile is anonymized aggregate data (purchase_frequency, rating
        # style, preference_tags, summary); not yet used for personalization.
        self._sessions[session_id] = SessionState()

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
                state.asked_attributes.clear()
            state.recent_messages.append(user_message)
            del state.recent_messages[:-RECENT_WINDOW]

        for attribute in FILTERABLE_ATTRIBUTES:
            value = extract_disclosed_value(attribute, user_message, self.attribute_index.price_buckets)
            if value is not None:
                state.disclosed[attribute] = value

        query = _build_query(state)
        candidate_pool = self._retrieve(query, max(top_k, ENTROPY_POOL_SIZE), state.disclosed)
        recommendations = candidate_pool[:top_k]

        signal = self.intent_classifier.classify(query) if self.intent_classifier is not None else classify_intent(query)
        logger.debug(
            "respond(%s, turn=%s): intent=%s score=%+.2f asked=%s",
            session_id, turn, signal.label, signal.score, state.asked_attributes,
        )

        attribute = _next_attribute(state, query, candidate_pool, self.attribute_index)
        if attribute is not None:
            state.asked_attributes.append(attribute)
            message = ATTRIBUTE_QUESTIONS[attribute]
        else:
            message = "Here are the closest matches I found so far."

        assert attribute is None or attribute in ALLOWED_ATTRIBUTES
        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": [{"parent_asin": pid} for pid in recommendations],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def _retrieve(
        self,
        query: str,
        top_k: int,
        disclosed: dict[str, str] | None = None,
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
        legs = [(bm25_ranked, BM25_FUSION_WEIGHT), (dense_ranked, DENSE_FUSION_WEIGHT)]
        rank_lists = [ranked for ranked, _ in legs if ranked]
        weights = [weight for ranked, weight in legs if ranked]
        if not rank_lists:
            return []
        fused = reciprocal_rank_fusion(rank_lists, top_n=top_k, weights=weights)

        if not disclosed:
            return fused

        return self._boost_by_disclosed(fused, disclosed)

    def _boost_by_disclosed(self, candidate_ids: list[str], disclosed: dict[str, str]) -> list[str]:
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
        def match_score(pid: str) -> int:
            score = 0
            for attribute, value in disclosed.items():
                actual = self.attribute_index.value_for(attribute, pid)
                if actual is None:
                    continue
                score += 1 if actual == value else -1
            return score

        return sorted(candidate_ids, key=match_score, reverse=True)
