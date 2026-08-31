"""Unit tests for starter/classifier.py.

Two layers, tested differently:

- The lexical/regex layer (_is_negated, classify_intent, detected_attributes,
  matches_override_cue, matches_defer_cue, classify_reply_lexically) is pure
  and tested directly -- this is also the fallback path used whenever the
  embedding model fails to load, so it must work standalone.
- The Embedding* classes need a real sentence-transformer to score meaning,
  which isn't something a unit test should download. What IS unit-testable
  without one is their control flow: the empty-input guards, the
  veto-before-encode short-circuit (detected_attributes trumps similarity),
  the union-not-vote combination in is_override, and the exception fallback.
  A tiny FakeModel stub with a controllable encode() isolates exactly that
  logic from actual embedding quality, which scripts/eval_override.py already
  measures on real data.
"""
from __future__ import annotations

import numpy as np
import pytest

from starter.classifier import (
    EmbeddingIntentClassifier,
    EmbeddingNonAnswerDetector,
    EmbeddingOverrideDetector,
    PROTOTYPE_BROWSING,
    PROTOTYPE_BUYING,
    PROTOTYPE_CONTINUATION,
    PROTOTYPE_NON_ANSWER,
    PROTOTYPE_OVERRIDE,
    _is_negated,
    classify_intent,
    classify_reply_lexically,
    detected_attributes,
    lead_clause,
    matches_defer_cue,
    matches_override_cue,
)


# ---------------------------------------------------------------------------
# _is_negated
# ---------------------------------------------------------------------------

def test_is_negated_true_within_same_clause():
    text = "not exactly what I had in mind"
    idx = text.index("exactly")
    assert _is_negated(text, idx) is True


def test_is_negated_false_across_clause_break():
    text = "I'm not picky. Exactly this color, please."
    idx = text.index("Exactly")
    assert _is_negated(text, idx) is False


def test_is_negated_apostrophe_less_contraction():
    text = "i dont want black shoes"
    idx = text.index("black")
    assert _is_negated(text, idx) is True


def test_is_negated_does_not_false_positive_on_want():
    # "want" ends in "nt" but must not be read as a negation marker itself.
    text = "i want black shoes"
    idx = text.index("black")
    assert _is_negated(text, idx) is False


def test_is_negated_pivot_cue_ends_the_negating_span():
    # "never mind" retracts the previous turn, not what follows it.
    text = "never mind i want white"
    idx = text.index("white")
    assert _is_negated(text, idx) is False


def test_is_negated_window_bounded():
    # A negation far outside NEGATION_WINDOW_CHARS shouldn't reach forward.
    text = "not " + ("x" * 50) + " black"
    idx = text.index("black")
    assert _is_negated(text, idx) is False


# ---------------------------------------------------------------------------
# detected_attributes
# ---------------------------------------------------------------------------

def test_detected_attributes_material():
    assert detected_attributes("I want cotton") == {"material"}


def test_detected_attributes_color():
    assert detected_attributes("black shoes please") == {"color"}


def test_detected_attributes_size():
    assert "size" in detected_attributes("I need a size 9")


def test_detected_attributes_budget():
    assert "budget" in detected_attributes("under $50")


def test_detected_attributes_multiple():
    result = detected_attributes("black cotton shirt under $50, size medium")
    assert result == {"color", "material", "budget", "size"}


def test_detected_attributes_negated_material_not_detected():
    assert detected_attributes("no leather please") == set()


def test_detected_attributes_empty_text():
    assert detected_attributes("") == set()


def test_detected_attributes_non_string_input():
    assert detected_attributes(None) == set()
    assert detected_attributes(123) == set()


# ---------------------------------------------------------------------------
# classify_intent (lexical fallback)
# ---------------------------------------------------------------------------

def test_classify_intent_explicit_buy_verb_is_buying():
    result = classify_intent("I want to buy shoes")
    assert result.label == "buying"


def test_classify_intent_negation_adjacent_to_buy_verb_falls_through():
    # Negation directly adjacent to the verb is caught.
    result = classify_intent("not buy")
    assert result.label == "browsing"


def test_classify_intent_negated_buy_verb_with_words_between_falls_through():
    # Matches BUYING_PHRASES's own worked example. Was failing at
    # NEGATION_WINDOW_CHARS=10 (too short to bridge "not ... to buy" once
    # words sit between the negation and the verb) until config.py raised
    # it to 20, measured score-neutral via scripts/sweep_negation_window.py.
    result = classify_intent("I'm not looking to buy yet")
    assert result.label == "browsing"


def test_classify_intent_negated_want_to_buy_falls_through():
    result = classify_intent("i don't want to buy this")
    assert result.label == "browsing"


def test_classify_intent_hedge_phrase_is_browsing():
    result = classify_intent("just browsing, not sure what style yet")
    assert result.label == "browsing"


def test_classify_intent_bare_category_mention_defaults_to_browsing():
    # 0/0 tie: the safer failure mode is one more question, not assumed specificity.
    result = classify_intent("shoes")
    assert result.label == "browsing"
    assert result.score == 0.0


def test_classify_intent_concrete_attribute_vocab_counts_as_buying_evidence():
    result = classify_intent("black leather boots, size 9, under $80")
    assert result.label == "buying"
    assert result.buying_evidence > 0


def test_classify_intent_non_string_input_treated_as_empty():
    result = classify_intent(None)
    assert result.label == "browsing"
    assert result.score == 0.0


# ---------------------------------------------------------------------------
# matches_override_cue / lead_clause
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "never mind, i want white shoes",
    "scratch that, show me boots instead",
    "actually no, let's go with sneakers",
    "forget what i said, i need a jacket",
])
def test_matches_override_cue_detects_clause_initial_cues(text):
    assert matches_override_cue(text) is True


@pytest.mark.parametrize("text", [
    "actually I also need it waterproof",
    "no preference there, whatever you think is best",
    "I forgot to mention I also need a belt",
])
def test_matches_override_cue_false_on_ordinary_continuations(text):
    # "actually" and "no" alone are not cues -- only the closed-list phrasings are.
    assert matches_override_cue(text) is False


def test_matches_override_cue_only_fires_clause_initial():
    # The cue must open a clause, not appear mid-sentence with no break before it.
    text = "please don't forget about the return policy in your reply"
    assert matches_override_cue(text) is False


def test_lead_clause_splits_on_punctuation():
    assert lead_clause("never mind, i want white shoes") == "never mind"


def test_lead_clause_returns_whole_text_without_break():
    text = "i want white shoes"
    assert lead_clause(text) == text


# ---------------------------------------------------------------------------
# matches_defer_cue
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "no preference, please use your judgment",
    "whatever you think is best",
    "it's totally up to you",
    "you know best",
])
def test_matches_defer_cue_detects_hand_back(text):
    assert matches_defer_cue(text) is True


def test_matches_defer_cue_false_on_ordinary_disclosure():
    assert matches_defer_cue("I'd like it in black, please") is False


def test_matches_defer_cue_non_string_input():
    assert matches_defer_cue(None) is False


# ---------------------------------------------------------------------------
# classify_reply_lexically
# ---------------------------------------------------------------------------

def test_classify_reply_lexically_pure_decline_is_non_answer():
    assert classify_reply_lexically("no preference there, whatever you think is best") == "non_answer"


def test_classify_reply_lexically_ordinary_disclosure_is_answer():
    assert classify_reply_lexically("cotton would be great") == "answer"


def test_classify_reply_lexically_decline_with_concrete_value_is_answer():
    # "I don't have a preference, but black is fine" discloses a color --
    # concrete content overrides the decline cue, not the other way round.
    assert classify_reply_lexically("I don't have a preference, but black is fine") == "answer"


def test_classify_reply_lexically_empty_text_is_answer():
    assert classify_reply_lexically("") == "answer"
    assert classify_reply_lexically(None) == "answer"


# ---------------------------------------------------------------------------
# FakeModel: deterministic stand-in for the sentence-transformer, used only
# to exercise the Embedding* classes' control flow, not embedding quality.
# ---------------------------------------------------------------------------

class FakeModel:
    """Maps known prototype sentences to fixed orthogonal unit vectors, and
    any other query text to a vector supplied by the test via `query_vectors`.

    encode() has the same signature the real SentenceTransformer exposes
    (sentences, normalize_embeddings, convert_to_numpy), so it's a drop-in
    for the classifier classes without pulling in torch/bge-small.
    """

    _PROTOTYPE_DIRECTIONS = {
        PROTOTYPE_BUYING: np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        PROTOTYPE_BROWSING: np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        PROTOTYPE_OVERRIDE: np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        PROTOTYPE_CONTINUATION: np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        PROTOTYPE_NON_ANSWER: np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    }
    _NEUTRAL = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def __init__(self, query_vectors: dict[str, np.ndarray] | None = None, raise_on_encode: bool = False):
        self.query_vectors = query_vectors or {}
        self.raise_on_encode = raise_on_encode
        self.encode_calls: list = []

    def _vector_for(self, sentence: str) -> np.ndarray:
        for prototypes, direction in self._PROTOTYPE_DIRECTIONS.items():
            if sentence in prototypes:
                return direction
        return self.query_vectors.get(sentence, self._NEUTRAL)

    def encode(self, sentences, normalize_embeddings=True, convert_to_numpy=True):
        self.encode_calls.append(sentences)
        if self.raise_on_encode:
            raise RuntimeError("encode should not have been called")
        if isinstance(sentences, str):
            return self._vector_for(sentences)
        return np.stack([self._vector_for(s) for s in sentences])


# ---------------------------------------------------------------------------
# EmbeddingIntentClassifier
# ---------------------------------------------------------------------------

def test_embedding_intent_classifier_empty_text_is_browsing_without_encoding():
    model = FakeModel()
    classifier = EmbeddingIntentClassifier(model)
    # __init__ already called encode() to build centroids; arm the tripwire
    # only now so the classify() call under test is what's being checked.
    model.raise_on_encode = True
    result = classifier.classify("")
    assert result.label == "browsing"
    assert result.score == 0.0


def test_embedding_intent_classifier_picks_nearer_centroid():
    model = FakeModel(query_vectors={"query": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])})
    classifier = EmbeddingIntentClassifier(model)
    result = classifier.classify("query")
    assert result.label == "buying"
    assert result.buying_evidence > result.browsing_evidence


def test_embedding_intent_classifier_falls_back_on_encode_failure():
    class BrokenModel(FakeModel):
        def encode(self, sentences, normalize_embeddings=True, convert_to_numpy=True):
            if isinstance(sentences, str):
                raise RuntimeError("boom")
            return super().encode(sentences, normalize_embeddings, convert_to_numpy)

    model = BrokenModel()
    classifier = EmbeddingIntentClassifier(model)
    result = classifier.classify("i want to buy shoes")
    # classify_intent's lexical fallback should have fired instead of raising.
    assert result.label == "buying"


# ---------------------------------------------------------------------------
# EmbeddingNonAnswerDetector
# ---------------------------------------------------------------------------

def test_non_answer_detector_veto_by_concrete_attribute_skips_encode():
    model = FakeModel()
    detector = EmbeddingNonAnswerDetector(model)
    model.encode_calls.clear()
    model.raise_on_encode = True
    # "black" is a concrete disclosure -- detected_attributes must veto
    # a non-answer verdict before any encode() call happens.
    assert detector.is_non_answer("no preference, but black is fine") is False
    assert model.encode_calls == []


def test_non_answer_detector_empty_text_is_false():
    model = FakeModel()
    detector = EmbeddingNonAnswerDetector(model)
    model.raise_on_encode = True
    assert detector.is_non_answer("") is False


def test_non_answer_detector_similarity_direction():
    model = FakeModel(query_vectors={"query": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])})
    detector = EmbeddingNonAnswerDetector(model)
    assert detector.is_non_answer("query") is True


def test_non_answer_detector_below_threshold_is_false():
    # Partial similarity to the decline direction, below NON_ANSWER_THRESHOLD
    # (0.68) -- the one-class floor, not a nearest-class contest, must reject it.
    model = FakeModel(query_vectors={"query": np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0])})
    detector = EmbeddingNonAnswerDetector(model)
    assert detector.is_non_answer("query") is False


def test_non_answer_detector_falls_back_to_lexical_on_encode_failure():
    class BrokenModel(FakeModel):
        def encode(self, sentences, normalize_embeddings=True, convert_to_numpy=True):
            if isinstance(sentences, str):
                raise RuntimeError("boom")
            return super().encode(sentences, normalize_embeddings, convert_to_numpy)

    model = BrokenModel()
    detector = EmbeddingNonAnswerDetector(model)
    assert detector.is_non_answer("no preference there, whatever you think is best") is True


# ---------------------------------------------------------------------------
# EmbeddingOverrideDetector
# ---------------------------------------------------------------------------

def test_override_detector_literal_cue_skips_encode():
    model = FakeModel()
    detector = EmbeddingOverrideDetector(model)
    model.encode_calls.clear()
    model.raise_on_encode = True
    assert detector.is_override("never mind, i want white shoes") is True
    assert model.encode_calls == []


def test_override_detector_empty_text_is_false():
    model = FakeModel()
    detector = EmbeddingOverrideDetector(model)
    model.raise_on_encode = True
    assert detector.is_override("") is False


def test_override_detector_similarity_path_when_no_literal_cue():
    model = FakeModel(query_vectors={"my priorities changed": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])})
    detector = EmbeddingOverrideDetector(model)
    assert detector.is_override("my priorities changed") is True


def test_override_detector_union_not_vote_across_full_text_and_lead_clause():
    # Only the lead clause looks override-like; the full message does not.
    # is_override must still return True (union, not requiring both to agree).
    full_text = "never-mind-cue-absent, actually let's keep going with this"
    lead = lead_clause(full_text)
    model = FakeModel(query_vectors={
        full_text: np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),  # looks like continuation
        lead: np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),        # looks like override
    })
    detector = EmbeddingOverrideDetector(model)
    assert detector.is_override(full_text) is True


def test_override_detector_falls_back_to_no_override_on_encode_failure():
    class BrokenModel(FakeModel):
        def encode(self, sentences, normalize_embeddings=True, convert_to_numpy=True):
            if isinstance(sentences, list) and sentences and sentences[0] not in (
                *PROTOTYPE_OVERRIDE, *PROTOTYPE_CONTINUATION, *PROTOTYPE_BUYING, *PROTOTYPE_BROWSING,
            ):
                raise RuntimeError("boom")
            return super().encode(sentences, normalize_embeddings, convert_to_numpy)

    model = BrokenModel()
    detector = EmbeddingOverrideDetector(model)
    assert detector.is_override("some ordinary sentence with no cue") is False
