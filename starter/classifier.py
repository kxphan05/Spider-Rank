"""Buying-vs-browsing intent classification for routing.

Two implementations sharing the IntentSignal interface: EmbeddingIntentClassifier
(zero-shot nearest-centroid over prototype utterances, primary) and
classify_intent (lexical/regex fallback if the embedding model can't load).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
from .config import (NEGATION_WINDOW_CHARS, TOP_PROTOTYPES)

logger = logging.getLogger(__name__)

# Necessity/constraint framing. Generic on purpose: verbs/phrases people use
# to assert a hard requirement, independent of which attribute it's about.
BUYING_PHRASES = (
    r"\bmust\b", r"\brequire[sd]?\b", r"\bneed(?:s|ed)?\s+(?:it\s+)?to\b",
    r"\bhas\s+to\s+be\b", r"\bspecifically\b", r"\bexactly\b",
    r"\bkey\s+requirement\b", r"\bshould\s+be\b", r"\bwith\s+an?\b",
    r"\bmade\s+of\b", r"\bin\s+size\b",
    # Purchase verbs, so "i want to buy shoes" scores as buying rather than
    # falling through to the browsing tie-break. _is_negated still catches
    # "not looking to buy yet".
    r"\bbuy(?:ing)?\b", r"\bpurchase\b", r"\bplace\s+an?\s+order\b",
)

# Hedging / exploratory framing.
BROWSING_PHRASES = (
    r"\bnot\s+sure\b", r"\b(?:browsing|looking)\b",
    r"\bstill\s+(?:exploring|deciding|looking|browsing)\b",
    r"\bwhatever\b", r"\bsomething\s+like\b", r"\bnot\s+(?:picky|particular)\b",
    r"\bopen\s+to\b", r"\bno\s+preference\b", r"\bhaven'?t\s+decided\b",
    r"\bexploring\s+options?\b", r"\bany\s+suggestions?\b",
    r"\bwhat\s+do\s+you\s+recommend\b", r"\bany\s+ideas?\b", r"\bnot\s+really\s+sure\b",
)

COLORS = (
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
    "purple", "yellow", "orange", "navy", "beige", "tan", "gold", "silver", "multicolor",
)
MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon",
    "denim", "suede", "canvas", "linen", "cashmere", "fleece", "mesh", "velvet",
)

SIZE_RE = re.compile(
    r"\b(?:size\s*\d+|us\s*\d+(?:\.\d+)?|x{1,3}[sl]\b|small\b|medium\b|large\b|"
    r"\d+\s*(?:mm|cm|in|inch|inches)\b)",
    re.I,
)
PRICE_RE = re.compile(r"(\$\s?\d+|\bunder\s+\$?\d+|\bbudget\s+(?:of\s+)?\$?\d+|\d+\s*dollars)", re.I)


# "n't" has no leading \b since the "n" in "don't"/"isn't" isn't preceded by
# a word boundary.
NEGATION_RE = re.compile(
    r"\b(?:not|no|never|without)\b|n't\b"
    # Apostrophe-less contractions as a closed list -- `n'?t\b` would also
    # match "want"/"print"/"garment", and typed input drops apostrophes often
    # enough that missing these left "i dont want black" un-negated.
    r"|\b(?:dont|doesnt|didnt|isnt|arent|wasnt|werent|wont|cant|cannot"
    r"|couldnt|wouldnt|shouldnt|havent|hasnt|hadnt)\b"
)
CLAUSE_BREAK_RE = re.compile(r"[.,;!?]")

# Discourse markers that open a pivot: the negation inside them retracts the
# previous turn, not what follows, so "never mind i want white" states white
# rather than declining it.
PIVOT_CUE_RE = re.compile(
    r"\b(?:never ?mind|no wait|wait no|nope|scratch that|forget (?:that|it)"
    r"|on second thought|come to think of it)\b"
)


def _is_negated(text: str, match_start: int) -> bool:
    """True if a negation marker precedes match_start within the same clause.

    Bounded by the nearest clause break so negation doesn't leak across
    sentences ("I'm not picky. Exactly this color." still counts).
    """
    window = text[max(0, match_start - NEGATION_WINDOW_CHARS):match_start]
    # A pivot cue ends the negating span exactly the way punctuation does:
    # whatever follows "never mind" is the new request, not part of what was
    # just retracted. Take whichever boundary sits closest to the match.
    last_break = None
    for pattern in (CLAUSE_BREAK_RE, PIVOT_CUE_RE):
        for break_match in pattern.finditer(window):
            if last_break is None or break_match.end() > last_break:
                last_break = break_match.end()
    if last_break is not None:
        window = window[last_break:]
    return bool(NEGATION_RE.search(window))


def _phrase_hits(patterns: tuple[str, ...], text: str) -> int:
    count = 0
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            if not _is_negated(text, match.start()):
                count += 1
                break  # at most one hit per pattern, same as before
    return count


def _vocab_hits(vocab: tuple[str, ...], text: str) -> int:
    return sum(1 for word in vocab if re.search(rf"\b{re.escape(word)}\b", text))


def _vocab_hits_positive(vocab: tuple[str, ...], text: str) -> int:
    """Like _vocab_hits, but a negated mention ("no leather", "not black")
    doesn't count -- the customer ruled a value out, they didn't disclose
    a preference for it.
    """
    count = 0
    for word in vocab:
        for match in re.finditer(rf"\b{re.escape(word)}\b", text):
            if not _is_negated(text, match.start()):
                count += 1
                break
    return count


@dataclass
class IntentSignal:
    label: str  # "buying" or "browsing"
    score: float  # positive leans buying, negative leans browsing; scale varies by implementation
    buying_evidence: float
    browsing_evidence: float


def detected_attributes(text: str) -> set[str]:
    """Attribute buckets already evidenced by concrete vocabulary in `text`.

    Used to avoid re-asking about an attribute the customer already stated.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    lowered = text.lower()
    found: set[str] = set()
    if _vocab_hits_positive(MATERIALS, lowered):
        found.add("material")
    if _vocab_hits_positive(COLORS, lowered):
        found.add("color")
    if SIZE_RE.search(lowered):
        found.add("size")
    if PRICE_RE.search(lowered):
        found.add("budget")
    return found


def classify_intent(text: str) -> IntentSignal:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    lowered = text.lower()

    buying_hits = max(
        _phrase_hits(BUYING_PHRASES, lowered)
        , _vocab_hits(COLORS, lowered)
        , _vocab_hits(MATERIALS, lowered)
        , (1 if SIZE_RE.search(lowered) else 0)
        , (1 if PRICE_RE.search(lowered) else 0)
    )
    browsing_hits = _phrase_hits(BROWSING_PHRASES, lowered)

    total = buying_hits + browsing_hits
    score = 0.0 if total == 0 else (buying_hits - browsing_hits) / total
    # Ties (including 0/0 — a bare, unmodified category mention) default to
    # browsing: the safer failure mode is asking one more question, not
    # assuming specificity that was never actually given.
    label = "buying" if score > 0 else "browsing"
    return IntentSignal(label=label, score=score, buying_evidence=float(buying_hits), browsing_evidence=float(browsing_hits))


# Hand-written, generic prototype utterances -- not copied from this repo's
# local evaluator templates -- used as zero-shot class anchors. Deliberately
# varied in phrasing/register so the centroid isn't keyed to one sentence shape.
PROTOTYPE_BUYING = (
    "I need a pair of black leather boots in size 9.",
    "Looking for a cotton t-shirt under $20.",
    "I want a waterproof winter jacket, it must be machine washable.",
    "I'm searching for a silver necklace with a small pendant.",
    "It has to be a slim fit, dark wash pair of jeans.",
    "I need running shoes with good arch support, size 10.5.",
    "Do you have this in a medium, navy blue?",
    "I want a wool sweater, crew neck, for under $50.",
    "Show me a leather handbag, brown, with a shoulder strap.",
    "I need a size 8 sandal, preferably suede.",
    "I want to buy a pair of shoes.",
    "I'm here to buy a watch today.",
    "Ready to purchase a handbag.",
)
PROTOTYPE_BROWSING = (
    "I'm just browsing for some new shoes, not sure what style yet.",
    "Looking for something nice to wear, no particular preference.",
    "I'm not sure what I want, just exploring options.",
    "Can you show me some jewelry, I'm open to anything.",
    "I want a jacket but haven't decided on the details.",
    "Not really sure yet, just looking around.",
    "What do you have in dresses? Nothing specific in mind.",
    "I'm shopping around, no fixed idea yet.",
    "Just curious what's available for accessories.",
    "Not picky, whatever looks good works for me.",
    "I want something for the summer.",
    "I need something for a wedding, open to suggestions.",
    "Something for the beach, whatever you think works.",
    "Looking for something for work, I don't mind what.",
)

PROTOTYPE_OVERRIDE = (
    "Actually, ignore my earlier preference. What I need is a leather jacket.",
    "Never mind what I said before, I want something different now.",
    "On second thought, forget the color I mentioned earlier.",
    "I changed my mind, let's go with a different style instead.",
    "Scratch that, I actually need running shoes instead.",
    "Wait, ignore my previous requirements, here's what I actually want.",
    "Let's start over, I'm looking for something else entirely.",
    "Forget what I said earlier, this is what I actually need.",
    "My priorities changed -- let's focus on this instead of what I said before.",
    "Hold on, I don't want that anymore, show me something different.",
    "Actually, disregard my last message.",
    "Sorry, I meant something completely different than what I said.",
)

PROTOTYPE_CONTINUATION = (
    "Yes, that sounds good, please continue.",
    "I also need it to be machine washable.",
    "Do you have this in a smaller size?",
    "That looks nice, can you show me more options like it?",
    "I'd also like it to be under $50.",
    "Great, and I prefer it in black.",
    "Can you narrow it down further?",
    "That works, what else do you have?",
    "I like the second one, tell me more about it.",
    "Sure, size medium works for me.",
    "It should also be waterproof.",
    "None of these quite fit, but keep the same style.",
    # Declining to state a preference reads as override-ish (negation/"don't")
    # but isn't, and is common enough to need explicit coverage here.
    "I don't have a particular preference for that, up to you.",
    "No preference there, whatever you think is best.",
    "I'm not picky about that detail, it's fine either way.",
    "No strong opinion on that one, doesn't matter much to me.",
    "That's not something I care about, anything works.",
    "I haven't thought about that, no requirement there.",
)



# An opening clause up to this many characters is scored on its own as well
# as in context -- see EmbeddingOverrideDetector.is_override.
LEAD_CLAUSE_RE = re.compile(r"^(.{0,60}?)[.,;:!?]\s")


def lead_clause(text: str) -> str:
    """The message's opening clause, or the whole message if it has no break.
    """
    match = LEAD_CLAUSE_RE.match(text.strip())
    return match.group(1) if match else text


# Explicit discard cues, as a closed list matched clause-initial only (a cue
# is a prefix to the new request, not something that can appear mid-sentence
# in ordinary catalog copy). Backs up the embedding similarity rule, which
# loses terse pivots like "never mind, white shoes" to the request half.
CLAUSE_SPLIT_RE = re.compile(r"[.,;:!?]+")

OVERRIDE_CUE_RE = re.compile(
    r"""\s*
    (?: never\s?mind
      | nvm\b
      | scratch\s+that
      | strike\s+that
      | cancel\s+that
      | forget\s+(?:that|it|those|what\s+i\s+said|my\b|the\s+\w+|about\b)
      | disregard\b
      | ignore\s+(?:that|it|all\s+that|what\s+i\s+said|my\b|the\s+\w+
                  |previous|earlier)
      | on\s+second\s+thoughts?
      | (?:i\s*(?:'ve|\s+have)?\s+)?changed\s+my\s+mind
      | (?:let'?s\s+)?start\s+over\b
      | actually,?\s*no\b
      | i\s+don'?t\s+want\s+(?:that|it|those|any\s+of\s+that)\s+anymore
      | instead\s+of\s+what\s+i\s+said
      | different\s+idea
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def matches_override_cue(text: str) -> bool:
    """True when any clause of `text` opens with a literal discard cue."""
    return any(
        OVERRIDE_CUE_RE.match(clause)
        for clause in CLAUSE_SPLIT_RE.split(text)
        if clause.strip()
    )


def _top_prototype_similarity(query_vec: np.ndarray, prototypes: np.ndarray) -> float:
    """Mean cosine similarity to the `TOP_PROTOTYPES` closest prototypes."""
    similarities = prototypes @ query_vec
    k = min(TOP_PROTOTYPES, similarities.shape[0])
    return float(np.sort(similarities)[-k:].mean())


class EmbeddingOverrideDetector:
    """Trimmed nearest-prototype detection of a mid-session preference reset.

    A false negative leaves stale disclosed-attribute state for the rest of
    the session, worse than a false positive's few wasted turns, so this is
    tuned toward recall (mean similarity to each class's closest prototypes).
    """

    def __init__(self, model) -> None:
        self.model = model
        self._override_protos = self._encode(PROTOTYPE_OVERRIDE)
        self._continuation_protos = self._encode(PROTOTYPE_CONTINUATION)
        self._buying_protos = self._encode(PROTOTYPE_BUYING)
        self._browsing_protos = self._encode(PROTOTYPE_BROWSING)

    def _encode(self, sentences: tuple[str, ...]) -> np.ndarray:
        return self.model.encode(
            list(sentences), normalize_embeddings=True, convert_to_numpy=True
        )

    def _leans_override(self, vec: np.ndarray) -> bool:
        return (
            _top_prototype_similarity(vec, self._override_protos)
            > max([_top_prototype_similarity(vec, self._continuation_protos),
                   _top_prototype_similarity(vec, self._buying_protos),
                   _top_prototype_similarity(vec, self._browsing_protos)
            ]
        ))

    def is_override(self, text: str) -> bool:
        if not isinstance(text, str) or not text.strip():
            return False
        # Union, not a vote: the closed list catches terse pivots, the
        # similarity catches paraphrases like "my priorities changed", and
        # the cue check runs first so it skips the embed call when it fires.
        if matches_override_cue(text):
            return True
        variants = [text]
        lead = lead_clause(text)
        if lead != text:
            variants.append(lead)
        try:
            query_vecs = self.model.encode(
                variants, normalize_embeddings=True, convert_to_numpy=True
            )
        except Exception:
            logger.exception("override detector failed on %r; defaulting to no-override", text)
            return False
        return any(self._leans_override(vec) for vec in query_vecs)


# Cue utterances for a reply that answers a clarifying question with *no new
# information* -- the customer declines to state a preference for the attribute
# that was asked about.
PROTOTYPE_NON_ANSWER = (
    "I don't have a preference for that, up to you.",
    "No preference there, whatever you think is best.",
    "I'm not picky about that detail, it's fine either way.",
    "No strong opinion on that one, doesn't matter much to me.",
    "That's not something I care about, anything works.",
    "I haven't thought about that, no requirement there.",
    "Doesn't matter to me, you choose.",
    "No idea honestly, surprise me.",
    "I'll leave that one to your judgment.",
    "Nothing specific in mind for that.",
    "Anything is fine on that front.",
    "I really don't mind either way.",
)

# The contrast class: a reply that *does* carry new constraint content. Kept
# separate from PROTOTYPE_CONTINUATION since that tuple's composition is
# load-bearing for EmbeddingOverrideDetector's measured numbers.
PROTOTYPE_INFORMATIVE = (
    "I also need it to be machine washable.",
    "For that, what matters is a waterproof outer shell.",
    "I'd like it in black, please.",
    "Size medium works for me.",
    "It should be under fifty dollars.",
    "Cotton, ideally something breathable.",
    "I need it for hiking in cold weather.",
    "A slim fit would be best.",
    "Leather, and preferably brown.",
    "Something with a zip pocket on the inside.",
    "I want a crew neck rather than a v-neck.",
    "Stainless steel, nothing that tarnishes.",
    # Terse catalog-jargon fragments: bare spec terms like these were reading
    # closer to the non-answer prototypes than to full-sentence disclosures,
    # so real answers like "Imported." were silently dropped from the query.
    "Imported.",
    "Button closure.",
    "Hand Wash Only.",
    "Pull On closure.",
    "Zipper closure.",
    "Tie closure.",
    "Machine Wash, Line Dry.",
    "Elastic waistband.",
    "Buckle closure.",
    "Non-slip rubber sole.",
    "Adjustable strap.",
    "Imported; Zipper closure.",
)

# Lexical floor for the non-answer detector, used when the embedding model is
# unavailable (that degrade is silent, so every embedding
# component needs a fallback or it disappears without a trace).
_DECLINE_RE = re.compile(
    r"\b(?:"
    r"no (?:strong )?(?:preference|opinion|requirement|idea|thoughts?)"
    r"|not? particular(?:ly)? (?:fussed|bothered)"
    r"|(?:don'?t|do not|doesn'?t|does not|never) (?:have|really have|mind|care|matter)"
    r"|not (?:picky|fussy|bothered|that fussed)"
    r"|(?:doesn'?t|does not) matter"
    r"|(?:up to|your) (?:you|judgment|judgement|call|choice)"
    r"|(?:anything|whatever|either way|nothing specific) (?:works|is fine|goes)"
    r"|haven'?t thought about"
    r"|surprise me"
    r")\b",
    re.IGNORECASE,
)


# The hand-back: a non-answer that also asks *us* to decide -- the only
# signal the agent gets that it's in a boundary scenario. Narrower than
# _DECLINE_RE, which would fire on most non-answers and put every session in it.
DEFER_CUE_RE = re.compile(
    r"\b(?:"
    r"(?:please )?(?:use|trust) your (?:own )?(?:judgment|judgement|discretion|expertise)"
    r"|(?:it'?s |that'?s )?(?:entirely |totally |completely )?up to you"
    r"|you (?:can |should |could )?(?:choose|decide|pick)"
    r"|your (?:call|choice|pick)"
    r"|whatever you (?:think|recommend|suggest|like)"
    r"|surprise me"
    r"|i'?ll leave (?:that|it|this) (?:one )?to your?"
    r"|you know best"
    r")\b",
    re.IGNORECASE,
)


def matches_defer_cue(text: str) -> bool:
    """True when the customer hands the decision back to the agent."""
    return bool(isinstance(text, str) and DEFER_CUE_RE.search(text))


def classify_reply_lexically(text: str) -> str:
    """Lexical floor: "non_answer" or "answer".

    Concrete attribute vocabulary always overrides a decline cue -- "I don't
    have a preference, but black is fine" still discloses a colour.
    """
    if not isinstance(text, str) or not text.strip():
        return "answer"
    if detected_attributes(text):
        return "answer"
    return "non_answer" if _DECLINE_RE.search(text) else "answer"


class EmbeddingNonAnswerDetector:
    """Trimmed nearest-prototype detection of a contentless clarifying reply.

    Tuned toward precision, not recall (opposite of EmbeddingOverrideDetector),
    since a false positive here discards a real disclosure from the query.
    """

    def __init__(self, model) -> None:
        self.model = model
        self._non_answer_protos = self._encode(PROTOTYPE_NON_ANSWER)
        self._informative_protos = self._encode(PROTOTYPE_INFORMATIVE)

    def _encode(self, sentences: tuple[str, ...]) -> np.ndarray:
        return self.model.encode(
            list(sentences), normalize_embeddings=True, convert_to_numpy=True
        )

    def is_non_answer(self, text: str) -> bool:
        if not isinstance(text, str) or not text.strip():
            return False
        # Concrete disclosure vetoes, before any embedding work: this is the
        # precision guard, and it must not be reachable around.
        if detected_attributes(text):
            return False
        try:
            vec = self.model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        except Exception:
            logger.exception("non-answer detector failed on %r; falling back to lexical", text)
            return classify_reply_lexically(text) == "non_answer"
        return (
            _top_prototype_similarity(vec, self._non_answer_protos)
            > _top_prototype_similarity(vec, self._informative_protos)
        )


class EmbeddingIntentClassifier:
    """Nearest-centroid buying-vs-browsing classification via sentence embeddings.

    Shares the caller's already-loaded bge-small instance rather than loading
    a second copy. No retrieval-style prefix -- utterance-to-utterance similarity.
    """

    def __init__(self, model) -> None:
        self.model = model
        self._buying_centroid = self._centroid(PROTOTYPE_BUYING)
        self._browsing_centroid = self._centroid(PROTOTYPE_BROWSING)

    def _centroid(self, sentences: tuple[str, ...]) -> np.ndarray:
        embeddings = self.model.encode(
            list(sentences), normalize_embeddings=True, convert_to_numpy=True
        )
        centroid = embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm > 0 else centroid

    def classify(self, text: str) -> IntentSignal:
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        if not text.strip():
            return IntentSignal(label="browsing", score=0.0, buying_evidence=0.0, browsing_evidence=0.0)
        try:
            query_vec = self.model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        except Exception:
            logger.exception("embedding classifier failed on %r; falling back to lexical", text)
            return classify_intent(text)
        sim_buying = float(query_vec @ self._buying_centroid)
        sim_browsing = float(query_vec @ self._browsing_centroid)
        score = sim_buying - sim_browsing
        label = "buying" if score > 0 else "browsing"
        return IntentSignal(label=label, score=score, buying_evidence=sim_buying, browsing_evidence=sim_browsing)
