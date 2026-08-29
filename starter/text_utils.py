from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# Ordinary English function words: frequent everywhere, so they carry no
# retrieval signal in either direction.
_FUNCTION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

# Request verbs and discourse markers -- the words a customer wraps a request
# in ("I want to *buy* shoes", "*actually*, *forget* that, *show* me X").
#
# These are stopwords for the opposite reason to the list above, and the
# distinction matters: they are not common in this catalog, they are RARE.
# Measured document frequencies over the 50k catalog:
#
#     actually 0.0018   forget 0.0032   buy 0.0261   find 0.0318
#     need 0.0404       show 0.0428     give 0.0453  get 0.0576
#
# A rare term is a high-IDF term, so BM25 hands the ranking to it. "i want to
# buy shoes" reduced to the terms {buy, shoes}, and since "shoes" sits in the
# catalog-universal root category (df 1.000, i.e. zero IDF -- see
# CATALOG_ROOT_MIN_SHARE in retrieval.py) the whole query was decided by "buy",
# which returned products from the store "Buy Caps and Hats". Conversational
# framing was outranking the only content word in the sentence.
#
# This is deliberately a list of ways to *ask*, never of things to ask about.
# CLAUDE.md #19 measured that pruning contentful customer text costs 0.041,
# because 89.7% of it is a verbatim substring of the target's own record; none
# of the words below can be part of a product description of anything.
_REQUEST_STOPWORDS = {
    "actually", "anything", "buy", "buying", "find", "forget", "get", "give",
    "hello", "hey", "hi", "instead", "need", "needs", "nevermind", "purchase",
    "recommend", "scratch", "show", "shopping", "something", "suggest",
    "thanks", "thank",
}

STOPWORDS = _FUNCTION_STOPWORDS | _REQUEST_STOPWORDS


# en-GB spellings mapped onto the catalog's en-US ones. The catalog is a US
# Amazon export, so a British spelling is not a rare term, it is a *wrong*
# term: "jewellery" matches 146 products (0.29%) while "jewelry" matches all
# 50,000, so "show me jewellery" searched the catalog for the 146 products
# that happen to spell it the British way and found no jewelry at all.
#
# Only genuine variants where the US form is the catalog's dominant one are
# listed. "grey" is deliberately absent: it occurs in 2,017 products against
# "gray"'s 751, so normalizing it would be the wrong direction.
SPELLING_VARIANTS = {
    "jewellery": "jewelry",
    "jewelery": "jewelry",
    "jewellry": "jewelry",
    "colour": "color",
    "colours": "colors",
    "coloured": "colored",
    "favourite": "favorite",
    "favourites": "favorites",
}


def normalize_query(text: str) -> str:
    """Rewrite customer text into the catalog's spelling conventions.

    Applied once in the agent's query builders so every leg -- BM25, phrase,
    PRF and dense -- sees the same normalized string. Word-boundary matched,
    like every other vocabulary match in this project (CLAUDE.md #10 is the
    bug that comes from forgetting that).
    """
    if not isinstance(text, str) or not text:
        return "" if text is None else str(text)

    def substitute(match: re.Match) -> str:
        word = match.group(0)
        replacement = SPELLING_VARIANTS[word.lower()]
        return replacement.upper() if word.isupper() else (
            replacement.capitalize() if word[0].isupper() else replacement
        )

    return _SPELLING_RE.sub(substitute, text)


_SPELLING_RE = re.compile(
    r"\b(?:" + "|".join(sorted(SPELLING_VARIANTS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


# Clause boundaries for phrase_clauses: sentence-final punctuation plus the
# separators the customer's own text uses to join independent statements.
CLAUSE_RE = re.compile(r"[.!?;:,]+")


def field_text(value: object) -> str:
    """Flatten a catalog field (str/list/dict/None) into plain text."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def phrase_tokens(text: str) -> list[str]:
    """Contiguous tokens with stopwords KEPT, for verbatim span matching.

    `terms()` drops stopwords, which is right for bag-of-words retrieval and
    wrong here: "pull on closure" would collapse to "pull closure" and stop
    matching the span as it actually appears in the catalog. A phrase is only
    a phrase with its function words in place.
    """
    return [token.lower() for token in TOKEN_RE.findall(text)]


def phrase_clauses(text: str) -> list[list[str]]:
    """Tokens grouped by clause, for span building that respects punctuation.

    `phrase_tokens` discards punctuation, so an n-gram window slides straight
    across a sentence boundary: "A key requirement is: nylon." yields the span
    "a key requirement is nylon", which occurs in no catalog product and still
    consumes one of the phrase leg's limited lookups. Catalog text never spans
    a customer's clause break, so neither should a candidate span.

    Splitting on punctuation rather than on known template wording is
    deliberate. Matching the local simulator's own phrasing would score well
    here and transfer nothing (CLAUDE.md #12/#13); clause structure is a
    property of written language, not of this evaluator.
    """
    return [tokens for tokens in
            ([token.lower() for token in TOKEN_RE.findall(part)]
             for part in CLAUSE_RE.split(text))
            if tokens]


def product_passage(product: dict, max_chars: int = 700) -> str:
    """Build the text a product is embedded from for dense retrieval.

    Order matters for truncation: title and category are the strongest
    disambiguating signal, so they go first and always survive the cap.
    """
    parts = [
        field_text(product.get("title")),
        field_text(product.get("categories")),
        field_text(product.get("store")),
        field_text(product.get("features")),
        field_text(product.get("details")),
        field_text(product.get("description")),
    ]
    text = " | ".join(part for part in parts if part)
    return text[:max_chars]
