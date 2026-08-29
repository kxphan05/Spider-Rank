from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


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
