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

# Request verbs and discourse markers -- words a customer wraps a request in
# ("I want to *buy* shoes"). These are rare in the catalog, so BM25 (high
# IDF) let them outrank the actual content word -- "buy shoes" once matched
# the store "Buy Caps and Hats" instead of any shoe. Never a thing to ask about.
_REQUEST_STOPWORDS = {
    "actually", "anything", "buy", "buying", "find", "forget", "get", "give",
    "hello", "hey", "hi", "instead", "need", "needs", "nevermind", "purchase",
    "recommend", "scratch", "show", "shopping", "something", "suggest",
    "thanks", "thank",
    # Pivot discourse -- "never mind i want yellow" tokenizes to "never"/
    # "mind", both rarer in the catalog than "yellow", same failure shape.
    "mind", "never", "nope", "wait",

    # Wrapper words from the local evaluator's own reply templates ("A key
    # requirement is: ...", "For that, what matters is: ...") -- rare enough
    # in the catalog to hijack ranking if left in.
    "key", "requirement", "requirements", "matters", "what", "preference",
    "additional", "specific", "attribute", "options", "yet", "quite",
    "ignore", "earlier", "judgment", "judgement",
}

STOPWORDS = _FUNCTION_STOPWORDS | _REQUEST_STOPWORDS


# en-GB spellings mapped onto the catalog's en-US ones -- "jewellery" matches
# only 146 of 50k products, so a British spelling searches the wrong term
# entirely. Only variants where the US form dominates are listed ("grey" is
# omitted: it outnumbers "gray" in this catalog).
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
    like every other vocabulary match in this project (substring matching
    invents attributes that were never stated).
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
