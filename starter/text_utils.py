from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


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
