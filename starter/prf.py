"""Pseudo-relevance feedback (Rocchio / RM3-style query expansion).

Assume the top few documents an initial query retrieves are relevant,
harvest the terms that distinguish them from the collection, and search
again with those added. This helps to improve the vocabulary of the query.
"""
from __future__ import annotations

import math
import sqlite3
from collections import Counter

from .text_utils import STOPWORDS, terms
# FEEDBACK_DOCS is re-exported deliberately: retrieval.py reads it as
# `prf.FEEDBACK_DOCS` (module attribute, late-bound) so a sweep can patch it
# on this module. Ruff cannot see that use, hence the noqa.
from .config import (EXPANSION_TERMS, FEEDBACK_DOCS, MIN_FEEDBACK_DF,  # noqa: F401
                     MIN_TERM_LENGTH)


def _catalog_df(connection: sqlite3.Connection, term: str, cache: dict[str, int]) -> int:
    """Number of catalog products containing `term`, memoised per index.

    Expansion candidates repeat heavily across turns/sessions, so the cache
    is what makes the per-term COUNT query affordable.
    """
    hit = cache.get(term)
    if hit is not None:
        return hit
    try:
        row = connection.execute(
            "SELECT count(*) FROM products WHERE products MATCH ?", (f'"{term}"',)
        ).fetchone()
        count = int(row[0]) if row else 0
    except sqlite3.OperationalError:
        # An FTS5 keyword (NEAR/AND/OR) as a bare term is an invalid
        # expression. Treat it as ubiquitous so it scores zero and is dropped.
        count = 0
    cache[term] = count
    return count


def expansion_terms(
    documents: dict[str, str],
    query: str,
    connection: sqlite3.Connection,
    df_cache: dict[str, int],
    catalog_size: int,
    limit: int = EXPANSION_TERMS,
) -> list[str]:
    """Pick the terms that distinguish the feedback documents from the catalog.

    Scored as `feedback_df * idf`: must be common among the assumed-relevant
    docs *and* rare overall. Either alone fails -- raw frequency returns
    ubiquitous words, raw rarity returns each doc's serial number.
    """
    if not documents:
        return []

    query_terms = set(terms(query))
    # track the number of documents where the term appears.
    feedback_df: Counter[str] = Counter()
    for text in documents.values():
        for term in set(terms(text)):
            if (
                len(term) >= MIN_TERM_LENGTH
                and term not in STOPWORDS
                and term not in query_terms
                and not term.isdigit()
            ):
                feedback_df[term] += 1

    scored: list[tuple[float, str]] = []
    for term, count in feedback_df.items():
        if count < MIN_FEEDBACK_DF:
            continue
        df = _catalog_df(connection, term, df_cache)
        if df <= 0:
            continue
        # Standard idf, floored at zero so a term in more than half the
        # catalog contributes nothing rather than a negative score.
        idf = max(0.0, math.log(catalog_size / df))
        if idf <= 0.0:
            continue
        scored.append((count * idf, term))

    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [term for _, term in scored[:limit]]
