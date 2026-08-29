"""Pseudo-relevance feedback (Rocchio / RM3-style query expansion).

The idea is classical: assume the top few documents an initial query
retrieves are relevant, harvest the terms that distinguish them from the
collection, and search again with those terms added. No model, no training,
no extra asset -- it reads the index this repo already builds.

Why it is worth trying on *this* task specifically. A buying session's entire
turn-1 signal is usually one very common material word: the median hard
constraint is one token, and the modal values are `cotton` (9,414 catalog
products, 18.8%), `polyester` (13.8%) and `leather` (12.6%). Buying scores
0.688 hit rate against browsing's 0.825 for exactly that reason -- the
bottleneck is *information*, not retrieval tolerance, and no amount of
re-weighting the existing legs manufactures signal the query never carried.
PRF is the one direction available that adds vocabulary rather than
redistributing weight: it turns "cotton" plus a category into the words that
actually co-occur in the products that query already ranks well.

Two honest caveats, stated before the measurement.

**PRF drifts when the initial results are wrong.** The classic failure mode is
query drift, and it is worse on a benchmark where the target is often already
in the top handful -- there is more to lose than to gain on the sessions that
were already working. This is why the expansion ships as its own fusion leg
with a sweepable weight whose 0.0 is the exact identity, rather than as an
edit to the existing BM25 query: a leg can be turned down continuously, and a
rewritten query cannot.

**This is RM3-flavoured, not textbook RM3.** True RM3 interpolates an
expansion language model into the original query model with weight lambda,
which needs per-term query weighting; FTS5's MATCH has no such thing. Fusing a
separate expansion leg by weighted RRF is the adaptation that fits the engine,
and the RRF weight plays lambda's role. Recorded as an adaptation so nobody
later reads a citation match into it.
"""
from __future__ import annotations

import math
import sqlite3
from collections import Counter

from .text_utils import STOPWORDS, terms

# How many top-ranked products are assumed relevant. Classic PRF uses 10-20;
# smaller is safer here because a wrong assumption is what causes drift.
FEEDBACK_DOCS = 10

# How many expansion terms survive into the second query.
EXPANSION_TERMS = 8

# A term must appear in at least this many feedback documents to be
# considered. A term in one document out of ten is that document's
# idiosyncrasy, not a property of the result set.
MIN_FEEDBACK_DF = 3

# Terms shorter than this are dropped -- FTS5 tokenises aggressively and short
# fragments are almost always noise.
MIN_TERM_LENGTH = 3


def _catalog_df(connection: sqlite3.Connection, term: str, cache: dict[str, int]) -> int:
    """Number of catalog products containing `term`, memoised per index.

    One COUNT per distinct term, cached for the process lifetime. The cache is
    what makes this affordable: expansion candidates repeat heavily across
    turns and sessions, so the query count amortises to near zero over a full
    evaluation even though a cold turn issues a few dozen.
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

    Scored as `feedback_df * idf`: a term must be common among the assumed-
    relevant documents *and* rare in the collection overall. Either half alone
    fails -- raw frequency returns "cotton" and "shirt" (already in the query,
    or ubiquitous), raw rarity returns each document's own serial number.
    """
    if not documents:
        return []

    query_terms = set(terms(query))
    # Document frequency within the feedback set, not raw term frequency: a
    # term repeated twenty times in one long description should not outrank a
    # term present once in every document.
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
