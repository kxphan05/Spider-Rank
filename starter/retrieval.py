"""Hybrid (BM25 + dense) retrieval with reciprocal rank fusion.

Both legs return ranked parent_asin lists; fusion combines them by rank
position rather than raw score, so no cross-scale normalization is needed
between BM25's unbounded score and dense cosine similarity.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from collections import Counter
from pathlib import Path

import numpy as np

from .paths import catalog_fingerprint, model_cache_dir
from . import prf
from .attributes import COLORS, MATERIALS
from .text_utils import (STOPWORDS, field_text, normalize_query, phrase_clauses,
                         phrase_tokens, terms)

logger = logging.getLogger(__name__)

# Words that describe a product rather than name a kind of product.
ATTRIBUTE_WORDS = {word.lower() for word in COLORS} | {word.lower() for word in MATERIALS}

DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DENSE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_embedding_model():
    """Load the shared bge-small model once; callers (DenseIndex, the
    embedding-based intent classifier) reuse the same instance rather than
    each loading their own copy of the model.
    """
    import torch  # deferred: slow import, only needed once a model actually loads
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    return SentenceTransformer(DENSE_MODEL_NAME, cache_folder=model_cache_dir())


# Verbatim-span matching parameters (BM25Index.phrase_search).
PHRASE_MIN_N = 2
PHRASE_MAX_N = 5
# Per-query budget on phrase lookups. Each is an indexed FTS5 phrase scan, so
# they are cheap, but an accumulated 10-turn query has a long token tail.
MAX_PHRASE_QUERIES = 24
# A span appearing in more than this many products carries no identifying
# information, so it is dropped rather than scored.
PHRASE_MAX_MATCHES = 150
# Build spans within a clause instead of across the whole query, and require
# both edge tokens to carry content.
#
# The budget above is spent longest-first, on the assumption that a longer span
# is a more specific one. That holds for catalog copy and fails for
# conversation: the longest spans of a 10-turn query are sentences of filler
# that match nothing. Measured on public_0008's final query -- 135 spans built,
# and all 24 that fit the budget matched zero products, while "bras everyday
# bras" (verbatim in the target) sat unqueried at 3-gram depth.
#
# Two rules, both properties of language rather than of this simulator's
# wording -- a template blacklist would score well locally and transfer nothing
# (CLAUDE.md #12/#13):
#   1. no span crosses a clause boundary, since catalog text never does
#   2. no span begins or ends on a stopword, which is the standard
#      phrase-extraction heuristic for a span that is a fragment of one
PHRASE_CLAUSE_SPANS = False


# BM25F field weights, in the FTS5 column order declared above:
#
#   parent_asin, title, categories, features, details, store, description
#
# parent_asin is UNINDEXED so its weight is inert and pinned at 0.0. The rest
# were hand-picked when the index was first built and have never been swept --
# the IR literature is consistent that BM25F field weights are collection- and
# query-dependent and need a grid search, so hand-picked values on a 50k
# clothing catalog are unlikely to be right. `scripts/sweep_bm25_fields.py`
# sweeps them; DEFAULT_FIELD_WEIGHTS is the identity and must reproduce the
# shipped score exactly before any swept point is believed.
DEFAULT_FIELD_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)

# Every product in this catalog hangs off one root category
# ("Clothing, Shoes & Jewelry", 49,990 of 50,000), so the three words in it
# are present on essentially every document. BM25 gives a term appearing in
# ~100% of documents an IDF of ~0, which means the customer's own category
# word was worth nothing:
#
#     term      df with root      df with root dropped
#     shoes           1.000                     0.235
#     jewelry         1.000                     0.111
#     clothing        1.000                     0.433
#
# So "show me shoes" could not rank a shoe above a t-shirt -- the only query
# word that mattered was whatever conversational filler came with it (see
# _REQUEST_STOPWORDS in text_utils.py). Dropping the root restores the IDF of
# the leaf category words, which are the ones that actually discriminate.
#
# Detected rather than hardcoded: the root is whatever value leads
# `categories` on at least CATALOG_ROOT_MIN_SHARE of the first
# CATALOG_ROOT_SAMPLE products. Below that threshold nothing is stripped, so a
# catalog without a universal root is left exactly as it was. The decision is
# made from the first insert batch, which is still in memory, so this costs no
# extra pass over the catalog file.
CATALOG_ROOT_SAMPLE = 1000
CATALOG_ROOT_MIN_SHARE = 0.95

# What counts as naming a product category, for names_category() below.
#
# Only category nodes that are a SINGLE word qualify ("Shoes", "Jewelry",
# "Dresses", "Watches"). Tokens drawn out of multi-word nodes do not, because
# they are modifiers rather than product types: "Water Shoes" contributes
# "water", "Hand Wash Only" contributes "only", and treating either as a
# category pivot fires on ordinary attribute talk. Measured against the
# evaluator's own 30 override turns, the loose token rule fired on 6 of them
# and the single-word-node rule fires on 1.
#
# The df floor drops store names and one-off merchandising nodes ("Westlake",
# "Toddler Test"), which are category-shaped strings naming no product type.
CATEGORY_TERM_MIN_DF = 20

# Category nodes that describe who a product is for, not what it is. A pivot
# naming only one of these has not changed what the customer is shopping for.
DEMOGRAPHIC_CATEGORIES = {"women", "men", "girls", "boys", "baby", "kids",
                          "unisex", "adult", "novelty"}

FIELD_WEIGHTS = DEFAULT_FIELD_WEIGHTS


def _bm25_expression() -> str:
    """Render the ORDER BY term from the current FIELD_WEIGHTS.

    Read at call time rather than baked in at import, so a sweep can rebind
    the module constant between runs without rebuilding the index.
    """
    return "bm25(products, " + ", ".join(f"{w:g}" for w in FIELD_WEIGHTS) + ")"


class BM25Index:
    """FTS5-backed keyword retrieval over catalog text fields."""

    def __init__(self, catalog_path: Path) -> None:
        # check_same_thread=False because the index is read-only after
        # _build() and callers are not guaranteed to be single-threaded. The
        # evaluator is, but any server front-end is not: Streamlit caches one
        # Agent and serves turns from a pool of worker threads, which made
        # every BM25 and phrase query raise ProgrammingError and silently fall
        # back to the dense leg alone. The failure was invisible in the score
        # because _retrieve catches per-leg exceptions and continues.
        #
        # sqlite3 itself is built serialized here, but the guard below makes
        # the invariant explicit rather than relying on the build flag: every
        # query holds the lock, so concurrent readers cannot interleave on one
        # connection. Contention is irrelevant -- queries are sub-millisecond
        # and the evaluator never contends at all.
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._lock = threading.Lock()
        self.size = 0
        # Catalog document frequencies, memoised across the process for the
        # PRF leg (prf.py). Lives on the index rather than in prf.py so it is
        # scoped to the index it describes, not to the module.
        self._df_cache: dict[str, int] = {}
        self._build(catalog_path)

    @staticmethod
    def _detect_root_category(samples: list[list]) -> str | None:
        """The category value leading nearly every product, if there is one.

        See CATALOG_ROOT_MIN_SHARE. Returns None when no single value clears
        the threshold, in which case nothing is stripped.
        """
        leads = [str(categories[0]) for categories in samples if categories]
        if not leads:
            return None
        candidate = Counter(leads).most_common(1)[0]
        if candidate[1] / len(samples) < CATALOG_ROOT_MIN_SHARE:
            return None
        return candidate[0]

    def _build(self, catalog_path: Path) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        # Rows are held with `categories` still a list until the root is
        # known, so the sample that decides it can itself be stripped.
        batch: list[tuple] = []
        category_df: Counter[str] = Counter()
        count = 0
        root: str | None = None
        root_decided = False

        def flush() -> None:
            rows = []
            for row in batch:
                stripped = self._strip_root(row[2], root)
                categories = field_text(stripped)
                category_df.update({
                    node.strip().lower() for node in map(str, stripped)
                    if node and " " not in node.strip() and node.strip().isalpha()
                })
                rows.append(row[:2] + (categories,) + row[3:])
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
            batch.clear()

        with catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                product = json.loads(line)
                categories = product.get("categories")
                batch.append(
                    (
                        str(product["parent_asin"]),
                        field_text(product.get("title")),
                        categories if isinstance(categories, list) else [categories],
                        field_text(product.get("features")),
                        field_text(product.get("details")),
                        field_text(product.get("store")),
                        field_text(product.get("description")),
                    )
                )
                count += 1
                if not root_decided and len(batch) >= CATALOG_ROOT_SAMPLE:
                    root = self._detect_root_category([row[2] for row in batch])
                    root_decided = True
                    logger.info("BM25Index: universal root category = %r", root)
                if root_decided and len(batch) >= CATALOG_ROOT_SAMPLE:
                    flush()
        if not root_decided:
            root = self._detect_root_category([row[2] for row in batch])
            logger.info("BM25Index: universal root category = %r", root)
        if batch:
            flush()
        self.connection.commit()
        self.size = count
        self.category_terms = {
            term for term, df in category_df.items()
            if df >= CATEGORY_TERM_MIN_DF and term not in DEMOGRAPHIC_CATEGORIES
        }
        logger.info(
            "BM25Index: indexed %d products, %d category terms", count, len(self.category_terms)
        )

    def names_category(self, text: str) -> set[str]:
        """Category words the text names, excluding attribute vocabulary.

        Used to tell a pivot that changes *what the customer is shopping for*
        ("show me jewellery") from one that changes an attribute of it
        ("what I need is: leather"). Material and colour words are excluded
        because several of them are also category path elements -- "Leather"
        is a node under handbags -- and treating an attribute as a category
        change is precisely the mistake that made the whole-history rewrite in
        CLAUDE.md #17 cost 0.058.
        """
        found = set()
        for term in terms(normalize_query(text)):
            if term in ATTRIBUTE_WORDS:
                continue
            # Category nodes are plural ("Watches", "Dresses"); customers name
            # them in the singular at least as often ("show me a watch").
            for form in (term, term + "s", term + "es"):
                if form in self.category_terms:
                    found.add(form)
                    break
        return found

    @staticmethod
    def _strip_root(categories: list, root: str | None) -> list:
        if root is not None and categories and str(categories[0]) == root:
            return categories[1:]
        return categories

    def document_text(self, parent_asins: list[str]) -> dict[str, str]:
        """Title + categories + features for the given products.

        The cross-encoder needs the product text, and the FTS table is already
        the one place it lives in memory -- rereading the catalog file per turn
        would be the only alternative. Ordered title-first because the reranker
        truncates, and the title is the strongest disambiguator.
        """
        if not parent_asins:
            return {}
        placeholders = ",".join("?" * len(parent_asins))
        with self._lock:
            rows = self.connection.execute(
                f"SELECT parent_asin, title, categories, features FROM products "  # noqa: S608
                f"WHERE parent_asin IN ({placeholders})",
                parent_asins,
            ).fetchall()
        return {str(r[0]): " ".join(str(part) for part in r[1:] if part) for r in rows}

    def phrase_search(self, query: str, top_n: int) -> list[str]:
        """Rank products by how much of the query they contain *verbatim*.

        Motivation is a measured property of this task, not a general IR
        preference: 89.7% of the local simulator's turn-1 hard constraints are
        verbatim substrings of the target product's own catalog text
        (CLAUDE.md #14). `search()` above dissolves that structure -- it ORs
        the query's unique tokens, so "Buckle closure" is two independent
        terms against 50k products and any item mentioning either scores.
        FTS5 can match the span itself, which is a far sharper signal when the
        span is genuinely present.

        Each contiguous n-gram is run as an FTS5 phrase query and contributes
        `len(gram) ** 2` to every product containing it, so a five-word span
        outweighs six unrelated two-word ones. A phrase matching more than
        `PHRASE_MAX_MATCHES` products is discarded as non-discriminative
        rather than counted, which is what keeps common filler ("shoes and",
        "for the") from drowning the rare spans that actually identify a
        product.
        """
        clauses = (phrase_clauses(query) if PHRASE_CLAUSE_SPANS
                   else [phrase_tokens(query)])
        grams: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        # Longest first, so the per-query budget is spent on the most
        # specific spans available.
        for size in range(PHRASE_MAX_N, PHRASE_MIN_N - 1, -1):
            for tokens in clauses:
                for start in range(len(tokens) - size + 1):
                    gram = tuple(tokens[start:start + size])
                    if gram in seen or all(token in STOPWORDS for token in gram):
                        continue
                    if PHRASE_CLAUSE_SPANS and (gram[0] in STOPWORDS or gram[-1] in STOPWORDS):
                        # A span hanging off a stopword is a fragment of a
                        # phrase, not a phrase.
                        continue
                    seen.add(gram)
                    grams.append(gram)
        if not grams:
            return []

        scores: dict[str, float] = {}
        for gram in grams[:MAX_PHRASE_QUERIES]:
            expression = '"' + " ".join(gram) + '"'
            try:
                with self._lock:
                    rows = self.connection.execute(
                        "SELECT parent_asin FROM products WHERE products MATCH ? "
                        f"ORDER BY {_bm25_expression()} LIMIT ?",
                        (expression, PHRASE_MAX_MATCHES + 1),
                    ).fetchall()
            except sqlite3.OperationalError:
                # A gram can still form an invalid MATCH expression (an FTS5
                # keyword like NEAR/AND/OR as a bare token). Skip it rather
                # than lose the whole leg.
                continue
            if len(rows) > PHRASE_MAX_MATCHES:
                continue
            weight = float(len(gram) ** 2)
            for row in rows:
                pid = str(row[0])
                scores[pid] = scores.get(pid, 0.0) + weight
        if not scores:
            return []
        return [pid for pid, _ in sorted(scores.items(), key=lambda kv: -kv[1])][:top_n]

    def search(self, query: str, top_n: int) -> list[str]:
        unique_terms = list(dict.fromkeys(terms(query)))[:40]
        if not unique_terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        with self._lock:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                f"ORDER BY {_bm25_expression()} LIMIT ?",
                (expression, top_n),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def prf_search(self, query: str, top_n: int, seed: list[str]) -> list[str]:
        """Re-retrieve with terms harvested from the top `seed` results.

        `seed` is the ranking an earlier leg already produced, so the feedback
        round costs no extra retrieval -- only the expansion query itself.
        Returns [] whenever expansion finds nothing worth adding, which makes
        the caller's fusion leg simply absent rather than degenerate.
        """
        if not seed:
            return []
        feedback = seed[:prf.FEEDBACK_DOCS]
        documents = self.document_text(feedback)
        # Held across the whole call: expansion_terms issues one COUNT per
        # uncached candidate term on this same connection.
        with self._lock:
            expansion = prf.expansion_terms(
                documents, query, self.connection, self._df_cache, max(1, self.size)
            )
        if not expansion:
            return []
        # Anchor the expansion to the original query rather than replacing it.
        # Searching the expansion terms alone drifts badly and visibly: for
        # "Men's Shoes ... leather" the harvested terms are `coats, jackets,
        # faux, jacket, genuine, collar, zipper`, because leather in this
        # catalog is dominated by outerwear -- so the leg would retrieve
        # jackets for a shoe query. RM3 interpolates the expansion *into* the
        # original query model for exactly this reason; ORing both term sets
        # is the FTS5-shaped version of that.
        original = list(dict.fromkeys(terms(query)))[:40]
        expression = " OR ".join(f'"{term}"' for term in original + expansion)
        try:
            with self._lock:
                rows = self.connection.execute(
                    "SELECT parent_asin FROM products WHERE products MATCH ? "
                    f"ORDER BY {_bm25_expression()} LIMIT ?",
                    (expression, top_n),
                ).fetchall()
        except sqlite3.OperationalError:
            logger.debug("PRF expansion produced an invalid MATCH expression: %r", expression)
            return []
        return [str(row[0]) for row in rows]


class DenseIndex:
    """Cosine-similarity retrieval over precomputed catalog embeddings.

    Requires `scripts/build_dense_index.py` to have been run first; this
    class only loads the cached artifacts, it never encodes the catalog.
    """

    def __init__(self, index_dir: Path, model=None, catalog_path: Path | None = None) -> None:
        embeddings_path = index_dir / "embeddings.npy"
        ids_path = index_dir / "ids.json"
        meta_path = index_dir / "meta.json"
        for path in (embeddings_path, ids_path, meta_path):
            if not path.exists():
                raise RuntimeError(
                    f"dense index artifact missing: {path}. "
                    "Run `uv run python3 scripts/build_dense_index.py` first."
                )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("model_name") != DENSE_MODEL_NAME:
            logger.warning(
                "dense index was built with model %r but agent expects %r",
                meta.get("model_name"), DENSE_MODEL_NAME,
            )
        if catalog_path is not None and catalog_path.exists():
            # Prefer the content hash. mtime is a property of *this* filesystem
            # and says nothing once the index travels to another machine -- a
            # prebuilt index shipped with a submission is always "older" than
            # the catalog the organizer checked out, so the mtime rule warned
            # on every correct setup while still missing a genuinely different
            # catalog. mtime remains the fallback for indexes built before the
            # hash was recorded.
            built_sha = meta.get("catalog_sha256")
            if built_sha is not None:
                current_sha = catalog_fingerprint(catalog_path)
                if current_sha != built_sha:
                    logger.warning(
                        "dense index at %s was built from a different %s "
                        "(index catalog_sha256=%s, current=%s); embeddings are stale. "
                        "Re-run scripts/build_dense_index.py to refresh.",
                        index_dir, catalog_path, built_sha[:12], current_sha[:12],
                    )
            else:
                current_mtime = catalog_path.stat().st_mtime
                built_mtime = meta.get("catalog_mtime")
                if built_mtime is not None and current_mtime > built_mtime:
                    logger.warning(
                        "dense index at %s predates %s and records no catalog_sha256 "
                        "(index catalog_mtime=%s, current mtime=%s); embeddings may be "
                        "stale. Re-run scripts/build_dense_index.py to refresh.",
                        index_dir, catalog_path, built_mtime, current_mtime,
                    )
        self.ids: list[str] = json.loads(ids_path.read_text(encoding="utf-8"))
        self.embeddings = np.load(embeddings_path)
        if self.embeddings.shape[0] != len(self.ids):
            raise RuntimeError(
                f"dense index corrupt: {self.embeddings.shape[0]} embeddings "
                f"but {len(self.ids)} ids"
            )
        logger.info(
            "DenseIndex: loaded %d embeddings (dim=%d) from %s",
            len(self.ids), self.embeddings.shape[1], index_dir,
        )

        self.model = model if model is not None else load_embedding_model()
        self._id_to_idx = {pid: i for i, pid in enumerate(self.ids)}

    def rank_subset(self, query: str, candidate_ids: list[str]) -> list[str]:
        """Cosine-rank a caller-supplied candidate set, not the whole catalog.

        Used after a structural-attribute filter has already narrowed the
        pool: fusion's rank-position blend doesn't apply to a subset this
        small/pre-selected, so this re-scores by raw dense cosine similarity
        directly instead.
        """
        if not query.strip() or not candidate_ids:
            return list(candidate_ids)
        query_vec = self.model.encode(
            DENSE_QUERY_PREFIX + query, normalize_embeddings=True, convert_to_numpy=True
        )
        known = [(cid, self._id_to_idx[cid]) for cid in candidate_ids if cid in self._id_to_idx]
        if not known:
            return list(candidate_ids)
        known_ids, indices = zip(*known, strict=True)
        scores = self.embeddings[list(indices)] @ query_vec
        order = np.argsort(-scores)
        ranked = [known_ids[i] for i in order]
        known_set = set(known_ids)
        missing = [cid for cid in candidate_ids if cid not in known_set]
        return ranked + missing

    def search(self, query: str, top_n: int) -> list[str]:
        if not query.strip():
            return []
        query_vec = self.model.encode(
            DENSE_QUERY_PREFIX + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        scores = self.embeddings @ query_vec
        top_n = min(top_n, len(self.ids))
        # argpartition is O(n) vs full sort O(n log n); fine for a 50k catalog per turn.
        top_idx = np.argpartition(-scores, top_n - 1)[:top_n]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [self.ids[i] for i in top_idx]

    def vectors_for(self, candidate_ids: list[str]) -> tuple[list[str], np.ndarray]:
        """Return (known_ids, embeddings) for the ids present in the index,
        preserving `candidate_ids` order. Ids not in the index are dropped
        rather than erroring, same "skip what's missing" contract as
        `rank_subset`'s known/missing split.
        """
        known = [(cid, self._id_to_idx[cid]) for cid in candidate_ids if cid in self._id_to_idx]
        if not known:
            return [], np.empty((0, self.embeddings.shape[1]))
        known_ids, indices = zip(*known, strict=True)
        return list(known_ids), self.embeddings[list(indices)]


def reciprocal_rank_fusion(
    rank_lists: list[list[str]], top_n: int, k: int = 60, weights: list[float] | None = None
) -> list[str]:
    """Merge ranked id lists by rank position: score = sum(weight / (k + rank)).

    Equal weights let a leg that's absent/weak on a given query (e.g. dense
    missing an item entirely) get outvoted by a competitor both legs rank
    only moderately -- measured empirically on this catalog: BM25 is the
    higher-precision leg on these near-exact-match queries, so it gets
    weighted higher by default.
    """
    if weights is None:
        weights = [1.0] * len(rank_lists)
    if len(weights) != len(rank_lists):
        raise ValueError(f"weights length {len(weights)} != rank_lists length {len(rank_lists)}")
    scores: dict[str, float] = {}
    for rank_list, weight in zip(rank_lists, weights, strict=True):
        for rank, item_id in enumerate(rank_list, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank)
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [item_id for item_id, _ in ranked[:top_n]]
