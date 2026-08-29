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
from pathlib import Path

import numpy as np

from .paths import catalog_fingerprint, model_cache_dir
from . import prf
from .text_utils import STOPWORDS, field_text, phrase_tokens, terms

logger = logging.getLogger(__name__)

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
        self.connection = sqlite3.connect(":memory:")
        self.size = 0
        # Catalog document frequencies, memoised across the process for the
        # PRF leg (prf.py). Lives on the index rather than in prf.py so it is
        # scoped to the index it describes, not to the module.
        self._df_cache: dict[str, int] = {}
        self._build(catalog_path)

    def _build(self, catalog_path: Path) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        count = 0
        with catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        field_text(product.get("title")),
                        field_text(product.get("categories")),
                        field_text(product.get("features")),
                        field_text(product.get("details")),
                        field_text(product.get("store")),
                        field_text(product.get("description")),
                    )
                )
                count += 1
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()
        self.size = count
        logger.info("BM25Index: indexed %d products", count)

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
        tokens = phrase_tokens(query)
        grams: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        # Longest first, so the per-query budget is spent on the most
        # specific spans available.
        for size in range(PHRASE_MAX_N, PHRASE_MIN_N - 1, -1):
            for start in range(len(tokens) - size + 1):
                gram = tuple(tokens[start:start + size])
                if gram in seen or all(token in STOPWORDS for token in gram):
                    continue
                seen.add(gram)
                grams.append(gram)
        if not grams:
            return []

        scores: dict[str, float] = {}
        for gram in grams[:MAX_PHRASE_QUERIES]:
            expression = '"' + " ".join(gram) + '"'
            try:
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
