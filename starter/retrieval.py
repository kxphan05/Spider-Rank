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
from .text_utils import (STOPWORDS, field_text, normalize_query,
                         phrase_tokens, terms)
from .config import (CATALOG_ROOT_MIN_SHARE, CATALOG_ROOT_SAMPLE, CATEGORY_TERM_MIN_DF, 
    DEFAULT_FIELD_WEIGHTS, DENSE_MODEL_NAME, DENSE_QUERY_PREFIX, MAX_PHRASE_QUERIES, 
    PHRASE_MAX_MATCHES, PHRASE_MAX_N, PHRASE_MIN_N)

logger = logging.getLogger(__name__)

# Words that describe a product rather than name a kind of product.
ATTRIBUTE_WORDS = {word.lower() for word in COLORS} | {word.lower() for word in MATERIALS}



def load_embedding_model():
    """Load the shared bge-small model once; callers (DenseIndex, the
    embedding-based intent classifier) reuse the same instance rather than
    each loading their own copy of the model.
    """
    import torch  # deferred: slow import, only needed once a model actually loads
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    return SentenceTransformer(DENSE_MODEL_NAME, cache_folder=model_cache_dir())







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
        # check_same_thread=False: a server front-end (unlike the evaluator)
        # serves turns from a worker-thread pool, so the connection is shared.
        # The lock below makes that safe explicitly rather than relying on sqlite3's build flag.
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._lock = threading.Lock()
        self.size = 0
        # Catalog document frequencies, memoised for the PRF leg (prf.py).
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

        Tells a pivot that changes *what* the customer shops for from one
        that changes an attribute ("Leather" can be both a color and category).
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

        Read from the in-memory FTS table. Ordered title-first since the
        reranker truncates and the title is the strongest disambiguator.
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

        89.7% of turn-1 constraints are verbatim substrings of the target's
        catalog text, so phrase n-grams (weighted `len(gram) ** 2`) beat search()'s OR-of-tokens.
        """
        clauses = ([phrase_tokens(query)])
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
                # A gram can form an invalid MATCH expression (e.g. an FTS5
                # keyword like NEAR/AND/OR as a bare token) -- skip it.
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

        `seed` is a ranking an earlier leg already produced, so this costs no
        extra retrieval beyond the expansion query itself.
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
        # Anchor to the original query rather than replacing it -- expansion
        # terms alone can drift onto an unrelated category (RM3-style).
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
            # Prefer the content hash: mtime doesn't survive the index
            # traveling to another machine. Kept as a fallback for indexes
            # built before the hash was recorded.
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
        pool, where fusion's rank-position blend no longer applies.
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
        """(known_ids, embeddings) for ids present in the index, order preserved.

        Ids not in the index are dropped, same as `rank_subset`.
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

    BM25 is the higher-precision leg on these near-exact-match queries, so
    it's weighted higher by default.
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
