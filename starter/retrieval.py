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

from .text_utils import field_text, terms

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
    return SentenceTransformer(DENSE_MODEL_NAME, cache_folder="./model")


class BM25Index:
    """FTS5-backed keyword retrieval over catalog text fields."""

    def __init__(self, catalog_path: Path) -> None:
        self.connection = sqlite3.connect(":memory:")
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
        logger.info("BM25Index: indexed %d products", count)

    def search(self, query: str, top_n: int) -> list[str]:
        unique_terms = list(dict.fromkeys(terms(query)))[:40]
        if not unique_terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
            (expression, top_n),
        ).fetchall()
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
            current_mtime = catalog_path.stat().st_mtime
            built_mtime = meta.get("catalog_mtime")
            if built_mtime is not None and current_mtime > built_mtime:
                logger.warning(
                    "dense index at %s was built from an older version of %s "
                    "(index catalog_mtime=%s, current mtime=%s); embeddings may be stale. "
                    "Re-run scripts/build_dense_index.py to refresh.",
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
        known_ids, indices = zip(*known)
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
    for rank_list, weight in zip(rank_lists, weights):
        for rank, item_id in enumerate(rank_list, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank)
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [item_id for item_id, _ in ranked[:top_n]]
