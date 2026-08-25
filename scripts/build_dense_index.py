"""Precompute dense (bge-small-en-v1.5) embeddings for the frozen catalog.

Run once (or whenever the catalog / model / passage format changes); the
agent loads the cached .npy/.json artifacts at init time instead of
re-encoding 50k products on every evaluator run.

Smoke test (fast, ~500 products) before committing to the full run:
    uv run python3 scripts/build_dense_index.py --limit 500

Full build:
    uv run python3 scripts/build_dense_index.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import os

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from starter.text_utils import product_passage  # noqa: E402

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_OUT_DIR = Path("data/dense_index")
MAX_CHARS = 700

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("build_dense_index")


def load_catalog(path: Path, limit: int | None) -> list[dict]:
    products: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                product = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
            if "parent_asin" not in product:
                raise ValueError(f"{path}:{line_no}: missing required field 'parent_asin'")
            products.append(product)
            if limit is not None and len(products) >= limit:
                break
    if not products:
        raise ValueError(f"{path}: no products loaded")
    return products


def build(catalog_path: Path, out_dir: Path, limit: int | None, batch_size: int) -> None:
    start = time.monotonic()
    logger.info("loading catalog from %s (limit=%s)", catalog_path, limit)
    products = load_catalog(catalog_path, limit)
    logger.info("loaded %d products in %.1fs", len(products), time.monotonic() - start)

    ids = [str(product["parent_asin"]) for product in products]
    passages = [product_passage(product, max_chars=MAX_CHARS) for product in products]
    empty_count = sum(1 for passage in passages if not passage)
    if empty_count:
        logger.warning("%d/%d products produced an empty passage text", empty_count, len(products))

    logger.info("loading embedding model %s", MODEL_NAME)
    import torch  # deferred: slow import
    from sentence_transformers import SentenceTransformer  # deferred: slow import

    torch.set_num_threads(os.cpu_count() or 1)
    model_start = time.monotonic()
    model = SentenceTransformer(MODEL_NAME)
    logger.info("model loaded in %.1fs", time.monotonic() - model_start)

    encode_start = time.monotonic()
    embeddings = model.encode(
        passages,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    encode_elapsed = time.monotonic() - encode_start
    logger.info(
        "encoded %d passages in %.1fs (%.1f/s), dim=%d",
        len(passages), encode_elapsed, len(passages) / max(encode_elapsed, 1e-6), embeddings.shape[1],
    )

    if embeddings.shape[0] != len(ids):
        raise RuntimeError(
            f"embedding count {embeddings.shape[0]} does not match id count {len(ids)}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings)
    (out_dir / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    meta = {
        "model_name": MODEL_NAME,
        "dim": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
        "max_chars": MAX_CHARS,
        "catalog_path": str(catalog_path),
        "catalog_mtime": catalog_path.stat().st_mtime,
        "limit": limit,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("wrote index to %s (total %.1fs)", out_dir, time.monotonic() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cached dense embeddings for the catalog")
    parser.add_argument("--catalog", default="data/catalog.jsonl", type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    parser.add_argument("--limit", type=int, default=None, help="Only embed the first N products (smoke test)")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    build(args.catalog, args.out_dir, args.limit, args.batch_size)


if __name__ == "__main__":
    main()
