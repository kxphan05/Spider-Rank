"""A/B: does prototype *sentence length* hurt EmbeddingIntentClassifier accuracy?

Hypothesis under test (from user): PROTOTYPE_BUYING / PROTOTYPE_BROWSING in
starter/classifier.py are long, multi-clause sentences that pack several
attributes into one embedding (e.g. "I want a wool sweater, crew neck, for
under $50" carries material + style + price), and that dilutes the buying/
browsing signal in the resulting vector, hurting nearest-centroid accuracy.

Design: hold the evaluation set fixed (turn-1 harvested customer utterances,
plus the hand-written OOD probes from train_intent_head.py) and vary only the
prototype set fed into the SAME centroid rule eval_intent.py already uses.

  A. baseline  -- current PROTOTYPE_BUYING / PROTOTYPE_BROWSING, verbatim.
  B. short     -- same 13/14 examples, same attributes, rewritten as short
                  fragments (content-matched, not just truncated -- truncating
                  mid-sentence would also strip the purchase-verb signal for
                  several of them, confounding length with cue-loss).
  C. truncated -- mechanical first-N-word truncation of A (no rewriting), to
                  isolate raw length from "someone rewrote these to be terser
                  and more idiomatic," which is a second, entangled variable
                  in B.

Read-only and Agent-free.

    uv run python3 scripts/ab_prototype_length.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from _common import DEFAULT_CATALOG, DEFAULT_DATASET, REPO_ROOT  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))

from starter.classifier import PROTOTYPE_BROWSING, PROTOTYPE_BUYING  # noqa: E402
from starter.retrieval import load_embedding_model  # noqa: E402
from train_intent_head import PROBE_BROWSING, PROBE_BUYING, harvest  # noqa: E402

# B: content-matched short rewrites, same order/index as the originals in
# classifier.py so each pair covers the same attributes.
SHORT_BUYING = (
    "black leather boots, size 9",
    "cotton t-shirt under $20",
    "waterproof, machine washable jacket",
    "silver necklace, small pendant",
    "slim fit dark wash jeans",
    "running shoes, arch support, size 10.5",
    "medium, navy blue",
    "wool crew neck sweater, under $50",
    "brown leather handbag, shoulder strap",
    "size 8 suede sandal",
    "buy shoes",
    "buy a watch",
    "purchase a handbag",
)
SHORT_BROWSING = (
    "just browsing shoes",
    "no particular preference",
    "not sure, exploring options",
    "open to anything, jewelry",
    "jacket, haven't decided",
    "not sure yet, just looking",
    "dresses, nothing specific in mind",
    "shopping around, no fixed idea",
    "curious about accessories",
    "not picky, whatever works",
    "something for summer",
    "wedding, open to suggestions",
    "beach, whatever works",
    "work, don't mind what",
)
assert len(SHORT_BUYING) == len(PROTOTYPE_BUYING)
assert len(SHORT_BROWSING) == len(PROTOTYPE_BROWSING)


def truncate(sentences: tuple[str, ...], n_words: int) -> tuple[str, ...]:
    out = []
    for s in sentences:
        words = s.rstrip(".?").split()
        out.append(" ".join(words[:n_words]))
    return tuple(out)


def centroid(vectors: np.ndarray) -> np.ndarray:
    mean = vectors.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


def mean_len(sentences: tuple[str, ...]) -> float:
    return float(np.mean([len(s.split()) for s in sentences]))


def main() -> None:
    print("loading embedding model...", file=sys.stderr)
    model = load_embedding_model()

    def encode(sentences):
        return model.encode(list(sentences), normalize_embeddings=True, convert_to_numpy=True)

    print("harvesting labelled turns...", file=sys.stderr)
    texts, label_list, _, stage_list = harvest(DEFAULT_DATASET, DEFAULT_CATALOG)
    labels = np.array(label_list)
    stages = np.array(stage_list)

    turn1_vecs = encode([t for t, s in zip(texts, stages, strict=True) if s == "turn1"])
    turn1_labels = labels[stages == "turn1"]

    probe_texts = list(PROBE_BUYING) + list(PROBE_BROWSING)
    probe_labels = np.array([1] * len(PROBE_BUYING) + [0] * len(PROBE_BROWSING))
    probe_vecs = encode(probe_texts)

    pools = {
        "turn-1": (turn1_vecs, turn1_labels),
        "OOD probe": (probe_vecs, probe_labels),
    }

    variants = {
        "A baseline (full)": (PROTOTYPE_BUYING, PROTOTYPE_BROWSING),
        "B short (rewritten)": (SHORT_BUYING, SHORT_BROWSING),
        "C truncated@4w": (truncate(PROTOTYPE_BUYING, 4), truncate(PROTOTYPE_BROWSING, 4)),
        "C truncated@6w": (truncate(PROTOTYPE_BUYING, 6), truncate(PROTOTYPE_BROWSING, 6)),
    }

    print(f"\n{'variant':>22}{'avg buy words':>15}{'avg brow words':>16}"
          + "".join(f"{name:>14}" for name in pools))
    print("-" * (22 + 15 + 16 + 14 * len(pools)))
    for name, (buy_protos, brow_protos) in variants.items():
        buy_vecs = encode(buy_protos)
        brow_vecs = encode(brow_protos)
        buy_c, brow_c = centroid(buy_vecs), centroid(brow_vecs)
        row = f"{name:>22}{mean_len(buy_protos):>15.1f}{mean_len(brow_protos):>16.1f}"
        for _, (vecs, expected) in pools.items():
            predicted = (vecs @ buy_c > vecs @ brow_c).astype(int)
            acc = float((predicted == expected).mean())
            row += f"{acc:>14.3f}"
        print(row)

    print(f"\n(pool sizes: turn-1 n={len(turn1_labels)}, OOD probe n={len(probe_labels)})")


if __name__ == "__main__":
    main()
