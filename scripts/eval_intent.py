"""Sweep scoring rules for EmbeddingIntentClassifier (buying vs browsing).

Companion to eval_override.py. Verdict: every trimmed-prototype
variant is worse than the centroid, and there is no headroom to chase -- the
centroid is at 0.988 on turn-1 and the label is worth ~nothing to the score.

Three pools, two of which mean anything. `turn-1` is the honest in-distribution
number. `accumulated` is reported for shape only and is **not** an accuracy:
scenario_type stops being ground truth once answers land, because the classifier
is built to drift buying-ward (classifier.py:14). `OOD probes` are hand-written
and hand-picked -- they can falsify a rule but must not rank two rules that both
pass. The lexical fallback's 1.000 on turn-1 is template memorization.

Read-only and Agent-free.

    uv run python3 scripts/eval_intent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from _common import DEFAULT_CATALOG, DEFAULT_DATASET, REPO_ROOT  # noqa: F401

# _common puts the repo root on sys.path as an import side effect, so the
# starter/evaluator imports below resolve when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from starter import classifier  # noqa: E402
from starter.classifier import (  # noqa: E402
    PROTOTYPE_BROWSING,
    PROTOTYPE_BUYING,
    _top_prototype_similarity,
    classify_intent,
)
from starter.retrieval import load_embedding_model  # noqa: E402
from train_intent_head import LABELS, PROBE_BROWSING, PROBE_BUYING, harvest  # noqa: E402


def centroid(vectors: np.ndarray) -> np.ndarray:
    mean = vectors.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


def main() -> None:
    print("loading embedding model...", file=sys.stderr)
    model = load_embedding_model()

    buying_protos = model.encode(list(PROTOTYPE_BUYING), normalize_embeddings=True, convert_to_numpy=True)
    browsing_protos = model.encode(list(PROTOTYPE_BROWSING), normalize_embeddings=True, convert_to_numpy=True)
    buying_centroid, browsing_centroid = centroid(buying_protos), centroid(browsing_protos)

    print("harvesting labelled turns...", file=sys.stderr)
    texts, label_list, _, stage_list = harvest(DEFAULT_DATASET, DEFAULT_CATALOG)
    labels = np.array(label_list)
    stages = np.array(stage_list)

    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=128)
    probe_texts = list(PROBE_BUYING) + list(PROBE_BROWSING)
    probe_labels = np.array([1] * len(PROBE_BUYING) + [0] * len(PROBE_BROWSING))
    probe_vectors = model.encode(probe_texts, normalize_embeddings=True, convert_to_numpy=True)

    pools = {
        "turn-1": (vectors[stages == "turn1"], labels[stages == "turn1"]),
        "accumulated": (vectors[stages == "accumulated"], labels[stages == "accumulated"]),
        "OOD probe": (probe_vectors, probe_labels),
    }

    def score_rule(vecs: np.ndarray, k: int | None) -> np.ndarray:
        """Predicted label index. k=None is the centroid rule (current)."""
        if k is None:
            return (vecs @ buying_centroid > vecs @ browsing_centroid).astype(int)
        return np.array([
            int(_top_prototype_similarity(v, buying_protos) > _top_prototype_similarity(v, browsing_protos))
            for v in vecs
        ])

    print(f"\n{'rule':>22}" + "".join(f"{name:>14}" for name in pools))
    print("-" * (22 + 14 * len(pools)))

    # Lexical fallback, as a floor.
    row = f"{'lexical (fallback)':>22}"
    for name, (_, expected) in pools.items():
        subset = [t for t, s in zip(texts, stages, strict=True) if (s == "turn1") == (name == "turn-1")] \
            if name != "OOD probe" else probe_texts
        if name == "accumulated":
            subset = [t for t, s in zip(texts, stages, strict=True) if s == "accumulated"]
        elif name == "turn-1":
            subset = [t for t, s in zip(texts, stages, strict=True) if s == "turn1"]
        predicted = np.array([LABELS.index(classify_intent(t).label) for t in subset])
        row += f"{float((predicted == expected).mean()):>14.3f}"
    print(row)

    original_k = classifier.TOP_PROTOTYPES
    for k in (None, 1, 2, 3, 4, 5, 6, 8, 10):
        classifier.TOP_PROTOTYPES = k if k is not None else original_k
        name_str = "centroid (current)" if k is None else f"top{k}-mean"
        row = f"{name_str:>22}"
        for _, (vecs, expected) in pools.items():
            if k is None:
                predicted = score_rule(vecs, None)
            else:
                predicted = np.array([
                    int(np.sort(vecs[i] @ buying_protos.T)[-k:].mean()
                        > np.sort(vecs[i] @ browsing_protos.T)[-k:].mean())
                    for i in range(len(vecs))
                ])
            row += f"{float((predicted == expected).mean()):>14.3f}"
        print(row)
    classifier.TOP_PROTOTYPES = original_k

    # Where does the current rule actually go wrong on turn-1?
    turn1_texts = [t for t, s in zip(texts, stages, strict=True) if s == "turn1"]
    turn1_vecs, turn1_labels = pools["turn-1"]
    predicted = score_rule(turn1_vecs, None)
    print("\n--- turn-1 errors, centroid rule (first 8 of each direction) ---")
    for want in (0, 1):
        wrong = [t for t, w, p in zip(turn1_texts, turn1_labels, predicted, strict=True) if w == want and p != want]
        print(f"  want {LABELS[want]} got {LABELS[1 - want]}: {len(wrong)} of {int((turn1_labels == want).sum())}")
        for text in wrong[:8]:
            print(f"    {text[:105]!r}")


if __name__ == "__main__":
    main()
