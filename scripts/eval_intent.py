"""Sweep scoring rules for EmbeddingIntentClassifier (buying vs browsing).

Companion to scripts/eval_override.py, same shape and same reason to exist:
the classifier scores against class *centroids*, and CLAUDE.md #7 documents
why that is fragile -- both centroids sit at 0.65-0.80 similarity to almost
anything, so the decision margin is inside the noise floor. The trimmed
nearest-prototype rule fixed exactly that for the override detector; this
measures whether it helps here too.

Note on what the classifier's output actually is: `signal.score` is used
nowhere except a log line (agent.py:310). Every consumer -- fusion weights,
the MMR diversity re-rank, the entropy threshold, _next_attribute -- branches
on `signal.label`. So this is a binary decision, not a continuous score, and
it is measured the same way the override detector is.

Three pools, and only two of them mean anything:

  turn-1        the simulator's opening message vs its scenario_type. This is
                the one honest in-distribution number.
  accumulated   opening message plus clarifying answers. Reported for shape
                only: scenario_type is NOT valid ground truth here, because
                the classifier is *designed* to drift buying-ward as concrete
                attributes land (classifier.py:14). A "browsing" session whose
                text now names a material and a size should read as buying,
                so disagreement in this pool is partly correct behaviour.
  OOD probes    hand-written utterances in neither template's vocabulary --
                the only evidence about the hidden grader's phrasing, and
                hand-picked, so a smoke test rather than a measurement.

Read-only and Agent-free. Run before touching the classifier.

Usage:
    uv run python3 scripts/eval_intent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
    texts, label_list, _, stage_list = harvest("data/public_set.jsonl", "data/catalog.jsonl")
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
        subset = [t for t, s in zip(texts, stages) if (s == "turn1") == (name == "turn-1")] \
            if name != "OOD probe" else probe_texts
        if name == "accumulated":
            subset = [t for t, s in zip(texts, stages) if s == "accumulated"]
        elif name == "turn-1":
            subset = [t for t, s in zip(texts, stages) if s == "turn1"]
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
    turn1_texts = [t for t, s in zip(texts, stages) if s == "turn1"]
    turn1_vecs, turn1_labels = pools["turn-1"]
    predicted = score_rule(turn1_vecs, None)
    print("\n--- turn-1 errors, centroid rule (first 8 of each direction) ---")
    for want in (0, 1):
        wrong = [t for t, w, p in zip(turn1_texts, turn1_labels, predicted) if w == want and p != want]
        print(f"  want {LABELS[want]} got {LABELS[1 - want]}: {len(wrong)} of {int((turn1_labels == want).sum())}")
        for text in wrong[:8]:
            print(f"    {text[:105]!r}")


if __name__ == "__main__":
    main()
