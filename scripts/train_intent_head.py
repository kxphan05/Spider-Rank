"""Train a buying-vs-browsing head on frozen bge-small embeddings. Do not ship it.

Scope: the encoder is never touched. TODO.md 4.3 bars fine-tuning base models
while intent modules are explicitly in scope, so a logistic head over the frozen
384-d output is the only allowed version of this idea.

**Measured verdict, CLAUDE.md #12: it is worse than the centroid it would
replace.** The simulator has exactly two turn-1 templates, so a high
in-distribution CV score is the head memorizing "still exploring". Three checks
confirm it: a regularization sweep is flat at chance on held-out surface forms
across five orders of magnitude of C; OOD accuracy *rises* with C, the opposite of
a generalization story; and training on the 20 hand-written prototypes alone, with
no simulator text, reaches OOD 1.000 while 640 labelled simulator turns drag it to
0.812.

Kept as a diagnostic. Re-run it only if the hidden set's phrasing turns out more
varied than the local templates -- the one condition that would change the answer.
Saving is opt-in.

    uv run python3 scripts/train_intent_head.py [--save]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from _common import DEFAULT_CATALOG, DEFAULT_DATASET, REPO_ROOT  # noqa: F401

# _common puts the repo root on sys.path as an import side effect, so the
# starter/evaluator imports below resolve when this script is run directly.

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from starter.classifier import EmbeddingIntentClassifier  # noqa: E402
from starter.retrieval import load_embedding_model  # noqa: E402

HEAD_PATH = Path("model/intent_head.npz")
LABELS = ("browsing", "buying")  # index 0 / 1, matching the trained head's sign

# Out-of-distribution probes: neither simulator template's vocabulary. Written
# to cover registers the templates never produce -- terse fragments, questions,
# and hedges that do not contain "still exploring".
PROBE_BUYING = (
    "size 9 black leather ankle boots, waterproof",
    "I need a stainless steel watch band, 22mm, brushed finish",
    "must be 100% merino wool, machine washable",
    "navy cotton oxford shirt, slim fit, 16.5 neck",
    "looking for a silver pendant necklace under $40",
    "do you carry these in a wide width, size 11",
    "I want the mesh running shoe with the carbon plate",
    "gold hoop earrings, hypoallergenic, small",
)
PROBE_BROWSING = (
    "what's popular in jewelry right now?",
    "just seeing what you have",
    "show me some ideas for a gift",
    "I could go either way on style, surprise me",
    "anything interesting in outerwear?",
    "hmm, let me see the options first",
    "no strong feelings, what would you pick?",
    "browsing for something to wear to a wedding, undecided",
)


def harvest(dataset: str, catalog: str) -> tuple[list[str], list[int], list[int], list[str]]:
    """Reconstruct labelled turn text from the evaluator's own reply templates.

    Returns (texts, labels, groups, stages). `groups` is the sample index, so
    cross-validation can split by sample and never put two turns from one
    session on both sides of a fold. `stages` marks "turn1" vs "accumulated"
    for the template-held-out split.
    """
    samples = load_jsonl(dataset)
    _, categories, products = catalog_index(catalog)
    texts: list[str] = []
    labels: list[int] = []
    groups: list[int] = []
    stages: list[str] = []

    for index, sample in enumerate(samples):
        scenario = sample["scenario_type"]
        # Only buying/browsing carry a ground-truth intent label. intent_override
        # and boundary sessions are a different axis (they are handled by the
        # override detector and by customer_reply's decline branch), and folding
        # them in would train the head on a label it cannot actually predict.
        if scenario not in ("buying", "browsing"):
            continue
        label = LABELS.index(scenario)
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        category = coarse_category(categories.get(target, []))

        turn1 = initial_message(effective, category, disclosed)
        texts.append(turn1); labels.append(label); groups.append(index); stages.append("turn1")

        # The agent classifies accumulated text, not the raw turn -- so train
        # on that shape too (agent.py: signal = classify(_build_query(state))).
        accumulated = turn1
        boundary_used = False
        for attribute in ("material", "color", "feature"):
            reply, boundary_used = customer_reply(effective, attribute, disclosed, boundary_used)
            accumulated += " " + reply
            texts.append(accumulated)
            labels.append(label); groups.append(index); stages.append("accumulated")
    return texts, labels, groups, stages


def fit_logistic(features: np.ndarray, targets: np.ndarray, seed: int = 0,
                 penalty: float = 1.0) -> tuple[np.ndarray, float]:
    """L2 logistic regression. scikit-learn is used offline only; the shipped
    artifact is plain numpy so the runtime keeps its current dependency set."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=penalty, max_iter=2000, random_state=seed)
    model.fit(features, targets)
    return model.coef_[0].astype(np.float32), float(model.intercept_[0])


def accuracy(weights: np.ndarray, bias: float, features: np.ndarray, targets: np.ndarray) -> float:
    return float(((features @ weights + bias > 0).astype(int) == targets).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--save", action="store_true",
                        help="write model/intent_head.npz. Off by default: the head measures WORSE "
                             "than the zero-shot centroid out of distribution (see module docstring).")
    args = parser.parse_args()

    print("loading frozen encoder (bge-small, not modified)...", file=sys.stderr)
    model = load_embedding_model()
    centroid_classifier = EmbeddingIntentClassifier(model)

    print("harvesting labelled turns...", file=sys.stderr)
    texts, label_list, group_list, stage_list = harvest(args.dataset, args.catalog)
    targets = np.array(label_list)
    groups = np.array(group_list)
    stages = np.array(stage_list)
    print(f"  {len(texts)} turns  ({int((targets == 1).sum())} buying / {int((targets == 0).sum())} browsing)",
          file=sys.stderr)

    features = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=128)
    probe_texts = list(PROBE_BUYING) + list(PROBE_BROWSING)
    probe_targets = np.array([1] * len(PROBE_BUYING) + [0] * len(PROBE_BROWSING))
    probe_features = model.encode(probe_texts, normalize_embeddings=True, convert_to_numpy=True)

    def centroid_accuracy(sentences, expected) -> float:
        predicted = np.array([LABELS.index(centroid_classifier.classify(t).label) for t in sentences])
        return float((predicted == expected).mean())

    print("\n=== 1. in-distribution, grouped 5-fold CV (expect ~1.0, ignore it) ===")
    from sklearn.model_selection import GroupKFold

    scores = []
    for train_index, test_index in GroupKFold(n_splits=5).split(features, targets, groups):
        weights, bias = fit_logistic(features[train_index], targets[train_index])
        scores.append(accuracy(weights, bias, features[test_index], targets[test_index]))
    print(f"  head      {np.mean(scores):.3f}")
    print(f"  centroid  {centroid_accuracy(texts, targets):.3f}")

    print("\n=== 2. template-held-out (train on one surface form, test on the other) ===")
    for train_stage, test_stage in (("turn1", "accumulated"), ("accumulated", "turn1")):
        train_mask, test_mask = stages == train_stage, stages == test_stage
        weights, bias = fit_logistic(features[train_mask], targets[train_mask])
        head = accuracy(weights, bias, features[test_mask], targets[test_mask])
        centroid = centroid_accuracy([t for t, m in zip(texts, test_mask, strict=True) if m], targets[test_mask])
        print(f"  train {train_stage:12s} -> test {test_stage:12s}  head {head:.3f}   centroid {centroid:.3f}")

    print("\n=== 3. out-of-distribution probes (the only number that matters) ===")
    weights, bias = fit_logistic(features, targets)
    head_probe = accuracy(weights, bias, probe_features, probe_targets)
    print(f"  head      {head_probe:.3f}")
    print(f"  centroid  {centroid_accuracy(probe_texts, probe_targets):.3f}")
    predictions = (probe_features @ weights + bias > 0).astype(int)
    for text, want, got in zip(probe_texts, probe_targets, predictions, strict=True):
        print(f"    {'ok ' if want == got else 'BAD'} want={LABELS[want]:8s} got={LABELS[got]:8s} {text!r}")

    print("\n=== 4. regularization sweep: is the OOD gap just a bad C? ===")
    print(f"  {'C':>8} {'in-dist CV':>11} {'turn1->accum':>13} {'OOD probe':>10}")
    from sklearn.model_selection import GroupKFold as _GKF

    turn1_mask, accum_mask = stages == "turn1", stages == "accumulated"
    for penalty in (0.001, 0.01, 0.1, 1.0, 10.0):
        cv = []
        for train_index, test_index in _GKF(n_splits=5).split(features, targets, groups):
            w, b = fit_logistic(features[train_index], targets[train_index], penalty=penalty)
            cv.append(accuracy(w, b, features[test_index], targets[test_index]))
        w, b = fit_logistic(features[turn1_mask], targets[turn1_mask], penalty=penalty)
        held = accuracy(w, b, features[accum_mask], targets[accum_mask])
        w, b = fit_logistic(features, targets, penalty=penalty)
        ood = accuracy(w, b, probe_features, probe_targets)
        print(f"  {penalty:>8} {np.mean(cv):>11.3f} {held:>13.3f} {ood:>10.3f}")

    print("\n=== 5. control: train on the hand-written PROTOTYPES only (no simulator text) ===")
    from starter.classifier import PROTOTYPE_BROWSING, PROTOTYPE_BUYING

    proto_texts = list(PROTOTYPE_BUYING) + list(PROTOTYPE_BROWSING)
    proto_targets = np.array([1] * len(PROTOTYPE_BUYING) + [0] * len(PROTOTYPE_BROWSING))
    proto_features = model.encode(proto_texts, normalize_embeddings=True, convert_to_numpy=True)
    for penalty in (0.01, 0.1, 1.0):
        w, b = fit_logistic(proto_features, proto_targets, penalty=penalty)
        print(f"  C={penalty:<6} OOD probe {accuracy(w, b, probe_features, probe_targets):.3f}"
              f"   simulator turns {accuracy(w, b, features, targets):.3f}")

    if not args.save:
        print("\nhead not written (pass --save to override; see module docstring "
              "for why the default is not to ship it)", file=sys.stderr)
        return
    HEAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        HEAD_PATH,
        weights=weights.astype(np.float32),
        bias=np.float32(bias),
        labels=np.array(LABELS),
        encoder=np.array("BAAI/bge-small-en-v1.5"),
        train_turns=np.int32(len(texts)),
    )
    print(f"\nwrote {HEAD_PATH} ({HEAD_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
