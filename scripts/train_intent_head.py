"""Train a buying-vs-browsing classifier head on frozen bge-small embeddings.

Scope note: TODO.md 4.3 puts "training or full-parameter fine-tuning of base
foundational LLMs" OUT of scope, so the encoder is never touched -- bge-small
stays frozen and is used exactly as it already is for dense retrieval. What is
trained here is a logistic-regression head over its 384-d output, which falls
under the explicitly IN-scope "designing highly sensitive intent-detection
modules to split traffic into Buying and Browsing tracks". The artifact is a
few KB of numpy weights (model/intent_head.npz); loading it at runtime needs
numpy only, not scikit-learn.

Why a trained head at all: EmbeddingIntentClassifier scores by nearest
*centroid*, and CLAUDE.md #7 documents the failure mode -- decision margins
are ~0.03 in a space where both centroids sit at 0.65-0.80 similarity to
almost anything, so the boundary is inside the noise floor. A learned boundary
can weight the dimensions that actually separate the classes instead of
averaging them away.

THE THING TO WATCH: the local simulator's turn-1 templates are trivially
separable --

    buying    "I'm looking for {category}. A key requirement is: {constraint}."
    browsing  "I'm looking for {category}, but I'm still exploring."

-- so a head trained on them can hit ~100% locally by memorizing "still
exploring" and still be worthless (or harmful) on the hidden grader's
phrasing. This script therefore reports three numbers, and only the last two
mean anything:

  in-dist CV      grouped 5-fold over samples. Expect ~1.0. Ignore it.
  template-held-out  train on turn-1 text only, test on accumulated multi-turn
                  text (and vice versa), so the test half has a different
                  surface form from the training half.
  OOD probes      hand-written utterances in neither template's vocabulary.
                  This is the only evidence about generalization, and it is
                  hand-picked -- treat it as a smoke test, not a measurement.

The centroid classifier is scored on the same three pools as a control. If the
head does not beat the centroid on the OOD probes, it has learned the
simulator and should not ship.

MEASURED CONCLUSION: it does not ship. Results as of 2026-08-27 --

    pool                        head    centroid
    in-distribution 5-fold CV   0.984      0.778     <- memorization, ignore
    train turn1 -> test accum   0.521      0.708     <- chance
    out-of-distribution probes  0.812      1.000

  regularization sweep (is the transfer gap just a bad C?)
        C   in-dist CV  turn1->accum  OOD probe
    0.001        0.511         0.537      0.688
     0.01        0.556         0.521      0.688
      0.1        0.883         0.519      0.750
      1.0        0.984         0.521      0.812
     10.0        0.988         0.523      0.812

  control: train on the 20 hand-written PROTOTYPES only, no simulator text
    C=0.01   OOD probe 1.000   simulator turns 0.761
    C=1.0    OOD probe 1.000   simulator turns 0.775

Three things follow. The held-out column is flat at chance across five orders
of magnitude of regularization, so the transfer failure is the data, not a
hyperparameter. OOD accuracy *rises* with C -- the more the head overfits the
templates, the better it does out of distribution, which is the opposite of a
generalization story and means the trend is noise on 16 probes. And the
control settles it: the 20 hand-written prototypes alone reach OOD 1.000,
while 640 labelled simulator turns drag it to 0.812. The simulator's labels
are real but its *surface forms* are two templates, so the head learns
"contains 'still exploring'" and nothing transferable.

Kept as a diagnostic, not a build step: re-run it if the hidden set's phrasing
turns out to be more varied than the local templates, which is the one
condition that would change the answer. Saving is opt-in (--save).

Usage:
    uv run python3 scripts/train_intent_head.py            # measure only
    uv run python3 scripts/train_intent_head.py --save     # also write the head
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    """Reconstruct labelled turn text the same way scripts/eval_classifier.py does.

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
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
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
        centroid = centroid_accuracy([t for t, m in zip(texts, test_mask) if m], targets[test_mask])
        print(f"  train {train_stage:12s} -> test {test_stage:12s}  head {head:.3f}   centroid {centroid:.3f}")

    print("\n=== 3. out-of-distribution probes (the only number that matters) ===")
    weights, bias = fit_logistic(features, targets)
    head_probe = accuracy(weights, bias, probe_features, probe_targets)
    print(f"  head      {head_probe:.3f}")
    print(f"  centroid  {centroid_accuracy(probe_texts, probe_targets):.3f}")
    predictions = (probe_features @ weights + bias > 0).astype(int)
    for text, want, got in zip(probe_texts, probe_targets, predictions):
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
