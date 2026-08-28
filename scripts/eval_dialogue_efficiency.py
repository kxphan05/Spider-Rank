"""How many clarifying questions come back empty, and can we tell?

Two read-only diagnostics, no Agent required:

1. **Sizing.** How many distinct answerable attribute buckets a session has
   left after turn 1, derived from the evaluator's own reply policy. This
   upper-bounds what any question-planning work can ever win, so it is worth
   knowing *before* building the planner rather than after.

2. **Detector quality.** Recall / false-positive rate for
   `EmbeddingNonAnswerDetector` against the simulator's own replies plus
   hand-written out-of-distribution phrasings, with the lexical floor
   (`classify_reply_lexically`) as a comparison row.

The error asymmetry is the opposite of the override detector's (CLAUDE.md #7):
a false positive here drops a *real* disclosure out of the query, while a false
negative merely leaves current behaviour unchanged. So **FPR is the column that
gates shipping**, and recall is the nice-to-have.

Ground-truth labelling for the simulator pool matches the two literal
non-answer templates. That is legitimate for scoring a rule but would be
cheating inside one -- see CLAUDE.md #12/#13 on template memorization.
"""
from __future__ import annotations

import argparse
import collections
import sys

from _common import DEFAULT_CATALOG, DEFAULT_DATASET  # noqa: F401

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index,
    classify_constraint,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from starter.classifier import (  # noqa: E402
    EmbeddingNonAnswerDetector,
    classify_reply_lexically,
)

ASKABLE = ("material", "color", "feature", "style", "size", "use_case", "budget")

# The simulator's two contentless replies. Used ONLY to label the harvested
# pool; never as detector input.
_NON_ANSWER_MARKERS = (
    "i don't have an additional preference for",
    "i don't have a preference for",
    "ask me about one specific attribute",
)

# Out-of-distribution declines: phrasings the hidden grader might use that
# share no wording with the local templates.
PROBE_NON_ANSWER = (
    "no strong feelings there honestly",
    "whatever you reckon is best",
    "hmm, not fussed about that one",
    "you pick, I trust you",
    "that's not really a factor for me",
    "open to anything on that",
    "no real requirement there",
    "eh, doesn't matter",
    "I'll go with your recommendation on that",
    "nothing in particular comes to mind",
)

# Informative replies that share decline-ish vocabulary but DO disclose. These
# are the false-positive traps -- the expensive error for this detector.
PROBE_INFORMATIVE = (
    "no preference on brand, but it must be leather",
    "I don't care about the colour as long as it's cotton",
    "not fussy, though size 10 please",
    "doesn't matter much, maybe something in black",
    "anything works if it's under $40",
    "for that, what matters is a waterproof shell",
    "I need it for hiking",
    "a slim fit would be best",
    "wool, ideally",
    "something with deep pockets",
)


def harvest(dataset: str, catalog: str):
    """Reconstruct simulator replies, labelled non-answer / answer."""
    samples = load_jsonl(dataset)
    _, categories, products = catalog_index(catalog)
    labelled: list[tuple[str, bool]] = []
    bucket_counts = collections.Counter()
    bucket_totals = collections.Counter()

    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        initial_message(effective, coarse_category(categories.get(target, [])), disclosed)

        constraints = [
            *[str(v) for v in card.get("hard_constraints", [])],
            *[str(v) for v in card.get("soft_preferences", [])],
        ]
        buckets = {classify_constraint(c) for c in constraints if c not in disclosed}
        bucket_counts[len(buckets)] += 1
        for bucket in buckets:
            bucket_totals[bucket] += 1

        # A fresh disclosed-set per ask order would be ideal, but the
        # simulator mutates it; walking ASKABLE once mirrors how a session
        # actually drains the card.
        boundary_used = False
        for attribute in ASKABLE:
            reply, boundary_used = customer_reply(effective, attribute, disclosed, boundary_used)
            lowered = reply.lower()
            is_non = any(marker in lowered for marker in _NON_ANSWER_MARKERS)
            labelled.append((reply, is_non))
    return labelled, bucket_counts, bucket_totals


def score(name: str, predict, pool: list[tuple[str, bool]]) -> None:
    tp = fp = fn = tn = 0
    for text, truth in pool:
        pred = predict(text)
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and truth:
            fn += 1
        else:
            tn += 1
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    print(f"  {name:<22} recall={recall:.3f}  FPR={fpr:.3f}  precision={precision:.3f}"
          f"   (FP={fp}/{fp + tn})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    args = parser.parse_args()

    print("harvesting simulator replies...", file=sys.stderr)
    labelled, bucket_counts, bucket_totals = harvest(args.dataset, args.catalog)
    n = sum(bucket_counts.values())

    print("\n=== sizing: answerable buckets remaining after turn 1 ===")
    for k in sorted(bucket_counts):
        print(f"  {k} bucket(s): {bucket_counts[k]:3d} sessions")
    mean = sum(k * v for k, v in bucket_counts.items()) / n
    print(f"  mean = {mean:.2f} distinct answerable buckets per session")
    print("\n  per-bucket answerability (share of sessions):")
    for bucket, count in bucket_totals.most_common():
        print(f"    {bucket:<10} {count:3d}  ({count / n:.3f})")

    non = sum(1 for _, truth in labelled if truth)
    print(f"\n  harvested replies: {len(labelled)}  non-answers: {non} ({non / len(labelled):.3f})")

    print("\n=== detector quality ===")
    print("\n  simulator pool (in-distribution; recall is easy by construction,")
    print("  FPR is the honest column):")
    score("lexical floor", lambda t: classify_reply_lexically(t) == "non_answer", labelled)

    probes = [(t, True) for t in PROBE_NON_ANSWER] + [(t, False) for t in PROBE_INFORMATIVE]
    print("\n  out-of-distribution probes (hand-written; a smoke test, not a")
    print("  measurement -- it can falsify a rule, not rank two that both pass):")
    score("lexical floor", lambda t: classify_reply_lexically(t) == "non_answer", probes)

    try:
        from starter.retrieval import load_embedding_model
    except Exception:
        print("\n  (no embedding model available; lexical rows only)")
        return
    try:
        model = load_embedding_model()
    except Exception as exc:
        print(f"\n  (embedding model failed to load: {exc}; lexical rows only)")
        return

    detector = EmbeddingNonAnswerDetector(model)
    print("\n  simulator pool, embedding rule:")
    score("trimmed-prototype", detector.is_non_answer, labelled)
    print("\n  out-of-distribution probes, embedding rule:")
    score("trimmed-prototype", detector.is_non_answer, probes)


if __name__ == "__main__":
    main()
