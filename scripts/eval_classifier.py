"""Sanity-check starter/classifier.py against the local dev templates.

Superseded for rule selection by eval_intent.py and eval_override.py, which sweep
candidate rules on three pools each. Read-only use of the evaluator's
message-generation helpers; it validates against local phrasing only and proves
nothing about the hidden simulator.

    uv run python3 scripts/eval_classifier.py
"""
from __future__ import annotations

from collections import Counter, defaultdict

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
from starter.classifier import EmbeddingIntentClassifier, classify_intent  # noqa: E402
from starter.retrieval import load_embedding_model  # noqa: E402


def main() -> None:
    samples = load_jsonl(DEFAULT_DATASET)
    catalog_ids, categories, products = catalog_index(DEFAULT_CATALOG)

    print("loading embedding model for EmbeddingIntentClassifier...")
    embedding_classifier = EmbeddingIntentClassifier(load_embedding_model())

    classifiers = {
        "lexical": classify_intent,
        "embedding": embedding_classifier.classify,
    }
    turn1_confusion: dict[str, dict[str, Counter[str]]] = {name: defaultdict(Counter) for name in classifiers}
    turn3_confusion: dict[str, dict[str, Counter[str]]] = {name: defaultdict(Counter) for name in classifiers}
    examples: list[tuple[str, str, dict[str, float]]] = []

    for sample in samples:
        scenario = sample["scenario_type"]
        target = str(sample["ground_truth"]["parent_asin"])
        intent_card, behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": intent_card, "behavior": behavior}
        disclosed: set[str] = set()

        category = coarse_category(categories.get(target, []))
        turn1_msg = initial_message(effective_sample, category, disclosed)

        scores = {}
        for name, classify in classifiers.items():
            signal1 = classify(turn1_msg)
            turn1_confusion[name][scenario][signal1.label] += 1
            scores[name] = signal1.score
        if len(examples) < 12:
            examples.append((scenario, turn1_msg, scores))

        # Simulate two rounds of attribute asking (mimicking what the agent
        # would do) to see whether accumulated text trends toward "buying"
        # as concrete answers accumulate -- the actual routing-relevant case.
        accumulated = turn1_msg
        boundary_used = False
        for attribute in ("material", "color"):
            reply, boundary_used = customer_reply(effective_sample, attribute, disclosed, boundary_used)
            accumulated += " " + reply
        for name, classify in classifiers.items():
            signal3 = classify(accumulated)
            turn3_confusion[name][scenario][signal3.label] += 1

    def print_confusion(title: str, table: dict[str, Counter[str]]) -> None:
        print(f"\n--- {title} ---")
        for scenario in sorted(table):
            counts = table[scenario]
            total = sum(counts.values())
            parts = ", ".join(f"{label}={n} ({n/total:.0%})" for label, n in counts.most_common())
            print(f"  {scenario:16s} n={total:3d}  {parts}")

    for name in classifiers:
        print(f"\n=== {name} classifier ===")
        print_confusion("turn-1 message only", turn1_confusion[name])
        print_confusion("after 2 simulated clarifying answers", turn3_confusion[name])

    print("\n=== sample turn-1 messages + scores (lexical vs embedding) ===")
    for scenario, msg, scores in examples:
        score_str = "  ".join(f"{name}={score:+.2f}" for name, score in scores.items())
        print(f"  [{scenario:16s}] {score_str}  {msg!r}")


if __name__ == "__main__":
    main()
