"""Can a local masked LM predict a session's target attribute, and is its entropy
a usable confidence gate?

Calibration matters more than accuracy: if accuracy is flat across the entropy
range, the gate is decoration. Ground truth is AttributeIndex's value for the
target, because that is exactly what _boost_by_disclosed compares against. The
context is the reconstructed turn-1 message -- what the agent actually holds when
it picks its first question.

Verdict: the gate works and is steep, so MAX_CONFIDENT_ENTROPY =
0.60 is calibrated rather than guessed. Colour is excluded -- it loses to an
always-"black" baseline and its gate runs backwards.

    uv run python3 scripts/eval_lm_confidence.py [--limit 60] [--attribute color]
"""
from __future__ import annotations

import argparse
import collections
import statistics
from pathlib import Path

from _common import DEFAULT_CATALOG, DEFAULT_DATASET, REPO_ROOT  # noqa: F401

# _common puts the repo root on sys.path as an import side effect, so the
# starter/evaluator imports below resolve when this script is run directly.

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index, coarse_category, initial_message, load_jsonl, materialize_hidden_fields,
)
from starter.attributes import AttributeIndex, COLORS, MATERIALS  # noqa: E402
from starter.lm_confidence import MaskedLMScorer, belief_for_attribute  # noqa: E402

VOCAB = {"material": MATERIALS, "color": COLORS}
# Entropy cut points for the calibration table.
BANDS = ((0.0, 0.6), (0.6, 0.75), (0.75, 0.85), (0.85, 0.95), (0.95, 1.01))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--attribute", action="append", choices=sorted(VOCAB), default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)[: args.limit]
    _, categories, products = catalog_index(args.catalog)
    index = AttributeIndex.build(Path(args.catalog))
    scorer = MaskedLMScorer()

    for attribute in args.attribute or sorted(VOCAB):
        vocab = VOCAB[attribute]
        scored: list[tuple[bool, float]] = []   # (correct, entropy)
        truths: list[str] = []
        predictions: collections.Counter = collections.Counter()

        for sample in samples:
            target = str(sample["ground_truth"]["parent_asin"])
            truth = index.value_for(attribute, target)
            if truth is None:
                continue
            card, behavior = materialize_hidden_fields(sample, products)
            effective = {**sample, "intent_card": card, "behavior": behavior}
            message = initial_message(effective, coarse_category(categories.get(target, [])), set())
            belief = belief_for_attribute(scorer, attribute, message, vocab)
            if belief is None:
                continue
            scored.append((belief.value == truth, belief.entropy))
            truths.append(truth)
            predictions[belief.value] += 1

        if not scored:
            print(f"\n=== {attribute}: no scorable samples ===")
            continue

        mode_value, mode_count = collections.Counter(truths).most_common(1)[0]
        accuracy = sum(correct for correct, _ in scored) / len(scored)
        baseline = mode_count / len(truths)

        print(f"\n=== {attribute} (n={len(scored)} targets with an extracted value) ===")
        print(f"  LM top-1 accuracy : {accuracy:.3f}")
        print(f"  always-guess-mode : {baseline:.3f}   (mode = {mode_value!r})")
        print(f"  delta             : {accuracy - baseline:+.3f}")
        print(f"  LM prediction spread: {len(predictions)} distinct values, "
              f"top: {[(v, c) for v, c in predictions.most_common(3)]}")

        print("  calibration -- accuracy by entropy band (the gate's premise):")
        for low, high in BANDS:
            band = [correct for correct, entropy in scored if low <= entropy < high]
            if not band:
                continue
            print(f"    H in [{low:.2f},{high:.2f}): n={len(band):3d}  accuracy={statistics.fmean(band):.3f}")
        confident = [c for c, h in scored if h < 0.85]
        unsure = [c for c, h in scored if h >= 0.85]
        if confident and unsure:
            print(f"  confident (H<0.85) accuracy {statistics.fmean(confident):.3f} (n={len(confident)})  vs  "
                  f"unsure {statistics.fmean(unsure):.3f} (n={len(unsure)})")
            print(f"  ==> gate is {'USEFUL' if statistics.fmean(confident) > statistics.fmean(unsure) else 'NOT useful'}")


if __name__ == "__main__":
    main()
