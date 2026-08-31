"""Calibrate an absolute similarity threshold for EmbeddingNonAnswerDetector.

Context: the shipped detector is a *two-class* nearest-prototype rule --
non-answer wins if closer to PROTOTYPE_NON_ANSWER than to
PROTOTYPE_INFORMATIVE. "Informative" is an unbounded space (any attribute
value a customer might state), so PROTOTYPE_INFORMATIVE had to be patched
reactively when terse catalog fragments ("Imported.", "Pull On closure.")
were found reading as non-answers (see .claude/skills/retrieval-experiments
entry 8) -- a pattern that will keep recurring and reads as overfit to this
catalog's specific vocabulary to anyone auditing classifier.py.

This script tests the alternative: a *one-class* rule. PROTOTYPE_NON_ANSWER
(declining a preference: "no preference", "up to you") is a small, closed,
genuinely generic set. Classify non-answer only when similarity to it clears
an absolute threshold; default everything else to "answer". This needs no
"informative" class at all, so no catalog-vocabulary list to keep patching.

Calibration set: harvested from the *catalog itself* (not the 200 public
sessions, to avoid circularity with the graded set and to proxy the hidden
800 fairly) via the same generative logic evaluator/local_evaluator.py uses
to build customer replies -- intent_card() + classify_constraint() -- over a
large random sample of products.

    uv run python3 scripts/sweep_nonanswer_threshold.py [--sample-products 4000] [--seed 7]
"""
from __future__ import annotations

import argparse
import random
import sys

from _common import DEFAULT_CATALOG, REPO_ROOT  # noqa: F401

sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluator.local_evaluator import ALLOWED_ATTRIBUTES, classify_constraint, intent_card, load_jsonl  # noqa: E402
from starter.classifier import PROTOTYPE_NON_ANSWER, _top_prototype_similarity, detected_attributes  # noqa: E402
from starter.retrieval import load_embedding_model  # noqa: E402

# classify_constraint never emits these two -- "other" isn't a real bucket
# here and "brand" has no source field in intent_card.
ASKABLE_ATTRIBUTES = sorted(set(ALLOWED_ATTRIBUTES) - {"category", "other", "brand"})

# The simulator's two non-answer templates are attribute-keyed, not
# product-keyed ("I don't have an additional preference for style." is byte-
# identical for every product asked about style), so harvesting them the
# same way as the answer side collapses to ~14 unique strings -- not enough
# to calibrate a threshold against. Scored separately below as a must-catch
# regression set instead.
DETERMINISTIC_NON_ANSWER_TEMPLATES = tuple(
    f"I don't have an additional preference for {attribute}." for attribute in ASKABLE_ATTRIBUTES
) + tuple(
    f"I don't have a preference for {attribute}; please use your judgment." for attribute in ASKABLE_ATTRIBUTES
) + (
    # Emitted whenever ask_attribute is None (session ran out of attributes,
    # or the agent stopped asking) -- carries no attribute content either.
    "Those options are not quite right yet. Ask me about one specific attribute.",
)

# Held-out decline paraphrases -- deliberately NOT in PROTOTYPE_NON_ANSWER
# (scoring against your own training prototypes is circular) and NOT copied
# from the simulator templates above. Same role as train_intent_head.py's
# PROBE_BUYING/PROBE_BROWSING: checks generalization past the 12 phrasings
# the centroid was actually built from.
OOD_NON_ANSWER_PROBES = (
    "Nah, doesn't matter to me either way.",
    "I'm indifferent about that one, honestly.",
    "No real preference on my end there.",
    "You pick, I trust your judgment on this.",
    "Meh, not fussed about that detail.",
    "Skip that one, it's not important to me.",
    "I've got no strong feelings about it.",
    "Eh, whatever's easiest for you.",
    "That part doesn't concern me much.",
    "I'll defer to you on that.",
    "Not something I've considered, no answer there.",
    "Zero preference from me on that front.",
)

# Regression check: the exact failure this project already diagnosed and
# fixed (.claude/skills/retrieval-experiments entry 8) -- terse catalog-
# jargon answers that used to read as non-answers. Reconstructed in the
# real customer_reply() template shape, not hand-picked wording.
REGRESSION_ANSWER_PROBES = (
    "For that, what matters is: Imported.",
    "For that, what matters is: Button closure; Hand Wash Only.",
    "For that, what matters is: Pull On closure.",
    "For that, what matters is: Zipper closure.",
    "For that, what matters is: Machine Wash, Line Dry.",
)


def harvest(catalog_path, rng: random.Random, n_products: int) -> tuple[list[str], list[int]]:
    """Return (texts, labels); label 0 == answer, harvested only.

    Mirrors evaluator/local_evaluator.py's "For that, what matters is: ..."
    template exactly, generated over a broad product sample rather than the
    200 graded targets, to test precision against realistically diverse
    catalog-derived free-text content without touching the graded set.
    """
    products = load_jsonl(catalog_path)
    rng.shuffle(products)
    texts: list[str] = []
    labels: list[int] = []
    seen: set[str] = set()
    for product in products[:n_products]:
        card = intent_card(product)
        constraints = [*card["hard_constraints"], *card["soft_preferences"]]
        by_attribute: dict[str, list[str]] = {attribute: [] for attribute in ASKABLE_ATTRIBUTES}
        for value in constraints:
            attribute = classify_constraint(value)
            if attribute in by_attribute:
                by_attribute[attribute].append(value)
        for attribute in ASKABLE_ATTRIBUTES:
            matches = by_attribute[attribute][:2]
            if not matches:
                continue
            text = "For that, what matters is: " + "; ".join(matches) + "."
            if text in seen:
                continue
            seen.add(text)
            texts.append(text)
            labels.append(0)
    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--sample-products", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[round(0.30 + 0.02 * i, 2) for i in range(21)])  # 0.30..0.70
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print(f"harvesting answer set from {args.sample_products} random products (seed {args.seed})...", file=sys.stderr)
    answer_texts, answer_labels = harvest(args.catalog, rng, args.sample_products)
    print(f"harvested {len(answer_texts)} unique diverse answer strings", file=sys.stderr)

    # The detected_attributes veto runs before any embedding call in the real
    # detector -- apply it here too, or the sweep would score text the real
    # detector never actually sends to the embedding path.
    keep = [i for i, text in enumerate(answer_texts) if not detected_attributes(text)]
    print(f"{len(answer_texts) - len(keep)} vetoed by detected_attributes (material/color/size/budget vocab) "
          f"-- excluded, matching production behavior", file=sys.stderr)
    answer_texts = [answer_texts[i] for i in keep]
    answer_labels = [answer_labels[i] for i in keep]
    print(f"scoring precision against {len(answer_texts)} diverse catalog-derived answer strings\n", file=sys.stderr)

    print("loading embedding model...", file=sys.stderr)
    model = load_embedding_model()
    non_answer_protos = model.encode(list(PROTOTYPE_NON_ANSWER), normalize_embeddings=True, convert_to_numpy=True)

    def sims_for(texts: list[str]) -> list[float]:
        vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=128)
        return [_top_prototype_similarity(vec, non_answer_protos) for vec in vectors]

    answer_sims = sims_for(answer_texts)
    template_sims = sims_for(list(DETERMINISTIC_NON_ANSWER_TEMPLATES))
    ood_sims = sims_for(list(OOD_NON_ANSWER_PROBES))
    regression_sims = sims_for(list(REGRESSION_ANSWER_PROBES))

    print(f"{'threshold':>10}  {'precision':>10} {'FP':>5}  {'template_recall':>16} {'OOD_recall':>11}  "
          f"{'regression_pass':>16}")
    best = (-1.0, None)
    for threshold in args.thresholds:
        fp = sum(1 for sim in answer_sims if sim > threshold)
        precision = (len(answer_sims) - fp) / len(answer_sims)
        template_recall = sum(1 for sim in template_sims if sim > threshold) / len(template_sims)
        ood_recall = sum(1 for sim in ood_sims if sim > threshold) / len(ood_sims)
        regression_pass = sum(1 for sim in regression_sims if sim <= threshold)
        print(f"{threshold:>10.2f}  {precision:>10.4f} {fp:>5}  {template_recall:>16.3f} {ood_recall:>11.3f}  "
              f"{regression_pass}/{len(regression_sims)}")
        # Selection rule: the deterministic templates (what the graded
        # simulator actually sends) must always be caught -- that's not
        # negotiable, it's the detector's whole job. Among thresholds that
        # do, prefer the one maximizing OOD recall + precision together.
        if template_recall == 1.0 and regression_pass == len(regression_sims):
            score = precision + ood_recall
            if score > best[0]:
                best = (score, threshold)

    print(f"\nbest (100% template recall + all regression probes pass, "
          f"max precision+OOD_recall): threshold={best[1]}")

    if best[1] is not None:
        print(f"\n--- false positives at threshold={best[1]} (diverse answer text read as non_answer) ---")
        shown = 0
        for text, sim in zip(answer_texts, answer_sims, strict=True):
            if sim > best[1]:
                print(f"    sim={sim:.3f}  {text[:100]!r}")
                shown += 1
                if shown >= 15:
                    break
        print(f"\n--- OOD probes missed at threshold={best[1]} ---")
        for text, sim in zip(OOD_NON_ANSWER_PROBES, ood_sims, strict=True):
            if sim <= best[1]:
                print(f"    sim={sim:.3f}  {text!r}")


if __name__ == "__main__":
    main()
