"""Sweep scoring rules for EmbeddingOverrideDetector.

Three pools:

    simulator  the evaluator's own turns. It emits exactly ONE override template,
               near-verbatim PROTOTYPE_OVERRIDE[0], so recall here is trivial by
               construction and the informative column is the false-positive rate.
    probe      hand-written out-of-distribution pivots -- the only evidence about
               phrasings the hidden grader might use, and hand-picked.
    probe_neg  continuations sharing override vocabulary ("don't", "no") without
               discarding anything.

Read-only and Agent-free. Run this before touching the detector.

    uv run python3 scripts/eval_override.py
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass

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
from starter.classifier import PROTOTYPE_CONTINUATION, PROTOTYPE_OVERRIDE  # noqa: E402
from starter.retrieval import load_embedding_model  # noqa: E402

# The attributes the agent actually asks about, so the harvested negatives
# cover both customer_reply() branches (a real disclosure vs. a decline).
ASKABLE = ("material", "color", "feature", "style", "size", "use_case", "budget")

# Out-of-distribution pivots. The first two are the reported failures.
PROBE_OVERRIDE = (
    "never mind, give me white shoes",
    "scratch that, white shoes",
    "forget it, show me sandals",
    "actually no, I want a dress",
    "never mind what I said, give me white shoes",
    "actually, give me white shoes instead",
    "I changed my mind, white shoes please",
    "on second thought, show me boots instead",
    "ignore all that, I want a silver bracelet",
    "wait, different idea -- wool coat",
)

# Continuations that share negation/discard vocabulary but discard nothing.
PROBE_CONTINUATION = (
    "no preference on that, whatever you think",
    "I don't really care about the material",
    "none of these are quite right, but keep looking",
    "not black, maybe something lighter",
    "I don't have a preference for size; please use your judgment.",
    "that's not what I meant, I want it in leather too",
    "no, show me more of the second one",
    "I'm not sure, what would you suggest?",
    "it should also be waterproof",
    "yes that works, anything else like it?",
)


def harvest(dataset: str, catalog: str) -> tuple[list[str], list[str]]:
    """Reconstruct the local simulator's override and continuation turns."""
    samples = load_jsonl(dataset)
    _, categories, products = catalog_index(catalog)
    positives: list[str] = []
    negatives: list[str] = []
    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        category = coarse_category(categories.get(target, []))
        # Turn 1 is never fed to the detector (agent.py only runs it from the
        # second message on), so it is deliberately not harvested here.
        initial_message(effective, category, disclosed)

        override = behavior.get("override") or {}
        if override.get("message"):
            positives.append(str(override["message"]))

        boundary_used = False
        for attribute in ASKABLE:
            reply, boundary_used = customer_reply(effective, attribute, disclosed, boundary_used)
            negatives.append(reply)
        # The "agent asked nothing" shape.
        reply, _ = customer_reply(effective, None, disclosed, boundary_used)
        negatives.append(reply)
    return positives, negatives


@dataclass
class Rule:
    name: str
    margin: float = 0.0

    def decide(self, sims_override: np.ndarray, sims_continuation: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CentroidRule(Rule):
    def decide(self, so, sc):
        return so.mean(axis=1) - sc.mean(axis=1) > self.margin


class MaxRule(Rule):
    def decide(self, so, sc):
        return so.max(axis=1) - sc.max(axis=1) > self.margin


class TopKMeanRule(Rule):
    k: int = 3

    def decide(self, so, sc):
        k = self.k
        top_o = np.sort(so, axis=1)[:, -k:].mean(axis=1)
        top_c = np.sort(sc, axis=1)[:, -k:].mean(axis=1)
        return top_o - top_c > self.margin


def topk_mean(k: int, margin: float = 0.0) -> Rule:
    rule = TopKMeanRule(name=f"top{k}-mean margin={margin:+.3f}", margin=margin)
    rule.k = k
    return rule


LEAD_CLAUSE_RE = re.compile(r"^(.{0,60}?)[.,;:!?]\s")


def lead_clause(text: str) -> str:
    """The message's opening clause, or the whole thing if it has no break.

    The simulator's override template is a short cue prefix followed by a
    verbatim product blob ("Actually, ignore my earlier preference. What I
    need is: <180 chars of catalog copy>"). Embedded whole, the blob
    dominates and the cue washes out -- which is exactly the shape of the
    two overrides every rule still misses. Scoring the lead clause as well
    and taking the more override-like of the two recovers those without
    changing terse pivots, which have no tail to strip.
    """
    match = LEAD_CLAUSE_RE.match(text.strip())
    return match.group(1) if match else text


def normalized_centroid(model, sentences) -> np.ndarray:
    embeddings = model.encode(list(sentences), normalize_embeddings=True, convert_to_numpy=True)
    centroid = embeddings.mean(axis=0)
    norm = np.linalg.norm(centroid)
    return centroid / norm if norm > 0 else centroid


def main() -> None:
    print("loading embedding model...", file=sys.stderr)
    model = load_embedding_model()

    override_vecs = model.encode(list(PROTOTYPE_OVERRIDE), normalize_embeddings=True, convert_to_numpy=True)
    continuation_vecs = model.encode(list(PROTOTYPE_CONTINUATION), normalize_embeddings=True, convert_to_numpy=True)
    # Centroid rules compare against the mean vector; express that as a
    # single-column "prototype" matrix so every rule shares one code path.
    override_centroid = normalized_centroid(model, PROTOTYPE_OVERRIDE)[None, :]
    continuation_centroid = normalized_centroid(model, PROTOTYPE_CONTINUATION)[None, :]

    print("harvesting simulator turns...", file=sys.stderr)
    sim_pos, sim_neg = harvest(DEFAULT_DATASET, DEFAULT_CATALOG)
    print(f"  {len(sim_pos)} override turns, {len(sim_neg)} continuation turns", file=sys.stderr)

    pools: dict[str, tuple[list[str], list[str]]] = {
        "simulator": (sim_pos, sim_neg),
        "probe": (list(PROBE_OVERRIDE), list(PROBE_CONTINUATION)),
    }

    encoded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    lead_encoded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (positives, negatives) in pools.items():
        encoded[name] = (
            model.encode(positives, normalize_embeddings=True, convert_to_numpy=True),
            model.encode(negatives, normalize_embeddings=True, convert_to_numpy=True),
        )
        lead_encoded[name] = (
            model.encode([lead_clause(s) for s in positives], normalize_embeddings=True, convert_to_numpy=True),
            model.encode([lead_clause(s) for s in negatives], normalize_embeddings=True, convert_to_numpy=True),
        )

    rules: list[tuple[Rule, np.ndarray, np.ndarray]] = [
        (CentroidRule(name="centroid (current)"), override_centroid, continuation_centroid),
        (MaxRule(name="max-prototype"), override_vecs, continuation_vecs),
    ]
    for k in (3, 4, 5, 6):
        rules.append((topk_mean(k), override_vecs, continuation_vecs))

    header = f"{'rule':32s}"
    for name in pools:
        header += f"  {name + ' rec':>14s}{name + ' fpr':>14s}"
    print("\n" + header)
    print("-" * len(header))
    for rule, proto_o, proto_c in rules:
        row = f"{rule.name:32s}"
        for name in pools:
            pos_vecs, neg_vecs = encoded[name]
            recall = float(rule.decide(pos_vecs @ proto_o.T, pos_vecs @ proto_c.T).mean())
            fpr = float(rule.decide(neg_vecs @ proto_o.T, neg_vecs @ proto_c.T).mean())
            row += f"  {recall:>14.3f}{fpr:>14.3f}"
        print(row)

    print("\n  (+ lead clause: decision OR'd with the same rule on the opening clause)")
    for rule, proto_o, proto_c in rules:
        row = f"{rule.name + ' +lead':32s}"
        for name in pools:
            pos_vecs, neg_vecs = encoded[name]
            lpos_vecs, lneg_vecs = lead_encoded[name]
            recall = float((rule.decide(pos_vecs @ proto_o.T, pos_vecs @ proto_c.T)
                            | rule.decide(lpos_vecs @ proto_o.T, lpos_vecs @ proto_c.T)).mean())
            fpr = float((rule.decide(neg_vecs @ proto_o.T, neg_vecs @ proto_c.T)
                         | rule.decide(lneg_vecs @ proto_o.T, lneg_vecs @ proto_c.T)).mean())
            row += f"  {recall:>14.3f}{fpr:>14.3f}"
        print(row)

    # Error detail for the candidate rules, on the simulator pool -- the
    # probes are hand-picked, the simulator errors are the ones that cost score.
    for rule, proto_o, proto_c in ((CentroidRule(name="centroid (current)"), override_centroid, continuation_centroid),
                                   (topk_mean(3), override_vecs, continuation_vecs)):
        print(f"\n--- simulator errors [{rule.name}] ---")
        pos_vecs, neg_vecs = encoded["simulator"]
        missed = ~rule.decide(pos_vecs @ proto_o.T, pos_vecs @ proto_c.T)
        flagged = rule.decide(neg_vecs @ proto_o.T, neg_vecs @ proto_c.T)
        print(f"  missed overrides ({int(missed.sum())}/{len(sim_pos)}):")
        for sentence in sorted({s for s, m in zip(sim_pos, missed, strict=True) if m})[:6]:
            print(f"    {sentence!r}")
        print(f"  false positives ({int(flagged.sum())}/{len(sim_neg)}):")
        for sentence in sorted({s for s, f in zip(sim_neg, flagged, strict=True) if f})[:6]:
            print(f"    {sentence!r}")

    print("\n--- probe detail [top3-mean] ---")
    rule, proto_o, proto_c = topk_mean(3), override_vecs, continuation_vecs
    for label, sentences in (("OVR", PROBE_OVERRIDE), ("CNT", PROBE_CONTINUATION)):
        vecs = model.encode(list(sentences), normalize_embeddings=True, convert_to_numpy=True)
        so, sc = vecs @ proto_o.T, vecs @ proto_c.T
        decisions = rule.decide(so, sc)
        for sentence, decision in zip(sentences, decisions, strict=True):
            want = decision if label == "OVR" else not decision
            print(f"    {'ok ' if want else 'BAD'} {label}  {sentence!r}")


if __name__ == "__main__":
    main()
