"""Local masked-LM belief over closed attribute vocabularies, with entropy
as the confidence gate.

The question this answers is the one `attributes.py`'s entropy selector cannot:
*given what the customer has actually said so far, do we already know what
they want?* The existing selector measures value-diversity in the retrieved
candidate pool -- a property of the catalog, not of the conversation. This
measures the model's belief given the dialogue text, and reports how sure it
is, so the agent can predict an attribute it can guess and spend its limited
turns asking about the ones it cannot.

Why a local masked LM rather than an API call: `docs/submission_rules.md`
§ Model Policy warns that "organizer policy may disable network access" for
official scoring, so anything on the inference path has to run offline. It
also has to expose token-level probabilities, which the hosted Claude
Messages API does not return -- there is no logprobs parameter -- while a
local `AutoModelForMaskedLM` gives the full vocabulary distribution directly.
distilbert-base-uncased (67M parameters, ~255MB) is small enough to ship as
the "lightweight local asset" the rules allow, and loads alongside the
bge-small encoder retrieval already depends on. bge itself cannot do this
job: its published weights carry no LM head (checked -- 200 tensors,
embeddings + encoder + pooler, no `cls.predictions.*`), so it can embed text
but cannot assign probability to a word.

Closed vocabulary, not free generation. The scorer never asks "what colour do
they want?" and reads a sentence back; it scores exactly the values in
`COLORS`/`MATERIALS` and renormalizes over them. That keeps every output
directly comparable with `AttributeIndex`'s extracted values (which are
compared for equality), and it makes the entropy meaningful -- it is entropy
over the same finite set the agent can actually act on.

Two approximations, both deliberate and both worth knowing when reading the
numbers this produces:

1. **Independent masks.** A value tokenizing to k word-pieces is scored with
   k `[MASK]`s in one forward pass, summing the per-position log-probabilities.
   That ignores dependence between the masked positions, which is the standard
   pseudo-log-likelihood approximation; the exact alternative costs k passes
   per value. Since every value competes under the same approximation and the
   result is only used to rank and to gate, the bias is largely shared.
2. **Length normalization.** Summed log-probabilities favour short values, so
   scores are divided by token count. Without it "tan" beats "polyester"
   almost regardless of context.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

LM_MODEL_NAME = "distilbert-base-uncased"
LM_CACHE_DIR = "./model"

# Beliefs at or above this normalized entropy are treated as "the model does
# not know". Calibrated, not chosen by feel: measured against the true
# target's extracted material over the 200-sample public set
# (scripts/eval_lm_confidence.py), top-1 accuracy by entropy band was
#
#     H < 0.60      n= 61   0.787
#     0.60-0.75     n=105   0.371
#     0.75-0.85     n= 14   0.000
#
# -- monotonic, and steep. 0.60 is where the belief is still worth acting on;
# above it the prediction is at or below the always-guess-the-mode baseline
# of 0.322, so acting on it would be worse than doing nothing.
MAX_CONFIDENT_ENTROPY = 0.60


@dataclass(frozen=True)
class LMBelief:
    """The model's belief over one attribute's closed value set."""
    value: str          # highest-probability value
    probability: float  # its probability, renormalized over the value set
    entropy: float      # normalized Shannon entropy over the set, in [0, 1]

    @property
    def confident(self) -> bool:
        return self.entropy < MAX_CONFIDENT_ENTROPY


class MaskedLMScorer:
    """Scores closed-vocabulary values in a `[MASK]`ed template."""

    def __init__(self, model_name: str = LM_MODEL_NAME, cache_dir: str = LM_CACHE_DIR) -> None:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self._model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir)
        self._model.eval()
        self._mask = self._tokenizer.mask_token
        self._token_cache: dict[str, list[int]] = {}
        logger.info("MaskedLMScorer: loaded %s", model_name)

    def _token_ids(self, value: str) -> list[int]:
        cached = self._token_cache.get(value)
        if cached is None:
            cached = self._tokenizer.encode(value, add_special_tokens=False)
            self._token_cache[value] = cached
        return cached

    def belief(self, template: str, values: tuple[str, ...]) -> LMBelief | None:
        """Distribution over `values` filling `{}` in `template`.

        One forward pass per distinct token length among the values (usually
        two or three), not one per value.
        """
        if not values or "{}" not in template:
            return None
        by_length: dict[int, list[str]] = {}
        for value in values:
            ids = self._token_ids(value)
            if ids:
                by_length.setdefault(len(ids), []).append(value)
        if not by_length:
            return None

        scores: dict[str, float] = {}
        torch = self._torch
        for length, group in by_length.items():
            prompt = template.format(" ".join([self._mask] * length))
            encoded = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            mask_positions = (
                encoded["input_ids"][0] == self._tokenizer.mask_token_id
            ).nonzero(as_tuple=True)[0]
            if len(mask_positions) != length:
                # Truncation ate some masks -- skip this length rather than
                # score against the wrong positions.
                continue
            with torch.no_grad():
                logits = self._model(**encoded).logits[0]
            log_probs = torch.log_softmax(logits[mask_positions], dim=-1)
            for value in group:
                ids = self._token_ids(value)
                total = sum(log_probs[slot, token_id].item() for slot, token_id in enumerate(ids))
                scores[value] = total / length  # length-normalized, see module docstring

        if not scores:
            return None
        return _to_belief(scores)


def _to_belief(scores: dict[str, float]) -> LMBelief:
    """Softmax over length-normalized log-scores, then entropy of the result."""
    highest = max(scores.values())
    weights = {value: math.exp(score - highest) for value, score in scores.items()}
    total = sum(weights.values())
    probabilities = {value: weight / total for value, weight in weights.items()}
    best = max(probabilities, key=probabilities.get)
    raw_entropy = -sum(p * math.log2(p) for p in probabilities.values() if p > 0)
    ceiling = math.log2(len(probabilities)) if len(probabilities) > 1 else 1.0
    return LMBelief(value=best, probability=probabilities[best], entropy=raw_entropy / ceiling)


# Templates put the attribute in a natural object position. Kept explicit
# per attribute rather than generated, because the phrasing measurably
# changes the distribution and these are the ones that were evaluated.
# Chosen by sweep, not by feel: on probes whose answer is stated in the
# context, "The shopper wants it in {} colour." and "It is {} in color."
# both scored 0/3 and collapsed onto "purple" for every input -- a degenerate
# attractor, not a belief. "I want the {} one." scored 2/3 on the same
# probes. Material was less sensitive (2-3 of 3 for every phrasing tried).
# Those probes only test whether the model can copy a stated value; the
# out-of-context predictive test is scripts/eval_lm_confidence.py.
ATTRIBUTE_TEMPLATES = {
    "material": "{context} The shopper wants it made of {}.",
}

# `color` is deliberately absent, and that is a measured decision rather than
# an omission. Over the same public set, LM top-1 colour accuracy was 0.189
# against an always-guess-"black" baseline of 0.432 -- materially worse than
# doing nothing -- and its entropy gate ran *backwards* (confident 0.172 vs
# unsure 0.250), so no threshold rescues it. It also collapsed onto a single
# attractor value, predicting "pink" for 14 of 37 targets, the same
# degenerate mode the rejected templates showed with "purple". Colour words
# are weakly determined by a shopping request in a way materials are not: a
# request for a winter coat implies wool far more than any colour, and the
# customer's own stated colour (which `extract_disclosed_value` already
# catches) is the only reliable source. Re-check with
# scripts/eval_lm_confidence.py before adding it back.


def belief_for_attribute(
    scorer: MaskedLMScorer, attribute: str, context: str, values: tuple[str, ...]
) -> LMBelief | None:
    template = ATTRIBUTE_TEMPLATES.get(attribute)
    if template is None:
        return None
    context = " ".join(context.split())[-600:]
    return scorer.belief(template.replace("{context}", context), values)
