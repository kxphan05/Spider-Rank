"""Local masked-LM belief over closed attribute vocabularies, with entropy
as the confidence gate.

Two approximations: independent masks (a k-token value gets k [MASK]s in one
pass, summing log-probs -- the standard pseudo-log-likelihood shortcut) and
length normalization (dividing by token count, since otherwise "tan" beats
"polyester" almost regardless of context).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from .paths import model_cache_dir
from .config import (LM_MODEL_NAME, MAX_CONFIDENT_ENTROPY)

logger = logging.getLogger(__name__)





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

    def __init__(self, model_name: str = LM_MODEL_NAME, cache_dir: str | None = None) -> None:
        cache_dir = cache_dir if cache_dir is not None else model_cache_dir()
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

ATTRIBUTE_TEMPLATES = {
    "material": "{context} The shopper wants it made of {}.",
    "color": "{context} The shopper wants item of colour {}."
}



def belief_for_attribute(
    scorer: MaskedLMScorer, attribute: str, context: str, values: tuple[str, ...]
) -> LMBelief | None:
    template = ATTRIBUTE_TEMPLATES.get(attribute)
    if template is None:
        return None
    context = " ".join(context.split())[-600:]
    return scorer.belief(template.replace("{context}", context), values)
