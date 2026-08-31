"""Dynamic, entropy-based attribute-question selection.

The agent picks and chooses the best attribute based on which attribute can
 split the search space the most (see Zou & Kanoulas, "Learning to
Ask: Question-based Sequential Bayesian Product Search", CIKM 2019, whose
core selection term is exactly a candidate-set relevance-mass split -- their
optional learned "reward" term needs purchase history we don't have, so we
only implement the data-free half).
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path

from .classifier import _is_negated
from .text_utils import field_text

logger = logging.getLogger(__name__)

MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon",
    "denim", "suede", "canvas", "linen", "cashmere", "fleece", "mesh", "velvet",
    "lace", "rubber", "sterling silver", "stainless steel", "metal",
)
COLORS = (
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
    "purple", "yellow", "orange", "navy", "beige", "tan", "gold", "silver", "multicolor",
)
# Attributes the dynamic (entropy-based) question selector is allowed to ask
# about. `budget` deliberately excluded most of the items are unlabelled
# this is unhelpful for the agent.
STRUCTURAL_ATTRIBUTES = ("material", "color")

# Attributes we can extract a structured value for and use to filter/rerank
# the candidate pool once disclosed.
FILTERABLE_ATTRIBUTES = ("material", "color", "budget")


def _vocab_matcher(vocab: tuple[str, ...]) -> tuple[re.Pattern[str], dict[str, int]]:
    """One word-boundary alternation over `vocab`, plus vocab-priority ranks.

    Longest-first in the alternation so "sterling silver" wins over a bare
    "silver" at the same position; the returned rank map then restores the
    original vocab order when picking among everything that matched, which is
    the selection rule the per-word loop this replaced used.
    """
    ordered = sorted(vocab, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(word) for word in ordered) + r")\b")
    return pattern, {word: rank for rank, word in enumerate(vocab)}


# Word-boundary matched, NOT substring matched. so words like red
# dont get matched to 'embroidered'
_MATERIAL_RE, _MATERIAL_RANK = _vocab_matcher(MATERIALS)
_COLOR_RE, _COLOR_RANK = _vocab_matcher(COLORS)


def _first_vocab_match(text: str, pattern: re.Pattern[str], rank: dict[str, int]) -> str | None:
    found = pattern.findall(text)
    return min(found, key=lambda word: rank[word]) if found else None


def _extract_material(product: dict) -> str | None:
    text = f"{field_text(product.get('title'))} {field_text(product.get('features'))}".lower()
    return _first_vocab_match(text, _MATERIAL_RE, _MATERIAL_RANK)


def _extract_color(product: dict) -> str | None:
    text = f"{field_text(product.get('title'))} {field_text(product.get('features'))}".lower()
    return _first_vocab_match(text, _COLOR_RE, _COLOR_RANK)


class AttributeIndex:
    """Precomputed per-product structural attribute values, keyed by parent_asin."""

    def __init__(self, values: dict[str, dict[str, str | None]],
                 prices: dict[str, float] | None = None,
                 popularity: dict[str, float] | None = None) -> None:
        self._values = values
        self._prices = prices
        self._popularity = popularity or {}

    @classmethod
    def build(cls, catalog_path: Path) -> AttributeIndex:
        values: dict[str, dict[str, str | None]] = {}
        prices: dict[str, float] = {}
        popularity: dict[str, float] = {}
        with catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                values[parent_asin] = {
                    "material": _extract_material(product),
                    "color": _extract_color(product),
                }
                price = product.get("price")
                if isinstance(price, (int, float)):
                    prices[parent_asin] = float(price)
                reviews = product.get("rating_number")
                popularity[parent_asin] = float(reviews) if isinstance(reviews, (int, float)) else 0.0

        logger.info(
            "AttributeIndex: extracted structural attributes for %d products (%d priced)",
            len(values), len(prices),
        )
        return cls(values, prices, popularity)

    def values_for(self, attribute: str, candidate_ids: list[str]) -> list[str | None]:
        return [self._values.get(pid, {}).get(attribute) for pid in candidate_ids]

    def value_for(self, attribute: str, parent_asin: str) -> str | None:
        return self._values.get(parent_asin, {}).get(attribute)

    def price_for(self, parent_asin: str) -> float | None:
        """Raw catalog price, or None for the 79.2% of products without one."""
        return self._prices.get(parent_asin)

    def popularity_for(self, parent_asin: str) -> float:
        """Review count, the only 100%-covered catalog field."""
        return self._popularity.get(parent_asin, 0.0)

    def by_popularity(self, candidate_ids: list[str]) -> list[str]:
        """`candidate_ids` reordered most-reviewed first."""
        return sorted(candidate_ids, key=self.popularity_for, reverse=True)


# Budget parsing.
_UNITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
          "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}

_AMOUNT_RE = re.compile(
    r"\$\s?(\d+(?:\.\d+)?)|\b(\d+(?:\.\d+)?)\s*(?:dollars?|bucks?|usd)\b",
    re.IGNORECASE,
)
_WORD_AMOUNT_RE = re.compile(
    r"\b((?:one|two|three|four|five|six|seven|eight|nine)\s+hundred|hundred|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)"
    r"(?:[\s-]+(one|two|three|four|five|six|seven|eight|nine))?\s*"
    r"(?:dollars?|bucks?|usd)\b",
    re.IGNORECASE,
)
_FLOOR_CUES = ("over", "above", "at least", "more than", "starting at",
               "no less than", "upwards of")
# Ceiling phrasings that *contain* a floor cue as a substring ("no more than"
# contains "more than", "nothing over" contains "over"). These are checked
# first, so a negated floor is not read as a floor.
_NEGATED_FLOOR_CUES = ("no more than", "not more than", "no higher than",
                       "not higher than", "no greater than", "nothing over",
                       "nothing above", "not over", "not above", "no over")


def _word_to_number(head: str, tail: str | None) -> float | None:
    head = head.lower().strip()
    if head.endswith("hundred"):
        multiplier = head.split()[0] if " " in head else "one"
        return float(_UNITS.get(multiplier, 1) * 100)
    base = _TENS.get(head)
    if base is not None:
        return float(base + (_UNITS.get((tail or "").lower(), 0)))
    return float(_UNITS[head]) if head in _UNITS else None


def parse_budget_constraint(text: str) -> str | None:
    """Parse a stated budget into a comparison token, e.g. "<=80" or ">=80"."""
    lowered = text.lower()
    match = _AMOUNT_RE.search(lowered)
    if match:
        amount = float(match.group(1) or match.group(2))
    else:
        match = _WORD_AMOUNT_RE.search(lowered)
        if not match:
            return None
        amount = _word_to_number(match.group(1), match.group(2))
        if amount is None:
            return None
    lead = lowered[: match.start()]
    if any(cue in lead for cue in _NEGATED_FLOOR_CUES):
        direction = "<="
    elif any(cue in lead for cue in _FLOOR_CUES):
        direction = ">="
    else:
        direction = "<="
    return f"{direction}{amount:g}"


def budget_constraint_satisfied(constraint: str, price: float) -> bool | None:
    """Test a price against a parse_budget_constraint token."""
    if constraint.startswith("<="):
        return price <= float(constraint[2:])
    if constraint.startswith(">="):
        return price >= float(constraint[2:])
    return None


def extract_disclosed_value(attribute: str, text: str) -> str | None:
    """Best-effort extraction of a concrete attribute value from FIRST query
    (a customer's answer to a question, or an unprompted mention).
    """
    if not isinstance(text, str) or not text.strip():
        return None
    lowered = text.lower()
    vocab = MATERIALS if attribute == "material" else COLORS if attribute == "color" else None
    if vocab is not None:
        for word in vocab:
            match = re.search(rf"\b{re.escape(word)}\b", lowered)
            if match and not _is_negated(lowered, match.start()):
                return word
        return None
    if attribute == "budget":
        return parse_budget_constraint(lowered)
    return None


def normalized_entropy(values: list[str | None]) -> float:
    """Shannon entropy over `values`, normalized to [0, 1] by log2(n_unique)."""
    if len(values) < 2:
        return 0.0
    counts = Counter(values)
    if len(counts) < 2:
        return 0.0
    total = len(values)
    raw = -sum((n / total) * math.log2(n / total) for n in counts.values())
    max_possible = math.log2(len(counts))
    return raw / max_possible if max_possible > 0 else 0.0


def _entropy_component(
    attribute: str,
    candidate_ids: list[str],
    attribute_index: AttributeIndex,
    min_entropy: float,
) -> float | None:
    """Normalized pool entropy for `attribute`, or None when there isn't one."""
    if attribute not in STRUCTURAL_ATTRIBUTES or not candidate_ids:
        return None
    score = normalized_entropy(attribute_index.values_for(attribute, candidate_ids))
    return score if score > min_entropy else None


def select_weighted_attribute(
    candidate_ids: list[str],
    attribute_index: AttributeIndex,
    excluded: set[str],
    answerability: dict[str, float],
    entropy_weight: float,
    answerability_weight: float,
    min_entropy: float = 0.15,
) -> str | None:
    """Pick the next question by a weighted mean of informativeness and answerability.

        score = (we * H + wa * A) / (we + wa)
    """
    denominator = entropy_weight + answerability_weight
    if denominator <= 0.0:
        return None
    ordered = list(dict.fromkeys(tuple(answerability) + STRUCTURAL_ATTRIBUTES))
    best_attribute: str | None = None
    best_score = float("-inf")
    for attribute in ordered:
        if attribute in excluded:
            continue
        entropy = _entropy_component(
            attribute, candidate_ids, attribute_index, min_entropy
        )
        answerable = answerability.get(attribute)
        if entropy is None and answerable is None:
            continue
        if entropy is None:
            score = answerable
        elif answerable is None:
            score = entropy
        else:
            score = (entropy_weight * entropy + answerability_weight * answerable) / denominator
        if score > best_score:
            best_score = score
            best_attribute = attribute
    return best_attribute


def select_dynamic_attribute(
    candidate_ids: list[str],
    attribute_index: AttributeIndex,
    excluded: set[str],
    min_entropy: float = 0.15,
) -> str | None:
    """Pick the structural attribute with the highest value-diversity among
    `candidate_ids`, or None if nothing clears `min_entropy` (meaning: for
    every structurally-measurable attribute, the pool already agrees, so
    asking wouldn't help narrow it further -- caller should fall back to
    non-structural attributes like style/size/use_case/feature instead).
    """
    if not candidate_ids:
        return None
    best_attribute: str | None = None
    best_entropy = min_entropy
    for attribute in STRUCTURAL_ATTRIBUTES:
        if attribute in excluded:
            continue
        values = attribute_index.values_for(attribute, candidate_ids)
        score = normalized_entropy(values)
        if score > best_entropy:
            best_entropy = score
            best_attribute = attribute
    return best_attribute
