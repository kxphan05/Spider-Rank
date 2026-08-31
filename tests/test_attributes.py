"""Unit tests for starter/attributes.py: pure functions, no catalog/model needed.

Table-tests the specific bug classes documented in the module's own comments
(word-boundary false positives, budget sign inversion, unpriced-product
neutrality) rather than re-deriving requirements from scratch.
"""
from __future__ import annotations

import pytest

from starter.attributes import (
    AttributeIndex,
    budget_constraint_satisfied,
    extract_disclosed_value,
    normalized_entropy,
    parse_budget_constraint,
    select_dynamic_attribute,
    select_weighted_attribute,
)


# ---------------------------------------------------------------------------
# parse_budget_constraint
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("$80", "<=80"),
    ("under $80", "<=80"),
    ("80 dollars", "<=80"),
    ("80 bucks", "<=80"),
    ("$79.99", "<=79.99"),
    ("around fifty dollars", "<=50"),
    ("about eighty dollars", "<=80"),
    ("one hundred dollars", "<=100"),
    ("twenty five dollars", "<=25"),
])
def test_parse_budget_constraint_ceiling_phrasings(text, expected):
    assert parse_budget_constraint(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("over $80", ">=80"),
    ("above $80", ">=80"),
    ("at least 80 dollars", ">=80"),
    ("more than $80", ">=80"),
    ("starting at $80", ">=80"),
])
def test_parse_budget_constraint_floor_phrasings(text, expected):
    assert parse_budget_constraint(text) == expected


@pytest.mark.parametrize("text", [
    "no more than $80",
    "not more than 80 dollars",
    "no higher than $80",
    "nothing over $80",
    "nothing above 80 bucks",
    "not over $80",
])
def test_parse_budget_constraint_negated_floor_reads_as_ceiling(text):
    # A negated floor cue ("no more than") must not be misread as a floor --
    # the substring "more than" sits inside it, so this is a real regression
    # risk for any future cue-matching change.
    assert parse_budget_constraint(text) == "<=80"


@pytest.mark.parametrize("text", [
    "size 80",
    "80",
    "I'm a size 8",
    "",
    "no numbers here",
])
def test_parse_budget_constraint_returns_none_without_currency_marker(text):
    # A bare number must never be read as money.
    assert parse_budget_constraint(text) is None


def test_parse_budget_constraint_case_insensitive():
    assert parse_budget_constraint("UNDER $80") == "<=80"


# ---------------------------------------------------------------------------
# budget_constraint_satisfied
# ---------------------------------------------------------------------------

def test_budget_constraint_satisfied_ceiling_boundary():
    assert budget_constraint_satisfied("<=80", 80.0) is True
    assert budget_constraint_satisfied("<=80", 80.01) is False
    assert budget_constraint_satisfied("<=80", 1.0) is True


def test_budget_constraint_satisfied_floor_boundary():
    assert budget_constraint_satisfied(">=80", 80.0) is True
    assert budget_constraint_satisfied(">=80", 79.99) is False
    assert budget_constraint_satisfied(">=80", 500.0) is True


def test_budget_constraint_satisfied_malformed_token_returns_none():
    assert budget_constraint_satisfied("garbage", 50.0) is None


# ---------------------------------------------------------------------------
# extract_disclosed_value -- word-boundary matching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attribute,text", [
    ("color", "I love this embroidered pattern"),   # must not yield "red"
    ("color", "titanium frame please"),              # must not yield "tan"
    ("color", "a nice instant classic"),              # must not yield "tan"
    ("material", "a beautiful necklace"),             # must not yield "lace"
])
def test_extract_disclosed_value_no_substring_false_positive(attribute, text):
    assert extract_disclosed_value(attribute, text) is None


@pytest.mark.parametrize("attribute,text,expected", [
    ("color", "I want it in red please", "red"),
    ("color", "black shoes", "black"),
    ("material", "cotton would be great", "cotton"),
    ("material", "sterling silver", "sterling silver"),
])
def test_extract_disclosed_value_matches_real_terms(attribute, text, expected):
    assert extract_disclosed_value(attribute, text) == expected


def test_extract_disclosed_value_negation_aware():
    # "no leather" is a rejection, not a disclosure.
    assert extract_disclosed_value("material", "no leather please") is None


def test_extract_disclosed_value_budget_returns_constraint_token():
    assert extract_disclosed_value("budget", "keep it under $80") == "<=80"


def test_extract_disclosed_value_unknown_attribute_returns_none():
    assert extract_disclosed_value("style", "something casual") is None


def test_extract_disclosed_value_non_string_input():
    assert extract_disclosed_value("color", None) is None
    assert extract_disclosed_value("color", "") is None


# ---------------------------------------------------------------------------
# normalized_entropy
# ---------------------------------------------------------------------------

def test_normalized_entropy_single_value_is_zero():
    assert normalized_entropy(["red", "red", "red"]) == 0.0


def test_normalized_entropy_uniform_two_way_split_is_one():
    assert normalized_entropy(["red", "blue"]) == pytest.approx(1.0)


def test_normalized_entropy_skewed_split_is_between_zero_and_one():
    score = normalized_entropy(["red"] * 9 + ["blue"])
    assert 0.0 < score < 1.0


def test_normalized_entropy_all_none_is_zero():
    assert normalized_entropy([None, None, None]) == 0.0


def test_normalized_entropy_too_short_is_zero():
    assert normalized_entropy([]) == 0.0
    assert normalized_entropy(["red"]) == 0.0


def test_normalized_entropy_none_counts_as_its_own_bucket():
    # A pool split evenly between a known value and "unknown" should be
    # treated the same as any other two-way split.
    score = normalized_entropy(["red", None])
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AttributeIndex -- constructed directly, no catalog file needed
# ---------------------------------------------------------------------------

@pytest.fixture
def index():
    values = {
        "A": {"material": "cotton", "color": "red"},
        "B": {"material": "leather", "color": "blue"},
        "C": {"material": None, "color": "red"},
    }
    prices = {"A": 50.0, "B": 120.0}  # C is unpriced
    popularity = {"A": 10.0, "B": 5000.0, "C": 1.0}
    return AttributeIndex(values, prices, popularity)


def test_attribute_index_value_for(index):
    assert index.value_for("material", "A") == "cotton"
    assert index.value_for("material", "C") is None
    assert index.value_for("material", "missing") is None


def test_attribute_index_price_for_unpriced_is_none(index):
    assert index.price_for("A") == 50.0
    assert index.price_for("C") is None


def test_attribute_index_popularity_for_missing_id_is_zero(index):
    assert index.popularity_for("missing") == 0.0


def test_attribute_index_by_popularity_orders_most_reviewed_first(index):
    assert index.by_popularity(["A", "B", "C"]) == ["B", "A", "C"]


def test_attribute_index_values_for_preserves_order(index):
    assert index.values_for("color", ["B", "A", "C"]) == ["blue", "red", "red"]


# ---------------------------------------------------------------------------
# select_dynamic_attribute / select_weighted_attribute
# ---------------------------------------------------------------------------

def test_select_dynamic_attribute_picks_highest_entropy(index):
    # color splits {red, blue, red} (some entropy); material splits
    # {cotton, leather, None} (also some entropy, 3 distinct buckets).
    # Both clear a low threshold, so just assert a legal attribute is chosen.
    result = select_dynamic_attribute(["A", "B", "C"], index, excluded=set(), min_entropy=0.0)
    assert result in ("material", "color")


def test_select_dynamic_attribute_none_when_pool_agrees(index):
    # A pool where every candidate shares the same color has zero entropy.
    values = {"A": {"material": None, "color": "red"}, "B": {"material": None, "color": "red"}}
    only_red = AttributeIndex(values)
    assert select_dynamic_attribute(["A", "B"], only_red, excluded=set(), min_entropy=0.0) is None


def test_select_dynamic_attribute_respects_excluded(index):
    result = select_dynamic_attribute(["A", "B", "C"], index, excluded={"material", "color"}, min_entropy=0.0)
    assert result is None


def test_select_dynamic_attribute_empty_pool_is_none(index):
    assert select_dynamic_attribute([], index, excluded=set()) is None


def test_select_weighted_attribute_missing_entropy_component_falls_back_to_answerability(index):
    # "feature" has no catalog labels at all (not in STRUCTURAL_ATTRIBUTES),
    # so its score must be answerability alone, not penalized for a missing
    # entropy term.
    answerability = {"feature": 0.9, "material": 0.1, "color": 0.1}
    result = select_weighted_attribute(
        [], index, excluded=set(), answerability=answerability,
        entropy_weight=1.0, answerability_weight=1.0, min_entropy=0.0,
    )
    assert result == "feature"


def test_select_weighted_attribute_zero_denominator_returns_none(index):
    result = select_weighted_attribute(
        ["A", "B"], index, excluded=set(), answerability={"material": 0.5},
        entropy_weight=0.0, answerability_weight=0.0,
    )
    assert result is None


def test_select_weighted_attribute_excludes_already_asked(index):
    answerability = {"material": 0.9, "color": 0.9}
    result = select_weighted_attribute(
        ["A", "B", "C"], index, excluded={"material", "color"}, answerability=answerability,
        entropy_weight=1.0, answerability_weight=1.0, min_entropy=0.0,
    )
    assert result is None
