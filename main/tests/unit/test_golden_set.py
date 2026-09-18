"""Tests for golden evaluation set coverage validation."""

from types import MappingProxyType

import pytest

from evaluation.dataset import QueryRecord
from evaluation.golden_set import (
    MAX_GOLDEN_SET_SIZE,
    MIN_GOLDEN_SET_SIZE,
    REQUIRED_CASE_TYPES,
    GoldenSetValidationError,
    classify_golden_set_coverage,
    validate_golden_set,
)


def make_query(query_id, difficulty="medium", scene_type="clean_product", category="outer"):
    return QueryRecord(
        query_id=query_id,
        image_path=None,
        category=category,
        difficulty=difficulty,
        scene_type=scene_type,
        split="dev",
        num_visible_items=1,
        background_complexity="low",
        important_features=(),
        labels=MappingProxyType({}),
    )


def make_full_coverage_queries(count=10):
    # Q1: easy, Q2: hard, Q3: multi_item, Q4: complex_background,
    # Q5: category boundary (via explicit id set), Q6: known previous
    # failure (via explicit id set); pad up to `count` with plain queries.
    queries = [
        make_query("Q1", difficulty="easy"),
        make_query("Q2", difficulty="hard"),
        make_query("Q3", scene_type="multi_item"),
        make_query("Q4", scene_type="complex_background"),
        make_query("Q5"),
        make_query("Q6"),
    ]
    for i in range(len(queries) + 1, count + 1):
        queries.append(make_query(f"Q{i}"))
    return queries


# --- classify_golden_set_coverage --------------------------------------------

def test_classify_golden_set_coverage_maps_difficulty_and_scene_type():
    queries = [make_query("Q1", difficulty="easy"), make_query("Q2", difficulty="hard")]
    coverage = classify_golden_set_coverage(queries)
    assert coverage["easy_case"] == ["Q1"]
    assert coverage["hard_case"] == ["Q2"]
    assert coverage["multi_item"] == []


def test_classify_golden_set_coverage_uses_explicit_ids_for_non_derivable_case_types():
    queries = [make_query("Q1"), make_query("Q2")]
    coverage = classify_golden_set_coverage(
        queries, category_boundary_ids={"Q1"}, known_previous_failure_ids={"Q2"}
    )
    assert coverage["category_boundary"] == ["Q1"]
    assert coverage["known_previous_failure"] == ["Q2"]


def test_classify_golden_set_coverage_a_query_can_satisfy_multiple_case_types():
    queries = [make_query("Q1", difficulty="hard", scene_type="multi_item")]
    coverage = classify_golden_set_coverage(queries)
    assert coverage["hard_case"] == ["Q1"]
    assert coverage["multi_item"] == ["Q1"]


def test_classify_golden_set_coverage_returns_every_required_case_type_key():
    coverage = classify_golden_set_coverage([])
    assert set(coverage) == set(REQUIRED_CASE_TYPES)


# --- validate_golden_set ------------------------------------------------------

def test_validate_golden_set_accepts_full_coverage():
    queries = make_full_coverage_queries()
    coverage = validate_golden_set(
        queries, category_boundary_ids={"Q5"}, known_previous_failure_ids={"Q6"}
    )
    assert all(coverage[case_type] for case_type in REQUIRED_CASE_TYPES)


def test_validate_golden_set_rejects_too_few_queries():
    queries = make_full_coverage_queries(count=MIN_GOLDEN_SET_SIZE - 1)
    with pytest.raises(GoldenSetValidationError, match="10-15"):
        validate_golden_set(queries, category_boundary_ids={"Q5"}, known_previous_failure_ids={"Q6"})


def test_validate_golden_set_rejects_too_many_queries():
    queries = make_full_coverage_queries(count=MAX_GOLDEN_SET_SIZE + 1)
    with pytest.raises(GoldenSetValidationError, match="10-15"):
        validate_golden_set(queries, category_boundary_ids={"Q5"}, known_previous_failure_ids={"Q6"})


def test_validate_golden_set_rejects_missing_case_type():
    # No hard-difficulty query anywhere -- "hard_case" coverage is missing.
    queries = [make_query(f"Q{i}") for i in range(1, MIN_GOLDEN_SET_SIZE + 1)]
    with pytest.raises(GoldenSetValidationError, match="hard_case"):
        validate_golden_set(queries)


def test_validate_golden_set_error_names_every_missing_case_type():
    queries = [make_query(f"Q{i}") for i in range(1, MIN_GOLDEN_SET_SIZE + 1)]
    with pytest.raises(GoldenSetValidationError) as exc_info:
        validate_golden_set(queries)
    message = str(exc_info.value)
    for case_type in REQUIRED_CASE_TYPES:
        assert case_type in message
