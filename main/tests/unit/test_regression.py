"""Tests for quality regression checks and critical-query rank assertions."""

import pytest

from evaluation.regression import (
    check_quality_regression,
    first_relevant_rank,
    passes_critical_query_rank,
    passes_quality_regression,
)


def result(product_id):
    return {"product_id": product_id}


# --- passes_quality_regression / check_quality_regression -------------------

def test_passes_quality_regression_true_when_within_tolerance():
    # 0.68 >= 0.70 * 0.97 (0.679) -- barely passes.
    assert passes_quality_regression(0.68, 0.70, tolerance=0.97)


def test_passes_quality_regression_false_when_below_tolerance():
    assert not passes_quality_regression(0.60, 0.70, tolerance=0.97)


def test_passes_quality_regression_true_when_new_value_improves():
    assert passes_quality_regression(0.80, 0.70)


def test_passes_quality_regression_none_when_either_value_is_none():
    assert passes_quality_regression(None, 0.70) is None
    assert passes_quality_regression(0.70, None) is None


def test_check_quality_regression_reports_per_metric():
    new_metrics = {"ndcg_at_10": 0.60, "mrr": 0.80}
    baseline_metrics = {"ndcg_at_10": 0.70, "mrr": 0.70}

    result_map = check_quality_regression(new_metrics, baseline_metrics, ["ndcg_at_10", "mrr"])

    assert result_map["ndcg_at_10"] is False
    assert result_map["mrr"] is True


# --- first_relevant_rank -----------------------------------------------------

def test_first_relevant_rank_finds_first_matching_rank():
    results = [result("X"), result("P1"), result("P2")]
    labels = {"P1": 2, "P2": 1}
    assert first_relevant_rank(results, labels) == 2


def test_first_relevant_rank_none_when_nothing_relevant_retrieved():
    results = [result("X"), result("Y")]
    labels = {"P1": 2}
    assert first_relevant_rank(results, labels) is None


def test_first_relevant_rank_respects_min_relevance():
    # P1 is grade 1 ("Relevant"), not grade 2 ("Highly Relevant") --
    # must not count when min_relevance=2.
    results = [result("P1")]
    labels = {"P1": 1}
    assert first_relevant_rank(results, labels, min_relevance=2) is None
    assert first_relevant_rank(results, labels, min_relevance=1) == 1


# --- passes_critical_query_rank ----------------------------------------------

def test_passes_critical_query_rank_true_within_bound():
    assert passes_critical_query_rank(5, max_rank=5)


def test_passes_critical_query_rank_false_beyond_bound():
    assert not passes_critical_query_rank(6, max_rank=5)


def test_passes_critical_query_rank_false_when_rank_is_none():
    assert not passes_critical_query_rank(None, max_rank=5)
