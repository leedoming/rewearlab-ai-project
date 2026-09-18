"""Tests for detection-threshold / padding / Top-K sensitivity metrics."""

import pytest

from evaluation.sensitivity import (
    average_detection_count,
    detection_success_rate,
    fallback_rate,
    summarize_sensitivity_sweep,
    wrong_object_rate,
)


# --- average_detection_count / detection_success_rate ----------------------

def test_average_detection_count():
    assert average_detection_count([0, 2, 4]) == pytest.approx(2.0)


def test_average_detection_count_rejects_empty_input():
    with pytest.raises(ValueError, match="must not be empty"):
        average_detection_count([])


def test_detection_success_rate_counts_nonzero_detection_queries():
    assert detection_success_rate([0, 1, 2, 0]) == pytest.approx(0.5)


def test_detection_success_rate_rejects_empty_input():
    with pytest.raises(ValueError, match="must not be empty"):
        detection_success_rate([])


# --- fallback_rate -----------------------------------------------------------

def test_fallback_rate_counts_true_flags():
    assert fallback_rate([True, False, True, False]) == pytest.approx(0.5)


def test_fallback_rate_rejects_empty_input():
    with pytest.raises(ValueError, match="must not be empty"):
        fallback_rate([])


# --- wrong_object_rate -------------------------------------------------------

def test_wrong_object_rate_flags_incompatible_non_fallback_selection():
    # "outer" category allows ["top", "outer"] (retrieval.category);
    # "bottom" is not compatible -- a confident but wrong-object selection.
    selections = [("outer", "bottom", False)]
    assert wrong_object_rate(selections) == pytest.approx(1.0)


def test_wrong_object_rate_ignores_fallback_selections():
    # Even though "bottom" is incompatible with "outer", this was a
    # fallback (nothing category-compatible was found) -- that's
    # fallback_rate's concern, not wrong_object_rate's.
    selections = [("outer", "bottom", True)]
    assert wrong_object_rate(selections) == pytest.approx(0.0)


def test_wrong_object_rate_zero_when_all_non_fallback_selections_compatible():
    selections = [("outer", "top", False), ("outer", "outer", False)]
    assert wrong_object_rate(selections) == pytest.approx(0.0)


def test_wrong_object_rate_zero_when_every_selection_was_a_fallback():
    selections = [("outer", "bottom", True), ("pants", "top", True)]
    assert wrong_object_rate(selections) == pytest.approx(0.0)


# --- summarize_sensitivity_sweep ---------------------------------------------

def test_summarize_sensitivity_sweep_groups_by_parameter_value():
    rows = [
        {"detection_threshold": 0.3, "metrics": {"ndcg_at_10": 0.2}},
        {"detection_threshold": 0.3, "metrics": {"ndcg_at_10": 0.4}},
        {"detection_threshold": 0.5, "metrics": {"ndcg_at_10": 0.8}},
    ]

    summary = summarize_sensitivity_sweep(rows, "detection_threshold")

    assert set(summary) == {0.3, 0.5}
    assert summary[0.3]["ndcg_at_10"]["mean"] == pytest.approx(0.3)
    assert summary[0.3]["ndcg_at_10"]["count"] == 2
    assert summary[0.5]["ndcg_at_10"]["mean"] == pytest.approx(0.8)


def test_summarize_sensitivity_sweep_handles_multiple_metric_names():
    rows = [
        {"top_k": 5, "metrics": {"precision_at_5": 0.6, "recall_at_5": 0.3}},
        {"top_k": 10, "metrics": {"precision_at_5": 0.4, "recall_at_5": 0.5}},
    ]

    summary = summarize_sensitivity_sweep(rows, "top_k")

    assert summary[5]["precision_at_5"]["mean"] == pytest.approx(0.6)
    assert summary[5]["recall_at_5"]["mean"] == pytest.approx(0.3)
    assert summary[10]["precision_at_5"]["mean"] == pytest.approx(0.4)


def test_summarize_sensitivity_sweep_empty_rows_returns_empty_summary():
    assert summarize_sensitivity_sweep([], "padding_ratio") == {}
