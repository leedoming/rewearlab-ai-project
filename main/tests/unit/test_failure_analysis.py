"""Tests for the failure taxonomy, poor-query extraction, and failure summary."""

import pytest

from evaluation.failure_analysis import (
    FAILURE_CATEGORIES,
    classify_failure,
    has_highly_relevant_in_top_k,
    is_poor_query,
    summarize_failures,
)


def result(product_id):
    return {"product_id": product_id}


# --- has_highly_relevant_in_top_k ------------------------------------------

def test_has_highly_relevant_in_top_k_true_when_grade_2_present():
    assert has_highly_relevant_in_top_k([result("P1")], {"P1": 2})


def test_has_highly_relevant_in_top_k_false_when_only_grade_1_present():
    # Grade 1 is "Relevant" (section 12), not "Highly Relevant" -- must not count.
    assert not has_highly_relevant_in_top_k([result("P1")], {"P1": 1})


def test_has_highly_relevant_in_top_k_ignores_beyond_k():
    assert not has_highly_relevant_in_top_k([result("P1"), result("P2")], {"P2": 2}, k=1)


# --- is_poor_query: section 31's three-way OR -------------------------------

def test_is_poor_query_true_for_low_ndcg():
    metrics = {"ndcg_at_10": 0.1, "mrr": 1.0}
    assert is_poor_query(metrics, [result("P1")], {"P1": 2})


def test_is_poor_query_true_for_low_mrr():
    metrics = {"ndcg_at_10": 1.0, "mrr": 0.1}
    assert is_poor_query(metrics, [result("P1")], {"P1": 2})


def test_is_poor_query_true_when_no_highly_relevant_in_top_k():
    metrics = {"ndcg_at_10": 1.0, "mrr": 1.0}
    assert is_poor_query(metrics, [result("P1")], {"P1": 1})


def test_is_poor_query_false_when_none_of_the_criteria_trigger():
    metrics = {"ndcg_at_10": 1.0, "mrr": 1.0}
    assert not is_poor_query(metrics, [result("P1")], {"P1": 2})


# --- classify_failure: upstream-first, per section 30 -----------------------

def test_classify_failure_detection_failure_when_no_detections():
    preprocessing = {"mode": "bbox", "fallback_used": True, "fallback_reason": "no_detections"}
    assert classify_failure(preprocessing, {"relevant_exclusion_rate": 0.0}) == "detection_failure"


def test_classify_failure_bbox_selection_failure_when_no_compatible_bbox():
    preprocessing = {
        "mode": "bbox",
        "fallback_used": True,
        "fallback_reason": "no_category_compatible_bbox",
    }
    assert classify_failure(preprocessing, {"relevant_exclusion_rate": 0.0}) == "bbox_selection_failure"


def test_classify_failure_raw_mode_never_reports_detection_or_bbox_failure():
    # E0/E8 are RAW: fallback_used is always False there, so a bbox-stage
    # failure is structurally impossible -- must fall through to a later check.
    preprocessing = {"mode": "raw", "selected_bbox": None, "fallback_used": False, "fallback_reason": None}
    assert classify_failure(preprocessing, {"relevant_exclusion_rate": 0.0}) == "embedding_similarity_failure"


def test_classify_failure_filter_exclusion_failure_when_real_exclusion_happened():
    preprocessing = {"mode": "bbox", "fallback_used": False, "fallback_reason": None}
    assert (
        classify_failure(preprocessing, {"relevant_exclusion_rate": 0.5})
        == "filter_exclusion_failure"
    )


def test_classify_failure_database_coverage_failure_when_flagged():
    preprocessing = {"mode": "bbox", "fallback_used": False, "fallback_reason": None}
    metrics = {"relevant_exclusion_rate": 0.0}
    assert (
        classify_failure(preprocessing, metrics, database_coverage_failure=True)
        == "database_coverage_failure"
    )


def test_classify_failure_database_coverage_checked_before_embedding_fallback():
    # database_coverage_failure=True must win even though filter exclusion
    # is 0 and nothing else upstream explains the failure -- ruling out "was
    # this item even indexed" must happen before blaming embedding quality.
    preprocessing = {"mode": "bbox", "fallback_used": False, "fallback_reason": None}
    metrics = {"relevant_exclusion_rate": 0.0}
    assert classify_failure(preprocessing, metrics, database_coverage_failure=True) != "embedding_similarity_failure"


def test_classify_failure_falls_back_to_embedding_similarity_failure_last():
    preprocessing = {"mode": "bbox", "fallback_used": False, "fallback_reason": None}
    metrics = {"relevant_exclusion_rate": 0.0}
    assert classify_failure(preprocessing, metrics) == "embedding_similarity_failure"


def test_classify_failure_never_fabricates_human_judgment_categories():
    # feature_loss/background_bias/category_prediction_failure/
    # semantic_category_boundary/ranking_failure require human review and
    # must never be auto-assigned by this function.
    manual_only = {
        "feature_loss",
        "background_bias",
        "category_prediction_failure",
        "semantic_category_boundary",
        "ranking_failure",
    }
    preprocessing = {"mode": "bbox", "fallback_used": False, "fallback_reason": None}
    for database_coverage_failure in (True, False):
        outcome = classify_failure(
            preprocessing, {"relevant_exclusion_rate": 0.0}, database_coverage_failure=database_coverage_failure
        )
        assert outcome not in manual_only


# --- summarize_failures: section 32's per-experiment distribution ----------

def test_summarize_failures_counts_and_ratios():
    summary = summarize_failures(["detection_failure", "detection_failure", "filter_exclusion_failure"])
    assert summary["detection_failure"] == {"count": 2, "ratio": pytest.approx(2 / 3)}
    assert summary["filter_exclusion_failure"] == {"count": 1, "ratio": pytest.approx(1 / 3)}


def test_summarize_failures_includes_zero_count_categories():
    summary = summarize_failures(["detection_failure"])
    assert summary["ranking_failure"] == {"count": 0, "ratio": 0.0}
    assert set(summary) == set(FAILURE_CATEGORIES)


def test_summarize_failures_empty_input_has_zero_ratios_not_a_crash():
    summary = summarize_failures([])
    assert all(entry == {"count": 0, "ratio": 0.0} for entry in summary.values())


def test_summarize_failures_rejects_unknown_category():
    with pytest.raises(ValueError, match="Unknown failure category"):
        summarize_failures(["not_a_real_category"])
