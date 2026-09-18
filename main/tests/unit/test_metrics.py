"""Unit tests for evaluation.metrics.

Covers IMPLEMENTATION_SPEC.md section 27's required cases (relevant at
rank 1, relevant at rank 2, no relevant result, multiple relevant items,
graded NDCG, K larger than result count) and section 25's explicit edge
cases (no relevant item -> Recall None, no relevant retrieval -> MRR 0,
fewer than K results -> Precision denominator stays K).
"""

import math

import pytest

from evaluation.metrics import (
    dcg_at_k,
    incompatible_category_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    relevant_exclusion_rate,
)


# --- precision_at_k ---------------------------------------------------

def test_precision_relevant_at_rank_1():
    assert precision_at_k([1, 0, 0, 0], k=4) == pytest.approx(0.25)


def test_precision_relevant_at_rank_2():
    assert precision_at_k([0, 1, 0, 0], k=4) == pytest.approx(0.25)


def test_precision_no_relevant_result():
    assert precision_at_k([0, 0, 0, 0], k=4) == 0.0


def test_precision_multiple_relevant_items():
    assert precision_at_k([1, 0, 1, 2], k=4) == pytest.approx(0.75)


def test_precision_fewer_than_k_results_denominator_stays_k():
    # Only 2 results retrieved (both relevant), but K=5: denominator is
    # still 5, not 2 (IMPLEMENTATION_SPEC.md section 25).
    assert precision_at_k([1, 1], k=5) == pytest.approx(2 / 5)


def test_precision_k_larger_than_result_count_with_no_relevant():
    assert precision_at_k([], k=3) == 0.0


def test_precision_rejects_non_positive_k():
    with pytest.raises(ValueError):
        precision_at_k([1, 0], k=0)


@pytest.mark.parametrize("metric,args", [
    (precision_at_k, ([1, 0], 2.5)),
    (recall_at_k, ([1, 0], 2.5, 1)),
    (dcg_at_k, ([1, 0], 2.5)),
    (ndcg_at_k, ([1, 0], 2.5)),
    (incompatible_category_rate_at_k, ([False], 2.5)),
])
def test_at_k_metrics_reject_non_integer_k(metric, args):
    with pytest.raises(ValueError, match="positive integer"):
        metric(*args)


# --- recall_at_k --------------------------------------------------------

def test_recall_relevant_at_rank_1():
    assert recall_at_k([1, 0, 0], k=3, total_relevant=1) == pytest.approx(1.0)


def test_recall_relevant_at_rank_2():
    assert recall_at_k([0, 1, 0], k=3, total_relevant=1) == pytest.approx(1.0)


def test_recall_no_relevant_result_found():
    assert recall_at_k([0, 0, 0], k=3, total_relevant=2) == 0.0


def test_recall_multiple_relevant_items():
    assert recall_at_k([1, 0, 1, 0], k=4, total_relevant=2) == pytest.approx(1.0)


def test_recall_partial_when_not_all_relevant_retrieved():
    assert recall_at_k([1, 0, 0], k=3, total_relevant=2) == pytest.approx(0.5)


def test_recall_no_relevant_item_returns_none():
    # IMPLEMENTATION_SPEC.md section 25: "No Relevant Item" -> None/NaN,
    # to be excluded from aggregation by the caller (not treated as 0).
    assert recall_at_k([0, 0, 0], k=3, total_relevant=0) is None


def test_recall_k_larger_than_result_count():
    assert recall_at_k([1], k=10, total_relevant=1) == pytest.approx(1.0)


def test_recall_rejects_negative_total_relevant():
    with pytest.raises(ValueError, match="total_relevant"):
        recall_at_k([1], k=1, total_relevant=-1)


# --- mrr -------------------------------------------------------------

def test_mrr_relevant_at_rank_1():
    assert mrr([1, 0, 0]) == pytest.approx(1.0)


def test_mrr_relevant_at_rank_2():
    assert mrr([0, 1, 0]) == pytest.approx(0.5)


def test_mrr_no_relevant_retrieval_returns_zero():
    # IMPLEMENTATION_SPEC.md section 25: "No Relevant Retrieval" -> MRR = 0.
    assert mrr([0, 0, 0]) == 0.0


def test_mrr_multiple_relevant_items_uses_first_only():
    assert mrr([0, 1, 1, 0]) == pytest.approx(0.5)


def test_mrr_empty_list_returns_zero():
    assert mrr([]) == 0.0


# --- dcg_at_k / ndcg_at_k -----------------------------------------------

def test_dcg_matches_manual_calculation():
    # DCG = 2/log2(2) + 1/log2(3) + 0/log2(4)
    expected = 2 / math.log2(2) + 1 / math.log2(3) + 0 / math.log2(4)
    assert dcg_at_k([2, 1, 0], k=3) == pytest.approx(expected)


def test_ndcg_perfect_ranking_is_one():
    # Already in ideal (descending) order -> NDCG@K == 1.0.
    assert ndcg_at_k([2, 1, 0], k=3) == pytest.approx(1.0)


def test_ndcg_graded_relevance_penalizes_wrong_order():
    # Highly-relevant (2) buried below relevant (1) and irrelevant (0):
    # DCG uses the given order, IDCG uses the best possible order (2,1,0).
    relevances = [0, 1, 2]
    ideal = [2, 1, 0]
    expected = dcg_at_k(relevances, 3) / dcg_at_k(ideal, 3)
    assert ndcg_at_k(relevances, k=3) == pytest.approx(expected)
    assert 0 < ndcg_at_k(relevances, k=3) < 1


def test_ndcg_no_relevant_items_returns_zero_not_none():
    assert ndcg_at_k([0, 0, 0], k=3) == 0.0


def test_ndcg_k_larger_than_result_count():
    # Only 1 result retrieved, K=5: missing slots contribute 0 gain to
    # both DCG and (self-ideal) IDCG, which are equal here since a single
    # relevant item is already "ideally" ordered -> NDCG == 1.0.
    assert ndcg_at_k([1], k=5) == pytest.approx(1.0)


def test_ndcg_accepts_explicit_ideal_relevances_from_full_ground_truth():
    # Retrieved only found the "relevant" (1) item, but the full labeled
    # pool for this query also contains a "highly relevant" (2) item that
    # this ranking failed to surface at all.
    retrieved = [1, 0, 0]
    ideal_from_ground_truth = [2, 1]
    result = ndcg_at_k(retrieved, k=3, ideal_relevances=ideal_from_ground_truth)
    expected = dcg_at_k(retrieved, 3) / dcg_at_k(sorted(ideal_from_ground_truth, reverse=True), 3)
    assert result == pytest.approx(expected)
    assert result < 1.0


def test_ndcg_rejects_ideal_that_cannot_dominate_retrieved_ranking():
    with pytest.raises(ValueError, match="ideal_relevances"):
        ndcg_at_k([2, 2, 0], k=3, ideal_relevances=[1])


def test_ndcg_rejects_non_numeric_explicit_ideal_grade():
    with pytest.raises(ValueError, match="ideal_relevances"):
        ndcg_at_k([1, 0], k=2, ideal_relevances=[2, None])


def test_dcg_and_ndcg_reject_non_positive_k():
    with pytest.raises(ValueError):
        dcg_at_k([1, 0], k=0)
    with pytest.raises(ValueError):
        ndcg_at_k([1, 0], k=-1)


# --- incompatible_category_rate_at_k -------------------------------------

def test_incompatible_category_rate_all_compatible():
    assert incompatible_category_rate_at_k([False, False, False], k=3) == 0.0


def test_incompatible_category_rate_some_incompatible():
    assert incompatible_category_rate_at_k([True, False, True, False], k=4) == pytest.approx(0.5)


def test_incompatible_category_rate_fewer_than_k_denominator_stays_k():
    assert incompatible_category_rate_at_k([True], k=4) == pytest.approx(0.25)


# --- relevant_exclusion_rate ---------------------------------------------

def test_relevant_exclusion_rate_none_excluded():
    assert relevant_exclusion_rate(total_relevant=5, excluded_relevant_count=0) == 0.0


def test_relevant_exclusion_rate_some_excluded():
    assert relevant_exclusion_rate(total_relevant=4, excluded_relevant_count=1) == pytest.approx(0.25)


def test_relevant_exclusion_rate_all_excluded():
    assert relevant_exclusion_rate(total_relevant=3, excluded_relevant_count=3) == pytest.approx(1.0)


def test_relevant_exclusion_rate_no_relevant_items_returns_none():
    assert relevant_exclusion_rate(total_relevant=0, excluded_relevant_count=0) is None


def test_relevant_exclusion_rate_rejects_impossible_counts():
    with pytest.raises(ValueError):
        relevant_exclusion_rate(total_relevant=2, excluded_relevant_count=3)


def test_relevant_exclusion_rate_rejects_impossible_count_when_total_is_zero():
    with pytest.raises(ValueError, match="cannot exceed"):
        relevant_exclusion_rate(total_relevant=0, excluded_relevant_count=3)


def test_relevant_exclusion_rate_rejects_negative_excluded_count():
    with pytest.raises(ValueError, match="excluded_relevant_count"):
        relevant_exclusion_rate(total_relevant=3, excluded_relevant_count=-1)
