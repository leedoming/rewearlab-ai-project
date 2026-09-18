"""Tests for the E5-E6 category-filter ablation infrastructure."""

from types import MappingProxyType

import pytest

from evaluation.dataset import EvaluationDataset, QueryRecord
from evaluation.evaluator import evaluate_query, run_category_filter_experiment


def make_query(query_id="Q001", category="outer", labels=None):
    return QueryRecord(
        query_id=query_id,
        image_path=None,
        category=category,
        difficulty="hard",
        scene_type="person_wearing",
        split="dev",
        num_visible_items=1,
        background_complexity="high",
        important_features=(),
        labels=MappingProxyType(labels or {"P1": 2, "P2": 1}),
    )


def make_config(experiment_id="E5", **overrides):
    config = {
        "experiment_id": experiment_id,
        "split": "dev",
        "preprocessing": {"mode": "bbox", "bbox_policy": "highest_confidence", "padding_ratio": 0.0},
        "category_filter": {"policy": "hard"},
        "retrieval": {"top_k": 10, "dedupe": True},
    }
    config.update(overrides)
    return config


def result(product_id, distance, collection="outer"):
    return {"product_id": product_id, "rank": 1, "raw_distance": distance, "collection": collection, "metadata": {}}


# --- evaluate_query: excluded_relevant_count stays backward compatible ---

def test_evaluate_query_defaults_to_zero_excluded_matching_e0_e4_behavior():
    query = make_query(labels={"P1": 2})
    metrics = evaluate_query(query, [result("P1", 0.1)])
    assert metrics["relevant_exclusion_rate"] == 0.0


def test_evaluate_query_reports_real_exclusion_count_when_given():
    query = make_query(labels={"P1": 2, "P2": 1})
    metrics = evaluate_query(query, [result("P1", 0.1)], excluded_relevant_count=1)
    assert metrics["relevant_exclusion_rate"] == pytest.approx(0.5)


# --- run_category_filter_experiment: config validation --------------------

def test_run_category_filter_experiment_rejects_unknown_experiment_id(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    with pytest.raises(ValueError, match="experiment_id"):
        run_category_filter_experiment(dataset, lambda q: ([], {}, 0), tmp_path, make_config("E9"))


def test_run_category_filter_experiment_rejects_non_bbox_mode(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    config = make_config(preprocessing={"mode": "raw", "bbox_policy": "highest_confidence"})
    with pytest.raises(ValueError, match="bbox"):
        run_category_filter_experiment(dataset, lambda q: ([], {}, 0), tmp_path, config)


def test_run_category_filter_experiment_rejects_invalid_bbox_policy(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    config = make_config(preprocessing={"mode": "bbox", "bbox_policy": "not_a_real_policy", "padding_ratio": 0.0})
    with pytest.raises(ValueError, match="bbox_policy"):
        run_category_filter_experiment(dataset, lambda q: ([], {}, 0), tmp_path, config)


def test_run_category_filter_experiment_accepts_any_valid_bbox_policy(tmp_path):
    # No "best" policy has been decided (no ablation evidence yet) -- E5/E6
    # must accept whichever of the four valid policies the config declares.
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    config = make_config(preprocessing={"mode": "bbox", "bbox_policy": "category_largest", "padding_ratio": 0.0})
    records, _ = run_category_filter_experiment(dataset, lambda q: ([], {}, 0), tmp_path, config)
    assert records[0]["config"]["preprocessing"]["bbox_policy"] == "category_largest"


def test_run_category_filter_experiment_rejects_nonzero_padding(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    config = make_config(preprocessing={"mode": "bbox", "bbox_policy": "highest_confidence", "padding_ratio": 0.2})
    with pytest.raises(ValueError, match="padding_ratio"):
        run_category_filter_experiment(dataset, lambda q: ([], {}, 0), tmp_path, config)


def test_run_category_filter_experiment_rejects_filter_policy_mismatch(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    # E5 must be "hard", not "soft".
    config = make_config("E5", category_filter={"policy": "soft"})
    with pytest.raises(ValueError, match="category_filter"):
        run_category_filter_experiment(dataset, lambda q: ([], {}, 0), tmp_path, config)


def test_run_category_filter_experiment_rejects_dedupe_false(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    config = make_config(retrieval={"top_k": 10, "dedupe": False})
    with pytest.raises(ValueError, match="dedupe"):
        run_category_filter_experiment(dataset, lambda q: ([], {}, 0), tmp_path, config)


# --- run_category_filter_experiment: happy path, real exclusion flows through

def test_run_category_filter_experiment_uses_real_exclusion_count(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(labels={"P1": 2, "P2": 1}),))
    config = make_config("E5")

    def pipeline(query):
        # Simulate: only P1 was retrieved, and P2 was excluded by the filter.
        return [result("P1", 0.1)], {"mode": "bbox", "fallback_used": False}, 1

    records, summary = run_category_filter_experiment(dataset, pipeline, tmp_path, config)

    assert records[0]["metrics"]["relevant_exclusion_rate"] == pytest.approx(0.5)
    assert summary["experiment_id"] == "E5"
