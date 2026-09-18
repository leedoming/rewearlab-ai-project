"""Tests for the E7-E8 padding ablation infrastructure."""

from types import MappingProxyType

import pytest

from evaluation.dataset import EvaluationDataset, QueryRecord
from evaluation.evaluator import run_padding_experiment, run_raw_filter_experiment


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


def make_e7_config(**overrides):
    config = {
        "experiment_id": "E7",
        "split": "dev",
        "preprocessing": {"mode": "bbox", "bbox_policy": "highest_confidence", "padding_ratio": 0.1},
        "category_filter": {"policy": "hard"},
        "retrieval": {"top_k": 10, "dedupe": True},
    }
    config.update(overrides)
    return config


def make_e8_config(**overrides):
    config = {
        "experiment_id": "E8",
        "split": "dev",
        "preprocessing": {"mode": "raw"},
        "category_filter": {"policy": "soft"},
        "retrieval": {"top_k": 10, "dedupe": True},
    }
    config.update(overrides)
    return config


def result(product_id, distance, collection="outer"):
    return {"product_id": product_id, "rank": 1, "raw_distance": distance, "collection": collection, "metadata": {}}


# --- run_padding_experiment (E7): config validation ------------------------

def test_run_padding_experiment_rejects_unknown_experiment_id(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    with pytest.raises(ValueError, match="experiment_id"):
        run_padding_experiment(dataset, lambda q: ([], {}, 0), tmp_path, make_e7_config(experiment_id="E9"))


def test_run_padding_experiment_rejects_non_bbox_mode(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    config = make_e7_config(preprocessing={"mode": "raw"})
    with pytest.raises(ValueError, match="bbox"):
        run_padding_experiment(dataset, lambda q: ([], {}, 0), tmp_path, config)


def test_run_padding_experiment_rejects_invalid_bbox_policy(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    config = make_e7_config(preprocessing={"mode": "bbox", "bbox_policy": "not_a_real_policy", "padding_ratio": 0.1})
    with pytest.raises(ValueError, match="bbox_policy"):
        run_padding_experiment(dataset, lambda q: ([], {}, 0), tmp_path, config)


def test_run_padding_experiment_accepts_any_valid_bbox_policy(tmp_path):
    # Same reasoning as E5/E6: no bbox ablation evidence exists yet, so E7
    # must accept whichever of the four valid policies the config declares.
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    config = make_e7_config(preprocessing={"mode": "bbox", "bbox_policy": "category_largest", "padding_ratio": 0.1})
    records, _ = run_padding_experiment(dataset, lambda q: ([], {}, 0), tmp_path, config)
    assert records[0]["config"]["preprocessing"]["bbox_policy"] == "category_largest"


def test_run_padding_experiment_rejects_zero_padding(tmp_path):
    # The whole point of E7 is padding=10%; 0% would silently duplicate E1-E6.
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    config = make_e7_config(preprocessing={"mode": "bbox", "bbox_policy": "highest_confidence", "padding_ratio": 0.0})
    with pytest.raises(ValueError, match="padding_ratio"):
        run_padding_experiment(dataset, lambda q: ([], {}, 0), tmp_path, config)


def test_run_padding_experiment_rejects_invalid_category_filter_policy(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    config = make_e7_config(category_filter={"policy": "not_a_real_policy"})
    with pytest.raises(ValueError, match="category_filter"):
        run_padding_experiment(dataset, lambda q: ([], {}, 0), tmp_path, config)


def test_run_padding_experiment_accepts_any_valid_category_filter_policy(tmp_path):
    # Same reasoning as the bbox policy above: no filter ablation evidence
    # exists yet either, so E7 must accept none/hard/soft.
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    config = make_e7_config(category_filter={"policy": "soft"})
    records, _ = run_padding_experiment(dataset, lambda q: ([], {}, 0), tmp_path, config)
    assert records[0]["config"]["category_filter"]["policy"] == "soft"


def test_run_padding_experiment_rejects_dedupe_false(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    config = make_e7_config(retrieval={"top_k": 10, "dedupe": False})
    with pytest.raises(ValueError, match="dedupe"):
        run_padding_experiment(dataset, lambda q: ([], {}, 0), tmp_path, config)


def test_run_padding_experiment_uses_real_exclusion_count(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(labels={"P1": 2, "P2": 1}),))
    config = make_e7_config()

    def pipeline(query):
        return [result("P1", 0.1)], {"mode": "bbox", "fallback_used": False}, 1

    records, summary = run_padding_experiment(dataset, pipeline, tmp_path, config)

    assert records[0]["metrics"]["relevant_exclusion_rate"] == pytest.approx(0.5)
    assert summary["experiment_id"] == "E7"


# --- run_raw_filter_experiment (E8): config validation ----------------------

def test_run_raw_filter_experiment_rejects_unknown_experiment_id(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    with pytest.raises(ValueError, match="experiment_id"):
        run_raw_filter_experiment(dataset, lambda q: ([], 0), tmp_path, make_e8_config(experiment_id="E9"))


def test_run_raw_filter_experiment_rejects_non_raw_mode(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    config = make_e8_config(preprocessing={"mode": "bbox", "bbox_policy": "highest_confidence"})
    with pytest.raises(ValueError, match="raw"):
        run_raw_filter_experiment(dataset, lambda q: ([], 0), tmp_path, config)


def test_run_raw_filter_experiment_rejects_filter_policy_mismatch(tmp_path):
    # E8 must be "soft", concretely -- not "Best", unlike E7.
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    config = make_e8_config(category_filter={"policy": "hard"})
    with pytest.raises(ValueError, match="category_filter"):
        run_raw_filter_experiment(dataset, lambda q: ([], 0), tmp_path, config)


def test_run_raw_filter_experiment_rejects_dedupe_false(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    config = make_e8_config(retrieval={"top_k": 10, "dedupe": False})
    with pytest.raises(ValueError, match="dedupe"):
        run_raw_filter_experiment(dataset, lambda q: ([], 0), tmp_path, config)


def test_run_raw_filter_experiment_never_fabricates_bbox_metadata(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(labels={"P1": 2, "P2": 1}),))
    config = make_e8_config()

    def pipeline(query):
        return [result("P1", 0.1)], 1

    records, summary = run_raw_filter_experiment(dataset, pipeline, tmp_path, config)

    assert records[0]["preprocessing"] == {
        "mode": "raw",
        "selected_bbox": None,
        "fallback_used": False,
        "fallback_reason": None,
    }
    assert records[0]["metrics"]["relevant_exclusion_rate"] == pytest.approx(0.5)
    assert summary["experiment_id"] == "E8"
