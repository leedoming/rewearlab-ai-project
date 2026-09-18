"""Tests for the E1-E4 bbox-policy ablation infrastructure.

Covers IMPLEMENTATION_SPEC.md section 19's two MUSTs: policies applied to
the same detection output (DetectionCache), and detection/selection kept
separate from the experiment's config validation (run_bbox_experiment).
"""

import json
from types import MappingProxyType

import pytest

from evaluation.detection_cache import DetectionCache
from evaluation.dataset import EvaluationDataset, QueryRecord
from evaluation.evaluator import run_bbox_experiment
from retrieval.category import get_allowed_labels
from retrieval.preprocessing import select_bbox


def make_query(query_id="Q001", category="outer"):
    return QueryRecord(
        query_id=query_id,
        image_path=f"/images/{query_id}.jpg",
        category=category,
        difficulty="hard",
        scene_type="person_wearing",
        split="dev",
        num_visible_items=1,
        background_complexity="high",
        important_features=(),
        labels=MappingProxyType({"P1": 2}),
    )


def make_config(experiment_id="E1", **overrides):
    config = {
        "experiment_id": experiment_id,
        "split": "dev",
        "preprocessing": {"mode": "bbox", "bbox_policy": "highest_confidence", "padding_ratio": 0.0},
        "category_filter": {"policy": "none"},
        "retrieval": {"top_k": 10, "dedupe": True},
    }
    config.update(overrides)
    return config


def result(product_id, distance, collection="outer"):
    return {"product_id": product_id, "rank": 1, "raw_distance": distance, "collection": collection, "metadata": {}}


# --- DetectionCache: the "same detection output" guarantee --------------

def test_detection_cache_runs_detector_once_per_unique_image():
    calls = []

    def detect(image_path):
        calls.append(image_path)
        return [{"bbox": [0, 0, 1, 1], "label": "top", "score": 0.9, "area": 1}]

    cache = DetectionCache(detect)

    # Simulate four "experiments" each asking for the same two images.
    for _ in range(4):
        cache.get("/images/Q001.jpg")
        cache.get("/images/Q002.jpg")

    assert calls == ["/images/Q001.jpg", "/images/Q002.jpg"]
    assert len(cache) == 2


def test_detection_cache_returns_the_identical_cached_object():
    cache = DetectionCache(lambda path: [{"bbox": [0, 0, 1, 1], "label": "top", "score": 0.9, "area": 1}])
    first = cache.get("/images/Q001.jpg")
    second = cache.get("/images/Q001.jpg")
    assert first is second


# --- run_bbox_experiment: config validation ------------------------------

def test_run_bbox_experiment_rejects_unknown_experiment_id(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    with pytest.raises(ValueError, match="experiment_id"):
        run_bbox_experiment(dataset, lambda q: ([], {}), tmp_path, make_config("E9"))


def test_run_bbox_experiment_rejects_non_bbox_mode(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    config = make_config(preprocessing={"mode": "raw", "bbox_policy": "highest_confidence"})
    with pytest.raises(ValueError, match="bbox"):
        run_bbox_experiment(dataset, lambda q: ([], {}), tmp_path, config)


def test_run_bbox_experiment_rejects_policy_mismatch_for_experiment_id(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    # E2 must be "largest", not "highest_confidence".
    config = make_config("E2", preprocessing={"mode": "bbox", "bbox_policy": "highest_confidence", "padding_ratio": 0.0})
    with pytest.raises(ValueError, match="bbox_policy"):
        run_bbox_experiment(dataset, lambda q: ([], {}), tmp_path, config)


def test_run_bbox_experiment_rejects_nonzero_padding(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    config = make_config(preprocessing={"mode": "bbox", "bbox_policy": "highest_confidence", "padding_ratio": 0.1})
    with pytest.raises(ValueError, match="padding_ratio"):
        run_bbox_experiment(dataset, lambda q: ([], {}), tmp_path, config)


def test_run_bbox_experiment_rejects_category_filtering(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    config = make_config(category_filter={"policy": "hard"})
    with pytest.raises(ValueError, match="category_filter"):
        run_bbox_experiment(dataset, lambda q: ([], {}), tmp_path, config)


def test_run_bbox_experiment_rejects_dedupe_false(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    config = make_config(retrieval={"top_k": 10, "dedupe": False})
    with pytest.raises(ValueError, match="dedupe"):
        run_bbox_experiment(dataset, lambda q: ([], {}), tmp_path, config)


# --- run_bbox_experiment: happy path, preprocessing metadata preserved --

def test_run_bbox_experiment_preserves_real_preprocessing_metadata(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))
    config = make_config("E1")
    real_metadata = {
        "mode": "bbox",
        "selected_bbox": [1, 2, 3, 4],
        "selected_label": "outer",
        "detection_score": 0.87,
        "bbox_area": 4,
        "padding_ratio": 0.0,
        "fallback_used": False,
        "fallback_reason": None,
    }

    def pipeline(query):
        return [result("P1", 0.1)], real_metadata

    records, summary = run_bbox_experiment(dataset, pipeline, tmp_path, config)

    assert records[0]["preprocessing"] == real_metadata
    assert records[0]["experiment_id"] == "E1"
    assert summary["experiment_id"] == "E1"
    saved = json.loads((tmp_path / "query_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert saved["preprocessing"]["selected_label"] == "outer"


# --- Regression: category-keyed bbox selection must actually filter -----

def test_category_confidence_policy_filters_using_dataset_collection_names():
    # Regression for a real cross-milestone bug found while wiring M5:
    # the evaluation dataset (M3) labels queries with a ChromaDB collection
    # name ("outer"), but retrieval.category's CATEGORY_LABEL_MAPPING was
    # keyed only by the Musinsa category string ("아우터"). Before the fix,
    # get_allowed_labels("outer") returned [] and category_confidence/
    # category_largest silently matched nothing for every dataset query.
    assert get_allowed_labels("outer") == ["top", "outer"]

    detections = [
        {"bbox": [0, 0, 10, 10], "label": "bottom", "score": 0.99, "area": 100},
        {"bbox": [0, 0, 5, 5], "label": "outer", "score": 0.5, "area": 25},
    ]
    selected = select_bbox(detections, "category_confidence", category="outer")
    assert selected is not None
    assert selected["label"] == "outer"
