"""Integration tests for failure_cases.jsonl / summary.json's failure_distribution.

IMPLEMENTATION_SPEC.md section 36 lists failure_cases.jsonl as one of every
experiment's four required outputs -- this was missing from every runner
until Milestone 8 (see docs/evidence/milestone-8.md section 2.1). These
tests cover the shared `_write_experiment_outputs` wiring through two of the
runners that use it, rather than duplicating the check for every runner
function (E0-E8 all share the exact same write path).
"""

import json
from types import MappingProxyType

import pytest

from evaluation.dataset import EvaluationDataset, QueryRecord
from evaluation.evaluator import run_baseline, run_category_filter_experiment


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
        labels=MappingProxyType(labels or {"P1": 2}),
    )


def result(product_id, distance, collection="outer"):
    return {"product_id": product_id, "rank": 1, "raw_distance": distance, "collection": collection, "metadata": {}}


def test_run_baseline_writes_failure_case_for_a_genuinely_poor_query(tmp_path):
    # No highly relevant item is ever returned -- P1 (grade 2) never appears
    # in the results, so this must be classified as a poor query.
    dataset = EvaluationDataset("eval-v1", True, (make_query(labels={"P1": 2}),))
    config = {
        "experiment_id": "E0",
        "split": "dev",
        "preprocessing": {"mode": "raw"},
        "retrieval": {"top_k": 10, "dedupe": True},
    }

    records, summary = run_baseline(dataset, lambda query: [result("Xnotrelevant", 0.1)], tmp_path, config)

    cases = [json.loads(line) for line in (tmp_path / "failure_cases.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(cases) == 1
    assert cases[0]["query_id"] == "Q001"
    assert cases[0]["experiment_id"] == "E0"
    # RAW mode can never be a detection/bbox-stage failure; no filter or DB
    # coverage issue was flagged either, so this must fall through to the
    # embedding-similarity fallback.
    assert cases[0]["primary_failure"] == "embedding_similarity_failure"

    assert summary["failure_distribution"]["embedding_similarity_failure"]["count"] == 1
    assert summary["failure_distribution"]["detection_failure"]["count"] == 0

    on_disk_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert on_disk_summary["failure_distribution"]["embedding_similarity_failure"]["count"] == 1


def test_run_baseline_writes_no_failure_cases_for_a_good_query(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(labels={"P1": 2}),))
    config = {
        "experiment_id": "E0",
        "split": "dev",
        "preprocessing": {"mode": "raw"},
        "retrieval": {"top_k": 10, "dedupe": True},
    }

    records, summary = run_baseline(dataset, lambda query: [result("P1", 0.1)], tmp_path, config)

    cases = (tmp_path / "failure_cases.jsonl").read_text(encoding="utf-8").splitlines()
    assert cases == []
    assert summary["failure_distribution"]["embedding_similarity_failure"]["count"] == 0


def test_run_category_filter_experiment_classifies_real_filter_exclusion(tmp_path):
    # P1 is the only relevant item and it's never retrieved (simulating the
    # filter having excluded it) -- exclusion count of 1 out of 1 relevant
    # item makes relevant_exclusion_rate > 0, which must classify as
    # filter_exclusion_failure, upstream of any embedding-quality verdict.
    dataset = EvaluationDataset("eval-v1", True, (make_query(labels={"P1": 2}),))
    config = {
        "experiment_id": "E5",
        "split": "dev",
        "preprocessing": {"mode": "bbox", "bbox_policy": "highest_confidence", "padding_ratio": 0.0},
        "category_filter": {"policy": "hard"},
        "retrieval": {"top_k": 10, "dedupe": True},
    }

    def pipeline(query):
        return [result("Xnotrelevant", 0.1)], {"mode": "bbox", "fallback_used": False}, 1

    records, summary = run_category_filter_experiment(dataset, pipeline, tmp_path, config)

    cases = [json.loads(line) for line in (tmp_path / "failure_cases.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(cases) == 1
    assert cases[0]["primary_failure"] == "filter_exclusion_failure"
    assert summary["failure_distribution"]["filter_exclusion_failure"]["count"] == 1
