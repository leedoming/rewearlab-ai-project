import csv
import json
from types import MappingProxyType

import pytest

from evaluation.aggregate import aggregate_metrics, summarize_values
from evaluation.dataset import EvaluationDataset, QueryRecord
from evaluation.evaluator import evaluate_query, run_baseline


def make_query(query_id="Q001", split="dev", labels=None):
    return QueryRecord(
        query_id=query_id,
        image_path=None,
        category="outer",
        difficulty="hard",
        scene_type="person_wearing",
        split=split,
        num_visible_items=1,
        background_complexity="high",
        important_features=(),
        labels=MappingProxyType(labels or {"P1": 2, "P2": 1}),
    )


def result(product_id, distance, collection="outer"):
    return {
        "product_id": product_id,
        "rank": 1,
        "raw_distance": distance,
        "collection": collection,
        "metadata": {"id": product_id},
    }


def test_evaluate_query_uses_pooled_labels_and_raw_ranking():
    metrics = evaluate_query(
        make_query(),
        [result("P2", 0.1), result("X", 0.2, "pants"), result("P1", 0.3)],
    )

    assert metrics["precision_at_5"] == pytest.approx(2 / 5)
    assert metrics["recall_at_10"] == 1.0
    assert metrics["mrr"] == 1.0
    assert 0 < metrics["ndcg_at_10"] < 1
    assert metrics["incompatible_category_rate_at_10"] == pytest.approx(1 / 10)
    assert metrics["relevant_exclusion_rate"] == 0.0


def test_aggregate_excludes_none_and_reports_slices():
    rows = [
        {"category": "outer", "difficulty": "hard", "scene_type": "person_wearing", "metrics": {"recall": None, "mrr": 0.0}},
        {"category": "outer", "difficulty": "easy", "scene_type": "clean_product", "metrics": {"recall": 1.0, "mrr": 1.0}},
    ]

    summary = aggregate_metrics(rows)

    assert summary["overall"]["recall"]["count"] == 1
    assert summary["overall"]["recall"]["mean"] == 1.0
    assert summary["overall"]["mrr"]["median"] == 0.5
    assert "outer" in summary["by_category"]
    assert summarize_values([])["mean"] is None


def test_run_baseline_serializes_dev_results(tmp_path):
    dataset = EvaluationDataset(
        version="eval-v1",
        pooled_ground_truth=True,
        queries=(make_query(), make_query("Q002", split="holdout")),
    )
    config = {
        "experiment_id": "E0",
        "split": "dev",
        "preprocessing": {"mode": "raw"},
        "retrieval": {"top_k": 10, "dedupe": True},
    }

    records, summary = run_baseline(
        dataset, lambda query: [result("P1", 0.1)], tmp_path, config
    )

    assert [record["query_id"] for record in records] == ["Q001"]
    assert records[0]["preprocessing"]["mode"] == "raw"
    assert records[0]["results"][0]["raw_distance"] == 0.1
    assert summary["query_count"] == 1

    lines = (tmp_path / "query_results.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["experiment_id"] == "E0"
    with (tmp_path / "metrics_by_query.csv").open(encoding="utf-8") as file:
        assert list(csv.DictReader(file))[0]["query_id"] == "Q001"
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["query_count"] == 1


def test_run_baseline_rejects_empty_selected_split(tmp_path):
    dataset = EvaluationDataset("eval-v1", True, (make_query(split="holdout"),))
    config = {
        "experiment_id": "E0",
        "split": "dev",
        "preprocessing": {"mode": "raw"},
        "retrieval": {"top_k": 10, "dedupe": True},
    }

    with pytest.raises(ValueError, match="no queries"):
        run_baseline(dataset, lambda query: [], tmp_path, config)


@pytest.mark.parametrize(
    "config,error",
    [
        ({"experiment_id": "E1"}, "experiment_id"),
        ({"experiment_id": "E0", "preprocessing": {"mode": "bbox"}}, "raw"),
        (
            {
                "experiment_id": "E0",
                "preprocessing": {"mode": "raw"},
                "retrieval": {"top_k": 5, "dedupe": True},
            },
            "top_k",
        ),
        (
            {
                "experiment_id": "E0",
                "preprocessing": {"mode": "raw"},
                "retrieval": {"top_k": 10, "dedupe": False},
            },
            "dedupe",
        ),
    ],
)
def test_run_baseline_rejects_config_that_changes_e0_controls(tmp_path, config, error):
    dataset = EvaluationDataset("eval-v1", True, (make_query(),))

    with pytest.raises(ValueError, match=error):
        run_baseline(dataset, lambda query: [], tmp_path, config)
