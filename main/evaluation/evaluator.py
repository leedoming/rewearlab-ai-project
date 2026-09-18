"""E0 RAW baseline evaluation and result serialization."""

import csv
import json
from pathlib import Path

from .aggregate import aggregate_metrics
from .metrics import (
    incompatible_category_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    relevant_exclusion_rate,
)


METRIC_FIELDS = (
    "precision_at_5",
    "precision_at_10",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "ndcg_at_10",
    "incompatible_category_rate_at_10",
    "relevant_exclusion_rate",
)


def evaluate_query(query, results):
    relevances = [query.labels.get(str(result.get("product_id")), 0) for result in results]
    total_relevant = sum(1 for grade in query.labels.values() if grade >= 1)
    incompatible = [result.get("collection") != query.category for result in results]
    return {
        "precision_at_5": precision_at_k(relevances, 5),
        "precision_at_10": precision_at_k(relevances, 10),
        "recall_at_5": recall_at_k(relevances, 5, total_relevant),
        "recall_at_10": recall_at_k(relevances, 10, total_relevant),
        "mrr": mrr(relevances),
        "ndcg_at_10": ndcg_at_k(
            relevances, 10, ideal_relevances=list(query.labels.values())
        ),
        "incompatible_category_rate_at_10": incompatible_category_rate_at_k(
            incompatible, 10
        ),
        "relevant_exclusion_rate": relevant_exclusion_rate(total_relevant, 0),
    }


def run_baseline(dataset, retrieve, output_dir, config):
    """Run E0 over one dataset split using an injected retrieval callable."""
    if config.get("experiment_id") != "E0":
        raise ValueError("RAW baseline runner only accepts experiment_id='E0'")
    if config.get("preprocessing", {}).get("mode") != "raw":
        raise ValueError("E0 preprocessing.mode must be 'raw'")
    retrieval_config = config.get("retrieval", {})
    top_k = retrieval_config.get("top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 10:
        raise ValueError("E0 retrieval.top_k must be an integer of at least 10")
    if retrieval_config.get("dedupe") is not True:
        raise ValueError("E0 retrieval.dedupe must be true")
    split = config.get("split", "dev")
    if split not in {"dev", "holdout", "all"}:
        raise ValueError("split must be dev, holdout, or all")
    queries = [query for query in dataset.queries if split == "all" or query.split == split]
    if not queries:
        raise ValueError(f"dataset contains no queries for split {split!r}")

    records = []
    for query in queries:
        results = retrieve(query)
        metrics = evaluate_query(query, results)
        records.append(
            {
                "query_id": query.query_id,
                "category": query.category,
                "difficulty": query.difficulty,
                "scene_type": query.scene_type,
                "experiment_id": "E0",
                "config": config,
                "preprocessing": {
                    "mode": "raw",
                    "selected_bbox": None,
                    "fallback_used": False,
                    "fallback_reason": None,
                },
                "results": results,
                "metrics": metrics,
            }
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "query_results.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    with (output_dir / "metrics_by_query.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        fieldnames = ["query_id", "category", "difficulty", "scene_type", *METRIC_FIELDS]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "query_id": record["query_id"],
                    "category": record["category"],
                    "difficulty": record["difficulty"],
                    "scene_type": record["scene_type"],
                    **record["metrics"],
                }
            )

    summary = {
        "experiment_id": "E0",
        "dataset_version": dataset.version,
        "config": config,
        **aggregate_metrics(records),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return records, summary
