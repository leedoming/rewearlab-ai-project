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

# IMPLEMENTATION_SPEC.md section 33's ablation matrix: E1-E4 vary only the
# bbox selection policy, holding everything else (dataset, detection model,
# threshold, category filter=none, padding=0%, top_k, dedupe) fixed.
EXPERIMENT_BBOX_POLICIES = {
    "E1": "highest_confidence",
    "E2": "largest",
    "E3": "category_confidence",
    "E4": "category_largest",
}

# IMPLEMENTATION_SPEC.md section 33: E5/E6 vary only the category filter
# policy (bbox policy held at whatever Milestone 8/10 eventually decides is
# "Best" -- not yet determined, see run_category_filter_experiment).
EXPERIMENT_CATEGORY_FILTER_POLICIES = {
    "E5": "hard",
    "E6": "soft",
}


def evaluate_query(query, results, excluded_relevant_count=0):
    """`excluded_relevant_count` is 0 by default (correct for E0-E4, which
    have no category filter to exclude anything) and should be the real
    count from `evaluation.label_lookup.count_relevant_excluded_by_filter`
    for category-filtering experiments (E5/E6)."""
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
        "relevant_exclusion_rate": relevant_exclusion_rate(total_relevant, excluded_relevant_count),
    }


def _validate_common_retrieval_controls(config, top_k_floor=10):
    """Shared controls that must stay fixed across every ablation experiment
    (IMPLEMENTATION_SPEC.md section 34: only one major variable changes per
    experiment)."""
    retrieval_config = config.get("retrieval", {})
    top_k = retrieval_config.get("top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < top_k_floor:
        raise ValueError(f"retrieval.top_k must be an integer of at least {top_k_floor}")
    if retrieval_config.get("dedupe") is not True:
        raise ValueError("retrieval.dedupe must be true")


def _select_split(dataset, config):
    split = config.get("split", "dev")
    if split not in {"dev", "holdout", "all"}:
        raise ValueError("split must be dev, holdout, or all")
    queries = [query for query in dataset.queries if split == "all" or query.split == split]
    if not queries:
        raise ValueError(f"dataset contains no queries for split {split!r}")
    return queries


def _write_experiment_outputs(records, output_dir, dataset, config, experiment_id):
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
        "experiment_id": experiment_id,
        "dataset_version": dataset.version,
        "config": config,
        **aggregate_metrics(records),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run_baseline(dataset, retrieve, output_dir, config):
    """Run E0 over one dataset split using an injected retrieval callable."""
    if config.get("experiment_id") != "E0":
        raise ValueError("RAW baseline runner only accepts experiment_id='E0'")
    if config.get("preprocessing", {}).get("mode") != "raw":
        raise ValueError("E0 preprocessing.mode must be 'raw'")
    _validate_common_retrieval_controls(config)
    queries = _select_split(dataset, config)

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

    summary = _write_experiment_outputs(records, output_dir, dataset, config, "E0")
    return records, summary


def run_bbox_experiment(dataset, pipeline, output_dir, config):
    """Run one of E1-E4 (bbox policy ablation) over one dataset split.

    `pipeline(query)` must return `(results, preprocessing_metadata)`, where
    `preprocessing_metadata` is exactly what
    `retrieval.preprocessing.preprocess_image` returned for that query, so
    `fallback_used`/`fallback_reason` are recorded, never fabricated.

    This function validates that the config only changes the bbox policy
    relative to E0 (IMPLEMENTATION_SPEC.md section 34: one variable at a
    time) -- it does NOT itself guarantee every policy saw the same
    detection output (spec section 19's other MUST). That guarantee comes
    from the caller building `pipeline` on top of a shared
    `evaluation.detection_cache.DetectionCache` (see
    `evaluation/run_bbox_experiments.py`), which this function has no way
    to verify from the outside.
    """
    experiment_id = config.get("experiment_id")
    expected_policy = EXPERIMENT_BBOX_POLICIES.get(experiment_id)
    if expected_policy is None:
        raise ValueError(
            f"bbox experiment runner only accepts experiment_id in {sorted(EXPERIMENT_BBOX_POLICIES)}"
        )
    if config.get("preprocessing", {}).get("mode") != "bbox":
        raise ValueError(f"{experiment_id} preprocessing.mode must be 'bbox'")
    bbox_policy = config.get("preprocessing", {}).get("bbox_policy")
    if bbox_policy != expected_policy:
        raise ValueError(
            f"{experiment_id} preprocessing.bbox_policy must be {expected_policy!r}, got {bbox_policy!r}"
        )
    padding_ratio = config.get("preprocessing", {}).get("padding_ratio", 0.0)
    if padding_ratio != 0.0:
        raise ValueError(
            f"{experiment_id} preprocessing.padding_ratio must be 0.0 "
            "(padding sensitivity is Milestone 9 scope)"
        )
    category_filter_policy = config.get("category_filter", {}).get("policy", "none")
    if category_filter_policy != "none":
        raise ValueError(
            f"{experiment_id} category_filter.policy must be 'none' "
            "(category filtering is Milestone 6 scope)"
        )
    _validate_common_retrieval_controls(config)
    queries = _select_split(dataset, config)

    records = []
    for query in queries:
        results, preprocessing_metadata = pipeline(query)
        metrics = evaluate_query(query, results)
        records.append(
            {
                "query_id": query.query_id,
                "category": query.category,
                "difficulty": query.difficulty,
                "scene_type": query.scene_type,
                "experiment_id": experiment_id,
                "config": config,
                "preprocessing": preprocessing_metadata,
                "results": results,
                "metrics": metrics,
            }
        )

    summary = _write_experiment_outputs(records, output_dir, dataset, config, experiment_id)
    return records, summary


def run_category_filter_experiment(dataset, pipeline, output_dir, config):
    """Run one of E5-E6 (category filter ablation) over one dataset split.

    `pipeline(query)` must return `(results, preprocessing_metadata,
    excluded_relevant_count)`, where `excluded_relevant_count` is the number
    of the query's labeled relevant items whose known collection was
    excluded by the category filter before search ever ran (see
    `evaluation.label_lookup`) -- NOT a placeholder 0 like E0-E4, since a
    real filter can and should be able to exclude relevant items, and
    `evaluate_query` needs the true count to report `relevant_exclusion_rate`
    honestly.

    Per IMPLEMENTATION_SPEC.md section 33, E5/E6 hold the bbox policy fixed
    at "Best" -- which policy that is has NOT been decided (no ablation
    result exists yet to justify one; see Milestone 5's evidence doc and
    IMPLEMENTATION_SPEC.md section 67 rule 7: "실험 결과가 없는 상태에서
    '최적' policy를 확정하지 않는다"). This function therefore accepts any
    of the four valid bbox policies rather than a hardcoded expectation --
    whatever the config declares is recorded in the output, not silently
    assumed correct.
    """
    experiment_id = config.get("experiment_id")
    expected_filter_policy = EXPERIMENT_CATEGORY_FILTER_POLICIES.get(experiment_id)
    if expected_filter_policy is None:
        raise ValueError(
            f"category filter experiment runner only accepts experiment_id in {sorted(EXPERIMENT_CATEGORY_FILTER_POLICIES)}"
        )
    if config.get("preprocessing", {}).get("mode") != "bbox":
        raise ValueError(f"{experiment_id} preprocessing.mode must be 'bbox'")
    bbox_policy = config.get("preprocessing", {}).get("bbox_policy")
    if bbox_policy not in EXPERIMENT_BBOX_POLICIES.values():
        raise ValueError(
            f"{experiment_id} preprocessing.bbox_policy must be one of "
            f"{sorted(set(EXPERIMENT_BBOX_POLICIES.values()))}, got {bbox_policy!r}"
        )
    padding_ratio = config.get("preprocessing", {}).get("padding_ratio", 0.0)
    if padding_ratio != 0.0:
        raise ValueError(
            f"{experiment_id} preprocessing.padding_ratio must be 0.0 "
            "(padding sensitivity is Milestone 9 scope)"
        )
    category_filter_policy = config.get("category_filter", {}).get("policy")
    if category_filter_policy != expected_filter_policy:
        raise ValueError(
            f"{experiment_id} category_filter.policy must be {expected_filter_policy!r}, "
            f"got {category_filter_policy!r}"
        )
    _validate_common_retrieval_controls(config)
    queries = _select_split(dataset, config)

    records = []
    for query in queries:
        results, preprocessing_metadata, excluded_relevant_count = pipeline(query)
        metrics = evaluate_query(query, results, excluded_relevant_count=excluded_relevant_count)
        records.append(
            {
                "query_id": query.query_id,
                "category": query.category,
                "difficulty": query.difficulty,
                "scene_type": query.scene_type,
                "experiment_id": experiment_id,
                "config": config,
                "preprocessing": preprocessing_metadata,
                "results": results,
                "metrics": metrics,
            }
        )

    summary = _write_experiment_outputs(records, output_dir, dataset, config, experiment_id)
    return records, summary
