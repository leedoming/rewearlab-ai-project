"""E0 RAW baseline evaluation and result serialization."""

import csv
import json
from pathlib import Path

from retrieval.config import BBOX_SELECTION_POLICIES, CATEGORY_FILTER_POLICIES

from .aggregate import aggregate_metrics
from .failure_analysis import classify_failure, is_poor_query, summarize_failures
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

# IMPLEMENTATION_SPEC.md section 33: E7 varies only padding_ratio (0% -> 10%),
# holding bbox policy and category filter policy at whatever Milestone 8/10
# eventually decides is "Best" for each -- neither has been decided yet (see
# run_padding_experiment below).
EXPERIMENT_PADDING_RATIOS = {
    "E7": 0.1,
}

# IMPLEMENTATION_SPEC.md section 33: unlike E7, E8's filter column is
# concretely "Soft" (not "Best"), paired with RAW preprocessing -- isolating
# "does category filtering alone help, with no bbox/padding at all" from
# E5-E7's bbox+filter combinations.
EXPERIMENT_RAW_FILTER_POLICIES = {
    "E8": "soft",
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

    # IMPLEMENTATION_SPEC.md section 36 lists failure_cases.jsonl as one of
    # every experiment's four required outputs, alongside section 32's MUST
    # for a per-experiment failure distribution -- neither existed before
    # Milestone 8 (see milestone-8.md section 2.1). Both are produced here,
    # in the one function every experiment runner (E0-E8) already shares,
    # rather than duplicated per runner.
    failure_cases = []
    for record in records:
        labels = dataset.by_id(record["query_id"]).labels
        if not is_poor_query(record["metrics"], record["results"], labels):
            continue
        failure_cases.append(
            {
                "query_id": record["query_id"],
                "experiment_id": record["experiment_id"],
                "primary_failure": classify_failure(record["preprocessing"], record["metrics"]),
                "secondary_failures": [],
                "notes": "",
                "evidence": {},
            }
        )

    with (output_dir / "failure_cases.jsonl").open("w", encoding="utf-8") as file:
        for case in failure_cases:
            file.write(json.dumps(case, ensure_ascii=False) + "\n")

    summary = {
        "experiment_id": experiment_id,
        "dataset_version": dataset.version,
        "config": config,
        **aggregate_metrics(records),
        "failure_distribution": summarize_failures(case["primary_failure"] for case in failure_cases),
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


def run_padding_experiment(dataset, pipeline, output_dir, config):
    """Run E7 (padding ablation) over one dataset split.

    Per IMPLEMENTATION_SPEC.md section 33, E7 holds bbox policy and category
    filter policy at "Best" -- neither has been decided (no ablation
    evidence exists yet for either; see `run_category_filter_experiment`'s
    own docstring and the Milestone 5/6 evidence docs), so this function
    accepts any of the four valid bbox policies and any of the three valid
    category filter policies, exactly like `run_category_filter_experiment`
    does for bbox policy. The one value the spec's ablation table does
    concretely pin for E7 is padding_ratio: it must be 0.1 (10%), not 0.0 --
    that's the entire point of this being a *padding* experiment.

    `pipeline(query)` must return `(results, preprocessing_metadata,
    excluded_relevant_count)`, the same shape `run_category_filter_experiment`
    requires, since E7 also filters collections by category before search.
    """
    experiment_id = config.get("experiment_id")
    expected_padding = EXPERIMENT_PADDING_RATIOS.get(experiment_id)
    if expected_padding is None:
        raise ValueError(
            f"padding experiment runner only accepts experiment_id in {sorted(EXPERIMENT_PADDING_RATIOS)}"
        )
    if config.get("preprocessing", {}).get("mode") != "bbox":
        raise ValueError(f"{experiment_id} preprocessing.mode must be 'bbox'")
    bbox_policy = config.get("preprocessing", {}).get("bbox_policy")
    if bbox_policy not in BBOX_SELECTION_POLICIES:
        raise ValueError(
            f"{experiment_id} preprocessing.bbox_policy must be one of "
            f"{sorted(BBOX_SELECTION_POLICIES)}, got {bbox_policy!r}"
        )
    padding_ratio = config.get("preprocessing", {}).get("padding_ratio")
    if padding_ratio != expected_padding:
        raise ValueError(
            f"{experiment_id} preprocessing.padding_ratio must be {expected_padding!r}, got {padding_ratio!r}"
        )
    category_filter_policy = config.get("category_filter", {}).get("policy")
    if category_filter_policy not in CATEGORY_FILTER_POLICIES:
        raise ValueError(
            f"{experiment_id} category_filter.policy must be one of "
            f"{sorted(CATEGORY_FILTER_POLICIES)}, got {category_filter_policy!r}"
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


def run_raw_filter_experiment(dataset, pipeline, output_dir, config):
    """Run E8 (RAW baseline + soft category filter) over one dataset split.

    Unlike E7, E8's category_filter.policy is concretely pinned to "soft"
    (IMPLEMENTATION_SPEC.md section 33 lists E8's filter column as "Soft",
    not "Best") while preprocessing.mode is "raw" -- no detection, no bbox,
    no padding. This isolates "does category filtering alone help, with no
    bbox preprocessing at all" from E5-E7's bbox+filter combinations.

    `pipeline(query)` must return `(results, excluded_relevant_count)` --
    there is no bbox preprocessing metadata to thread through from the
    caller; this function constructs the fixed "raw" metadata itself
    (mirroring `run_baseline`'s E0 metadata), so a caller cannot
    accidentally fabricate bbox fields for an experiment that never ran
    detection.
    """
    experiment_id = config.get("experiment_id")
    expected_filter_policy = EXPERIMENT_RAW_FILTER_POLICIES.get(experiment_id)
    if expected_filter_policy is None:
        raise ValueError(
            f"raw filter experiment runner only accepts experiment_id in {sorted(EXPERIMENT_RAW_FILTER_POLICIES)}"
        )
    if config.get("preprocessing", {}).get("mode") != "raw":
        raise ValueError(f"{experiment_id} preprocessing.mode must be 'raw'")
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
        results, excluded_relevant_count = pipeline(query)
        metrics = evaluate_query(query, results, excluded_relevant_count=excluded_relevant_count)
        records.append(
            {
                "query_id": query.query_id,
                "category": query.category,
                "difficulty": query.difficulty,
                "scene_type": query.scene_type,
                "experiment_id": experiment_id,
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
    if bbox_policy not in BBOX_SELECTION_POLICIES:
        raise ValueError(
            f"{experiment_id} preprocessing.bbox_policy must be one of "
            f"{sorted(BBOX_SELECTION_POLICIES)}, got {bbox_policy!r}"
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
