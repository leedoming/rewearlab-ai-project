"""Failure taxonomy, poor-query extraction, and per-experiment failure summaries.

IMPLEMENTATION_SPEC.md section 28 (Phase 7 - Failure Taxonomy), section 29
(Failure Record), section 30 (Error Analysis Order), section 31 (Poor Query
Extraction), and section 32 (Failure Summary).

Kept dependency-free like `evaluation.metrics`, so it can run without torch/
chromadb/transformers being installed.
"""

# IMPLEMENTATION_SPEC.md section 28's ten primary failure categories, snake_cased
# to match section 29's example failure record ("primary_failure": "bbox_selection").
FAILURE_CATEGORIES = (
    "detection_failure",
    "bbox_selection_failure",
    "feature_loss",
    "background_bias",
    "category_prediction_failure",
    "semantic_category_boundary",
    "filter_exclusion_failure",
    "embedding_similarity_failure",
    "ranking_failure",
    "database_coverage_failure",
)

HIGHLY_RELEVANT_GRADE = 2

# IMPLEMENTATION_SPEC.md section 31's example thresholds.
DEFAULT_POOR_QUERY_THRESHOLDS = {"ndcg_at_10": 0.4, "mrr": 0.25}


def has_highly_relevant_in_top_k(results, labels, k=10):
    """IMPLEMENTATION_SPEC.md section 31's third poor-query criterion:
    "No highly relevant item in Top-10" (relevance grade 2, section 12)."""
    top_k = results[:k]
    return any(
        labels.get(str(result.get("product_id")), 0) >= HIGHLY_RELEVANT_GRADE
        for result in top_k
    )


def is_poor_query(metrics, results, labels, thresholds=DEFAULT_POOR_QUERY_THRESHOLDS, k=10):
    """Section 31's three-way OR: low NDCG@10, low MRR, or no highly relevant
    item in the top K. `results`/`labels` are needed only for the third
    check -- `metrics` alone (as already computed by `evaluate_query`)
    covers the first two."""
    if metrics["ndcg_at_10"] < thresholds["ndcg_at_10"]:
        return True
    if metrics["mrr"] < thresholds["mrr"]:
        return True
    return not has_highly_relevant_in_top_k(results, labels, k=k)


def classify_failure(preprocessing, metrics, database_coverage_failure=False):
    """Classify a poor query's primary failure, per IMPLEMENTATION_SPEC.md
    section 30's MUST: investigate upstream first, and do not default to
    "embedding failure" without ruling out earlier stages.

    This only auto-classifies the failure categories the codebase has a
    real, structural signal for:

    - `detection_failure` / `bbox_selection_failure`: from
      `retrieval.preprocessing.preprocess_image`'s own `fallback_used` /
      `fallback_reason` metadata (never fabricated -- see
      `preprocessing.py`'s docstrings).
    - `filter_exclusion_failure`: from `evaluate_query`'s real
      `relevant_exclusion_rate` (Milestone 6), not a placeholder.
    - `database_coverage_failure`: from a caller-supplied boolean, since
      determining it requires a live ChromaDB lookup
      (`evaluation.label_lookup.resolve_label_collections`) this pure
      function has no client to perform itself; defaults to `False` (not
      auto-detected) rather than silently assuming it never happens -- see
      the milestone-8 evidence doc's "Known Limitations".

    Everything else in the taxonomy (`feature_loss`, `background_bias`,
    `category_prediction_failure`, `semantic_category_boundary`,
    `ranking_failure`) requires human judgment (or, for
    `category_prediction_failure`, a predicted-category component that
    doesn't exist yet -- see milestone-6.md section 3) this function cannot
    fabricate. A human reviewer assigns those via the failure record's own
    `secondary_failures`/`notes` fields (section 29), not this function.

    Falls back to `"embedding_similarity_failure"` only when nothing
    upstream explains the poor query -- database coverage is checked before
    that fallback specifically because section 30 warns against blaming
    embedding quality for an item that was never indexed in the first
    place, even though section 30's pipeline diagram lists "DB Coverage"
    after "Embedding/Retrieval": that diagram is the request-time data
    flow, not the order to trust a verdict in, and ruling out "was this
    item even in the database" has to come before accepting an embedding
    failure verdict, not after.
    """
    if preprocessing.get("mode") == "bbox" and preprocessing.get("fallback_used"):
        if preprocessing.get("fallback_reason") == "no_detections":
            return "detection_failure"
        return "bbox_selection_failure"
    if (metrics.get("relevant_exclusion_rate") or 0.0) > 0.0:
        return "filter_exclusion_failure"
    if database_coverage_failure:
        return "database_coverage_failure"
    return "embedding_similarity_failure"


def summarize_failures(primary_failures):
    """IMPLEMENTATION_SPEC.md section 32's MUST: a per-experiment failure
    distribution (count and ratio), covering every known category -- not
    just the ones that happened to occur -- so a reader can see which
    failure modes had zero occurrences, not just the ones that didn't.

    `primary_failures` is an iterable of `primary_failure` category strings
    (one per poor query already classified by `classify_failure`).
    """
    failures = list(primary_failures)
    total = len(failures)
    counts = dict.fromkeys(FAILURE_CATEGORIES, 0)
    for failure in failures:
        if failure not in FAILURE_CATEGORIES:
            raise ValueError(f"Unknown failure category: {failure!r}")
        counts[failure] += 1
    return {
        category: {"count": count, "ratio": (count / total) if total else 0.0}
        for category, count in counts.items()
    }
