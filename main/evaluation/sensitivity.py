"""Detection-threshold / padding / Top-K sensitivity metrics and comparison.

IMPLEMENTATION_SPEC.md section 41 (Detection Threshold), section 42
(Padding Sensitivity), section 43 (Top-K Sensitivity).

Kept dependency-free like `evaluation.metrics`/`evaluation.failure_analysis`
(only imports `retrieval.category`, itself dependency-free), so it can run
without torch/chromadb/transformers being installed.

Scope note: this module computes the sensitivity metrics that are
*objectively* derivable from already-real preprocessing signals (detection
counts, fallback usage, category-compatible labels) or that are already
implemented elsewhere and simply need comparing across parameter values
(NDCG, MRR -- see `evaluation.metrics`). Section 42 also lists "Feature
Loss" and "Background Bias" for padding sensitivity; the spec gives no
computable formula for either, and unlike, say, `relevant_exclusion_rate`
(Milestone 6) there is no existing structural signal to derive them from --
they read as qualitative/human-judgment metrics, the same category of thing
`evaluation.failure_analysis.classify_failure` already declines to
fabricate for `feature_loss`/`background_bias`. This module does not invent
a formula for them; see docs/evidence/milestone-9.md.
"""

from retrieval.category import is_label_allowed_for_category

from .aggregate import summarize_values


def average_detection_count(detection_counts):
    """Section 41: 'Average Detection Count' at a fixed detection threshold.

    `detection_counts` is the raw number of detections found per query
    (before bbox selection) -- one int per query, all at the same
    threshold.
    """
    counts = list(detection_counts)
    if not counts:
        raise ValueError("detection_counts must not be empty")
    return sum(counts) / len(counts)


def detection_success_rate(detection_counts):
    """Section 41: fraction of queries with at least one detection."""
    counts = list(detection_counts)
    if not counts:
        raise ValueError("detection_counts must not be empty")
    return sum(1 for count in counts if count > 0) / len(counts)


def fallback_rate(fallback_used_flags):
    """Section 41: fraction of queries where bbox selection fell back.

    `fallback_used_flags` should come straight from
    `retrieval.preprocessing.preprocess_image`'s own `fallback_used` field
    (never fabricated -- see that module's docstrings), not re-derived.
    """
    flags = list(fallback_used_flags)
    if not flags:
        raise ValueError("fallback_used_flags must not be empty")
    return sum(1 for used in flags if used) / len(flags)


def wrong_object_rate(selections):
    """Section 41: fraction of *non-fallback* bbox selections whose label
    isn't compatible with the query's category -- i.e. detection succeeded
    and a bbox was confidently chosen, but it's the wrong kind of clothing
    item (can happen under `highest_confidence`/`largest`, which don't
    filter by category at all).

    Fallback selections are excluded deliberately: a fallback is already
    counted by `fallback_rate`, and "wrong object" is about a policy
    confidently choosing the wrong thing, not about there being nothing
    category-compatible to choose from in the first place.

    `selections` is an iterable of `(category, selected_label,
    fallback_used)` tuples, one per query.
    """
    non_fallback = [
        (category, selected_label)
        for category, selected_label, fallback_used in selections
        if not fallback_used
    ]
    if not non_fallback:
        return 0.0
    wrong = sum(
        1
        for category, selected_label in non_fallback
        if not is_label_allowed_for_category(selected_label, category)
    )
    return wrong / len(non_fallback)


def summarize_sensitivity_sweep(rows, parameter_field):
    """Sections 41-43's shared "compare across parameter values" need,
    generalized once instead of duplicated for detection threshold, padding,
    and Top-K separately.

    `rows` is an iterable of dicts, each shaped like
    `{parameter_field: <swept value>, "metrics": {...}}` -- one row per
    (parameter value, query) pair, e.g. `{"detection_threshold": 0.3,
    "metrics": {"ndcg_at_10": 0.6, "mrr": 0.5}}`. Returns, for every swept
    value, the same mean/median/stddev summary
    (`evaluation.aggregate.summarize_values`) every other summary in this
    project already uses, for every metric name present in any row.
    """
    rows = list(rows)
    values = sorted({row[parameter_field] for row in rows})
    metric_names = sorted({name for row in rows for name in row["metrics"]})
    return {
        value: {
            name: summarize_values(
                [row["metrics"].get(name) for row in rows if row[parameter_field] == value]
            )
            for name in metric_names
        }
        for value in values
    }
