"""Quality regression checks and critical-query assertions.

IMPLEMENTATION_SPEC.md section 54 (Regression Tests - the "Quality" half;
the "Functional" half is exercised directly against `retrieval.*` in
`tests/regression/test_golden_set.py`, not here), section 56 (Quality
Regression Threshold), and section 57 (Critical Query Tests).

Kept dependency-free like `evaluation.metrics`.
"""

# IMPLEMENTATION_SPEC.md section 56's example tolerance
# ("new_ndcg >= baseline_ndcg * 0.97"). The spec itself says the real
# tolerance is decided after a real baseline exists (section 56: "실제
# tolerance는 baseline 결과 후 결정한다") -- this default is a starting
# point, not a finalized decision.
DEFAULT_QUALITY_REGRESSION_TOLERANCE = 0.97


def passes_quality_regression(new_value, baseline_value, tolerance=DEFAULT_QUALITY_REGRESSION_TOLERANCE):
    """Section 56's formula: `new_value >= baseline_value * tolerance`.

    Returns `None` (not `True`/`False`) when either value is `None` --
    e.g. a metric like `recall_at_k`/`relevant_exclusion_rate` that is
    itself `None` when there was nothing relevant to measure
    (`evaluation.metrics`' own "no relevant item" convention). `None`
    means "this comparison doesn't apply," which a caller must not treat
    as either a pass or a regression.
    """
    if new_value is None or baseline_value is None:
        return None
    return new_value >= baseline_value * tolerance


def check_quality_regression(new_metrics, baseline_metrics, metric_names, tolerance=DEFAULT_QUALITY_REGRESSION_TOLERANCE):
    """Apply `passes_quality_regression` across several named metrics at once
    (e.g. `["ndcg_at_10", "mrr"]`, per section 56's own example), returning
    `{metric_name: True | False | None}` rather than a single boolean, so a
    caller can see exactly which metric(s) regressed instead of only
    "something did."
    """
    return {
        name: passes_quality_regression(new_metrics.get(name), baseline_metrics.get(name), tolerance)
        for name in metric_names
    }


def first_relevant_rank(results, labels, min_relevance=1):
    """The 1-based rank of the first result meeting `min_relevance`, or
    `None` if none does -- IMPLEMENTATION_SPEC.md section 57's example
    ("Q017 first relevant rank <= 5") needs exactly this rank, not just
    whether one exists (that's already `mrr`'s job).
    """
    for rank, result in enumerate(results, start=1):
        if labels.get(str(result.get("product_id")), 0) >= min_relevance:
            return rank
    return None


def passes_critical_query_rank(rank, max_rank):
    """Section 57's assertion shape: `rank <= max_rank`. `rank=None` (no
    relevant item retrieved at all) never passes, regardless of
    `max_rank` -- there is no rank to compare.
    """
    if rank is None:
        return False
    return rank <= max_rank
