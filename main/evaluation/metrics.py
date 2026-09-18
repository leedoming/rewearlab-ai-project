"""Pure retrieval evaluation metrics.

No dependency on torch, chromadb, or streamlit (IMPLEMENTATION_SPEC.md
section 26: "metrics.py는 다음 dependency를 가져서는 안 된다: torch,
chromadb, streamlit"). This module operates only on already-computed
relevance / compatibility signals (plain ints/bools in rank order), so it
can be unit tested without loading any model or database, and reused by
any future retrieval/ablation experiment regardless of how the ranking
was produced.

Relevance encoding (IMPLEMENTATION_SPEC.md section 12):
    0 = Not Relevant
    1 = Relevant
    2 = Highly Relevant
Binary relevance (used by Precision/Recall/MRR) is `relevance >= 1`.
Graded relevance (0/1/2 as-is) is used by NDCG.

Metric edge cases (IMPLEMENTATION_SPEC.md section 25):
    No Relevant Item      -> Recall is None (caller excludes it from
                              aggregation rather than treating it as 0).
    No Relevant Retrieval -> MRR is 0.
    Fewer than K Results  -> Precision's denominator stays K (a missing
                              slot counts as not relevant, it is not
                              dropped from the average).
"""

import math


def _is_relevant(relevance):
    return relevance >= 1


def precision_at_k(relevances, k):
    """Precision@K: fraction of the K result slots holding a relevant item.

    `relevances` is the ranked list of (binary-thresholded) relevance
    grades for the retrieved results, rank 1 first. The denominator is
    always `k`, even when fewer than `k` items were actually retrieved
    (IMPLEMENTATION_SPEC.md section 25: "Fewer than K Results" ->
    Precision denominator stays K) -- missing slots simply contribute no
    relevant hits, they do not shrink the denominator.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = relevances[:k]
    relevant_count = sum(1 for r in top_k if _is_relevant(r))
    return relevant_count / k


def recall_at_k(relevances, k, total_relevant):
    """Recall@K: fraction of all relevant items that appear in the top K.

    `total_relevant` is the total number of relevant items that exist for
    this query (per the ground truth), independent of how many were
    retrieved. Returns None when `total_relevant` is 0
    (IMPLEMENTATION_SPEC.md section 25: "No Relevant Item" -> None/NaN,
    excluded from aggregation by the caller) since recall is undefined
    when there is nothing to find.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if total_relevant == 0:
        return None
    top_k = relevances[:k]
    relevant_count = sum(1 for r in top_k if _is_relevant(r))
    return relevant_count / total_relevant


def mrr(relevances):
    """Reciprocal rank of the first (binary) relevant item in `relevances`.

    Returns 0.0 if no relevant item appears anywhere in `relevances`
    (IMPLEMENTATION_SPEC.md section 25: "No Relevant Retrieval" -> MRR = 0).
    This is a single-query score; averaging it over queries to get a mean
    reciprocal rank is the caller's job.
    """
    for rank, relevance in enumerate(relevances, start=1):
        if _is_relevant(relevance):
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevances, k):
    """Discounted Cumulative Gain@K using graded relevance (0/1/2) as-is."""
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = relevances[:k]
    return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(top_k, start=1))


def ndcg_at_k(relevances, k, ideal_relevances=None):
    """Normalized DCG@K using graded relevance (0/1/2) as-is
    (IMPLEMENTATION_SPEC.md section 12: "NDCG에서는 0/1/2를 그대로 사용한다").

    `ideal_relevances` should be the best-possible ordering of relevance
    grades for this query -- e.g. every known relevant/highly-relevant
    item for the query (from the pooled ground truth), sorted descending.
    That full ground-truth pool does not exist yet (it is Milestone 3's
    dataset work), so when `ideal_relevances` is omitted this falls back
    to sorting `relevances` itself in descending order, i.e. treating the
    retrieved set as the full universe of known-relevant items for this
    query. That fallback keeps this metric usable now, and is a strict
    no-op once a real `ideal_relevances` (a superset of what was
    retrieved) is supplied later -- it does not need to change shape.

    Returns 0.0 (not None) when the ideal DCG is 0, i.e. no relevant item
    exists at all for this query: there is nothing to rank, so there is
    no ranking error to report either.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    ideal = sorted(ideal_relevances, reverse=True) if ideal_relevances is not None else sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(relevances, k) / ideal_dcg


def incompatible_category_rate_at_k(is_incompatible_flags, k):
    """Fraction of the top-K retrieved items whose category is incompatible
    with the query's intended category.

    `is_incompatible_flags` is a list of booleans in rank order (True =
    this item's category is incompatible), computed by the caller (e.g.
    via `retrieval.category` compatibility rules) -- this module has no
    domain knowledge of what "compatible" means. The denominator is fixed
    at `k`, consistent with `precision_at_k`'s "fewer than K" handling.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = is_incompatible_flags[:k]
    incompatible_count = sum(1 for flag in top_k if flag)
    return incompatible_count / k


def relevant_exclusion_rate(total_relevant, excluded_relevant_count):
    """Fraction of truly relevant items that a filtering step (e.g. a hard
    category filter) removed before they had any chance to be ranked or
    retrieved.

    `total_relevant`: total relevant items for this query, per the ground
    truth (independent of any filtering). `excluded_relevant_count`: how
    many of those were removed by the filter under evaluation.

    Returns None when `total_relevant` is 0, matching `recall_at_k`'s
    "nothing to reason about" convention for that same case.
    """
    if total_relevant == 0:
        return None
    if excluded_relevant_count > total_relevant:
        raise ValueError("excluded_relevant_count cannot exceed total_relevant")
    return excluded_relevant_count / total_relevant
