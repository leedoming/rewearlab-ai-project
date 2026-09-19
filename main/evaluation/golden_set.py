"""Golden evaluation set coverage validation.

IMPLEMENTATION_SPEC.md section 55 (Golden Evaluation Set): a small
(10-15 query) regression set that MUST cover six required case types.

Four of these map directly to fields `evaluation.dataset.loader.QueryRecord`
already has (`difficulty`, `scene_type`); the other two ("category
boundary", "known previous failure") don't correspond to any existing
field -- the manifest schema has no "tags" concept, and guessing at a
heuristic (e.g. "any query whose category participates in
SOFT_CATEGORY_FILTER_MAPPING is a category-boundary case") would silently
mislabel queries the author never intended as boundary cases. The caller
supplies these two explicitly as sets of query_ids instead.

Kept dependency-free like `evaluation.metrics`/`evaluation.failure_analysis`.
"""

# IMPLEMENTATION_SPEC.md section 55's SHOULD.
MIN_GOLDEN_SET_SIZE = 10
MAX_GOLDEN_SET_SIZE = 15

REQUIRED_CASE_TYPES = (
    "easy_case",
    "hard_case",
    "multi_item",
    "background_heavy",
    "category_boundary",
    "known_previous_failure",
)


class GoldenSetValidationError(ValueError):
    """Raised when a candidate golden set violates section 55's requirements."""


def classify_golden_set_coverage(queries, category_boundary_ids=frozenset(), known_previous_failure_ids=frozenset()):
    """Return `{case_type: [query_id, ...]}` for every `REQUIRED_CASE_TYPES`.

    A query can satisfy more than one case type at once (e.g. a hard,
    multi-item query) -- this reports every match, it does not force a
    single label per query.
    """
    coverage = {case_type: [] for case_type in REQUIRED_CASE_TYPES}
    for query in queries:
        if query.difficulty == "easy":
            coverage["easy_case"].append(query.query_id)
        if query.difficulty == "hard":
            coverage["hard_case"].append(query.query_id)
        if query.scene_type == "multi_item":
            coverage["multi_item"].append(query.query_id)
        if query.scene_type == "complex_background":
            coverage["background_heavy"].append(query.query_id)
        if query.query_id in category_boundary_ids:
            coverage["category_boundary"].append(query.query_id)
        if query.query_id in known_previous_failure_ids:
            coverage["known_previous_failure"].append(query.query_id)
    return coverage


def validate_golden_set(queries, category_boundary_ids=frozenset(), known_previous_failure_ids=frozenset()):
    """IMPLEMENTATION_SPEC.md section 55: 10-15 queries, covering every
    required case type at least once.

    Raises `GoldenSetValidationError` naming exactly which requirement
    failed (size, or which case type(s) are missing), rather than a
    generic assertion, so a golden-set author gets an actionable message.
    Returns the coverage mapping on success.
    """
    count = len(queries)
    if not (MIN_GOLDEN_SET_SIZE <= count <= MAX_GOLDEN_SET_SIZE):
        raise GoldenSetValidationError(
            f"golden set must have {MIN_GOLDEN_SET_SIZE}-{MAX_GOLDEN_SET_SIZE} queries, got {count}"
        )
    coverage = classify_golden_set_coverage(queries, category_boundary_ids, known_previous_failure_ids)
    missing = sorted(case_type for case_type, ids in coverage.items() if not ids)
    if missing:
        raise GoldenSetValidationError(f"golden set is missing required case types: {missing}")
    return coverage
