# ADR-002: BBox Selection Policy

## Status

**Decision: TBD** — pending real ablation results. See "Decision" below.

## Context

When more than one object is detected in a query image (multi-item scenes, occlusion, background
clutter), the pipeline must pick exactly one bbox to crop and embed. The original pipeline used
an unexamined default; this ADR is about which of the four policies built in Milestone 1/5
should be the actual default going forward.

## Alternatives

`main/retrieval/preprocessing.py::select_bbox` implements all four (IMPLEMENTATION_SPEC.md
section 18):

1. **highest_confidence** — highest detector confidence score, regardless of category.
2. **largest** — largest bbox area, regardless of category.
3. **category_confidence** — highest confidence among detections compatible with the query's
   category (`retrieval.category.get_allowed_labels`).
4. **category_largest** — largest area among category-compatible detections.

## Evidence

- **E1-E4** (Milestone 5, `main/evaluation/run_bbox_experiments.py`) run all four against the
  same detection output (one shared `DetectionCache`, per section 19's MUST), holding every
  other control fixed (section 34).
- A real cross-milestone bug was found and fixed while wiring E1-E4: `get_allowed_labels`
  (used by the two category-aware policies) was keyed only by Musinsa category strings, while
  the evaluation dataset labels queries by ChromaDB collection name — `category_confidence`/
  `category_largest` would have silently matched zero detections for every dataset query before
  the fix (`docs/evidence/milestone-5.md`, section on `COLLECTION_LABEL_MAPPING`). This is now
  covered by a regression test (`main/tests/regression/test_golden_set.py::
  test_category_mapping_resolves_known_collection_names`).
- No NDCG/MRR/failure-distribution comparison across the four policies exists yet — same root
  cause as ADR-001 (empty dataset, no ML dependencies installed here).

## Decision

**TBD.** No policy is finalized without the E1-E4 comparison Milestone 5 built the infrastructure
for. `docs/decisions/final_config.yaml`'s `bbox_policy` field stays `TBD`.

## Reason

N/A until Decision is made.

## Trade-offs

- **Confidence-based** (1, 3) picks what the detector is most sure about, which may not be the
  largest/most-visible instance of the garment in a multi-item scene.
- **Area-based** (2, 4) picks the most visually dominant object, which may be a non-garment
  object the detector confidently but incorrectly flagged.
- **Category-aware** (3, 4) is strictly more conservative than its non-category counterpart (1,
  2) — it can only select from a subset of the same detections — at the cost of a fallback path
  when nothing category-compatible is found (`preprocess_image`'s `fallback_policy`).

## Rejected Alternatives

None rejected yet — all four remain candidates pending the ablation comparison.

## Future Work

Run E1-E4 for real; compare NDCG@10/MRR and each policy's failure distribution (Milestone 8),
paying particular attention to `bbox_selection_failure`/`detection_failure` rates (a policy that
wins on average NDCG but has a much higher failure rate on hard/multi-item queries is a
meaningfully different trade-off than one that's simply better everywhere).
