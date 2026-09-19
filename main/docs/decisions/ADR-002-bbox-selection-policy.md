# ADR-002: BBox Selection Policy

## Status

> **⚠️ Superseded by `docs/evidence/milestone-10-transform-bug.md`.** Same root cause as
> ADR-001: the Pilot Evidence below used embeddings computed with a real preprocessing bug. The
> corrected re-run still has `category_confidence`/`category_largest` (E3/E4, now identical to
> each other) beating `highest_confidence`/`largest` (E1/E2) — that part of this ADR's reasoning
> survives — **but RAW (ADR-001) now beats all four BBox policies**, so the practical
> recommendation of this ADR only matters if BBox is used at all, which ADR-001 no longer
> supports. Kept below as the historical record.

**Provisional (partially retracted — see correction above): `category_confidence` (E3).** Based
on a real but small (N=7 query) pilot — see "Pilot Evidence" below and ADR-001's own Pilot
Evidence section (same run). Not final: needs confirmation against the full golden set (section
55) before `docs/decisions/final_config.yaml` is filled in for real.

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
- Until now, no NDCG/MRR/failure-distribution comparison across the four policies existed — see
  "Pilot Evidence" below for what's changed.

## Pilot Evidence (real data, N=7 queries)

Same real pilot run as ADR-001 (`main/evaluation/pilot/`, see that ADR for the full setup and
docs/evidence/milestone-10-pilot.md for the complete writeup):

| Policy | NDCG@10 | MRR | Recall@10 | Incompatible Category Rate@10 | Failures (of 7) |
|---|--:|--:|--:|--:|--:|
| highest_confidence (E1) | 0.603 | 0.714 | 0.655 | 0.314 | 2 embedding_similarity |
| largest (E2) | 0.601 | 0.714 | 0.667 | 0.314 | 2 embedding_similarity |
| category_confidence (E3) | **0.738** | 0.857 | 0.810 | 0.271 | 1 embedding_similarity |
| category_largest (E4) | 0.714 | 0.857 | 0.810 | 0.257 | 1 embedding_similarity |

The two category-aware policies (E3/E4) both beat both category-agnostic policies (E1/E2) on
every metric in this pilot, and `category_confidence` (E3) edges out `category_largest` (E4) on
NDCG/recall. The mechanism is directly inspectable in the pooled candidates
(docs/evidence/milestone-10-pilot.md): for the two multi-item/person-wearing queries in this
pilot (Q003, a knit-sweater-plus-visible-jeans photo; Q006, two people each wearing a cardigan),
`highest_confidence`/`largest` selected a detection from the *wrong* collection entirely (e.g.
cropping to jeans for a knit-sweater query) in nearly every pooled result, while
`category_confidence`/`category_largest` — constrained to category-compatible detections via
`retrieval.category.get_allowed_labels` — did not make that mistake. This is precisely the
`wrong_object_rate` failure mode `evaluation.sensitivity` (Milestone 9) was built to measure,
now observed on real photos rather than only defined in the abstract.

## Decision

**Provisional: `category_confidence` (E3).** It has the best NDCG@10, ties for best MRR, ties
for best recall, and has the lowest failure count in this pilot. `category_largest` (E4) is a
close second and would be a reasonable runner-up candidate in ADR-005's final shortlist.

This is provisional, not final, for the same reasons as ADR-001: N=7 is below section 55's
10-15 query target, only 3 of 4 collections are represented, and relevance labels come from a
single AI-assisted rater rather than a validated human process.

## Reason

See "Pilot Evidence" above; full per-query detail in docs/evidence/milestone-10-pilot.md.

## Trade-offs

- **Confidence-based** (1, 3) picks what the detector is most sure about, which may not be the
  largest/most-visible instance of the garment in a multi-item scene.
- **Area-based** (2, 4) picks the most visually dominant object, which may be a non-garment
  object the detector confidently but incorrectly flagged.
- **Category-aware** (3, 4) is strictly more conservative than its non-category counterpart (1,
  2) — it can only select from a subset of the same detections — at the cost of a fallback path
  when nothing category-compatible is found (`preprocess_image`'s `fallback_policy`).

## Rejected Alternatives

**highest_confidence (E1) and largest (E2)**, provisionally — both underperformed the
category-aware policies on every metric in the pilot, for the specific, inspectable reason above
(wrong-collection detections chosen in multi-item scenes). Not rejected with full confidence
given N=7, but there is no pilot evidence favoring either of them over E3/E4.

## Future Work

Grow the golden set (section 55's 10-15 queries, all four collections) and get a second rater,
then re-run E1-E4 to confirm `category_confidence` still wins before finalizing
`docs/decisions/final_config.yaml`'s `bbox_policy` field. Pay particular attention to whether a
larger, more diverse sample keeps `bbox_selection_failure`/`detection_failure` rates low for the
category-aware policies specifically (this pilot never observed either fallback failure mode at
all — worth checking that's not an artifact of the pilot's uniformly clean product photos rather
than genuine robustness).
