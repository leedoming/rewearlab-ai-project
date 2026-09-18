# ADR-001: RAW vs. BBox Preprocessing

## Status

**Decision: TBD** — pending real ablation results. See "Decision" below.

## Context

The original pipeline (pre-refactor) always cropped the query image to a detected bbox before
embedding, with no configurable RAW path and no measurement of whether cropping actually helps.
`IMPLEMENTATION_SPEC.md` section 47 (Adaptive Preprocessing) and section 48 (Complexity
Principle) both require this choice to be evidence-driven: don't add complexity (a detector
dependency, a fallback path, bbox-selection logic) unless it measurably improves retrieval
quality by more than a small, complexity-justifying margin.

## Alternatives

1. **RAW only** — embed the full query image, no detection/cropping at all. Simplest possible
   pipeline; no detector dependency, no fallback logic, no bbox-selection policy to choose.
2. **BBox only** — always crop to a detected bbox before embedding (the original pipeline's
   behavior, now made configurable and measurable via `preprocessing.mode` in every experiment
   config, Milestone 4-7).
3. **Adaptive** — pick RAW or BBox per-query based on some signal (e.g. dominant-bbox ratio,
   per section 47's example). Explicitly gated behind evidence: section 47's MUST NOT forbids
   adding this "근거 없이" (without justification).

## Evidence

- **E0** (Milestone 4, `main/evaluation/evaluator.py::run_baseline`) implements the RAW baseline
  path, fully tested (`main/tests/unit/test_baseline_runner.py`).
- **E1-E4** (Milestone 5) implement the BBox path across four selection policies, sharing one
  `DetectionCache` per section 19's MUST so all four policies see identical detection output.
- **E7-E8** (Milestone 7) add a BBox+10%-padding variant (E7) and a RAW+soft-filter variant (E8),
  isolating whether category filtering alone (without any bbox preprocessing) helps.
- None of E0/E1-E4/E7/E8 has been run against real data: `main/evaluation/dataset/labels.json`
  is still `{"queries": {}}` (Milestone 3's dataset is an empty scaffold), and
  `torch`/`transformers`/`open_clip` aren't installed in this dev environment. There is no NDCG/
  MRR number for RAW vs. any BBox policy to compare yet.

## Decision

**TBD.** Per section 50's MUST ("실험 이전에 TODO 값을 임의로 확정하지 않는다" — do not
finalize TODO values before the experiments that justify them), this ADR does not pick RAW or
BBox without the E0 vs. E1-E4 NDCG/MRR comparison that Milestone 4-5's infrastructure was built
to produce. `docs/decisions/final_config.yaml`'s `preprocessing` field stays `TBD` until then.

## Reason

N/A until Decision is made — the reasoning will cite the actual E0 vs. E1-E4 aggregate NDCG/MRR
(`main/evaluation/results/*/summary.json`, once produced) and the failure distribution
(Milestone 8) each candidate produces, per section 49's comparison criteria (Retrieval Quality,
Robustness, Failure Rate, Complexity, Maintainability, Explainability, Service Fit).

## Trade-offs

- RAW: simplest, no detector dependency or latency, but embeds background/occlusion/other people
  in multi-item or person-wearing scenes (section 1's original motivating problem).
- BBox: isolates the target garment, but adds a detection-model dependency, a fallback path for
  when detection fails (Milestone 1's `preprocess_image`), and per-policy selection complexity
  (Milestone 5's four policies).

## Rejected Alternatives

- **Adaptive (RAW/BBox chosen per-query)**: not rejected outright, but deliberately not built
  yet — section 47 requires the pattern to already be visible in ablation results
  ("clean image → RAW better, complex image → BBOX better") before implementing it, and no
  ablation results exist. Revisit only if/when E0 vs. E1-E4 results show that pattern.

## Future Work

Once a real dataset and ML dependencies are available: run E0 and E1-E4 for real, compare
NDCG@10/MRR (section 37: mean, median, stddev, not just mean), check the failure distribution
each produces (Milestone 8's `failure_distribution`), and revisit this ADR with the actual
numbers.
