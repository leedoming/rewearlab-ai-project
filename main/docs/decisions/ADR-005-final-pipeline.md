# ADR-005: Final Pipeline Selection

## Status

**Decision: TBD** — pending real ablation, failure, and sensitivity results. See "Decision"
below.

## Context

`IMPLEMENTATION_SPEC.md` section 49 (Phase 10 — Final Pipeline Selection) asks for 2-3 final
candidate pipelines, compared on Retrieval Quality, Robustness, Failure Rate, Complexity,
Maintainability, Explainability, and Service Fit — not just whichever single experiment had the
highest NDCG. This ADR is the final synthesis step across ADR-001 through ADR-004.

## Alternatives

Every alternative in ADR-001 (RAW vs. BBox), ADR-002 (bbox policy), ADR-003 (category filter),
and ADR-004 (padding) combine into a large configuration space; section 49 asks for 2-3 *complete
pipeline* candidates to be shortlisted from it, not an exhaustive cross-product comparison. A
reasonable shortlisting approach (once real data exists): take the winning value from each of
ADR-001/002/003/004 individually as one candidate, plus 1-2 near-runner-up combinations that
differ from it in exactly one dimension (per section 34's "one variable at a time" discipline)
and are worth a final head-to-head look — e.g. if `category_confidence` (a category-aware bbox
policy) and `soft` (category filter) both win independently, it's worth checking whether
combining them compounds or duplicates the same benefit before finalizing both.

## Evidence

None yet. This ADR's "Decision" section is intentionally empty of a chosen pipeline because
ADR-001 through ADR-004 are all still `TBD` — there is nothing to synthesize until they aren't.
Every piece of infrastructure needed to produce that evidence already exists and is tested:

- Ablation experiments E0-E8 (Milestones 4-7): `main/evaluation/run_bbox_experiments.py`,
  `run_category_filter_experiments.py`, `run_padding_ablation_experiments.py`.
- Failure taxonomy and per-experiment failure distribution (Milestone 8):
  `main/evaluation/failure_analysis.py`, wired into every experiment's `summary.json` and
  `failure_cases.jsonl`.
- Sensitivity metrics (Milestone 9): `main/evaluation/sensitivity.py`.
- Golden-set coverage validation and regression checks (this milestone):
  `main/evaluation/golden_set.py`, `main/evaluation/regression.py`,
  `main/tests/regression/test_golden_set.py`, `main/tests/regression/test_critical_queries.py`.

## Decision

**TBD.** `docs/decisions/final_config.yaml` remains fully `TBD`. Per section 50's MUST, this
project does not finalize a pipeline before the ablation results that would justify it exist.

## Reason

N/A until Decision is made — will be written against section 49's seven comparison criteria once
real E0-E8 results, failure distributions, and sensitivity sweeps exist.

## Trade-offs

N/A until a candidate is selected — see ADR-001 through ADR-004 for the per-dimension trade-offs
that will compose into this decision.

## Rejected Alternatives

**Adaptive pipeline** (branching preprocessing per-query based on image complexity, section 47)
is not a rejected alternative but a deliberately deferred one, same as in ADR-001: section 47
requires the RAW-vs-BBox pattern to already be visible in real ablation results before adding
that complexity, and section 48 (Complexity Principle) requires the resulting quality gain to be
weighed explicitly against the added complexity, not assumed worthwhile. Neither can happen
without ADR-001's evidence existing first.

## Future Work

Once ADR-001 through ADR-004 each have a real decision:

1. Shortlist 2-3 final candidate pipelines per section 49's criteria (not just the single
   best-NDCG combination).
2. Fill in `docs/decisions/final_config.yaml` with the selected values, replacing every `TBD`.
3. Curate the real 10-15 query golden set (`main/evaluation/golden_set.py`'s
   `validate_golden_set` is ready to check it covers every required case type — easy, hard,
   multi-item, background-heavy, category boundary, known previous failure).
4. Run the final pipeline against the golden set to establish the real regression baseline, then
   use `evaluation.regression.check_quality_regression` (this milestone) to gate future changes
   against that baseline at the section 56 tolerance (0.97, itself provisional until a real
   baseline exists to tune it against).
5. Write the README sections section 64 requires (Problem through Future Work), now backed by
   real numbers instead of the infrastructure-only story this milestone can honestly tell.
