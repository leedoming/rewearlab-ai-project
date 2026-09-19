# ADR-004: BBox Padding Ratio

## Status

**Decision: TBD** — pending real ablation results. See "Decision" below.

## Context

A tight crop to the detected bbox can cut off part of the garment (sleeve, hem) that carries
distinguishing detail — but padding the crop outward re-introduces some of the background/
context the bbox crop was meant to remove in the first place. `IMPLEMENTATION_SPEC.md` section
20 and section 42 require comparing at least 0%/10%, and 0%/10%/20% where possible.

## Alternatives

`main/retrieval/preprocessing.py::crop_image` supports an arbitrary `padding_ratio`, clamped to
image bounds (regression-tested:
`main/tests/regression/test_golden_set.py::test_bbox_coordinates_stay_within_image_bounds_even_with_padding`).
Three concrete values are in scope:

1. **0%** — tight crop to the detected bbox (E1-E6's fixed control value).
2. **10%** — `main/evaluation/configs/e7_padding_ablation.json` (Milestone 7's E7).
3. **20%** — not yet wired to any experiment config; see "Future Work."

## Evidence

- **E7** (Milestone 7) is the only experiment that varies padding away from 0% so far, and only
  to 10% — no experiment currently exercises 20%.
- Section 42 also asks for two qualitative metrics, "Feature Loss" and "Background Bias." No
  formula for either exists anywhere in the spec, and this project's `main/evaluation/
  sensitivity.py` (Milestone 9) deliberately does not invent one — see
  `docs/evidence/milestone-9.md` section 2.1. Any padding decision this ADR eventually makes can
  cite NDCG/MRR at each padding value, but not a Feature Loss/Background Bias score, since none
  exists.
- No real NDCG/MRR comparison across 0%/10%/20% exists yet — same root cause as ADR-001/002/003.

## Decision

**TBD.** `docs/decisions/final_config.yaml`'s `padding_ratio` field stays `TBD`.

## Reason

N/A until Decision is made.

## Trade-offs

- **0%**: no risk of re-introducing background, but the tightest crop is also the most likely to
  cut off a defining feature near the bbox edge.
- **10%/20%**: recovers some cut-off detail at the cost of more background/context leaking into
  the embedding; 20% recovers more detail than 10% but also leaks more background — the
  trade-off's shape (is it monotonic, does it plateau, does it reverse past some point) is
  exactly what section 42's 0/10/20 sweep is meant to reveal, and isn't knowable without running
  it.

## Rejected Alternatives

None rejected yet — 0%/10%/20% all remain candidates pending the sweep.

## Future Work

- Run E7 (10%) for real, and add a 20% config/experiment once real data exists, to complete the
  0/10/20 sweep section 42 asks for.
- If "Feature Loss"/"Background Bias" turn out to matter for the final decision, define a
  concrete, evidence-backed way to measure them (e.g. a small human-annotated sample) rather than
  inferring them indirectly from NDCG/MRR alone.
