# ADR-001: RAW vs. BBox Preprocessing

## Status

**Decided: RAW.** Confirmed on a spec-compliant golden set (N=11, all 4 collections, passes
`evaluation.golden_set.validate_golden_set()`) with a real, deterministic, reproducible pipeline,
on a catalog rebalanced to remove a sub-style sampling skew — see
`docs/evidence/milestone-10-catalog-rebalance.md` (current numbers: **NDCG@10 0.812**, MRR 0.848,
Precision@5 0.636, all beating every BBox policy). `docs/decisions/final_config.yaml`'s
`preprocessing` field is set to `raw`.

This decision took three corrections to reach:
1. The original N=7 pilot (see "Pilot Evidence" below) found category-aware BBox (E3) winning —
   computed with a real embedding-pipeline bug present (`retrieval/models.py` used a random-crop
   training transform instead of the deterministic validation one; see
   `docs/evidence/milestone-10-transform-bug.md`).
2. The bug-fixed re-run on the same N=7 set reversed that to RAW.
3. The N=11, 4-collection expansion (`milestone-10-final-decision.md`) confirmed RAW by a wider
   margin (0.735 vs. 0.659 NDCG@10) — then its own per-category breakdown revealed the `pants`/
   `outer` catalogs were built from a skewed random sample (95%/97% dominated by one sub-style).
4. `docs/evidence/milestone-10-catalog-rebalance.md` resampled the catalog with a per-sub-style
   quota and relabeled the affected pooled candidates, widening RAW's lead further still
   (0.812 vs. 0.650 NDCG@10) — the strongest and cleanest evidence for this ADR to date.

The original (buggy) Pilot Evidence, the N=7 bug-fixed re-run, and the N=11 pre-rebalance numbers
are all kept below/linked as the historical record of what was found and when;
`milestone-10-catalog-rebalance.md` is the current, decision-grade evidence.

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
- Until now, none of E0/E1-E4/E7/E8 had been run against real data. This has since changed —
  see "Pilot Evidence" below.

## Pilot Evidence (real data, N=7 queries)

After this ADR was first written, real product photos became available (a separate personal
project's Musinsa crawl: `musinsa_pants_1000`/`musinsa_upper_2000`, ~2900 images across
pants/top/outer — no dress_skirts images existed in that source, so this pilot excludes that
collection entirely). `main/evaluation/pilot/run_pilot.py` built a real 180-item ChromaDB catalog
(60 per collection, `category_confidence`+RAW-fallback ingestion, matching
`main/embedding/musinsa_to_chromadb.py`'s production ingestion policy exactly) and held out 7
query images, hand-labeled (by the AI assistant driving this session, viewing each image — not a
professional human labeler; see "Known Limitations" in docs/evidence/milestone-10-pilot.md)
against the pooled top-5 results from E0 and all four bbox policies. `main/evaluation/pilot/run_experiments.py`
then ran the real, unmodified `evaluation.evaluator.run_baseline`/`run_bbox_experiment` against
this catalog and dataset (now committed for real at `main/evaluation/dataset/`).

Real NDCG@10 (mean over 7 queries):

| Experiment | Policy | NDCG@10 | MRR | Recall@10 | Incompatible Category Rate@10 |
|---|---|--:|--:|--:|--:|
| E0 | RAW | 0.712 | 0.857 | 0.881 | 0.243 |
| E1 | highest_confidence | 0.603 | 0.714 | 0.655 | 0.314 |
| E2 | largest | 0.601 | 0.714 | 0.667 | 0.314 |
| E3 | category_confidence | **0.738** | 0.857 | 0.810 | 0.271 |
| E4 | category_largest | 0.714 | 0.857 | 0.810 | 0.257 |

A real, fixable bug was found and fixed while producing this table: `retrieval/search.py`'s
`search_collection` passed `list(query_embedding)` to ChromaDB, which is a list of numpy
`float32` scalars (since `embed_image` returns a numpy array) — the installed `chromadb==1.5.9`
rejects that shape even though it accepts a list of native Python floats or a numpy array
directly. Fixed to `[float(value) for value in query_embedding]`. No prior milestone caught this
because none had run a real embedding through real ChromaDB before.

## Decision

**Provisional: BBox, specifically with a category-aware selection policy (E3, see ADR-002) — not
RAW, and not a category-agnostic bbox policy either.** This pilot's own data explains why RAW
still beats naive BBox (E1/E2) but loses to category-aware BBox (E3/E4): two of the seven queries
(Q003, Q006) are person-wearing/multi-item photos where a category-agnostic policy
(`highest_confidence`/`largest`) selected a *different, wrong-collection* garment in the photo
(e.g. cropping to visible jeans in a knit-sweater query) — this is exactly the "wrong object"
failure mode `evaluation.sensitivity.wrong_object_rate` (Milestone 9) was built to measure, now
observed for real rather than only hypothesized. RAW never makes this particular mistake because
it never crops at all, but it also never removes background/other-garment noise either. A
category-aware policy gets the crop right *and* avoids embedding irrelevant background, which is
why E3 edges out both RAW and naive BBox here.

This is marked **provisional, not final**, because N=7 is far below section 55's 10-15 query
target, only 3 of 4 collections are represented (no dress_skirts data existed in the source
photos), and the relevance labels come from one AI-assisted rater's visual judgment, not a
professional or multi-rater human process (section 65: "Human relevance subjectivity" — this
pilot's version of that limitation is arguably worse, not better, than the spec anticipated).

## Reason

See "Pilot Evidence" above for the real numbers this reasoning is based on, and
docs/evidence/milestone-10-pilot.md for the full writeup including per-query pooled candidates
and every relevance judgment made.

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

- **Done:** grew the golden set to N=11 across all four collections (dress_skirts sourced from a
  second personal project — see `milestone-10-final-decision.md`); `final_config.yaml`'s
  `preprocessing` field is now `raw`, not `TBD`.
- **Still open:** get a second rater (ideally a human, not the same AI assistant that ran the
  pipeline) to relabel at least a sample of the pooled candidates, to check for rater bias.
- **Still open:** the catalog is still capped at 60 items/collection (9 for dress_skirts) — a
  larger, full-catalog validation would be the gold standard beyond this project's available data.
- **Still open:** `outer`'s category-aware bbox result didn't recover to RAW level unlike `top`
  (`milestone-10-final-decision.md` section 4) — worth another look with more than 2 outer
  queries before concluding anything about that category specifically.
