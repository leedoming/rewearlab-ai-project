# Milestone 10 — Final Decision & Regression: Evidence Record

**Branch:** `feat/m10-final-decision-regression` (created from `main` after PR #1–#9 were merged)
**PR:** _(filled in after the PR is opened)_
**Repository:** fork `leedoming/rewearlab-ai-project` (upstream: `GeeYun086/rewearlab-ai-project`)

---

## 1. Objective

Per `IMPLEMENTATION_SPEC.md` section 69 ("Milestone 10 — Final Decision & Regression: Final
config, ADR, Golden set, Regression tests, README") and sections 49–58 (Phase 10 — Final Pipeline
Selection, ADRs, Regression Tests, Golden Evaluation Set): close out the ten-milestone refactor
by building the final-decision infrastructure (config template, ADRs), the golden-set/regression
infrastructure, and updating the README — while being honest about which of these can actually
be *decided* versus merely *built*, given the dataset is still empty and no ML dependency is
installed in this environment.

---

## 2. Design Decisions

### 2.1 No final pipeline is declared — every ADR's Decision is `TBD`

**Decision:** `docs/decisions/final_config.yaml` and all five ADRs (`ADR-001` through `ADR-005`)
explicitly record `Decision: TBD`, not a chosen configuration.

**Reason:** `IMPLEMENTATION_SPEC.md` section 50's MUST is unambiguous: "실험 이전에 TODO 값을
임의로 확정하지 않는다" (do not arbitrarily finalize TODO values before the experiments that
justify them). Milestones 4–9 built the complete infrastructure to produce every number a final
decision would need — RAW vs. four bbox policies (E0–E4), category filtering (E5–E6), padding
(E7–E8), a failure taxonomy with per-experiment distributions (Milestone 8), and sensitivity
metrics (Milestone 9) — but none of it has been run against real data:
`main/evaluation/dataset/labels.json` is still `{"queries": {}}`, and
`torch`/`chromadb`/`transformers`/`open_clip` are not installed here. Declaring a final pipeline
without that evidence would be the exact fabrication this project's evidence docs have refused at
every prior milestone (see, e.g., milestone-6.md section 2.4, milestone-7.md section 2.1,
milestone-9.md section 2.1 — all making the identical "don't declare a winner without evidence"
call). This milestone's honest deliverable is the *decision infrastructure*, not a decision.

**Verification:** N/A (a documentation-scope decision) — `final_config.yaml`'s every field and
every ADR's Status/Decision section literally reads `TBD`, checkable by inspection.

### 2.2 Golden set: build coverage *validation*, not a curated golden set itself

**Decision:** `main/evaluation/golden_set.py` implements `validate_golden_set()` /
`classify_golden_set_coverage()` — functions that check whether a *given* list of queries
satisfies section 55's requirements (10–15 queries, six required case types) — rather than
producing 10–15 actual curated queries.

**Reason:** a real golden set needs real product photos and real human relevance judgments for
each of the six required case types (easy, hard, multi-item, background-heavy, category
boundary, known previous failure) — exactly the kind of content Milestone 3's dataset docs
already identified as requiring manual curation, and which doesn't exist yet (the dataset is
still an empty scaffold). Building the *validator* now means that whenever a real 10–15 query
golden set is curated, there's already a tested, mechanical way to confirm it actually satisfies
section 55's requirements, rather than eyeballing it.

**Reason for the two explicit-only case types:** "category boundary" and "known previous
failure" don't correspond to any field the existing `QueryRecord` schema
(`evaluation/dataset/loader.py`, Milestone 3) already has — unlike `difficulty`/`scene_type`,
which map directly. Guessing at a heuristic (e.g. "any query in a category that appears in
`SOFT_CATEGORY_FILTER_MAPPING` is a boundary case") would silently mislabel queries the golden
set's actual author never intended as boundary cases just because they happen to share a
category. `classify_golden_set_coverage()` instead takes these two as explicit `query_id` sets
the caller supplies — an honest "this project doesn't know which queries these are automatically"
rather than a fabricated guess.

**Verification:** `test_validate_golden_set_accepts_full_coverage`,
`test_validate_golden_set_rejects_missing_case_type`,
`test_classify_golden_set_coverage_uses_explicit_ids_for_non_derivable_case_types`.

### 2.3 Quality regression checker is generic and dependency-free, not tied to a real baseline

**Decision:** `main/evaluation/regression.py`'s `check_quality_regression()` takes
`new_metrics`/`baseline_metrics`/`metric_names` as plain dicts/lists — it doesn't read a
committed "baseline.json" file or assume any specific metric names exist.

**Reason:** section 56 itself says the real tolerance is decided *after* a real baseline exists
("실제 tolerance는 baseline 결과 후 결정한다") — there is no real baseline yet (same root cause
as 2.1), so hardcoding one now would be fabricating baseline numbers. The function is built to
be immediately usable the moment a real baseline does exist, without needing to change its
signature.

**Verification:** `test_check_quality_regression_reports_per_metric`,
`test_passes_quality_regression_none_when_either_value_is_none`.

### 2.4 `tests/regression/test_golden_set.py` tests functional invariants, not retrieval quality

**Decision:** the five tests in `tests/regression/test_golden_set.py` map 1:1 to section 54's
"Functional" regression list (BBox coordinates valid, Fallback works, Category mapping works,
Top-K result shape valid, Metric computation correct) using synthetic, deterministic fixtures —
not real product images or a claim about retrieval quality.

**Reason:** section 54 explicitly separates "Functional" regression (code-behavior invariants,
testable today with synthetic inputs) from "Quality" regression (NDCG/MRR/Critical Query Rank
against a real baseline, which needs real data — see 2.3). Conflating the two in one file would
either understate what's actually verified today (functional invariants, fully covered) or
overstate it (implying a quality claim that doesn't exist).

**Verification:** the file's own five tests, e.g.
`test_bbox_coordinates_stay_within_image_bounds_even_with_padding`,
`test_category_mapping_resolves_known_collection_names` (a direct regression guard for the real
Milestone 5 bug).

### 2.5 `tests/regression/test_critical_queries.py` uses an explicitly fictional query ID

**Decision:** section 57's own example ("Q017 first relevant rank <= 5") is a real production
case from a real dataset. This project's `test_critical_queries.py` uses `"P_TARGET"` and a
docstring explicitly stating this is a stand-in, not a real finding.

**Reason:** there is no real "Q017" — `labels.json` has zero queries. Naming a synthetic test
case "Q017" or inventing a plausible-sounding real query ID would misrepresent a fabricated
example as a genuine critical-query finding, exactly the kind of thing this project's "no
fabricated results" discipline exists to prevent.

**Verification:** the module docstring itself, plus `test_synthetic_critical_query_within_rank_bound`/
`test_synthetic_critical_query_beyond_rank_bound_fails`.

### 2.6 README gets an honest infrastructure summary, not a fabricated 15-section narrative

**Decision:** `main/README.md` gains one new section ("Retrieval 평가 시스템") summarizing what
was built across Milestones 1–10 and explicitly stating the current "never run against real
data" status — not a rewrite implementing every one of section 64's fifteen README sections
(Problem, Original Pipeline, Observed Failures, ... Future Work).

**Reason:** several of those fifteen sections (Observed Failures, Ablation Study results,
Failure Analysis findings, Sensitivity Analysis findings, Final Architecture) can only be
honestly written from real experiment output — writing them now would mean inventing failure
rates, ablation winners, and sensitivity conclusions that don't exist, the same fabrication this
whole record has refused throughout. The section actually added is scoped to what's genuinely
true today: what infrastructure exists, where it lives, and that it hasn't been run for real yet
— consistent with `docs/decisions/ADR-005-final-pipeline.md`'s own "Future Work" item 5, which
explicitly defers the full README rewrite until real numbers exist.

**Verification:** N/A (documentation) — the added section is checkable by reading it; it makes
no claim beyond what `docs/evidence/milestone-*.md` and `docs/decisions/ADR-*.md` already
establish.

---

## 3. What This Milestone Does NOT Build

- **No final pipeline decision** — see 2.1. Every ADR and `final_config.yaml` stays `TBD`.
- **No real curated golden set** (10–15 actual queries) — see 2.2. Only the coverage validator.
- **No real quality regression baseline** — see 2.3. Only the generic checker.
- **No full section-64 README rewrite** — see 2.6. Only an honest infrastructure summary.
- **No new experiment IDs or CLI runners** — this milestone is decision/regression
  infrastructure, not new ablation experiments (those were Milestones 4–7).

---

## 4. Fixes Made

None — this is new, additive infrastructure and documentation; no existing behavior was changed.

---

## 5. Tests Added

| File | Covers |
|---|---|
| `main/tests/unit/test_regression.py` | `passes_quality_regression()`/`check_quality_regression()`: within/below tolerance, improvement, `None` propagation when either value is `None`, per-metric reporting. `first_relevant_rank()`: finds the correct rank, `None` when nothing relevant, respects `min_relevance`. `passes_critical_query_rank()`: within bound, beyond bound, `None` rank never passes. |
| `main/tests/unit/test_golden_set.py` | `classify_golden_set_coverage()`: maps difficulty/scene_type correctly, uses explicit id sets for the two non-derivable case types, a query can satisfy multiple case types, always returns every required key. `validate_golden_set()`: accepts full coverage, rejects too few/too many queries, rejects a missing case type, error message names every missing case type. |
| `main/tests/regression/test_golden_set.py` | Section 54's five functional regression checks: bbox coordinates valid (with and without padding), fallback works (no detections, no category-compatible bbox), category mapping works (including the real Milestone 5 bug's regression guard), Top-K result shape valid (bounded, ranked, correct keys, fewer-than-K case), metric computation correct (precision/recall/mrr/ndcg against hand-computed expectations, bbox policy selection). |
| `main/tests/regression/test_critical_queries.py` | `first_relevant_rank()`/`passes_critical_query_rank()` against an explicitly-fictional synthetic query, both within and beyond the rank bound. |

---

## 6. Exact Validation Commands and Results

```bash
python -m compileall -q main/evaluation main/tests main/retrieval
```
Result: exit 0, no output.

```bash
cd main && python -m pytest tests/ -q
```
Result: `226 passed` (194 from Milestones 1–9 + 32 new). Zero failures, zero skipped.

Test environment: Python 3.13.1, pytest 9.1.1, on Windows. `torch`, `chromadb`, `transformers`,
`open_clip` remain **not installed**; `evaluation.regression`/`evaluation.golden_set` have zero
third-party dependencies (like `evaluation.metrics`/`evaluation.failure_analysis`/
`evaluation.sensitivity`), so every test here runs without any of them. The functional
regression tests in `tests/regression/test_golden_set.py` exercise real `retrieval.*` code
against synthetic PIL images and a fake ChromaDB client — never a real model or database.

---

## 7. Known Limitations

- **No final pipeline has been decided** — see 2.1. This is the single largest honest limitation
  of the whole ten-milestone project: every ablation/failure/sensitivity infrastructure piece
  built since Milestone 4 is real and tested, but none of it has produced a real number, because
  the dataset is empty and the ML dependencies aren't installed here.
- **No real golden set exists** — see 2.2.
- **No real quality regression baseline exists** — see 2.3.
- **No real critical queries exist** — see 2.5.
- **The README's evaluation section is an infrastructure summary, not the full section-64
  narrative** — see 2.6.
- This is the last of ten milestones' evidence docs; taken together with milestone-1.md through
  milestone-9.md, this is a complete, honest account of what was built, what was verified, and
  what remains explicitly undecided pending real data — consistent with the stated goal of this
  whole exercise (demonstrating genuine engineering judgment, not exhaustive completion).

---

## 8. Commit SHAs

```
a8039e0  feat: add quality regression checks and golden-set coverage validation
1511fbf  test: cover regression/golden-set modules and add functional regression suite
c063e5d  chore: remove accidentally committed __pycache__ files
7a4ac5f  docs: add final_config.yaml, five ADRs, and a README evaluation summary
```
(This evidence record is itself added in a further, following commit — see the PR for its exact SHA.)

## 9. PR

_(filled in after the PR is opened)_
