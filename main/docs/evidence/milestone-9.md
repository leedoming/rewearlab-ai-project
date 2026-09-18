# Milestone 9 — Sensitivity Analysis: Evidence Record

**Branch:** `feat/m9-sensitivity-analysis` (created from `main` after PR #1–#8 were merged)
**PR:** _(filled in after the PR is opened)_
**Repository:** fork `leedoming/rewearlab-ai-project` (upstream: `GeeYun086/rewearlab-ai-project`)

---

## 1. Objective

Per `IMPLEMENTATION_SPEC.md` section 69 ("Milestone 9 — Sensitivity: threshold, padding, Top-K")
and sections 41–43 (Phase 9 — Sensitivity Analysis): build the metrics needed to compare
detection threshold (0.3/0.4/0.5), padding ratio (0%/10%/20%), and Top-K (5/10/20) sweeps against
each other, as `main/evaluation/sensitivity.py`.

---

## 2. Design Decisions

### 2.1 Scope: compute what's objectively derivable; don't fabricate the rest

**Decision:** `sensitivity.py` implements `average_detection_count`, `detection_success_rate`,
`fallback_rate`, and `wrong_object_rate` (all from section 41's Detection Threshold metric list)
plus a generic `summarize_sensitivity_sweep()` for comparing any already-existing metric (NDCG,
MRR, precision, recall) across swept parameter values. It does **not** implement "Feature Loss"
or "Background Bias" (section 42's Padding Sensitivity metrics).

**Reason:** every metric this module does implement has a concrete, already-real signal to
derive it from: `fallback_rate`/`wrong_object_rate` read `retrieval.preprocessing`'s own
`fallback_used`/`selected_label` fields (never fabricated, per that module's own docstrings), and
`average_detection_count`/`detection_success_rate` read raw detection counts a caller gets
directly from `retrieval.detection.detect_fashion_items`. "Feature Loss" and "Background Bias"
have no such signal anywhere in this codebase, and the spec gives no formula for computing
either from data — they read as qualitative, human-judgment assessments (would a human say this
crop lost a defining design detail? does the background visibly bias the embedding?), the exact
same category of thing `evaluation.failure_analysis.classify_failure` (Milestone 8) already
declines to auto-assign for `feature_loss`/`background_bias` in the failure taxonomy. Inventing
a formula for either here would be fabricating a metric with no evidence behind it — precisely
what section 67 rule 7 and this project's whole "no fabricated results" discipline forbid.
Padding sensitivity's *quantitative* half (NDCG/MRR across 0%/10%/20%) is still fully covered —
by reusing the exact same `evaluation.metrics.ndcg_at_k`/`mrr` already built in Milestone 2, run
across padding values, via `summarize_sensitivity_sweep()`.

**Verification:** every function in `sensitivity.py` is unit-tested against synthetic inputs
(`test_sensitivity.py`); no test claims a Feature Loss/Background Bias score exists.

### 2.2 No new CLI runner script or fixed experiment IDs for the sensitivity sweeps

**Decision:** unlike Milestones 4–7 (`run_bbox_experiments.py`, `run_category_filter_experiments.py`,
`run_padding_ablation_experiments.py`), this milestone does not add a
`run_sensitivity_experiments.py` CLI script or new experiment IDs (e.g. "S1"/"S2"/"S3") to the
ablation matrix.

**Reason:** `IMPLEMENTATION_SPEC.md` section 33's ablation matrix (E0–E8) is a fixed set of
named experiments with specific pinned controls; sections 41–43's sensitivity sweeps are a
different kind of thing — an open-ended parameter sweep (three detection thresholds, three
padding ratios, three Top-K values) the spec gives no experiment-ID naming convention for at
all. More importantly, every ML-touching CLI script built so far (`run_bbox_experiments.py`
onward) has been honestly documented as **never run against a live detector/ChromaDB/dataset**
in this dev environment (`torch`/`chromadb`/`transformers`/`open_clip` aren't installed here,
and the dataset is still an empty scaffold). Writing a sixth such script that also could never be
run or verified here would just be more unverifiable code, which cuts against this project's own
per-milestone workflow (section 70, step 5: "Run relevant tests") and the task's explicit
instruction to verify solidly whatever gets built. `sensitivity.py`'s pure functions, by
contrast, are fully computable and testable today with synthetic inputs, and are written so that
whenever a live sweep does become possible, the exact same functions apply directly to real
per-query detection counts / preprocessing metadata / metrics.

**Verification:** N/A (a design-scope decision, not a code behavior) — see docs section 3 for
what remains explicitly out of scope.

### 2.3 `wrong_object_rate` excludes fallback selections deliberately

**Decision:** `wrong_object_rate()` only considers selections where `fallback_used` is `False`.

**Reason:** a fallback selection (nothing category-compatible was found, so the policy fell back
to `raw` or `largest`) is already counted by `fallback_rate()` — counting it again in
`wrong_object_rate()` would double-count the same underlying event under two different metric
names and inflate both. "Wrong object" is specifically about a policy *confidently* choosing an
incompatible bbox when a choice was actually made (possible under `highest_confidence`/`largest`,
which don't filter by category at all — see `retrieval.preprocessing.select_bbox`), not about
there being nothing to choose from in the first place.

**Verification:** `test_wrong_object_rate_ignores_fallback_selections`,
`test_wrong_object_rate_zero_when_every_selection_was_a_fallback`.

### 2.4 `summarize_sensitivity_sweep` is generic across all three sensitivity axes

**Decision:** one function, parameterized by `parameter_field`, rather than three separate
functions (`summarize_threshold_sweep`, `summarize_padding_sweep`, `summarize_top_k_sweep`).

**Reason:** the actual computation — group rows by a swept value, summarize every metric with
`evaluation.aggregate.summarize_values` — is identical for all three axes; only the field name
being grouped on differs. Three near-identical functions would be exactly the kind of
duplicated-for-no-reason code this project's own conventions (e.g. Milestone 6/7's shared
`_validate_common_retrieval_controls`/`_write_experiment_outputs`) already avoid elsewhere.

**Verification:** `test_summarize_sensitivity_sweep_groups_by_parameter_value` (detection
threshold),`test_summarize_sensitivity_sweep_handles_multiple_metric_names` (top_k) — same
function, two different `parameter_field`s.

---

## 3. What This Milestone Does NOT Build

- **No live sensitivity sweep has been run** — same root cause as every milestone since M4: no
  detector/ChromaDB/dataset available in this environment. `sensitivity.py`'s functions are
  ready to consume real per-query data the moment that becomes possible.
- **No CLI runner script or new experiment IDs** — see 2.2.
- **No "Feature Loss"/"Background Bias" score** — see 2.1. These remain human-judgment
  assessments, not automated metrics, exactly like Milestone 8's non-auto-assigned failure
  categories.
- **No comparison report across actual threshold/padding/Top-K values** — `summarize_sensitivity_sweep()`
  is the infrastructure to produce one; producing the actual report requires real experiment
  data, which doesn't exist yet.

---

## 4. Fixes Made

None — this is new, additive infrastructure; no existing behavior was changed.

---

## 5. Tests Added

| File | Covers |
|---|---|
| `main/tests/unit/test_sensitivity.py` | `average_detection_count()`/`detection_success_rate()`: correct computation and empty-input rejection. `fallback_rate()`: correct computation and empty-input rejection. `wrong_object_rate()`: flags an incompatible non-fallback selection, ignores fallback selections (both when some and when all selections were fallbacks), zero when every non-fallback selection was compatible. `summarize_sensitivity_sweep()`: groups by an arbitrary parameter field, handles multiple metric names per row, empty input returns an empty summary rather than crashing. |

---

## 6. Exact Validation Commands and Results

```bash
python -m compileall -q main/evaluation main/tests main/retrieval
```
Result: exit 0, no output.

```bash
cd main && python -m pytest tests/unit -q
```
Result: `194 passed` (181 from Milestones 1–8 + 13 new). Zero failures, zero skipped.

Test environment: Python 3.13.1, pytest 9.1.1, on Windows. `torch`, `chromadb`, `transformers`,
`open_clip` remain **not installed**; `sensitivity.py` has zero third-party dependencies (like
`evaluation.metrics`/`evaluation.failure_analysis`), so every test here runs without any of them.

---

## 7. Known Limitations

- **Never run against a live detector/ChromaDB/dataset** — see section 3.
- **Feature Loss / Background Bias are not automated** — see 2.1; these require human review of
  actual cropped images, which doesn't exist yet (no real dataset).
- **No actual sensitivity comparison numbers exist** — this milestone builds the capability to
  compute one; it does not claim any threshold/padding/Top-K value is more or less sensitive
  than another, since no real sweep has been run.
- No performance numbers, benchmark results, or "which parameter value is best" claims are made
  anywhere in this record or in the code — none exist yet, and none were fabricated.

---

## 8. Commit SHAs

```
c4b7bd7  feat: add detection-threshold/padding/Top-K sensitivity metrics
4152ed1  test: cover detection-threshold/padding/Top-K sensitivity metrics
```
(This evidence record is itself added in a further, following commit — see the PR for its exact SHA.)

## 9. PR

_(filled in after the PR is opened)_
