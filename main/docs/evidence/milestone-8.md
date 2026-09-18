# Milestone 8 — Failure Analysis: Evidence Record

**Branch:** `feat/m8-failure-analysis` (created from `main` after PR #1–#7 were merged)
**PR:** _(filled in after the PR is opened)_
**Repository:** fork `leedoming/rewearlab-ai-project` (upstream: `GeeYun086/rewearlab-ai-project`)

---

## 1. Objective

Per `IMPLEMENTATION_SPEC.md` section 69 ("Milestone 8 — Failure Analysis: failure taxonomy,
poor-query extraction, summary generation") and sections 28–32 (Phase 7): build the ten-category
failure taxonomy, an automatic poor-query extractor, an upstream-first failure classifier, and a
per-experiment failure distribution — as the new `main/evaluation/failure_analysis.py` module
listed in the spec's own directory structure (section 5).

---

## 2. Design Decisions

### 2.1 A real cross-milestone gap found: `failure_cases.jsonl` was never written

**Problem:** `IMPLEMENTATION_SPEC.md` section 36 has listed `failure_cases.jsonl` as one of
every experiment's four required outputs (`query_results.jsonl`, `metrics_by_query.csv`,
`summary.json`, `failure_cases.jsonl`) since the spec was written — including for E0, built in
Milestone 4. No runner (`run_baseline`, `run_bbox_experiment`, `run_category_filter_experiment`,
`run_padding_experiment`, `run_raw_filter_experiment`) has ever produced this file; it was simply
missed until this milestone's own module gave a reason to look for it.

**Evidence:** grepping `main/evaluation/evaluator.py` prior to this milestone shows
`_write_experiment_outputs` writing exactly three files
(`query_results.jsonl`/`metrics_by_query.csv`/`summary.json`); `failure_cases.jsonl` appears
nowhere in the codebase before this commit.

**Decision / Fix:** rather than adding failure-case writing separately to each of the five
runner functions (risking one of them being missed, or drifting out of sync with the others),
the fix goes into `_write_experiment_outputs` itself — the one function every runner already
shares. This means the fix applies retroactively to E0 through E8 in a single change, and any
future experiment runner (E9/E10, if ever added) gets it automatically by using the same shared
helper, rather than needing to remember to wire it in again.

**Verification:** `test_run_baseline_writes_failure_case_for_a_genuinely_poor_query`,
`test_run_baseline_writes_no_failure_cases_for_a_good_query`,
`test_run_category_filter_experiment_classifies_real_filter_exclusion`.

### 2.2 `classify_failure` only auto-assigns categories with a real, structural signal

**Decision:** of the ten categories in section 28's taxonomy, `classify_failure()` only ever
returns four of them automatically: `detection_failure`, `bbox_selection_failure`,
`filter_exclusion_failure`, and `database_coverage_failure` — plus the
`embedding_similarity_failure` fallback when none of those explain a poor query. The other five
(`feature_loss`, `background_bias`, `category_prediction_failure`, `semantic_category_boundary`,
`ranking_failure`) are never auto-assigned.

**Reason:** each of the four auto-assigned categories has a concrete, already-computed signal to
key off — `retrieval.preprocessing.preprocess_image`'s own `fallback_used`/`fallback_reason`
metadata (never fabricated, per its own docstring), and `evaluate_query`'s real
`relevant_exclusion_rate` (Milestone 6). The other five categories require either human
judgment this codebase has no way to compute (is the crop missing a defining feature? is the
background visually biasing the embedding?) or a component that doesn't exist yet
(`category_prediction_failure` needs a predicted-category classifier — Milestone 6's evidence
doc, section 3, already established that only Oracle category exists). Auto-assigning any of
these would mean fabricating a diagnosis without evidence — exactly what the project's own
engineering-process rules (section 67 rule 7, applied here by analogy) forbid. The failure
record schema (section 29) still has room for a human reviewer to add these via
`secondary_failures`/`notes` — `classify_failure()` just never claims to do that job itself.

**Verification:** `test_classify_failure_never_fabricates_human_judgment_categories`.

### 2.3 `database_coverage_failure` is caller-supplied, not auto-detected, in this milestone

**Decision:** `classify_failure(preprocessing, metrics, database_coverage_failure=False)` takes
this as an explicit boolean parameter rather than computing it internally, and defaults to
`False`.

**Reason:** determining whether a labeled relevant product was never indexed in ChromaDB at all
(as opposed to being excluded by a category filter — a distinction Milestone 6's
`evaluation.label_lookup` module already draws, see its own test
`test_count_relevant_excluded_by_filter_ignores_unresolved_products`) requires a live ChromaDB
`.get()` call. `failure_analysis.py` is deliberately kept dependency-free (like
`evaluation.metrics`), so it has no client to make that call itself — and `_write_experiment_outputs`,
where `classify_failure` is invoked, only has already-computed `records`/`dataset`, not a live
`client` either (unlike the CLI runner scripts, which do hold one). Wiring a live DB-coverage
check into the shared write path would mean giving a pure output-serialization function a
network dependency it doesn't otherwise need, for a signal only the CLI scripts could actually
supply. Left as an explicit, honestly-defaulted parameter rather than silently assumed `False`
forever — a future milestone (or the CLI runners themselves) can thread the real value through
once it's worth the added coupling.

**Verification:** `test_classify_failure_database_coverage_failure_when_flagged`,
`test_classify_failure_database_coverage_checked_before_embedding_fallback`.

### 2.4 Classification order follows section 30's upstream-first MUST, not the pipeline diagram's literal order

**Decision:** `classify_failure()` checks in this order: detection → bbox selection → filter
exclusion → database coverage → embedding similarity (fallback). Section 30's pipeline diagram
lists stages as `Input → Detection → BBox Selection → Crop → Category → Filter →
Embedding/Retrieval → DB Coverage` — DB Coverage appears *after* Embedding/Retrieval there.

**Reason:** that diagram describes the request-time data flow (an image literally passes through
detection before search), not the order a diagnosis should be trusted in. Section 30's actual
MUST is "처음부터 embedding failure로 단정하지 않는다" (don't conclude embedding failure from the
start) — and an item that was never indexed at all is the clearest case of a poor result that
has nothing to do with embedding quality. Checking database coverage *before* falling back to
`embedding_similarity_failure`, rather than after, is what actually satisfies that MUST; matching
the diagram's left-to-right order literally would mean claiming an embedding failure first and
only reconsidering it afterward, which is the exact anti-pattern section 30 warns against.

**Verification:** `test_classify_failure_database_coverage_checked_before_embedding_fallback`.

### 2.5 `summarize_failures` reports every category, including zero-count ones

**Decision:** `summarize_failures()` always returns all ten `FAILURE_CATEGORIES` keys, with
`{"count": 0, "ratio": 0.0}` for categories that didn't occur, rather than omitting them.

**Reason:** section 32's example table lists specific failure rows regardless of whether they
occurred in a given experiment run — a reader comparing experiments (section 40's cross-experiment
comparison table) needs to see "zero `bbox_selection_failure` this run" as a real data point, not
infer it from the row's absence. An omitted row is ambiguous between "zero occurrences" and "this
tool doesn't track that category."

**Verification:** `test_summarize_failures_includes_zero_count_categories`,
`test_summarize_failures_empty_input_has_zero_ratios_not_a_crash`.

---

## 3. What This Milestone Does NOT Build

- **No automatic `database_coverage_failure` detection** — see 2.3. The hook exists
  (`classify_failure`'s parameter); wiring a live ChromaDB lookup into it is future work.
- **No human-authored `secondary_failures`/`notes`/`evidence`** — every auto-generated
  `failure_cases.jsonl` entry has these fields present but empty (`[]`/`""`/`{}`), exactly
  matching section 29's schema shape, ready for a human reviewer to fill in during actual
  failure investigation once real dataset queries exist.
- **No cross-experiment failure comparison report** (section 40's `RAW | BBOX | BBOX+Soft`
  table) — that requires comparing multiple experiments' already-generated `summary.json` files
  against each other and is Milestone 10 (Final Decision & Regression) scope, once real
  experiment results exist to compare.

---

## 4. Fixes Made

See section 2.1 — `failure_cases.jsonl` is now written by every experiment runner (E0–E8),
retroactively, via the one function they all share. No other existing behavior changed;
`summary.json` gained one new key (`failure_distribution`) and no existing key was removed or
altered, which is why no test asserting on `summary.json`'s previously-existing keys needed to
change (confirmed by the full existing suite still passing unmodified).

---

## 5. Tests Added

| File | Covers |
|---|---|
| `main/tests/unit/test_failure_analysis.py` | `has_highly_relevant_in_top_k()`: grade-2 detection, grade-1 correctly excluded, respects `k`. `is_poor_query()`: each of section 31's three OR conditions individually, and the "none trigger" negative case. `classify_failure()`: detection/bbox-selection/filter-exclusion/database-coverage signals each classified correctly, RAW mode structurally can't report a bbox-stage failure, database coverage is checked before the embedding fallback, and the five human-judgment-only categories are never auto-assigned. `summarize_failures()`: counts/ratios, zero-count categories included, empty input doesn't crash, unknown category rejected. |
| `main/tests/unit/test_failure_cases_output.py` | End-to-end through `run_baseline`/`run_category_filter_experiment`: a genuinely poor query produces exactly one `failure_cases.jsonl` entry with the correct `primary_failure`, a good query produces none, and `summary.json`'s `failure_distribution` reflects the real count in both cases. |

---

## 6. Exact Validation Commands and Results

```bash
python -m compileall -q main/evaluation main/tests main/retrieval
```
Result: exit 0, no output.

```bash
cd main && python -m pytest tests/unit -q
```
Result: `181 passed` (159 from Milestones 1–7 + 22 new). Zero failures, zero skipped.

Test environment: Python 3.13.1, pytest 9.1.1, on Windows. `torch`, `chromadb`, `transformers`,
`open_clip` remain **not installed**; `failure_analysis.py` itself has zero third-party
dependencies (like `evaluation.metrics`), so every test here runs without any of them.

---

## 7. Known Limitations

- **`database_coverage_failure` is never auto-detected** in this milestone — see 2.3. Every
  `failure_cases.jsonl` entry generated so far will report either an upstream bbox/filter
  failure or fall through to `embedding_similarity_failure`, never `database_coverage_failure`,
  until a caller threads a real value through.
- **Five of the ten taxonomy categories are never auto-assigned** (2.2) — they exist in
  `FAILURE_CATEGORIES` and `summarize_failures()`'s output for completeness, but only a human
  reviewer populates them, and no human review of real queries has happened yet (the dataset is
  still an empty scaffold, same limitation as every milestone since M4).
- **No comparison across experiments' failure distributions is made or claimed** — see section 3.
- No performance numbers, failure counts, or "which stage fails most" claims are made anywhere in
  this record or in the code beyond what the (currently empty) dataset would produce — none exist
  yet, and none were fabricated.

---

## 8. Commit SHAs

```
85219ad  feat: add failure taxonomy and wire failure_cases.jsonl into all experiment outputs
ddc2d0f  test: cover failure taxonomy and failure_cases.jsonl/failure_distribution output
```
(This evidence record is itself added in a further, following commit — see the PR for its exact SHA.)

## 9. PR

_(filled in after the PR is opened)_
