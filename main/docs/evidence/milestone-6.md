# Milestone 6 — Category Filter Experiments (E5–E6): Evidence Record

**Branch:** `feat/m6-category-experiments` (created from `main` after PR #1–#5 were merged)
**PR:** _(filled in after the PR is opened)_
**Repository:** fork `leedoming/rewearlab-ai-project` (upstream: `GeeYun086/rewearlab-ai-project`)

---

## 1. Objective

Per `IMPLEMENTATION_SPEC.md` section 69 ("Milestone 6 — Category Experiments: E5~E6") and
section 21 (Phase 5 — Category Filtering): build the infrastructure to run the `hard` (E5) and
`soft` (E6) category-filter policies — which ChromaDB collections get searched, given the
query's category — against the dataset, holding bbox policy and every other control fixed
(section 34: one variable at a time relative to whichever bbox experiment eventually becomes
"Best"). This milestone does **not** decide which bbox policy is best (that needs real
ablation results — Milestone 8/10), and does not build "predicted category" — see section 4.

---

## 2. Design Decisions

### 2.1 Category filtering is a distinct concept from bbox category-compatibility

**Decision:** Added a new, separate set of constants/function
(`CATEGORY_FILTER_POLICIES`, `HARD_CATEGORY_FILTER_MAPPING`, `SOFT_CATEGORY_FILTER_MAPPING`,
`get_filtered_collections()`), rather than reusing `CATEGORY_LABEL_MAPPING`/
`get_allowed_labels()` from Milestone 1/5.

**Reason:** these answer different questions. `get_allowed_labels(category)` (Milestone 1,
extended in Milestone 5) decides which *detection labels* are compatible with a category, for
bbox selection during preprocessing. `get_filtered_collections(category, policy)` (this
milestone) decides which *ChromaDB collections* get queried at all, before any bbox/embedding
work happens. Conflating them would mean a single mapping trying to serve two different
"compatible with what" questions — the spec's own soft-mapping example in section 21
(`{"pants": ["pants"], "top": ["top", "outer"], ...}`) is explicitly about collections, not
detection labels, confirming these are meant to be separate concerns.

### 2.2 Unknown category falls back to "search everything," not "search nothing"

**Decision:** `get_filtered_collections()` returns all collections when `category` isn't in the
`hard`/`soft` mapping, rather than an empty list.

**Reason:** IMPLEMENTATION_SPEC.md section 28 lists "Category Prediction Failure" and "Filter
Exclusion Failure" as distinct failure modes to diagnose later. If an unrecognized category
silently excluded every collection, every such query would score zero on every metric for a
reason unrelated to retrieval quality — polluting the very ablation data Milestone 8's failure
analysis needs. Falling back to unfiltered search means an unrecognized category degrades to
"as if no filter were applied," which is the safer default and matches `category_filter.policy
= "none"`'s own behavior.

**Verification:** `test_get_filtered_collections_unknown_category_falls_back_to_no_filter`.

### 2.3 Computing `relevant_exclusion_rate` honestly, not as a placeholder

**Problem:** Milestones 0–5's `evaluate_query()` hardcoded `relevant_exclusion_rate(total_relevant,
0)` — correct for E0–E4, which have no category filter, but would silently under-report
exclusion for E5/E6, which *do* filter and can legitimately exclude relevant items.

**Evidence:** `main/evaluation/dataset/labels.json`'s schema (`labels.schema.json`, Milestone 3)
only records `{product_id, relevance}` per query — it does not record which ChromaDB collection
each labeled product actually belongs to. Without that, there's no way to tell whether a
missing labeled item was filtered out or just ranked outside the results for an unrelated
reason.

**Decision / Fix:** Added `evaluation/label_lookup.py`, which resolves each labeled product's
true collection via a direct ChromaDB `.get(ids=...)` metadata lookup (cheap, not a similarity
search) rather than extending the Milestone 3 dataset schema. `evaluate_query()` gained an
optional `excluded_relevant_count` parameter (default `0`, so E0–E4's behavior and tests are
completely unchanged) that E5/E6's pipeline populates with the real count from
`count_relevant_excluded_by_filter()`.

**Alternative considered and rejected:** Extending `labels.json`'s schema to record each
labeled product's collection directly. Rejected because it would require re-validating and
re-authoring Milestone 3's dataset format for a value ChromaDB's own metadata already has —
duplicating data that can drift out of sync, for no benefit over a cheap lookup at evaluation
time.

**Verification:** `test_label_lookup.py` (resolution across collections, early-stop once
everything's found, missing ids omitted, exclusion counting logic) and
`test_run_category_filter_experiment_uses_real_exclusion_count` (end-to-end through
`run_category_filter_experiment`).

### 2.4 E5/E6 accept any valid bbox policy — "Best" has not been decided

**Decision:** `run_category_filter_experiment()` validates `preprocessing.bbox_policy` is *one
of* the four valid policies, not a single hardcoded expectation (unlike `run_bbox_experiment()`,
which pins E1–E4 to one specific policy each).

**Reason:** `IMPLEMENTATION_SPEC.md` section 33's ablation matrix lists E5/E6's bbox column as
"Best" — but no bbox-policy ablation has actually been run against real data yet (Milestone 5's
evidence doc: the dataset is still an empty scaffold). Section 67 rule 7 explicitly forbids
declaring an "optimal" policy without evaluation evidence. The shipped configs
(`e5_hard_category_filter.json`, `e6_soft_category_filter.json`) use `highest_confidence` as an
explicitly-labeled placeholder (`_bbox_policy_note` field) — not a claim about what's best, just
a config value that has to be something until Milestone 8/10 produce real evidence.

**Verification:** `test_run_category_filter_experiment_accepts_any_valid_bbox_policy` and
`test_run_category_filter_experiment_rejects_invalid_bbox_policy` (garbage values are still
rejected — this is "any of the four known policies," not "anything goes").

---

## 3. What This Milestone Does NOT Build: Predicted Category

`IMPLEMENTATION_SPEC.md` section 22 distinguishes **Oracle Category** (human ground-truth,
measures the filtering architecture's own effect) from **Predicted Category** (the system's own
guess, measures end-to-end performance). This milestone only implements the Oracle path —
`get_filtered_collections()` is called with `query.category`, which comes directly from the
Milestone 3 dataset's human-labeled ground truth.

There is no category-prediction component anywhere in this codebase to source a "predicted"
category from (the closest candidate — the selected detection's `label`, e.g. "outer" — is a
detection label, not validated as a category classifier, and conflating the two would be
exactly the kind of undocumented assumption the task's engineering-process rules ask to avoid).
Building and validating a real category predictor is out of this milestone's scope; the Oracle
path is genuinely all that can be built honestly right now.

---

## 4. Review Findings

No code review has been run against this milestone's diff yet. Will be updated (or a follow-up
fix commit + evidence update added) if/when one is performed, per the process used for
Milestones 1–5.

---

## 5. Fixes Made

None — this is new infrastructure; no existing behavior was changed. (`evaluate_query()`'s new
`excluded_relevant_count` parameter defaults to `0`, preserving E0–E4's exact prior behavior —
verified by the existing Milestone 4/5 tests passing unchanged.)

---

## 6. Tests Added

| File | Covers |
|---|---|
| `main/tests/unit/test_category.py` | `get_filtered_collections()`: none/hard/soft policies, unknown-category fallback, unknown-policy rejection |
| `main/tests/unit/test_label_lookup.py` | `resolve_label_collections()`: finds ids across collections, stops early once everything is found, omits unresolvable ids; `count_relevant_excluded_by_filter()`: only counts relevant+excluded items, ignores unresolved products, zero when nothing was filtered out |
| `main/tests/unit/test_category_filter_experiment_runner.py` | `evaluate_query()`'s `excluded_relevant_count` stays backward-compatible (defaults to 0); `run_category_filter_experiment()` rejects unknown experiment_id, non-`"bbox"` mode, an invalid bbox policy, nonzero padding, a category-filter-policy mismatch, and `dedupe: false`; accepts any of the four valid bbox policies; a happy-path run threads a real exclusion count through to `relevant_exclusion_rate` |

---

## 7. Exact Validation Commands and Results

```bash
python -m compileall -q main/evaluation main/tests main/retrieval
```
Result: exit 0, no output.

```bash
cd main && python -m pytest tests/unit -q
```
Result: `145 passed` (124 from Milestones 1–5 + 21 new). Zero failures, zero skipped.

Test environment: Python 3.13.1, pytest 9.1.1, on Windows. `torch`, `chromadb`, `transformers`,
`open_clip` remain **not installed**; `run_category_filter_experiments.py`'s real
detection/embedding/ChromaDB calls are exercised only through fakes (`FakeCollection`/
`FakeClient` for label lookup, direct-call tests with fake pipelines for the experiment runner).

---

## 8. Known Limitations

- **Never run against a live detector/ChromaDB/dataset**, same root cause as every milestone
  since M4: the dataset is still an empty scaffold and the ML/DB dependencies aren't installed
  here.
- **No Predicted Category path** (see section 3) — only Oracle Category filtering is
  implemented. Building a category predictor and re-running E5/E6 (or new experiments) against
  it is future work, not silently substituted here.
- **`highest_confidence` in the E5/E6 configs is a placeholder**, explicitly marked as such in
  each config's `_bbox_policy_note` field — not a claim that it's the best bbox policy.
- **No comparison across E5/E6 (or against E1–E4) is made or claimed.** Deciding whether hard
  or soft filtering helps, or hurts, requires the real dataset and is Milestone 8+ territory.
- No performance numbers, benchmark results, or "which filter is better" claims are made
  anywhere in this record or in the code — none exist yet, and none were fabricated.

---

## 9. Commit SHAs

```
0b5828a  feat: add category filter policies (none/hard/soft)
67af79c  feat: add E5-E6 category-filter ablation experiment runner
a53a5be  test: cover category filtering and E5-E6 experiment runner
```
(This evidence document is added in a fourth, following commit — see the PR for its exact SHA.)

## 10. PR

_(filled in after the PR is opened)_
