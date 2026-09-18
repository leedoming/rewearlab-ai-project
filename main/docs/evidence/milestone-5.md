# Milestone 5 — BBox Policy Experiments (E1–E4): Evidence Record

**Branch:** `feat/m5-bbox-experiments` (created from `main` after PR #1–#4 were merged)
**PR:** _(filled in after the PR is opened)_
**Repository:** fork `leedoming/rewearlab-ai-project` (upstream: `GeeYun086/rewearlab-ai-project`)

---

## 1. Objective

Per `IMPLEMENTATION_SPEC.md` section 69 ("Milestone 5 — BBox Experiments: E1~E4") and
section 19 (Phase 4 — BBox Policy Experiments): build the infrastructure to run the four
bbox-selection policies — `highest_confidence` (E1), `largest` (E2), `category_confidence`
(E3), `category_largest` (E4) — against the same dataset, holding every other variable fixed
(no category filter, 0% padding, same `top_k`/`dedupe` as E0), per section 34's "one variable
at a time" rule. This milestone does **not** decide which policy is best, add category
filtering (Milestone 6), or vary padding (Milestone 9).

---

## 2. Problem Found While Wiring This Milestone

### Problem
`category_confidence` and `category_largest` — two of the four policies this milestone must
exercise — silently never matched anything when given a query from the Milestone 3 evaluation
dataset.

### Evidence
`retrieval/category.py`'s `get_allowed_labels(category)` looked up `category` only in
`CATEGORY_LABEL_MAPPING`, which is keyed by the original Musinsa category string ("바지",
"상의", "아우터", "원피스_스커트" — from `main/embedding/musinsa_to_chromadb.py`'s crawl
pipeline). But `main/evaluation/dataset/loader.py` (Milestone 3) constrains every query's
`category` field to a ChromaDB **collection name** instead (`CATEGORIES =
frozenset(COLLECTION_NAMES)` = `{"pants", "top", "outer", "dress_skirts"}`). Verified directly:

```python
>>> get_allowed_labels("outer")
[]          # before the fix — silently empty for every dataset query
>>> get_allowed_labels("아우터")
['top', 'outer']   # the Musinsa-keyed lookup this same category maps to
```

Since `retrieval/preprocessing.py`'s `_filter_by_category` treats an empty allowed-labels list
as "nothing compatible," every E3/E4 query would have silently fallen through to the raw
fallback for **every single query**, making E3 and E4 indistinguishable from a policy that
never selects anything by category at all — not a crash, just silently wrong results that
would have looked like a legitimate (if strange) experiment outcome.

### Decision / Fix
Added `COLLECTION_LABEL_MAPPING` to `retrieval/config.py`, **derived** from the existing
`CATEGORY_LABEL_MAPPING` + `CATEGORY_COLLECTION_MAPPING` (not hand-typed again, so the two
namespaces can't silently drift apart later), and updated `get_allowed_labels()` to check the
Musinsa-keyed mapping first, then fall back to the collection-keyed one. This is additive: any
existing caller using the Musinsa category strings (e.g. `musinsa_to_chromadb.py`) is
unaffected, since that lookup still takes priority.

**Alternative considered and rejected:** Changing the Milestone 3 dataset loader to store
Musinsa category strings instead of collection names. Rejected because the loader's choice is
actually the more useful one for evaluation — `evaluate_query()` (Milestone 4) already compares
`result["collection"] != query.category` directly to compute `incompatible_category_rate`, and
that comparison requires the collection namespace. Making `retrieval.category` bilingual is a
smaller, more local fix than changing the dataset schema and every M4 caller.

### Verification
`main/tests/unit/test_category.py::test_get_allowed_labels_accepts_collection_names_too` and
`main/tests/unit/test_bbox_experiment_runner.py::test_category_confidence_policy_filters_using_dataset_collection_names`
(the latter runs the real `select_bbox(..., policy="category_confidence", category="outer")`
end to end and asserts it actually selects the compatible detection, not just that the label
lookup returns a non-empty list).

---

## 3. Design Decisions

### 3.1 `DetectionCache` makes "same detection output" structural, not conventional

**Decision:** Added `main/evaluation/detection_cache.py::DetectionCache`, a simple
memoize-by-image-path wrapper around the (expensive) detector call, and a dedicated CLI
(`run_bbox_experiments.py`) that builds **one** `DetectionCache` and runs all four experiment
configs against it in a single process.

**Reason:** `IMPLEMENTATION_SPEC.md` section 19 MUST: "모든 policy는 동일한 detection
output에 적용한다. Detector를 policy별로 다시 돌려 결과가 달라지는 구조를 피한다." Running
each of the four config JSON files through a generic single-experiment CLI (one process per
config, as `run_experiment.py` does for E0) would silently violate this — each process would
re-run detection independently, and nothing would catch it if detection turned out to be
non-deterministic (e.g. any future change that makes the detector's post-processing
order-sensitive).

**Alternative considered and rejected:** Documenting "run detection once, reuse the result"
as a runbook instruction instead of enforcing it in code. Rejected because a documentation-only
convention is exactly the kind of thing this refactor project (see Milestone 1) has already
seen fail silently once — a structural guarantee (cache class + one shared CLI) is cheap here
and removes the failure mode entirely, verified in
`test_detection_cache_runs_detector_once_per_unique_image`.

### 3.2 `run_bbox_experiment` is a new function, not a generalized `run_baseline`

**Decision:** `evaluator.py` keeps `run_baseline` (E0-specific) and adds a separate
`run_bbox_experiment` (E1–E4), refactoring only the genuinely shared parts (`_write_experiment_outputs`,
`_select_split`, `_validate_common_retrieval_controls`) into private helpers both call.

**Reason:** E0's `retrieve(query) -> results` and E1–E4's `pipeline(query) -> (results,
preprocessing_metadata)` have different shapes because E0 has no bbox-selection step to report
metadata for (`preprocessing.mode` is unconditionally `"raw"`). Forcing both into one function
with a mode flag would mean scattering `if experiment_id == "E0": ... else: ...` branches
through the query loop itself, rather than keeping each experiment family's config validation
and per-query record shape next to its own call site — the existing `run_baseline` and its
Milestone 4 tests were also left completely untouched, so this milestone provably didn't
regress E0.

### 3.3 Config validation enforces the "one variable at a time" rule for E1–E4

Each of E1–E4's config is validated to have `preprocessing.mode == "bbox"`,
`preprocessing.bbox_policy` matching exactly what that experiment ID means (E1 =
highest_confidence, ..., E4 = category_largest — see `EXPERIMENT_BBOX_POLICIES`),
`preprocessing.padding_ratio == 0.0`, `category_filter.policy == "none"`, and the same
`top_k`/`dedupe` controls as E0. This directly implements section 34 ("한 번에 한 주요
variable만 변경한다") as an enforced check rather than a config-authoring convention — a
config file that accidentally turns on category filtering, or sets padding to something other
than 0%, fails loudly with a specific error instead of quietly running a different experiment
than its filename claims to be.

---

## 4. Review Findings

No code review has been run against this milestone's diff yet. Will be updated (or a follow-up
fix commit + evidence update added) if/when one is performed, per the process used for
Milestones 1–4.

---

## 5. Fixes Made

- Fixed `retrieval/category.py`'s `get_allowed_labels()` to resolve both the Musinsa category
  namespace and the ChromaDB collection-name namespace (see section 2 above). This is the one
  correctness fix in this milestone; everything else is new infrastructure.

---

## 6. Tests Added

| File | Covers |
|---|---|
| `main/tests/unit/test_category.py` | `get_allowed_labels()` now resolves collection names ("pants"/"top"/"outer"/"dress_skirts"), not just Musinsa category strings |
| `main/tests/unit/test_bbox_experiment_runner.py` | `DetectionCache` calls the detector at most once per unique image path across multiple simulated "experiments"; `run_bbox_experiment` rejects an unknown experiment ID, non-`"bbox"` mode, a `bbox_policy` that doesn't match its experiment ID, nonzero `padding_ratio`, non-`"none"` `category_filter.policy`, and `dedupe: false`; a happy-path run preserves real (non-fabricated) `preprocess_image` metadata verbatim through to `query_results.jsonl`; an end-to-end regression test that `select_bbox(policy="category_confidence", category="outer")` actually selects a category-compatible detection |

---

## 7. Exact Validation Commands and Results

Run from the repository root (`2025-AI-REWEARLab/`) unless noted.

```bash
python -m compileall -q main/evaluation main/tests main/retrieval
```
Result: exit 0, no output.

```bash
cd main && python -m pytest tests/unit -v
```
Result: `124 passed` (113 from Milestones 1–4 + 11 new: 1 in `test_category.py`, 10 in
`test_bbox_experiment_runner.py`). Zero failures, zero skipped.

Test environment: Python 3.13.1, pytest 9.1.1, on Windows. `torch`, `chromadb`, `transformers`,
`open_clip` remain **not installed** in this environment (same as every prior milestone) —
`detect_fashion_items`/`load_detection_model`/`load_embedding_model`/ChromaDB calls in
`run_bbox_experiments.py` are exercised only through fakes/direct-call tests
(`DetectionCache` with a fake `detect`, `run_bbox_experiment` with a fake `pipeline`), never
through the real model/DB stack.

---

## 8. Known Limitations

- **The E1–E4 CLI (`run_bbox_experiments.py`) has never actually been run against a live
  detector/ChromaDB/dataset.** Same root cause as Milestone 4: the evaluation dataset
  (`main/evaluation/dataset/labels.json`) is still an empty scaffold (`"queries": {}` — no
  human has collected the 20–30 real queries yet), and `torch`/`chromadb`/`open_clip` aren't
  installed here. `run_bbox_experiment()`'s logic is verified through tests with fakes;
  end-to-end correctness of the actual detection→crop→embed→search pipeline still needs a
  human run once real data and dependencies are available.
- **No comparison across E1–E4 is made or claimed.** This milestone only builds the ability to
  run the four experiments consistently; deciding which bbox policy performs best (or
  producing any Precision/Recall/NDCG numbers) requires the real dataset and is explicitly
  Milestone 8+ (Failure Analysis) / 10 (Final Decision) territory, not this one.
- The `DetectionCache` fix in section 3.1 only prevents *this codebase's* scripts from
  re-running detection per policy; it doesn't (and can't) verify the underlying
  `AutoModelForObjectDetection` inference is itself deterministic across calls — assumed true
  based on it being eval-mode inference with no dropout/randomness, not independently verified
  here.
- No performance numbers, benchmark results, or "which policy is better" claims are made
  anywhere in this record or in the code — none exist yet, and none were fabricated.

---

## 9. Commit SHAs

```
c64606a  fix: resolve category-label lookup for the evaluation dataset's collection-name namespace
a0e8aa1  feat: add E1-E4 bbox-policy ablation experiment runner
bc81e93  test: cover bbox experiment runner and detection cache
```
(This evidence document is added in a fourth, following commit — see the PR for its exact SHA.)

## 10. PR

_(filled in after the PR is opened)_
