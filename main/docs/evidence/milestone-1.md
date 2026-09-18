# Milestone 1 — Retrieval Core Refactor: Evidence Record

**Branch:** `feat/m1-retrieval-core-refactor`
**PR:** https://github.com/leedoming/rewearlab-ai-project/pull/1
**Repository:** fork `leedoming/rewearlab-ai-project` (upstream: `GeeYun086/rewearlab-ai-project`)

---

## 1. Objective

Per `IMPLEMENTATION_SPEC.md` section 6 ("Phase 1 — Retrieval Core Refactoring"): separate
retrieval core logic (detection, bbox selection, preprocessing, embedding, ChromaDB search)
from Streamlit/UI code, while preserving existing behavior. This milestone does **not**
introduce an evaluation dataset, metrics, ablation experiments, or a final bbox/category
policy decision — those are explicitly out of scope until later milestones.

Three existing Streamlit scripts duplicated this logic with inconsistent behavior:

| File | Role | Original bbox-selection behavior |
|---|---|---|
| `main/embedding/musinsa_to_chromadb.py` | DB embedding pipeline | category-compatible labels → highest confidence → crop |
| `main/search-app/musinsa_detect.py` | interactive search app | all detections → sorted by area → user manually picks via selectbox |
| `main/main-app/app.py` | combined defect-detection + search + LLM app | all detections → sorted by area → auto-pick largest |

---

## 2. Design Decisions

### 2.1 Package layout

Created `main/retrieval/` with one module per responsibility, matching the boundaries the
spec requires to stay separate: `config.py`, `models.py`, `category.py`, `detection.py`,
`preprocessing.py`, `embedding.py`, `search.py`, `__init__.py`.

### 2.2 Preserve, don't unify, bbox-selection policy per app

**Decision:** Route all three apps through the same `retrieval.preprocessing.select_bbox`
function, but keep each app's own policy explicit (`category_confidence` for DB embedding,
manual user-selection over all detections for search-app, `largest` for main-app) rather than
forcing one shared policy.

**Reason:** `IMPLEMENTATION_SPEC.md` section 3 and the task instructions explicitly said the
two (in practice three) existing behaviors must not be silently unified during this
milestone — comparative evaluation of policies is deferred to later milestones (Phase 4).

**Alternative considered and rejected:** Picking one "best" policy (e.g. `highest_confidence`
everywhere) and migrating all three apps to it. Rejected because no evaluation data exists
yet to justify a choice, and `IMPLEMENTATION_SPEC.md` section 67 explicitly forbids declaring
an "optimal" policy without experiment evidence.

### 2.3 Heavy dependencies are lazily imported inside functions

`torch`, `transformers`, `open_clip`, `chromadb`, and `numpy` are imported inside the
functions that need them (`detection.py`, `models.py`, `embedding.py`, `search.py`), not at
module top level.

**Reason:** none of these packages are installed in this development environment (verified:
`pip show torch/chromadb/open_clip/transformers` all report "not found"; only `streamlit`,
`numpy`, and `pillow` are present). Lazy imports let `main/retrieval/__init__.py` and the
pure-logic modules (`config.py`, `category.py`, `preprocessing.py`) import and unit-test
successfully without those dependencies installed, and satisfy the acceptance criterion
"retrieval core can be imported without importing Streamlit" without also silently requiring
a full ML stack just to import the package.

**Alternative considered and rejected:** Module-level imports (simpler, more conventional)
were rejected because they would make the package fail to import at all in this environment,
blocking both the acceptance-criteria check and the unit test suite.

### 2.4 Explicit fallback policy with recorded metadata

`retrieval.preprocessing.preprocess_image()` returns `(image, metadata)`, where `metadata`
only contains fabricated selection fields (`selected_bbox`, `selected_label`,
`detection_score`, `bbox_area`) when a selection actually happened; a raw fallback leaves
those `None` and instead records `fallback_used` / `fallback_reason`. This directly
implements `IMPLEMENTATION_SPEC.md` section 8 ("Do not fabricate metadata if no selection
occurred") and section 9 (fallback policies: `raw` / `largest` / `fail`, must be recorded).

### 2.5 `search_collections(..., dedupe=...)` instead of two separate functions

**Decision:** One `search_collections()` function with a `dedupe: bool` parameter, rather
than two separately-named functions, to represent search-app's identity-dedup merge vs.
main-app's non-deduping merge.

**Alternative considered:** Two distinct functions/named policies (e.g.
`search_collections_deduped` / `search_collections_raw`, or a `DEDUPE_POLICY` map per app in
`config.py`). Not adopted in the initial implementation; the review (section 3.4 below)
flagged this as a design risk (implicit default, no test coverage), which was addressed by
adding explicit tests pinning both behaviors (see section 5) rather than restructuring the
API, since restructuring would have been a larger change than necessary to fix the actual
risk (untested behavior, not incorrect behavior).

---

## 3. Review Process

After the initial refactor (commits `b63d5c0`..`2679b8c`), a structured code review was run
against the PR diff: 8 independent finder angles (correctness ×3, reuse, simplification,
efficiency, altitude, conventions) followed by a 1-vote verification pass per surviving
candidate, using the repository's `code-review` skill at "high" effort.

### 3.1 Findings reported (post-verification)

| # | Finding | File | Verdict |
|---|---|---|---|
| 1 | `search_collections()` swallowed per-collection query exceptions with no logging | `retrieval/search.py` | CONFIRMED |
| 2 | search-app's no-detection path fabricated a synthetic detection instead of using the new `preprocess_image()` fallback, so no `fallback_used`/`fallback_reason` was recorded for that app | `search-app/musinsa_detect.py` | CONFIRMED |
| 3 | `detect_fashion_items`'s lazy-load guard didn't handle "processor+model supplied, device omitted" — `device` silently stayed `None`, contradicting the docstring | `retrieval/detection.py` | CONFIRMED |
| 4 | Per-collection diagnostic log (item count before querying) was dropped during extraction, with no replacement | `retrieval/search.py` | CONFIRMED |
| 5 | `dedupe` boolean differs by call site (search-app relies on default `True`, main-app passes `False`) with zero test coverage of either behavior | `retrieval/search.py` | CONFIRMED |
| 6 | `numpy`/`chromadb` imports left dead in `main-app/app.py` after the refactor | `main-app/app.py` | CONFIRMED |

### 3.2 Candidates investigated and explicitly rejected (not new regressions)

| Candidate | Why rejected |
|---|---|
| `search_collections()` re-embeds the query image once per collection (redundant CLIP forward passes) when called via `query_image=` | Verified via `git diff` against the pre-refactor code: the original `musinsa_detect.py` already looped over collections and called `collection.query(query_images=[...])` once per collection before this refactor existed. Inherited, pre-existing behavior — out of scope for a "preserve existing behavior" refactor; left unchanged (`0.` — Milestone 1 does not include performance work). |
| Dedup logic collides distinct items into one when ChromaDB metadata lacks both `id` and `product_id` (`product_id` resolves to `None` for all of them) | Verified the pre-refactor `musinsa_detect.py` had the identical collision structure (`item.get('id', '') or item.get('product_id', '')`, colliding on `''` instead of `None`). Also verified `create_metadata()` in `musinsa_to_chromadb.py` always sets both `id` and `product_id` from the same source value and skips products with no ID before that point — so this cannot occur for the current dataset. Inherited, non-triggering for real data — not fixed. |
| The one `preprocess_image()` call site in `main-app/app.py` has no surrounding `try/except` | On inspection, the entire enclosing `else:` block in that file has no try/except anywhere (no `main()` function exists in this script; it is bare top-level code) — every other statement in the same block carries identical risk. Not a defect introduced by, or specific to, this refactor. Not fixed. |

None of the review's findings concerned fabricated benchmark numbers, performance claims, or
policy decisions — this milestone made none of those claims.

---

## 4. Fixes Made

### Finding 1 — Silent collection-search failures

- **Problem:** `except Exception: continue` in `search_collections()` (`main/retrieval/search.py`)
  dropped the exception with no trace, whereas all three original implementations logged
  `logger.error(f"컬렉션 '{collection_name}' 검색 중 오류: {e}")` before continuing.
- **Evidence:** Confirmed via `git diff main...feat/m1-retrieval-core-refactor` that the
  removed lines included the `logger.error(...)` call in both `musinsa_detect.py` and
  `app.py`, and that `retrieval/search.py` had no `logging` import or usage at all before the fix.
- **Decision / Fix:** Added `logger = logging.getLogger(__name__)` to `search.py` (matching
  the existing convention already used in `retrieval/category.py`) and changed the except
  block to `except Exception as exc: logger.error(f"컬렉션 '{collection_name}' 검색 중 오류: {exc}"); continue`.
  Per-collection isolation (one failing collection does not stop the others) was preserved —
  only the silent-drop was fixed.
- **Verification:** `main/tests/unit/test_search.py::test_failed_collection_is_isolated_and_logged`
  — a fake client with one collection that raises and one that succeeds; asserts the good
  collection's results still come back *and* the failure is present in the logs (via `caplog`).

### Finding 2 — search-app fallback bypassed the new metadata tracking

- **Problem:** `musinsa_detect.py`'s `main()` still hand-built a synthetic full-image
  detection (`{'bbox': [0,0,w,h], 'label': 'original', 'score': 0.0, ...}`) when
  `detect_fashion_items()` returned empty, instead of calling the new `preprocess_image()`.
  This meant `fallback_used`/`fallback_reason` were never recorded for this app.
- **Evidence:** `git diff` and direct inspection confirmed the file only imported
  `crop_image` from `retrieval.preprocessing`, never `preprocess_image`.
- **Decision / Fix:** Split `main()`'s post-detection logic into two branches:
  - `if detected_items:` — unchanged manual-selection UX (candidate preview grid,
    selectbox, crop the user's chosen bbox). No behavior change here.
  - `else:` — no candidate list is built. A new `resolve_fallback_image(image, detected_items)`
    helper calls `preprocess_image(image, [], policy="largest", fallback_policy="raw")` and
    the resulting `fallback_used`/`fallback_reason` are logged; the UI shows the same
    "탐지하지 못함, 원본으로 검색" message as before.
  - Constraint respected: the manual-selection UX for the normal (detections-found) case was
    not touched or redesigned.
- **Verification:** `main/tests/unit/test_search_app_fallback.py` — imports the real
  `musinsa_detect` module (with `chromadb` stubbed via `sys.modules`, since it isn't
  installed and isn't needed for this code path) and asserts `resolve_fallback_image([])`
  returns the original image with `mode="raw"`, `fallback_used=True`,
  `fallback_reason="no_detections"`, and that it delegates to `preprocess_image` with the
  expected arguments.

### Finding 3 — `device=None` contract gap in `detect_fashion_items`

- **Problem:** `if image_processor is None or model is None: ... = load_detection_model(...)`
  only resolved `device` when reloading everything. A caller supplying `image_processor` and
  `model` but leaving `device=None` (a documented default) kept `device=None`, contradicting
  the docstring's claim that "if any is None, the model is loaded on demand."
- **Evidence:** Confirmed no current call site (all three apps) triggers this — they all pass
  all three arguments or none — but the function's own documented contract was inconsistent
  with its code, which matters for future milestones reusing this core module.
- **Decision / Fix:** Extracted `_resolve_detection_components(image_processor, model, device, model_name)`
  with an explicit, documented contract:
  - both omitted → lazily load all three via `load_detection_model`;
  - both supplied, device omitted → infer device from the model's own parameters
    (`next(model.parameters()).device`);
  - both supplied, device given → use as-is;
  - exactly one of the two supplied → raise `ValueError` (an inconsistent combination has no
    correct guess).
- **Verification:** `main/tests/unit/test_detection.py` — six cases covering all four
  branches above, using a fake model object (`FakeModel.parameters()` yields a fake tensor
  with a `.device` attribute) so no real torch/transformers install is required.

### Finding 4 — Dropped per-collection diagnostic log

- **Problem:** The original `musinsa_detect.py` logged
  `"컬렉션 '{name}'에서 검색 중... (총 {count}개 아이템)"` before every collection query; this
  was dropped with no replacement when the query loop moved into `retrieval/search.py`.
- **Decision / Fix:** Added the same log line inside `search_collection()` in
  `retrieval/search.py` (the shared module now owns it once, instead of each app duplicating
  it).
- **Verification:** `test_search.py::test_search_collection_logs_item_count_diagnostic`.

### Finding 5 — Untested `dedupe` asymmetry

- **Problem:** No test in the suite exercised `search_collections(..., dedupe=True)` or
  `dedupe=False` at all; a future edit could silently flip either app's behavior.
- **Decision / Fix:** Per the task's explicit instruction, the *behavior* was not changed
  (both apps' existing semantics are preserved), only test coverage was added.
- **Verification:** `test_search.py` — `test_dedupe_true_collapses_duplicate_product_ids_across_collections`,
  `test_dedupe_false_keeps_duplicate_product_ids`, `test_dedupe_true_and_false_both_respect_top_k_truncation`.

### Finding 6 — Dead imports

- **Fix:** Removed `import numpy as np` and `import chromadb` from `main/main-app/app.py`
  (confirmed via grep that neither `np.` nor `chromadb.` appears anywhere else in the file).

---

## 5. Tests Added / Changed

| File | Covers |
|---|---|
| `main/tests/unit/test_bbox_selection.py` | `select_bbox` policies: highest_confidence, largest, category_confidence, category_largest, empty detections, unknown policy |
| `main/tests/unit/test_crop.py` | bbox crop boundary clamping, including padding and a documented pre-existing out-of-bounds edge case |
| `main/tests/unit/test_fallback.py` | `preprocess_image` fallback policies: raw / largest / fail, metadata never fabricated when no selection occurred |
| `main/tests/unit/test_category.py` | category↔label/collection mapping helpers |
| `main/tests/unit/test_core_imports.py` | `retrieval` package imports without Streamlit; public API surface present |
| `main/tests/unit/test_detection.py` *(added during fix round)* | `_resolve_detection_components` argument contract (6 cases) |
| `main/tests/unit/test_search.py` *(added during fix round)* | collection failure isolation+logging, diagnostic logging, dedupe True/False, top-k truncation/ranking |
| `main/tests/unit/test_search_app_fallback.py` *(added during fix round)* | search-app's no-detection path delegates to the shared fallback mechanism |

---

## 6. Exact Validation Commands and Results

Run from the repository root (`2025-AI-REWEARLab/`) unless noted.

```bash
python -m compileall -q main/retrieval main/tests \
    main/embedding/musinsa_to_chromadb.py \
    main/search-app/musinsa_detect.py \
    main/main-app/app.py
```
Result: exit 0, no output (no syntax/import errors).

```bash
cd main && python -m pytest tests/unit -q
```
Result (after initial refactor, before fixes): `28 passed in 0.12s`
Result (after fix commits `e1e0c95`, `de47295`): `41 passed in 1.05s`

Test environment: Python 3.13.1, pytest 9.1.1, on Windows. `torch`, `chromadb`,
`transformers`, and `open_clip` are **not installed** in this environment — confirmed via
`pip show <package>` returning "not found" for each. All 41 tests pass without them because
the pure-logic modules (`config`, `category`, `preprocessing`) and the heavy-dependency
modules (`detection`, `models`, `embedding`, `search`) both lazily import their heavy
dependencies only inside the functions that need them, and every test either exercises pure
logic directly or substitutes a fake/stub for the heavy dependency (fake detection
model/processor, fake ChromaDB client/collection, stubbed `chromadb` module for the
search-app import test).

---

## 7. Known Limitations

- **No real-model/database smoke test.** Because `torch`/`transformers`/`open_clip`/`chromadb`
  are not installed in this environment, the three Streamlit apps were never actually run
  end-to-end against a live model or ChromaDB instance in this session. Correctness there
  rests on: (a) unit tests of the extracted pure logic, (b) `compileall` syntax/import
  validation, and (c) manual line-by-line diff review confirming each app's call sites match
  the new function signatures exactly. This is a real gap — UI-level manual verification by
  a human with the full dependency stack installed is recommended before relying on this in
  production.
- **Pre-existing `crop_image` edge case, not fixed.** A bbox entirely outside the image
  bounds raises `ValueError` in PIL (`crop_image([200,200,300,300])` on a 100×100 image)
  because independently clamping `x1↑`/`x2↓` can leave `x2 < x1`. This bug already existed,
  identically, in all three original implementations. Real detector output never produces
  such a box (it's always within the source image), so it was preserved rather than "fixed"
  mid-refactor without a decision to change behavior. Documented and explicitly tested for
  in `test_crop.py::test_crop_fully_out_of_bounds_bbox_raises` (asserts the `ValueError`,
  rather than silently expecting it to have been fixed).
- **Redundant per-collection query embedding**, inherited from the pre-refactor code (see
  section 3.2) — not addressed in this milestone.
- **`product_id` collision on missing metadata**, inherited from the pre-refactor code (see
  section 3.2) — not addressed in this milestone; does not occur with the current dataset.
- No dataset, metrics, or ablation work exists yet — by design; that begins at Milestone 2.

---

## 8. Commit SHAs

```
b63d5c0  refactor: add shared retrieval configuration
a75e842  refactor: extract detection and bbox selection
9143c30  refactor: extract embedding and search core
e601844  test: cover retrieval preprocessing policies
2679b8c  refactor: extract retrieval core from Streamlit apps
e1e0c95  fix: address retrieval core review findings
de47295  test: cover search fallback and dedupe behavior
```

## 9. PR

https://github.com/leedoming/rewearlab-ai-project/pull/1 — open, not merged, targets the
fork's own `main` (not upstream `GeeYun086/rewearlab-ai-project`).
