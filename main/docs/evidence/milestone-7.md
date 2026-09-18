# Milestone 7 — Padding Ablation (E7–E8): Evidence Record

**Branch:** `feat/m7-padding-ablation` (created from `main` after PR #1–#6 were merged)
**PR:** https://github.com/leedoming/rewearlab-ai-project/pull/7
**Repository:** fork `leedoming/rewearlab-ai-project` (upstream: `GeeYun086/rewearlab-ai-project`)

---

## 1. Objective

Per `IMPLEMENTATION_SPEC.md` section 69 ("Milestone 7 — Padding / Ablation: E7~E8") and
section 33's ablation matrix: build the infrastructure to run E7 (BBOX, padding=10%, bbox
policy and category filter policy both held at "Best") and E8 (RAW, category filter pinned to
"Soft") against the dataset. `crop_image()`'s `padding_ratio` support already existed from
Milestone 1 (`main/retrieval/preprocessing.py`) — this milestone's actual work is the
evaluation-layer plumbing: config validation, exclusion-rate accounting, and a CLI runner, not
new cropping logic.

---

## 2. Design Decisions

### 2.1 E7 accepts any valid bbox policy AND any valid category filter policy

**Decision:** `run_padding_experiment()` validates `preprocessing.bbox_policy` is one of the
four valid bbox policies and `category_filter.policy` is one of the three valid filter
policies — neither is pinned to a single hardcoded value.

**Reason:** `IMPLEMENTATION_SPEC.md` section 33 lists E7's BBox Policy *and* Filter columns
both as "Best." As documented in Milestone 5's and Milestone 6's evidence docs, no ablation
result has actually been run against real data yet, so neither "best" is known.
`run_category_filter_experiment()` (Milestone 6) already established the pattern of accepting
any valid policy rather than declaring one "optimal" without evidence (section 67 rule 7) —
`run_padding_experiment()` applies the identical pattern to both of E7's two undetermined
dimensions. The only dimension E7's ablation table pins concretely is `padding_ratio = 0.1`,
so that is the one value this function rejects any deviation from.

**Verification:** `test_run_padding_experiment_accepts_any_valid_bbox_policy`,
`test_run_padding_experiment_accepts_any_valid_category_filter_policy`,
`test_run_padding_experiment_rejects_invalid_bbox_policy`,
`test_run_padding_experiment_rejects_invalid_category_filter_policy`.

### 2.2 E7 rejects `padding_ratio = 0.0`, not just non-numeric garbage

**Decision:** `run_padding_experiment()` treats `padding_ratio != 0.1` as an error, including
the valid-but-wrong value `0.0`.

**Reason:** `0.0` is a legitimate `padding_ratio` for every other bbox experiment (E1–E6) — a
config that silently defaulted to `0.0` here would produce a JSON output that *looks* like a
real E7 result but is actually a duplicate of an E1–E6 run, defeating the entire purpose of a
padding ablation experiment without raising any error. Padding is the one variable this
experiment exists to isolate (section 34: one variable at a time), so it's the one value
validated most strictly, unlike bbox/filter policy which are deliberately left open (2.1).

**Verification:** `test_run_padding_experiment_rejects_zero_padding`.

### 2.3 E8 pins `category_filter.policy` to `"soft"` concretely, unlike E7

**Decision:** `run_raw_filter_experiment()` requires `category_filter.policy == "soft"`
exactly (an error for `"none"`, `"hard"`, or any other value) — the opposite validation
strategy from E7's permissive one (2.1).

**Reason:** `IMPLEMENTATION_SPEC.md` section 33 lists E8's Filter column as the concrete value
"Soft," not "Best" — this is a deliberate spec choice, not an oversight to be smoothed over by
reusing E7's permissive validation. Distinguishing "the spec pins this value" (E8's filter,
E7's padding) from "the spec says 'Best' and no evidence exists yet" (E7's bbox/filter) in the
validation logic itself means a future reader can tell which experiment values are load-bearing
spec requirements and which are open questions, without cross-referencing the spec by hand.

**Verification:** `test_run_raw_filter_experiment_rejects_filter_policy_mismatch`.

### 2.4 E8 constructs its own "raw" preprocessing metadata; the pipeline never supplies it

**Decision:** `run_raw_filter_experiment()`'s `pipeline(query)` contract returns
`(results, excluded_relevant_count)` — no preprocessing metadata parameter at all. The function
itself hardcodes `{"mode": "raw", "selected_bbox": None, "fallback_used": False,
"fallback_reason": None}` for every record.

**Reason:** E8 never runs detection or bbox selection (`preprocessing.mode` is validated as
`"raw"` in 2.3) — there is no real metadata for a pipeline to report, only fabricated-looking
placeholders a caller *could* mistakenly invent (e.g. copy-pasting a bbox experiment's
pipeline and forgetting to clear `selected_bbox`). Constructing the metadata inside
`run_raw_filter_experiment()` itself, the same way `run_baseline()` does for E0, makes that
class of mistake structurally impossible rather than relying on the pipeline author to remember.

**Verification:** `test_run_raw_filter_experiment_never_fabricates_bbox_metadata`.

### 2.5 E7/E8 share the same `label_collections_cache`, mirroring E5/E6's fix

**Decision:** `run_padding_ablation_experiments.py` builds one `label_collections_cache = {}`
dict and threads it through both E7's and E8's pipelines via a shared
`_resolve_exclusion_count()` helper.

**Reason:** This is the exact pattern fixed on PR #6's review round 1 for E5/E6 (see
`main/docs/evidence/milestone-6.md` section 4) — a query's ground-truth labels' true ChromaDB
collection doesn't change between E7 and E8 any more than it changes between E5 and E6, so
resolving it twice (once per experiment) would repeat the same redundant ChromaDB lookup this
milestone's own review process already flagged as wasteful once before. Applying the lesson
immediately here, rather than shipping the same mistake again and fixing it in a second review
round, is the point of writing that finding into the evidence record in the first place.

---

## 3. What This Milestone Does NOT Decide

Same limitation as Milestone 6 section 3 and section 4 of `run_category_filter_experiment`'s
docstring: **no bbox policy or category filter policy has been declared "best."** E7's config
(`e7_padding_ablation.json`) uses `highest_confidence`/`hard` as explicitly-labeled placeholders
(`_bbox_policy_note`/`_category_filter_note` fields) — not claims about what performs best, just
values that have to be something until Milestone 8/10 produce real ablation evidence.

---

## 4. Fixes Made

None — this is new infrastructure; no existing behavior was changed. `crop_image()`'s
`padding_ratio` parameter (Milestone 1) required no changes; this milestone only adds the
evaluation-layer config validation and CLI wiring around it.

---

## 5. Tests Added

| File | Covers |
|---|---|
| `main/tests/unit/test_padding_experiment_runner.py` | `run_padding_experiment()` (E7): rejects unknown experiment_id, non-`"bbox"` mode, invalid bbox policy, zero padding, invalid category filter policy, `dedupe: false`; accepts any of the four valid bbox policies and any of the three valid filter policies; a happy-path run threads a real exclusion count through. `run_raw_filter_experiment()` (E8): rejects unknown experiment_id, non-`"raw"` mode, a filter-policy mismatch (anything but `"soft"`), `dedupe: false`; a happy-path run never fabricates bbox metadata and threads a real exclusion count through. |

---

## 6. Exact Validation Commands and Results

```bash
python -m compileall -q main/evaluation main/tests main/retrieval
```
Result: exit 0, no output.

```bash
cd main && python -m pytest tests/unit -q
```
Result: `159 passed` (145 from Milestones 1–6 + 14 new). Zero failures, zero skipped.

Test environment: Python 3.13.1, pytest 9.1.1, on Windows. `torch`, `chromadb`, `transformers`,
`open_clip` remain **not installed**; `run_padding_ablation_experiments.py`'s real
detection/embedding/ChromaDB calls are exercised only through fakes (direct-call tests with
fake pipelines for both experiment runner functions).

---

## 7. Known Limitations

- **Never run against a live detector/ChromaDB/dataset**, same root cause as every milestone
  since M4: the dataset is still an empty scaffold and the ML/DB dependencies aren't installed
  here.
- **`highest_confidence`/`hard` in the E7 config are placeholders**, explicitly marked as such
  in `_bbox_policy_note`/`_category_filter_note` — not a claim that either is the best policy.
- **No comparison across E7/E8 (or against E0–E6) is made or claimed.** Deciding whether 10%
  padding or RAW+soft-filter helps, or hurts, requires the real dataset and is Milestone 8+
  territory.
- No performance numbers, benchmark results, or "which config is better" claims are made
  anywhere in this record or in the code — none exist yet, and none were fabricated.

---

## 8. Commit SHAs

```
803d12d  feat: add E7-E8 padding ablation experiment runner
32439c1  test: cover E7-E8 padding ablation experiment runner
```
(This evidence record is itself added in a further, following commit — see the PR for its exact SHA.)

## 9. PR

https://github.com/leedoming/rewearlab-ai-project/pull/7
