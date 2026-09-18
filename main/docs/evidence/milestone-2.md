# Milestone 2 — Metric Foundation: Evidence Record

**Branch:** `feat/m2-metric-foundation` (created from `main` at `ff39fce`, per the
per-milestone rule "start from updated main" — independent of the still-unmerged
Milestone 1 branch)
**PR:** _(filled in after the PR is opened — see section 8)_
**Repository:** fork `leedoming/rewearlab-ai-project` (upstream: `GeeYun086/rewearlab-ai-project`)

---

## 1. Objective

Per `IMPLEMENTATION_SPEC.md` section 69 ("Milestone 2 — Metric Foundation") and section 23
(Phase 6 — Retrieval Metrics): implement the retrieval quality metrics as pure, testable
functions, with unit tests, before any evaluation dataset or experiment infrastructure exists.
This milestone does **not** implement a dataset, an evaluator, or any experiment — those are
Milestones 3+.

Required metrics (section 23): Precision@5, Precision@10, Recall@5, Recall@10, MRR, NDCG@10,
Incompatible Category Rate@K, Relevant Exclusion Rate. Primary metric per section 24 is
NDCG@10 (secondary: MRR, Precision@5, Recall@10) — this milestone implements the metric
functions themselves at a general `@K`, not a decision about which K/metric to report where;
that reporting decision belongs to the evaluator (Milestone 4+).

---

## 2. Design Decisions

### 2.1 No dependency on torch / chromadb / streamlit

**Decision:** `main/evaluation/metrics.py` imports only the standard-library `math` module.

**Reason:** `IMPLEMENTATION_SPEC.md` section 26 explicitly requires this ("Pure evaluation
logic이어야 한다"). Verified directly: `grep -n "^import\|^from" main/evaluation/metrics.py`
shows only `import math`.

**Evidence:** all 34 unit tests for this module pass in this development environment, which
does **not** have `torch`, `chromadb`, `transformers`, or `open_clip` installed (same
environment/verification as Milestone 1 — see `main/docs/evidence/milestone-1.md` section 6).

### 2.2 Functions operate on pre-computed signals, not raw retrieval results

**Decision:** Every function takes plain Python values already reduced to what the metric
needs — a list of relevance grades (`0`/`1`/`2`) in rank order, a list of booleans, or plain
counts — rather than ChromaDB result objects, product metadata, or category strings.

**Reason:** keeps the module domain-agnostic and trivially unit-testable with plain fixtures
(no fakes/mocks needed for ChromaDB or the retrieval pipeline, unlike Milestone 1's
`search.py`/`detection.py` tests). The computation of "is this relevance grade" or "is this
category incompatible" is the evaluator's job (a later milestone), once the dataset/ground
truth format exists.

**Alternative considered and rejected:** Passing full retrieved-result dicts (with
`metadata`, `collection`, etc.) into the metric functions and having them look up relevance
via a ground-truth lookup table passed alongside. Rejected because it would require the
metrics module to know about the ground-truth/dataset schema — which doesn't exist yet (that
is Milestone 3's job) — and because it would make the "pure, dependency-free" unit tests
require constructing realistic-looking fake retrieval results instead of plain integers/bools.

### 2.3 `ndcg_at_k`'s optional `ideal_relevances` parameter

**Decision:** `ndcg_at_k(relevances, k, ideal_relevances=None)`. When `ideal_relevances` is
omitted, IDCG is computed by sorting `relevances` itself in descending order (i.e. treating
the retrieved set as the full universe of relevant items for the query).

**Reason:** A correct NDCG requires the ideal ranking over *all* relevant items for the query,
not just the ones a given ranking happened to retrieve — but that full ground-truth pool
(built via candidate pooling, `IMPLEMENTATION_SPEC.md` section 14) is Milestone 3's
deliverable and doesn't exist yet. Making `ideal_relevances` optional, with a documented,
correct-by-construction fallback (self-sorting the retrieved list), means: (a) the function is
usable and testable today without inventing a fake dataset schema, and (b) once Milestone 3's
evaluator can supply the real pooled `ideal_relevances`, no signature or call-site change is
needed — only the argument value changes.

**Alternative considered and rejected:** Making `ideal_relevances` a required argument.
Rejected because it would force this milestone to also design and stub out the ground-truth
pooling format just to make the function callable, which is explicitly Milestone 3 scope
(`IMPLEMENTATION_SPEC.md` section 69's milestone ordering, and the task instruction "Do not
implement future milestones early").

**Verification:** `test_ndcg_accepts_explicit_ideal_relevances_from_full_ground_truth` in
`test_metrics.py` exercises the non-default path explicitly, confirming the function produces
a strictly-lower NDCG when a ground-truth item was missed by the ranking than when no external
ideal is supplied.

### 2.4 Extending spec's explicit edge-case conventions to the two non-specified metrics

`IMPLEMENTATION_SPEC.md` section 25 only explicitly mandates edge-case behavior for Recall, MRR,
and Precision. `incompatible_category_rate_at_k` and `relevant_exclusion_rate` are new metrics
this milestone had to give a concrete edge-case behavior to, since the spec doesn't state one.

- **Decision:** `incompatible_category_rate_at_k` uses the same "denominator is always K"
  rule as `precision_at_k` (both are "fraction of K retrieved slots meeting some per-item
  condition" metrics — the same reasoning applies).
- **Decision:** `relevant_exclusion_rate` returns `None` when `total_relevant == 0`, the same
  convention as `recall_at_k`'s "No Relevant Item" case (both are undefined, not zero, when
  there's nothing relevant to reason about).

These are documented in the docstrings and covered by tests
(`test_incompatible_category_rate_fewer_than_k_denominator_stays_k`,
`test_relevant_exclusion_rate_no_relevant_items_returns_none`), so the convention is explicit
and auditable rather than an undocumented implementation detail.

---

## 3. Review Findings

No code review has been run against this milestone's diff yet as of this record. This section
will be updated (or a follow-up fix commit + evidence update added) if/when one is performed,
per the same process used for Milestone 1 (see `main/docs/evidence/milestone-1.md` section 3).

---

## 4. Fixes Made

None — this is a from-scratch implementation, not a fix to existing code.

---

## 5. Tests Added

| File | Covers |
|---|---|
| `main/tests/unit/test_metrics.py` | `precision_at_k` (rank 1/2, none, multiple, fewer-than-K, k-larger-than-results, invalid k); `recall_at_k` (rank 1/2, none, multiple, partial, no-relevant-item→None, k-larger-than-results); `mrr` (rank 1/2, none→0, multiple, empty list); `dcg_at_k`/`ndcg_at_k` (manual DCG check, perfect ranking→1.0, graded-relevance reordering penalty, no-relevant→0.0 not None, k-larger-than-results, explicit `ideal_relevances` from a fuller ground truth, invalid k); `incompatible_category_rate_at_k` (none/some/fewer-than-K); `relevant_exclusion_rate` (none/some/all excluded, no-relevant→None, invalid count) |

Also added (fresh on this branch, since Milestone 1's scaffolding lives only on the
still-unmerged `feat/m1-retrieval-core-refactor` branch): `main/tests/__init__.py`,
`main/tests/conftest.py` (adds `main/` to `sys.path`), `main/tests/unit/__init__.py`.

---

## 6. Exact Validation Commands and Results

Run from the repository root (`2025-AI-REWEARLab/`) unless noted.

```bash
grep -n "^import\|^from" main/evaluation/metrics.py
```
Result: `27:import math` — the only import in the file, confirming no
torch/chromadb/streamlit dependency.

```bash
python -m compileall -q main/evaluation main/tests
```
Result: exit 0, no output (no syntax/import errors).

```bash
cd main && python -m pytest tests/unit -v
```
Result: `34 passed in 0.07s`. All 34 tests listed by name in the commit history / CI log;
none skipped, none xfailed.

Test environment: Python 3.13.1, pytest 9.1.1, on Windows. As with Milestone 1, `torch`,
`chromadb`, `transformers`, and `open_clip` are **not installed** in this environment — this
milestone's module has no dependency on any of them, so that is not a limitation for this
particular test run (unlike Milestone 1's `detection`/`search` modules, which needed
fakes/stubs specifically because of this).

---

## 7. Known Limitations

- **No real dataset to validate against.** These metric functions are verified against
  hand-computed arithmetic (e.g. manual DCG calculation, known Precision/Recall fractions),
  not against a real labeled query set — because no such dataset exists yet
  (`IMPLEMENTATION_SPEC.md` Phase 2, Milestone 3). Correctness here means "implements the
  documented formula/edge-case correctly," not "produces validated real-world numbers."
- **`ndcg_at_k`'s self-sorting fallback is an approximation** when `ideal_relevances` is
  omitted (see section 2.3): it can only be as good as the retrieved set itself, and will
  overstate NDCG relative to the true pooled ground truth whenever the ranking missed a
  relevant item entirely. This is a known, documented, and intentional limitation of using
  the metric ahead of the dataset milestone — not a bug.
- **No aggregation logic yet** (mean/median/stdev across queries, category/difficulty
  slicing) — that is `IMPLEMENTATION_SPEC.md` section 37 (Milestone 8+ territory), out of
  scope here.
- **No experiment manifest / caching / versioning** — those are later-milestone
  infrastructure (sections 60-63) that this milestone does not touch.
- No performance numbers, benchmark results, or policy comparisons are claimed anywhere in
  this record or in the code — none exist yet, and none were fabricated.

---

## 8. Commit SHAs

```
8d84688  feat: add retrieval evaluation metrics (Milestone 2)
2934a83  test: cover metrics edge cases
```
(This evidence document is added in a third, following commit — see the PR for its exact SHA.)

## 9. PR

_(To be filled in immediately after the PR is opened.)_
