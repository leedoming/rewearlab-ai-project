# Milestone 10 Cluster Validation — Densified Relevant Sets (N=14): Evidence Record

**Branch:** `feat/m10-final-decision-regression`
**Repository:** fork `leedoming/rewearlab-ai-project` (upstream: `GeeYun086/rewearlab-ai-project`)
**Extends:** `docs/evidence/milestone-10-catalog-rebalance.md` (N=11) with 3 additional queries.

---

## 1. Objective

The user's follow-up question after the rebalance results: is Precision@5 low partly because the
pooled golden set simply doesn't have 5+ labeled-relevant items for most queries? Checked directly
— yes: mean *achievable* P@5 across the N=11 set is 0.709 (bounded by `min(1, total_relevant/5)`
per query), against an achieved 0.636, i.e. RAW already reaches ~90% of what the label density
allows. The user's proposal: build a small number of queries where the catalog is *deliberately*
stocked with a dense, moderately-similar (not near-duplicate, not disjoint) relevant set, so a
low score can't be blamed on label scarcity — if the pipeline still misses obviously-similar items,
that's a real signal to chase, not a golden-set artifact.

---

## 2. Method: finding real "moderate similarity" clusters, not synthetic ones

For each of `pants`, `top`, `outer` (categories with abundant raw source data — 164-773 images per
sub-style, see `milestone-10-catalog-rebalance.md` section 2), a real anchor image and its natural
neighbors were found by:

1. Reusing the RAW-embedding cache from `evaluation/pilot/hubness_analysis.py` (2,160 real Musinsa
   images, embedded once, cosine similarity computed exactly) — no new embedding pass needed for
   discovery.
2. For ~40 candidate anchors per collection (excluding any image already used as a catalog or
   query source), computing cosine similarity to every other image in the same collection and
   keeping the ones landing in a **moderate band (0.55–0.90)** — high enough to be genuinely
   similar in style, low enough to exclude near-duplicate reposts and to leave room for the
   ranking task to be non-trivial. The anchor with the largest such band (≥14 candidates) was kept.
3. Viewing the anchor + top-16 candidates as a montage and grading each 0/1/2 by the same
   category→silhouette→design detail→pattern→color rubric as every other label in this project —
   not accepted uncritically from the embedding similarity score, which is a *discovery* heuristic,
   not the ground truth.

| New query | Category | Anchor | Relevant (≥1) of 16 pooled candidates |
|---|---|---|--:|
| Q012 | pants | wide-leg medium-wash denim, person wearing | 15/16 |
| Q013 | top | black long-sleeve graphic tee | 12/16 |
| Q014 | outer | light grey plain oversized hoodie | 15/16 |

All 16 candidates per query were also **inserted into the real ChromaDB catalog** (not just used
for labeling) — `pants`/`top`/`outer` each grew from 60 to 76 items — so the retrieval pipeline
actually has a chance to find them, this isn't a labels-only exercise.

`evaluation.golden_set.validate_golden_set()` still passes on the resulting N=14 set with the same
`category_boundary_ids={'Q007'}`, `known_previous_failure_ids={'Q006'}`.

---

## 3. A real infrastructure bug found and fixed along the way

Adding 16 items to an *already-populated* ChromaDB collection via a second, separate `add()` call
(in a new process, after the original 60-item catalog had already been built and persisted)
corrupted that collection's on-disk HNSW index in a specific, reproducible way: **the first query
against the collection in a new process succeeds, and every query after that fails** with `Error
creating hnsw segment reader: Nothing found on disk`. `retrieval/search.py::search_collections`
catches and logs a per-collection failure and continues with the remaining collections by design
(so the run doesn't crash) — but that resilience silently produced a **wrong evaluation run**: the
first full E0-E4 re-run after adding the cluster queries put every query's `pants` results at
`NDCG@10 = 0.0` for Q012 (a pants query!) because `pants` had silently dropped out of every search
after the very first one.

This was caught, not shipped, by reproducing the exact call sequence standalone and noticing the
failure pattern started at the *second* query, not randomly:

```
Q001 collections_returned: {'pants'}          # first query: fine
Q002 collections_returned: {'top', 'outer'}   # every query after: pants silently missing
...
```

**Fix:** rebuilt all 4 collections from a clean single `add()` call each (read back every existing
`(id, embedding, metadata)` triple with `collection.get(include=[...])`, delete the collection,
recreate it, add everything in one call). Re-running the same reproduction script afterward showed
zero errors and `pants` results present in every query that should have them. **Lesson for this
codebase going forward: never incrementally `add()` to an existing persistent ChromaDB collection
across separate process runs — rebuild it in one shot, or defensively rebuild after any incremental
add, before trusting evaluation numbers from it.** The N=11 numbers in
`milestone-10-catalog-rebalance.md` were unaffected (that catalog was always built with exactly one
`add()` call per collection, never grown incrementally), so they stand as reported.

---

## 4. Results

### Overall (mean across 14 queries, clean catalog)

| Metric | E0 (RAW) | E1 | E2 | E3 | E4 |
|---|--:|--:|--:|--:|--:|
| NDCG@10 | **0.640** | 0.420 | 0.489 | 0.552 | 0.552 |

RAW still wins outright with 3 more queries and a larger (76-item) catalog per collection — the
core ADR-001 conclusion is unaffected. The overall mean dropped from N=11's 0.812 mostly because
Q012 (below) pulls it down, not because the added queries are uniformly harder.

### The 3 new densified queries specifically (RAW/E0)

| Query | Category | Relevant available | NDCG@10 | Precision@5 | Recall@10 |
|---|---|--:|--:|--:|--:|
| Q012 | pants | 15/16 | **0.355** | 0.6 | **0.333** |
| Q013 | top | 12/16 | 0.635 | 0.8 | 0.417 |
| Q014 | outer | 15/16 | **0.906** | 0.8 | 0.533 |

**This is the anomaly the user was looking for.** Q012 and Q014 both have 15 of 16 catalog
candidates genuinely relevant — label scarcity cannot explain a low score on either. Yet Q012
scores dramatically worse (NDCG 0.355, only a third of its relevant items recalled in the top 10)
than Q014 (0.906). This is the same `pants` weakness flagged in
`milestone-10-final-decision.md` section 5.1 and `milestone-10-catalog-rebalance.md` section 4 —
but this time it isn't explained away by sparse labels or catalog composition skew: the catalog
now has an abundant, verified-relevant pool of wide-leg denim in various washes (cosine similarity
0.82–0.88 to the query by construction), and RAW retrieval still only surfaces a third of them.
**The fine-grained denim discrimination weakness identified earlier is now confirmed on a set
purpose-built to rule out every other explanation.**

---

## 5. Decision

No change to ADR-001 (RAW still wins). This record adds two things to the evidence base:
1. A methodology for building non-sparse validation queries from real data (reusable for any
   future category), which the user should keep in mind if `pants` embedding quality gets
   revisited — a fine-tuned or denim-specialized embedding model could be validated directly
   against Q012 and expected to substantially beat NDCG 0.355 if it actually helps.
2. A concrete, load-bearing regression risk in the pilot's ChromaDB usage (section 3) — any future
   incremental catalog growth must rebuild the collection, not append to it, or risk silently
   dropping a whole collection from results without any test catching it (existing tests don't run
   against the real persistent catalog).

---

## 6. Reproduction

```
cd main
python -m pytest tests/ -q   # 227 passed
python -c "from evaluation.dataset import load_dataset; from evaluation.golden_set import validate_golden_set; d=load_dataset('evaluation/dataset'); print(validate_golden_set(d.queries, category_boundary_ids={'Q007'}, known_previous_failure_ids={'Q006'}))"
python evaluation/pilot/run_experiments.py
```
