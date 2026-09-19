# Milestone 10 Final Decision — Expanded Golden Set (N=11, 4 Collections): Evidence Record

**Branch:** `feat/m10-final-decision-regression`
**Repository:** fork `leedoming/rewearlab-ai-project` (upstream: `GeeYun086/rewearlab-ai-project`)
**Supersedes:** the E0-E4 numbers in `milestone-10-transform-bug.md` section 3.1 (N=7, 3
collections) with a larger, spec-compliant golden set.

---

## 1. Objective

After `milestone-10-transform-bug.md` corrected a real embedding bug and reversed the RAW-vs-BBox
conclusion, the user asked directly: can this project reach a *clear* final pipeline decision?
The honest answer at that point was no — N=7 covering only 3 of 4 collections is pilot-scale, not
decision-scale. This record closes that specific gap: find real dress_skirts photos (the one
collection with zero coverage everywhere in this project until now), expand the golden set past
section 55's 10-15 query minimum, and re-run the ablation with the bug already fixed.

---

## 2. Finding a real dress_skirts source

Neither `itda` (the source for pants/top/outer throughout this project) nor `Marqo/KAGL`
(considered and rejected in `milestone-10-hubness.md` section 2.1, wrong task shape) had usable
dress/skirt photos. A second, unrelated personal project —
`C:/Users/smn07/Desktop/glacier-project/itda-fashion-detect` — turned out to have 원피스 (dress,
25 images) and 치마 (skirt, 14 images) folders.

**This source needed its own filtering pass.** Unlike `itda`'s folders, several of these
filenames listed many garment types at once (e.g. one literally named
"...원피스블라우스가디건티셔츠후드집업니트청바지..." — dress+blouse+cardigan+tee+hoodie+knit+jeans
all in one title), and viewing them confirmed these are promotional collage banners showing 2-3
different outfits side by side, not single products. Running real detection over all 39 images
(`scan_dress_skirt_source.py`) and keeping only images with ≤2 detections and a `dress`/`bottom`
label present filtered this down to 13 genuine single-product photos automatically — the exact
same "messy vs. clean" detection-count heuristic from `milestone-10-hubness.md` section 2, reused
here for data curation instead of hub analysis. Two more were dropped by hand from the 13 despite
passing that filter (one lingerie-adjacent listing, one the detector mislabeled).

Of the resulting pool: 9 went into the catalog, 4 were held out as new golden-set queries
(Q008-Q011).

---

## 3. Expanded golden set: now spec-compliant

**N=11 queries, all 4 collections, 189-item catalog** (60 pants / 60 top / 60 outer / 9
dress_skirts — dress_skirts is catalog-limited by real source availability, not by choice).

Running `evaluation.golden_set.validate_golden_set()` against this set — designating Q007 (the
grey hoodie) as the `category_boundary` case (outer/top are modeled as adjacent throughout this
project, e.g. `CATEGORY_LABEL_MAPPING["아우터"] = ["top", "outer"]`) and Q006 (the two-cardigan
photo, where earlier pilots first observed the wrong-category-detection failure mode) as
`known_previous_failure` — **passes for the first time in this project's history**:

```
easy_case:             [Q002, Q004, Q009]
hard_case:             [Q001, Q003, Q006, Q011]
multi_item:            [Q006, Q010]
background_heavy:      [Q008]
category_boundary:     [Q007]
known_previous_failure:[Q006]
→ validate_golden_set(...) passes
```

This is the first golden set in the project's ten-milestone history to actually satisfy section
55's requirements, rather than being explicitly documented as a smaller stand-in for one.

---

## 4. Results (N=11, 4 collections, bug-fixed, deterministic)

| Experiment | Policy | NDCG@10 | MRR | Precision@5 | Recall@10 | Incompatible Cat. Rate |
|---|---|--:|--:|--:|--:|--:|
| **E0** | **RAW** | **0.7347** | **0.8283** | **0.5091** | **0.8833** | 0.3727 |
| E1 | highest_confidence | 0.6080 | 0.7455 | 0.4545 | 0.6712 | 0.4091 |
| E2 | largest | 0.5983 | 0.7455 | 0.4545 | 0.6712 | 0.4000 |
| E3 | category_confidence | 0.6592 | 0.7909 | 0.4909 | 0.7621 | 0.3636 |
| E4 | category_largest | 0.6592 | 0.7909 | 0.4909 | 0.7621 | 0.3636 |

**RAW wins outright, by a wider margin than the N=7 result** (0.735 vs. 0.659 for the
category-aware policies — a larger gap than N=7's 0.624 vs. 0.545). This is the same direction as
`milestone-10-transform-bug.md`'s corrected N=7 result, now confirmed on a set that actually meets
section 55's size and coverage requirements.

### Per-category breakdown (NDCG@10)

| Category | E0 (RAW) | E1/E2 | E3/E4 |
|---|--:|--:|--:|
| dress_skirts | 0.928 | 0.888 | 0.888 |
| outer | 0.754 | 0.396 | 0.396 |
| pants | 0.254 | 0.254–0.307 | 0.254 |
| top | 0.785 | 0.576 | 0.799 |

Two things worth flagging honestly, not smoothing over:
- **`top` is where category-aware bbox actually recovers RAW-level quality** (0.799 vs. RAW's
  0.785, vs. category-agnostic's 0.576) — the mechanism identified in `milestone-10-hubness.md`
  (avoiding wrong-collection crops in multi-item photos) is visible here specifically for `top`.
- **`outer` does NOT recover** — E3/E4 score identically to E1/E2 (0.396) for this category, well
  below RAW (0.754). With only 2 outer queries (Q006, Q007) this could be noise, or it could mean
  category-aware selection doesn't help `outer` the way it helps `top` — not resolved here,
  flagged as-is rather than papered over.
- **`pants` scores low across every experiment (0.254–0.307), including RAW.** This is not a
  bbox/preprocessing story at all — it's consistent across every policy. Investigated below.

### 5.1 Follow-up: why is `pants` low under every policy?

Not a labeling artifact and not a bbox/policy story — a real embedding-ranking weakness, found by
re-running Q001/Q002 with `top_k=60` (the full `pants` collection) instead of the usual `top_k=10`,
so the actual rank of every labeled product could be inspected, not just whether it made the
saved top-10:

| Query | Product | Relevance | Rank (of 60) |
|---|---|--:|--:|
| Q001 | pants-0012 | **2** | 25 |
| Q001 | pants-0011 | **2** | 47 |
| Q001 | pants-0007 | **2** | 34 |
| Q001 | pants-0006 | 1 | 2 |
| Q002 | pants-0022 | **2** | 43 |
| Q002 | pants-0031 | 0 | 1 |
| Q002 | pants-0001 | 0 | 3 |

Every relevance=2 (near-duplicate) item is buried past rank 25, while relevance=0/1 items occupy
the top ranks. This lines up with a real, structural cause in the catalog itself:
`FOLDER_TO_COLLECTION` maps *two* source folders to the single `pants` collection —
`musinsa_pants_1000/청바지` (jeans, 164 images) and `musinsa_pants_1000/바지` (general pants, 18
images) — combined then randomly capped to 60. At that ratio the sampled 60-item `pants` catalog
ends up roughly 90% denim jeans. So `pants` isn't testing "find similar pants" so much as
"distinguish this exact jeans wash/fit/length from ~54 other jeans" — a fine-grained
intra-style discrimination task. A general CLIP-family embedding is good at coarse
category/style separation (which is why `dress_skirts`/`outer`/`top` all score well) but weak at
this kind of fine-grained denim discrimination, and that weakness shows up identically under
every bbox policy because none of them touch the embedding model itself.

This is a catalog-composition/embedding-capability limitation, not a preprocessing bug — out of
scope for ADR-001, but worth flagging for anyone growing the catalog: either source a
style-balanced pants set (not jeans-dominated), or accept that fine-grained denim ranking needs a
different embedding model/fine-tuning to work well.

---

## 5. Decision

**ADR-001 (RAW vs. BBox): RAW.** Confirmed on a spec-compliant N=11, 4-collection, deterministic,
reproducible golden set — not just the N=7 pilot. This is the strongest evidence this project has
produced for either side of this question.

**ADR-002 (BBox policy, conditional on BBox being used at all): `category_confidence`/
`category_largest`** — still tied with each other, still ahead of the category-agnostic policies
on the aggregate and on `top` specifically, though not on `outer`. Since ADR-001 now favors RAW
outright, ADR-002's practical relevance is secondary — it matters only if a future decision
revisits RAW vs. BBox with different data.

This is a **real final decision for this project's available data**, not another "provisional."
It is still bounded by what section 8 below discloses (catalog size, single-rater labeling,
random 60-item sampling) — a larger, human-labeled, full-catalog validation would still be the
gold standard — but within this project's honest constraints, RAW is the recommended default and
`docs/decisions/final_config.yaml`'s `preprocessing` field is updated accordingly.

---

## 6. Exact Validation Commands and Results

```bash
cd main && python evaluation/pilot/scan_dress_skirt_source.py
```
Result: 39 dress/skirt images scanned; 14 usable (≤2 detections, dress/bottom label present).

```bash
cd main && python evaluation/pilot/run_pilot.py       # rebuild catalog: 189 items, 4 collections; 11 queries
cd main && python evaluation/pilot/run_experiments.py  # E0-E4, deterministic
```
Result: see section 4.

```bash
cd main && python -c "from evaluation.dataset import load_dataset; from evaluation.golden_set import validate_golden_set; ..."
```
Result: `validate_golden_set()` passes with `category_boundary_ids={'Q007'}`,
`known_previous_failure_ids={'Q006'}`.

```bash
cd main && python -m pytest tests/ -q
```
Result: `226 passed`.

---

## 7. Updated `docs/decisions/final_config.yaml`

```yaml
final_pipeline:
  preprocessing: raw   # ADR-001, confirmed N=11/4-collection re-run
  bbox_policy: TBD      # moot given preprocessing=raw; see ADR-002 if BBox is revisited
  padding_ratio: TBD    # moot given preprocessing=raw

  category_filter:
    policy: TBD          # E5/E6 never run against real data (see Known Limitations)

  detection:
    threshold: TBD       # Milestone 9 sensitivity sweep never run against real data

  retrieval:
    top_k: 10             # matches every experiment's controlled retrieval.top_k
```

---

## 8. Known Limitations

- **Catalog is still 189 items** (60/collection cap, 9 for dress_skirts limited by real source
  availability) — real production catalogs are orders of magnitude larger; this result describes
  behavior at pilot scale, not proof it holds at full scale.
- **dress_skirts catalog (9 items) is far smaller than the other three (60 each)** — its very
  high NDCG (0.888–0.928) may partly reflect a small, easier-to-search candidate pool rather than
  a genuinely stronger retrieval signal for that category specifically.
- **Labeling is still single-rater, AI-assisted** — same disclosed limitation as
  `milestone-10-pilot.md` section 2.6, now covering 11 queries instead of 7.
- **`outer`'s failure to benefit from category-aware bbox (section 4) is based on only 2 queries**
  — not enough to conclude the mechanism doesn't generalize to `outer`, only enough to flag that
  it didn't show up here.
- **E5-E8 (category filtering, padding) were never run against this or any real data** — this
  decision covers preprocessing mode (RAW vs. BBox) and bbox policy only.
- **`category_filter.policy`, `detection.threshold`, `bbox_policy`/`padding_ratio` remain `TBD`**
  in `final_config.yaml` — this record only resolves the RAW-vs-BBox question, per its own scope.
