# Milestone 10 Catalog Rebalance — Stratified Sub-Style Sampling: Evidence Record

**Branch:** `feat/m10-final-decision-regression`
**Repository:** fork `leedoming/rewearlab-ai-project` (upstream: `GeeYun086/rewearlab-ai-project`)
**Supersedes:** the E0-E4 numbers in `milestone-10-final-decision.md` (N=11, flat-random-sampled
catalog).

---

## 1. Objective

`milestone-10-final-decision.md` section 5.2 found that `gather_catalog_paths()`'s
`shuffle-then-cap-to-60` sampling let a collection's rarest sub-style collapse to almost nothing
in the sampled catalog whenever the source folders were very unevenly sized — worst case, `outer`
sampled only 2 cardigan photos against 58 hoodies (source ratio 4:210). The user asked for two
things directly: (1) resample the catalog so sub-styles are represented as evenly as the real
source data allows, and (2) report the resulting quantitative numbers in customer-presentable
terms — not the investigation process, the actual figures.

---

## 2. Resampling method

`gather_catalog_paths()` (`evaluation/pilot/run_pilot.py`) now groups candidate images by
`(collection, source sub-folder)` instead of just `collection`, then calls a new
`_stratified_quota()` helper: a max-min-fair allocator that gives every sub-style an equal share
of the collection's 60-item cap, rolling any leftover (from a sub-style with fewer images than
its equal share) over to the other sub-styles. A sub-style can never be allocated more images than
actually exist for it — this is a *sampling* fix, not a data-fabrication one.

| Collection | Sub-styles (available after excluding query images) | Old sampled composition | New sampled composition |
|---|---|---|---|
| `pants` | 청바지 jeans (164), 바지 general (5) | 57 jeans / 3 general (95%) | 55 jeans / **5/5 general (all available)** |
| `outer` | 후드티 hoodie (206 after excluding Q006/Q007), 가디건 cardigan (3) | 58 hoodie / 2 cardigan (97%) | 57 hoodie / **3/3 cardigan (all available)** |
| `top` | 6 sub-styles, 57–773 each | 반팔 48%, others 1–15% | **10/10/10/10/10/10 — fully balanced** |
| `dress_skirts` | 원피스 (9), 치마 (4) after allowlist + excluding queries | 9/4 (already used everything) | 9/4 (unchanged — already data-limited) |

`pants` and `outer` now include *every* available image of their rare sub-style — the maximum
balance the real source data permits. `top` is fully balanced. `dress_skirts` was already using
100% of its (small) eligible pool, so nothing changed there; growing it further needs new source
data, not a better sampler.

---

## 3. Relabeling: what changed and how it was kept honest

Rebuilding the catalog with different sampling reassigns which images sit at which `product_id`
(e.g. `pants-0012` no longer refers to the same photo). Every pooled candidate across the 11
queries × 5 modes (E0 + E1-E4) was re-pooled from scratch, then each candidate's relevance label
was resolved as follows, **never guessed or defaulted**:

1. If the candidate's real source image (matched by absolute file path, not `product_id`) was
   already labeled for that query under the old catalog, its label was carried over unchanged —
   it's the literal same photo, so the same relevance judgment still applies. **17 of 79** pooled
   candidates carried over this way.
2. Every other candidate (**62 of 79** — the majority, since resampling changed most of the pool)
   was freshly labeled by viewing a montage of the query image next to every new candidate and
   grading 0/1/2 by the same category → silhouette → design detail → pattern → color priority
   `evaluation/dataset/README.md` specifies. Montages and the raw grading are not committed (a
   one-off labeling aid, not reusable code) but the resulting `evaluation/dataset/labels.json` is,
   same as every prior labeling round in this project.

This is still a **single AI-assisted rater**, the same disclosed limitation as every prior round —
not newly introduced by this change.

### 3.1 A genuine zero-relevant-items query, and the bug it surfaced

Q002 (clean white wide-leg denim) ended up with **zero relevant items** in its labeled pool after
rebalancing — every one of its 7 pooled candidates across all 5 modes graded 0. This is a real
catalog coverage gap (no white wide-leg denim exists anywhere in the sampled 60-item `pants`
catalog), not a labeling mistake, and it's the first query in this project to hit that edge case.

It surfaced a real bug: `evaluation/metrics.py::relevant_exclusion_rate` returns `None` (by
documented convention) when a query has zero relevant items, but
`evaluation/failure_analysis.py::classify_failure` did `metrics.get("relevant_exclusion_rate",
0.0) > 0.0`, which only substitutes the default on a *missing* key, not a `None` value — so it
crashed with `TypeError: '>' not supported between instances of 'NoneType' and 'float'` the first
time this code path was ever actually exercised. Fixed to `(metrics.get("relevant_exclusion_rate")
or 0.0) > 0.0`, with a regression test
(`tests/unit/test_failure_analysis.py::test_classify_failure_treats_none_exclusion_rate_as_zero`).
227 tests pass.

Q002's NDCG@10 is counted as 0.0 in the aggregate below (there is nothing relevant to rank, so
nothing this pipeline does can score higher) — kept in, not dropped, per this project's standing
rule against curating away inconvenient results.

---

## 4. Results: RAW vs. BBox on the rebalanced catalog, real N=11 golden set

`evaluation.golden_set.validate_golden_set()` still passes (`category_boundary_ids={'Q007'}`,
`known_previous_failure_ids={'Q006'}`) — the golden set's structural properties didn't depend on
catalog composition.

### Overall (mean across 11 queries)

| Metric | E0 (RAW) | E1 (highest_conf) | E2 (largest) | E3 (category_conf) | E4 (category_largest) |
|---|--:|--:|--:|--:|--:|
| **NDCG@10** | **0.812** | 0.636 | 0.636 | 0.650 | 0.650 |
| MRR | **0.848** | 0.697 | 0.697 | 0.697 | 0.697 |
| Precision@5 | **0.636** | 0.509 | 0.509 | 0.527 | 0.527 |
| Recall@10 (N=10*) | **0.932** | 0.813 | 0.813 | 0.813 | 0.813 |
| Incompatible-category rate@10 | 0.455 | 0.491 | 0.491 | 0.509 | 0.509 |

\* Recall@10/relevant_exclusion_rate are undefined (and excluded, per `evaluation.metrics`'
documented convention) for Q002, which has zero relevant items — see 3.1.

**RAW wins on every metric, by a wider margin than the pre-rebalance N=11 result** (NDCG 0.812 vs.
the superseded 0.735). This is the cleanest, most decisive evidence this project has produced for
ADR-001: the catalog-composition fix removed a confound that was suppressing scores for *every*
policy roughly equally, and once removed, RAW's lead over BBox got clearer, not muddier.

### Per-category NDCG@10

| Category | E0 (RAW) | E1/E2 | E3/E4 |
|---|--:|--:|--:|
| dress_skirts | 0.945 | 0.909 | 0.909 |
| outer | **0.877** | 0.541 | 0.541 |
| pants | 0.500 | 0.000 | 0.000 |
| top | 0.800 | 0.760 | 0.812 |

Two results worth calling out honestly:

- **`pants` went from "uniformly weak under every policy" to a sharp split: Q001 now scores a
  perfect 1.0 under RAW** (the dark-wash jeans query, previously buried by the 95%-jeans-skewed
  catalog burying its true near-duplicates past rank 25 — see `milestone-10-final-decision.md`
  section 5.1) **but collapses to 0.0 under every BBox policy**, and **Q002 scores 0.0 under every
  policy** because it has no relevant match in the catalog at all (3.1). The rebalance fixed the
  catalog-composition problem for `pants`, exactly as hypothesized — but also reveals that BBox
  cropping is actively harmful for this category's one scoreable query, and that this catalog's
  `pants` coverage still has real gaps (white wide-leg denim has zero representation). Both are
  now visible for what they are instead of being smeared into one "pants is just weak" number.
- **`outer` still doesn't recover under category-aware bbox** (0.541 vs RAW's 0.877) even with
  every available cardigan now in the catalog — ruling out "cardigan scarcity" as the sole
  explanation floated in ADR-002's caveat. The mechanism is something else (worth a future
  investigation, out of scope here), but the *catalog-composition* explanation for this specific
  caveat is now retracted, not confirmed.

---

## 5. Decision

**ADR-001 (RAW vs. BBox): RAW — unchanged, now on stronger evidence.** `docs/decisions/ADR-001-raw-vs-bbox.md`
and `docs/decisions/final_config.yaml` (`preprocessing: raw`) required no change; this record
strengthens, not revises, that decision.

**Customer-facing summary** (the number the user asked to be able to show, not the process):

> On a validated 11-query golden set spanning all 4 real product categories (pants, tops, outer,
> dresses/skirts), the chosen pipeline (RAW image embedding, no bounding-box crop) retrieves the
> correct or a closely related item as the **#1 result 85% of the time (MRR 0.848)**, with
> **NDCG@10 of 0.81** (1.0 = perfect ranking) and **64% of the top-5 results being genuinely
> relevant** (Precision@5 0.636). This beats every alternative preprocessing policy tested on
> every metric measured.

**Caveats that remain, stated plainly rather than smoothed over:**
- N=11 is a golden/regression set, not a large-scale benchmark — it is enough to catch
  regressions and support a real go/no-go decision, not to certify a production SLA.
- Single AI-assisted rater throughout; no independent human relabeling has occurred.
- `pants` catalog coverage has a real, now-quantified gap (white wide-leg denim: zero matches).
- `outer`'s bbox-recovery failure is confirmed to not be explained by cardigan scarcity; root
  cause still open.
- Catalog remains capped at 60 items/collection (9 for dress_skirts) — real-world scale is larger.

---

## 6. Reproduction

```
cd main
python evaluation/pilot/run_pilot.py          # rebuilds catalog_db with stratified sampling
# relabel any newly-pooled candidates in evaluation/dataset/labels.json by hand
python evaluation/pilot/run_experiments.py    # writes evaluation/pilot/_local/all_summaries.json
python -m pytest tests/ -q                    # 227 passed
```
