# Milestone 10 Padding Sweep — Testing ADR-004's Unresolved `padding_ratio`

**Branch:** `feat/m10-final-decision-regression`
**Repository:** fork `leedoming/rewearlab-ai-project` (upstream: `GeeYun086/rewearlab-ai-project`)
**Related:** `docs/evidence/milestone-10-letterbox.md` (the other pants-anomaly hypothesis tested,
which didn't hold up); `ADR-004` (padding_ratio, previously `TBD`, never tested against real data).

---

## 1. Objective

After the letterbox ablation ruled out aspect-ratio distortion as the dominant explanation for the
Q001-vs-Q012 pants anomaly, the user's next concrete question: does a small context margin around
the bbox crop (instead of a razor-tight 0%) help the model discriminate similar garments better,
without reintroducing enough background noise to hurt? `padding_ratio` has existed in the codebase
since Milestone 1 (`crop_image`'s own parameter) but was never swept against real data — ADR-004
has stood at `TBD` this entire project.

---

## 2. Method

Swept `padding_ratio ∈ {0%, 10%, 20%, 30%, 50%}` on the `category_confidence` bbox policy (the
policy ADR-002 already found best among the four). For each value, built a **separate** ChromaDB
catalog (catalog and query embedded with the same padding — apples-to-apples) from the same source
images in `catalog_manifest.json`, then computed NDCG@10/Precision@5/Recall@10/MRR directly via
`evaluation.metrics` functions with `ideal_relevances=list(query.labels.values())` (the query's
full pooled label set — matching `evaluate_query`'s exact convention; the strict `run_bbox_experiment`
/`run_padding_experiment` runners in `evaluation/evaluator.py` deliberately reject any padding_ratio
outside their spec-pinned values, so this pilot script computes metrics directly instead, the same
pattern other non-spec pilot scripts in this project use).

**Two real infrastructure issues were caught during this sweep, not shipped:**

1. **A metric bug in the first draft of this script**: it called `ndcg_at_k(relevances, 10)` without
   passing `ideal_relevances`, silently falling back to `ndcg_at_k`'s documented convention of
   treating the retrieved top-10 as the entire universe of relevant items. For a query like Q001
   (5 labeled-relevant items across the full catalog, only 1 of which lands in the top 10), this
   *inflates* NDCG by understating what a perfect ranking would look like (0.301 measured vs. 0.102
   the correct way). Caught by noticing the padding=0% result didn't match the already-known
   official E3 result for the same catalog and policy — it should have been identical and wasn't.
   Fixed by passing the full label set, exactly as `evaluate_query` does.
2. **The same ChromaDB HNSW flakiness from `milestone-10-cluster-validation.md` recurred** — this
   time on a freshly single-`add()`-built catalog (`padding_ratio=0.3`), silently zeroing out both
   pants queries (Q001, Q002) after the metric bug fix made it visible as a suspicious exact 0.0 for
   both. **This means "always `add()` once" is not a sufficient fix on its own** — the underlying
   ChromaDB local persistence issue can recur even without incremental growth. Added
   `heal_flaky_collections()` to `run_padding_sweep.py`: after opening/building each catalog, every
   collection is queried 3 times with one of its own stored embeddings, and any collection that
   fails even one of those checks is rebuilt (delete + recreate + single bulk `add()` from its own
   stored embeddings) before any real evaluation query runs. This is a pilot-script-local mitigation,
   not a fix to the underlying ChromaDB behavior, which remains unexplained.

---

## 3. Results

### Overall (N=14, mean NDCG@10)

| padding_ratio | 0% | 10% | 20% | 30% | 50% |
|---|--:|--:|--:|--:|--:|
| Overall | 0.552 | **0.562** | **0.567** | 0.541 | 0.501 |

A small, real peak at 10-20%, degrading by 30% and clearly worse by 50%. Nowhere near closing the
gap to RAW (0.640 on this same N=14 set, `milestone-10-cluster-validation.md`).

### `pants` specifically — the category this was aimed at

| padding_ratio | 0% | 10% | 20% | 30% | 50% |
|---|--:|--:|--:|--:|--:|
| pants (mean of Q001/Q002/Q012) | 0.222 | **0.327** | 0.235 | 0.182 | 0.208 |
| Q001 alone | 0.102 | **0.388** | 0.121 | 0.0 | 0.121 |
| Q002 alone | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Q012 alone | 0.564 | **0.593** | 0.583 | 0.545 | 0.504 |

**10% padding gives a real, meaningful lift specifically to `pants`** — Q001 nearly quadruples
(0.102 → 0.388), Q012 also improves slightly, and it's the single best padding value for this
category by a wide margin over every other value tested. Q002 stays at exactly 0.0 regardless of
padding, confirming (again) that its problem is a genuine catalog coverage gap
(`milestone-10-catalog-rebalance.md` section 3), not something any crop treatment can fix.

### Other categories: no clear win, mild cost

| padding_ratio | 0% | 10% | 20% | 30% | 50% |
|---|--:|--:|--:|--:|--:|
| top | **0.574** | 0.527 | 0.533 | 0.456 | 0.457 |
| outer | 0.449 | 0.486 | 0.464 | **0.518** | 0.458 |
| dress_skirts | 0.854 | 0.828 | **0.928** | 0.911 | 0.797 |

`top` is clearly best at 0% and monotonically worse with more padding — for tops, a tight crop is
already right, and extra margin mostly adds background/other-garment noise. `outer`/`dress_skirts`
are noisy (small N: 2 and 4 queries respectively) without a clean trend.

---

## 4. Interpretation

Unlike the letterbox ablation (a clean rejection), this one is a **real, if modest, positive
result for `pants`, and a wash-to-mild-negative everywhere else**. The pattern is consistent with
what the user's original intuition was aiming at: a razor-tight crop with 0% margin cuts off
information (drape, hem length, exact rise/fit context) that matters for discriminating between
visually-similar pants, but this is category-specific — `top`'s tighter crops were already
adequate, and more padding just adds noise there. A single global `padding_ratio` cannot be optimal
for every category simultaneously; a per-category padding value is the more defensible design if
this were pursued further, not a single blanket constant.

This does **not** close the pants gap to RAW (0.327 vs. RAW's 0.435-0.478 from earlier N=14 runs)
or reverse ADR-001. It is a real, quantified partial improvement worth keeping in mind, not a
decided pipeline change.

---

## 5. Decision

**Not adopted into `final_config.yaml`** (`padding_ratio` stays `TBD`) — a 5-point, N=14 sweep is
not enough to justify a specific value, especially one that helps one category and hurts another.
Recorded as real evidence for a future per-category padding decision, should `pants` retrieval
quality become a priority to fix at the pipeline-config level rather than via a better embedding
model (`milestone-10-cluster-validation.md`'s standing recommendation).

---

## 6. Reproduction

```
cd main
python -m pytest tests/ -q                     # 233 passed (no code changes in retrieval/, pilot-script only)
python evaluation/pilot/run_padding_sweep.py    # writes catalog_db_pad0XX/ and padding_sweep_summaries.json
```
