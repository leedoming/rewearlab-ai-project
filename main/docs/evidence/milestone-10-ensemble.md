# Milestone 10 Ensemble Experiment — Combining RAW + BBox Query Representations

**Branch:** `feat/m10-final-decision-regression`
**Repository:** fork `leedoming/rewearlab-ai-project` (upstream: `GeeYun086/rewearlab-ai-project`)
**Related:** proposed directly from the Q001-vs-Q012 divergence in
`docs/evidence/milestone-10-cluster-validation.md` (both pants queries, opposite winners between
RAW and bbox) and `milestone-10-padding-sweep.md` (a single global preprocessing choice can't be
optimal for every category/query).

---

## 1. Objective

Every prior ablation in this project (ADR-001, the letterbox test, the padding sweep) picks *one*
query-side preprocessing for every query. But the evidence keeps showing that's the wrong frame:
Q001 wants RAW (NDCG 0.951 vs bbox's 0.102) and Q012 wants bbox (0.564 vs RAW's 0.355) — opposite
winners, same category. The user's proposal: instead of choosing one policy for everyone, combine
both query representations (RAW full image + `category_confidence` bbox crop) at retrieval time
and see whether the combination captures whichever signal is stronger per query, without having to
decide in advance.

---

## 2. Method

For each of the 14 golden-set queries, both a RAW and a `category_confidence`-cropped embedding of
the query were computed, then compared against the **same, already-built catalog** (`catalog_db/`,
237 items — no rebuild needed, no new flakiness risk introduced) via **exact, brute-force cosine
similarity** (`collection.get(include=["embeddings"])` + plain numpy — deliberately not ChromaDB's
`.query()`, sidestepping the HNSW flakiness documented in the last two evidence docs; this catalog
is small enough that brute force costs nothing). Three fusion rules were tested per catalog item:

- **max**: `max(sim_raw, sim_bbox)` — take whichever view is more confident about this item.
- **mean**: `(sim_raw + sim_bbox) / 2` — average the two similarity scores.
- **RRF** (reciprocal rank fusion): `1/(60+rank_raw) + 1/(60+rank_bbox)` — combine by each view's
  *rank* rather than its raw score, standard in IR when two scores aren't on a guaranteed-comparable
  scale (Cormack et al. 2009's constant of 60).

`raw` and `bbox` alone were also recomputed the same brute-force way as a sanity check: they
reproduced the already-known official numbers (`raw` 0.640, `bbox` 0.552 vs. the official 0.640/0.552
from `milestone-10-cluster-validation.md`) — confirms the brute-force reimplementation is faithful.

---

## 3. Result: no fusion beats plain RAW overall

### Overall (N=14, mean NDCG@10)

| raw | bbox | fused (max) | fused (mean) | fused (RRF) |
|--:|--:|--:|--:|--:|
| **0.640** | 0.552 | 0.611 | 0.579 | 0.572 |

Every fusion variant lands *between* RAW and bbox, closer to RAW, but none beats RAW outright.

### Per-query (NDCG@10) — where fusion helps and where it doesn't

| Query | raw | bbox | max | mean | RRF |
|---|--:|--:|--:|--:|--:|
| Q001 | **0.951** | 0.102 | 0.399 | 0.530 | 0.530 |
| Q006 | 0.840 | 0.160 | **0.890** | 0.541 | 0.192 |
| Q009 | **1.000** | 0.840 | 0.840 | 0.882 | 1.000 |
| Q012 | 0.355 | **0.564** | 0.564 | 0.430 | 0.387 |
| Q014 | 0.906 | 0.956 | 0.916 | **0.963** | 0.962 |

Fusion *does* occasionally win outright — `max` beats both individual views on Q006 (0.890 vs.
0.840/0.160), `mean` edges out both on Q014 (0.963 vs. 0.906/0.956) — so the underlying intuition
(the two views carry complementary information) is real and sometimes pays off. But **Q001 is the
clearest counter-case**: RAW's near-perfect 0.951 collapses to 0.399-0.530 under every fusion rule.
None of the fusion rules can tell, item by item, "trust RAW's opinion here" vs. "trust bbox's
opinion here" — `max`/`mean` blend in bbox's *wrong* candidates (bbox's own top matches for Q001
are largely irrelevant, per `milestone-10-cluster-validation.md`), and RRF's rank-based combination
has the same problem in a different form (an item ranked #1 by RAW but poorly by bbox still gets
diluted by bbox's competing #1).

---

## 4. Interpretation

**Naive score/rank fusion isn't the fix.** The core problem this experiment surfaces: fusion helps
when both views roughly agree on the *ranking* but differ on fine details (Q006, Q014), but hurts
when the two views fundamentally disagree about which items are relevant at all (Q001, where bbox's
crop apparently loses the discriminating signal RAW's full-body context captures — an open question,
not resolved here). A fixed 50/50 (or any single global) combination rule can't distinguish these
two situations query-by-query; it needs either (a) a *learned* combiner (e.g. a per-query confidence
signal deciding how much to trust each view), which is real ML-engineering scope beyond a pilot
sweep, or (b) accepting that RAW alone is still the best *simple* choice, consistent with every
ADR-001 result so far.

---

## 5. Decision

**Not adopted.** RAW alone (ADR-001's existing decision) remains the better simple choice on this
evidence — no fusion rule tested here beats it overall, and the one query it loses badly on (Q002,
NDCG 0.0 under every method including fusion) is a genuine catalog coverage gap, not something any
query-side combination can fix. This does not reverse or weaken ADR-001. If ensemble retrieval is
revisited later, a learned or confidence-weighted combiner (not a fixed max/mean/RRF rule) is the
more promising direction, given `max` and `mean` each won on a *different* query here — the ceiling
looks real, a fixed rule just can't reach it.

---

## 6. Reproduction

```
cd main
python -m pytest tests/ -q                        # 233 passed (no retrieval/ code changes, pilot-script only)
python evaluation/pilot/run_ensemble_experiment.py # reads catalog_db/ directly, writes ensemble_summaries.json
```
