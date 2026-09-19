# Milestone 10 Letterbox Ablation — Testing the Aspect-Ratio-Distortion Hypothesis

**Branch:** `feat/m10-final-decision-regression`
**Repository:** fork `leedoming/rewearlab-ai-project` (upstream: `GeeYun086/rewearlab-ai-project`)
**Related:** `docs/evidence/milestone-10-cluster-validation.md` section 4 (the Q001-vs-Q012 anomaly
this investigates).

---

## 1. Objective

`milestone-10-cluster-validation.md` found that a *correctly*-selected `category_confidence` bbox
crop (verified by inspecting the actual cropped image, not just the metadata) still scores far
worse than RAW for Q001 (NDCG 0.102 vs 0.951) while scoring *better* than RAW for Q012 (0.564 vs
0.355) — both are `pants` queries with a correct "bottom" crop. Root-caused to a real, generalizable
mechanism: the embedding model's own preprocessing (`open_clip`'s val transform) is a bare
`Resize((224, 224))` with **no aspect-ratio preservation** — it stretches whatever it's given to
fit a square. A tall, narrow bbox crop (e.g. a full-length pants crop, aspect ratio ≈0.37) gets
stretched far more severely than a roughly-portrait raw photo (≈0.83), which is a plausible reason
bbox crops would systematically hurt pants specifically. The user asked to test this directly
rather than leave it as a hypothesis.

---

## 2. Implementation

- `retrieval/preprocessing.py::letterbox_to_square(image, fill=(128,128,128))`: pads an image to a
  square canvas, centered, with a neutral mid-gray fill (matching common object-crop letterboxing
  practice) — preserves aspect ratio so the model's own resize only scales, never distorts.
- `retrieval/embedding.py::embed_image(..., letterbox=False)`: new opt-in parameter (default off,
  so no existing caller's behavior changes) that applies `letterbox_to_square` before calling the
  model's preprocessing.
- `evaluation/pilot/run_letterbox_experiment.py`: builds a **separate** ChromaDB catalog
  (`_local/catalog_db_letterbox/`, not touching the real `catalog_db/`) from the same source
  images already recorded in `catalog_manifest.json`, with every embedding (catalog *and* query,
  both RAW and every bbox policy) computed with `letterbox=True`, then re-runs E0-E4 against it.
  Both catalog and query use letterboxing consistently — an apples-to-apples ablation, not a
  mismatched one.
- Unit tests: `tests/unit/test_crop.py` (4 new tests for `letterbox_to_square`) and the new
  `tests/unit/test_embedding.py` (2 tests, with a fake model/preprocess so no real model load is
  needed) verify the padding math and that `embed_image` only letterboxes when asked. 233 tests
  pass total.

---

## 3. Result: real effect, but small and inconsistent — not a clean win

### Overall (N=14, mean NDCG@10)

| | E0 (RAW) | E1 | E2 | E3 | E4 |
|---|--:|--:|--:|--:|--:|
| No letterbox | 0.640 | 0.420 | 0.489 | 0.552 | 0.552 |
| **Letterboxed** | 0.622 | 0.427 | 0.506 | 0.567 | 0.567 |

RAW still wins outright either way — this does not change ADR-001. Letterboxing moved every bbox
policy up slightly (+0.007 to +0.017) and RAW down slightly (-0.018), a small net convergence, not
a reversal.

### pants specifically (the category the hypothesis was about)

| | E0 (RAW) | E3 (category_confidence) |
|---|--:|--:|
| No letterbox | 0.435 | 0.222 |
| Letterboxed | 0.478 | 0.237 |

Both improved slightly — consistent with the hypothesis in direction, but a +0.04/+0.015 shift is
small next to the query-to-query swings seen below.

### Q001 and Q012 individually — the two queries the hypothesis was built to explain

| Query | Policy | No letterbox NDCG | Letterboxed NDCG |
|---|---|--:|--:|
| Q001 | E0 (RAW) | 0.951 | **0.616** (down) |
| Q001 | E3 (bbox) | 0.102 | **0.233** (up, still far below RAW) |
| Q012 | E0 (RAW) | 0.355 | **0.818** (up a lot) |
| Q012 | E3 (bbox) | 0.564 | **0.479** (down) |

**This is the honest, load-bearing finding: letterboxing changed RAW's own scores by more than it
changed bbox's**, and in opposite directions for the two queries (Q001's RAW result got much worse,
Q012's RAW result got much better). If the aspect-ratio-distortion theory were the dominant
mechanism, letterboxing should have helped bbox catch up to RAW consistently, not moved RAW around
unpredictably too — RAW images aren't nearly as distorted to begin with (aspect ratio ≈0.83, vs.
bbox crops' ≈0.37), yet they were *more* sensitive to the letterbox padding here, not less.

## 4. Interpretation

The aspect-ratio-distortion mechanism is real (the model's `Resize((224,224))` genuinely has no
aspect-ratio handling, that part is not in question) but it is **not the dominant explanation** for
why bbox underperforms RAW on `pants`. The bigger driver is more likely this specific embedding
model's general sensitivity to exact framing/padding for this style category — adding uniform gray
padding shifts embeddings by more than expected in directions that aren't simple "less distortion
is better." This is a real, tested result, not a rejected-without-trying hypothesis, but on this
evidence it does not justify shipping letterboxing as a pipeline change.

## 5. Decision

**Not adopted.** `letterbox_to_square` and `embed_image`'s `letterbox` parameter stay in the
codebase (tested, off by default) as an available building block, not wired into the real pipeline
or `final_config.yaml`. ADR-001 (RAW vs. BBox) is unaffected — RAW wins under both conditions
tested. This closes the specific "is it just aspect-ratio squish" question the Q001-vs-Q012
anomaly raised; the anomaly itself (pants/denim fine-grained discrimination is genuinely weak,
`milestone-10-cluster-validation.md`) remains open and would need a different fix (e.g. a
denim-specialized or fine-tuned embedding model) than a preprocessing change.

## 6. Reproduction

```
cd main
python -m pytest tests/ -q                          # 233 passed
python evaluation/pilot/run_letterbox_experiment.py  # builds catalog_db_letterbox/, writes all_summaries_letterbox.json
```
