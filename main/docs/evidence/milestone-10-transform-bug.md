# Milestone 10 Correction — The `preprocess_train`/`preprocess_val` Bug: Evidence Record

**Branch:** `feat/m10-final-decision-regression`
**Repository:** fork `leedoming/rewearlab-ai-project` (upstream: `GeeYun086/rewearlab-ai-project`)
**Supersedes:** the E0-E4 numbers in `milestone-10-pilot.md` and the Gini/partial-centering
numbers in `milestone-10-hubness.md` — both were computed with the bug described here still
present. This document is the correction; the other two remain as the historical record of what
was found and when, not rewritten.

---

## 1. What happened

While chasing the reproducibility noise flagged in `milestone-10-hubness.md` section 3.4 (two
identical re-embeddings of the same 187 images producing NDCG@10 values 0.03–0.05 apart), a
deeper root cause was found: **`retrieval/models.py::load_embedding_model` was feeding every
image — every catalog item ever indexed, every query ever embedded, across every milestone since
M4 — through the wrong preprocessing transform.**

`open_clip.create_model_and_transforms()` returns exactly `(model, preprocess_train,
preprocess_val)`, per its own source. The code (present since before the Milestone 1 refactor —
neither the original `main-app/app.py` nor `embedding/musinsa_to_chromadb.py` avoided it either)
unpacked this as:

```python
model, preprocess_val, _ = open_clip.create_model_and_transforms(model_name)
```

The variable is *named* `preprocess_val`, but it actually receives `preprocess_train` — the
augmented training-time transform. The real `preprocess_val` was discarded into `_`. Printing what
this actually was:

```
Compose(
    RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=bicubic)
    MaybeConvertMode()
    MaybeToTensor()
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
)
```

Every single embedding this entire project has ever computed — including the original
pre-refactor MVP — was run through a **random crop** (a random 90–100% window at a random aspect
ratio between 0.75 and 1.33), not a deterministic center crop. Confirmed the model's own forward
pass is bit-identical on an identical input tensor; only the "preprocess" step varied between
calls on the same image.

---

## 2. The fix

`main/retrieval/models.py::load_embedding_model`:

```python
model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name)
```

Also added `.eval()` to both `load_embedding_model` and `load_detection_model` (defensive —
confirmed this specific model has no active dropout, so it wasn't the cause here, but leaving
either model in the PyTorch `train()` default is a latent bug waiting for a future model swap that
does have dropout/batchnorm).

**Verification:** re-embedding the same image 3 times in a single process now produces
bit-identical output (`np.array_equal` true, `max_abs_diff = 0.0`) — before the fix, the same test
showed `max_abs_diff` up to 2.8e-2, non-trivial for a normalized embedding vector.

---

## 3. Impact: every prior number in this milestone needs re-reading

This is not a cosmetic fix. Random-crop jitter at **both indexing and query time** means:
- The same catalog product could embed differently depending on which random crop happened to be
  drawn when it was indexed.
- The same query photo could rank differently depending on which random crop was drawn at request
  time — i.e., **the original production system's search results were not fully reproducible for
  the same uploaded photo**, a real, user-facing quality issue independent of anything about
  bbox/detection.

### 3.1 Corrected E0-E4 (supersedes `milestone-10-pilot.md` section 4 / ADR-001 / ADR-002)

Re-ran `run_pilot.py` (full catalog rebuild) + `run_experiments.py` against the identical
7-query golden set, twice, confirming bit-identical results both times:

| Experiment | Policy | NDCG@10 | MRR | Precision@5 | Recall@10 | Incompatible Cat. Rate |
|---|---|--:|--:|--:|--:|--:|
| **E0** | **RAW** | **0.6242** | 0.7302 | 0.4857 | **0.8167** | **0.2429** |
| E1 | highest_confidence | 0.4642 | 0.6429 | 0.4000 | 0.4833 | 0.3000 |
| E2 | largest | 0.4491 | 0.6429 | 0.4000 | 0.4833 | 0.3000 |
| E3 | category_confidence | 0.5447 | 0.7143 | 0.4571 | 0.6262 | 0.2429 |
| E4 | category_largest | 0.5447 | 0.7143 | 0.4571 | 0.6262 | 0.2429 |

**RAW now wins outright** — the opposite of the buggy run's conclusion (which had E3=0.738 beating
E0=0.712). E3/E4 are now identical to four decimal places, meaning `highest_confidence` and
`largest` selection agreed on every single query once restricted to category-compatible
detections (consistent with a small catalog/query set where usually at most one category-matching
detection exists per image).

**ADR-001 and ADR-002's "provisional: category-aware BBox" call is retracted.** With the bug fixed,
this N=7 pilot now favors RAW. Given N=7's already-documented limitations (section 6 of
`milestone-10-pilot.md`), this is itself still provisional in the other direction, not a new final
decision — but the previous provisional direction is no longer supported by this evidence.

### 3.2 Corrected hubness numbers (supersedes `milestone-10-hubness.md` section 3.1/3.4) — core finding holds

| | RAW (buggy) | RAW (fixed) | BBox (buggy) | BBox (fixed) |
|---|--:|--:|--:|--:|
| Gini | 0.5287 | 0.5295 | 0.5267 | 0.5240 |
| Max hub count | 133 | 195 | 143 | 142 |

Gini is essentially unchanged (the bug added noise, not a systematic bias toward or away from
hubness) — **the core finding that hubs are generic/plain-design garments, not messy thumbnails,
and that BBox cropping doesn't meaningfully reduce hub concentration, is robust to this bug and
still holds.** The RAW max count did shift meaningfully (133 → 195); not investigated further here,
but it doesn't change the Gini-based conclusion.

### 3.3 Corrected partial-centering sweep — the previous "α≈0.4 sweet spot" was pure noise

| α | Hub Gini | NDCG@10 (buggy, noisy) | NDCG@10 (fixed, deterministic) |
|--:|--:|--:|--:|
| 0.0 | 0.5295 | 0.6041 | 0.5431 |
| 0.3 | 0.4887 | 0.6020 | 0.5501 |
| 0.4 | 0.4689 | **0.6355** (looked best) | 0.5363 |
| 0.5 | 0.4456 | 0.6057 | 0.5305 |
| 0.7 | 0.3925 | 0.5967 | 0.5193 |
| 1.0 | 0.3351 | 0.5066 | 0.4673 |

With determinism restored, **NDCG@10 decreases monotonically (with noise-free flatness from
α=0.0–0.3) as α increases — there is no α that improves NDCG.** The buggy run's apparent "α=0.4
beats α=0.0" result was an artifact of embedding non-determinism, not a real effect. **This
retracts `milestone-10-hubness.md`'s "α≈0.4–0.5 is a promising candidate" recommendation.** The
real, corrected picture is a plain trade-off: some hub reduction is available at α=0.1–0.3 for
close to zero NDCG cost (0.5431 → 0.5501, actually flat-to-slightly-up in this run, within
plausible small-sample noise even now), but nothing beyond that is free — larger α trades
real relevance quality for hub suppression, with no shortcut.

*(Note: this table's α=0.0 baseline (0.5431, from `sweep_partial_centering.py`'s own
from-scratch catalog re-embedding) differs from section 3.1's E0 baseline (0.6242, from
`run_experiments.py`'s real `search_collections`/ChromaDB path) by a real methodological
difference, not noise: the sweep script embeds the golden-set catalog RAW/uncropped for
simplicity, while the actual E0 pipeline's catalog was ingested with `category_confidence`
bbox-cropping, matching production convention — see `milestone-10-pilot.md` section 2.4. Both
numbers are now individually reproducible; they are simply answering slightly different
questions (RAW-query-vs-RAW-catalog vs. RAW-query-vs-bbox-cropped-catalog).)*

---

## 4. What this means going forward

1. **Ship this fix regardless of anything else in this record.** A production embedding pipeline
   that silently applies random-crop augmentation at inference time is a real bug independent of
   any ablation conclusion — it means repeated searches of the identical photo can return
   different results, which is a user-facing correctness issue on its own.
2. **ADR-001/002 need to be re-run, not just re-read**, once a larger golden set exists — this
   correction doesn't hand them a new final answer, it removes confidence in the old one.
3. **The hubness finding (section 3.2) survives** — it wasn't sensitive to this bug, so
   `milestone-10-hubness.md`'s core conclusion and its recommendations (section 4) stand.
4. **The partial-centering recommendation does not survive** — no α is a free win; picking one
   requires deciding how much NDCG the team is willing to trade for how much hub reduction, which
   is a product decision, not something this record can resolve on its own.

---

## 5. Exact Validation Commands and Results

```bash
cd main && python evaluation/pilot/check_determinism.py
```
Before fix: `max_abs_diff` up to 2.8e-2 across repeated embeddings of the same image. After fix:
`identical=True`, `max_abs_diff=0.0`, both repeats.

```bash
cd main && python -m pytest tests/ -q
```
Result: `226 passed` — the fix touches only `retrieval/models.py`, which no existing unit test
exercises against the real `open_clip`/`transformers` packages (all mocked/faked), so no test
needed updating.

```bash
cd main && python evaluation/pilot/run_pilot.py      # rebuild catalog with fixed transform
cd main && python evaluation/pilot/run_experiments.py  # re-run E0-E4, twice, bit-identical both times
cd main && python evaluation/pilot/hubness_analysis.py
cd main && python evaluation/pilot/hubness_mitigation.py
cd main && python evaluation/pilot/sweep_partial_centering.py
```
Results: see sections 3.1–3.3 above.

---

## 6. Known Limitations

- **This correction was itself found through the specific act of chasing a small (N=7) golden
  set's reproducibility noise** — a reminder that "the numbers don't reproduce" is itself a
  signal worth investigating, not just something to average away.
- All limitations already disclosed in `milestone-10-pilot.md` (N=7, no dress_skirts, single
  AI-assisted rater) and `milestone-10-hubness.md` still apply to the corrected numbers here —
  this record fixes a measurement bug, not the sample-size/coverage/labeling limitations.
- The E8/E5-E6/E7 experiments (category filtering, padding) from Milestones 6-7 were never run
  against real data in either the buggy or fixed state — this correction only covers what the M10
  pilot and hubness record actually executed (E0-E4).
