# Milestone 10 Follow-up — Hubness Validation: Evidence Record

**Branch:** `feat/m10-final-decision-regression` (same branch as PR #10 and the real-data pilot in
`milestone-10-pilot.md`)
**Repository:** fork `leedoming/rewearlab-ai-project` (upstream: `GeeYun086/rewearlab-ai-project`)
**Portfolio dashboard:** https://claude.ai/artifact/Jkp74Hc888kSmN3Pv4AeW8 (private; see the user's
own presentation of this link for sharing)

---

## 1. Objective

This follows directly from a question the user raised after `milestone-10-pilot.md`: is Marqo's
own published benchmark (text-to-image / category-to-product, on clean stock photos) a meaningful
validation for this project? The user's answer — no — came with the actual origin story of why
this whole evaluation effort exists:

1. The original MVP embedded raw product images with `Marqo/marqo-fashionSigLIP` directly.
2. Visibly dissimilar items showed up in Top-K search results.
3. Running every catalog item as a query and tallying Top-K appearance counts revealed a small
   number of items being retrieved anomalously often, across unrelated queries.
4. Inspecting those items' thumbnails found the cause: unlike normal single-product photos, they
   showed multiple garments together or a full-body/person-wearing shot.
5. Detection + bbox cropping was added to fix this — but the fix was never quantitatively
   validated. Segmentation was considered and rejected (less consistent structure than bbox,
   risk of losing fine garment detail like wrinkles/logos, which matters in this domain).

This is the real problem the ten-milestone refactor and every prior pilot exists to eventually
validate. This record quantitatively tests whether step 5's fix actually addresses step 4's cause,
using real data at a scale (2,160 images) an order of magnitude larger than the M10 pilot's N=7.

---

## 2. Method

### 2.1 Why not Marqo's own benchmark datasets (e.g. `Marqo/KAGL`)

Investigated and rejected. Marqo's six public benchmark datasets are re-hosted, license-free, on
HuggingFace (`Marqo/KAGL`, etc.), which made them attractive at first. But two mismatches make
them unsuitable for validating *this* fix specifically:

- **Task shape mismatch:** Marqo's benchmarks are text-to-image and category(text)-to-product —
  this project's actual task is image-to-image (a photo in, similar photos out). None of Marqo's
  three benchmarked task types is image-to-image.
- **The exact bug can't occur in that data:** `Marqo/KAGL` and the other five datasets are already
  single-item, clean product photos (Kaggle/Myntra-style catalog shots) — that data structurally
  cannot contain the "multiple garments/full-body" thumbnails whose existence is the entire
  premise of this investigation. Testing hubness reduction there would test nothing.

Real, messy Musinsa-style catalog photos were the only data that could actually stress-test the
fix — hence reusing the same real image source as `milestone-10-pilot.md`
(`C:/Users/smn07/Desktop/glacier-project/itda`), scanned exhaustively this time rather than
hand-picked.

### 2.2 Messy-thumbnail scan (`scan_messy_thumbnails.py`)

Ran real detection (`yainage90/fashion-object-detection`, the same threshold/min-area as
production: `DETECTION_THRESHOLD=0.4`, `MIN_BBOX_AREA=100`) over every image in the 11 non-empty
category folders across `musinsa_pants_1000`/`musinsa_upper_2000` (2,160 images total — a full
census of the usable source data, not a sample). Heuristic: **1 detection = a clean single-product
photo; 2+ detections = a messy thumbnail** (multiple garments, or a full-body shot showing
top+bottom+shoes together). Spot-checked visually against the top-scoring images (full-body
outfit shots with blazer+sweater+jeans+bag+hat, exactly matching the user's own description) and
against low-scoring ones (flat single-garment product shots) — the heuristic held up.

**Result:** 24.8% of the pool (536/2,160) is "messy" by this definition — a real, substantial
fraction, confirming the phenomenon the user described is not rare. Messy rate varies sharply by
category: 청바지(jeans) 79.3% (a full-body shot is the norm for showing how jeans fit) down to
후드티(hoodies) 10.0% (usually shown flat).

### 2.3 Hubness experiment (`hubness_analysis.py`)

Embedded all 2,160 images two ways: **RAW** (whole image) and **BBox** (`highest_confidence`
policy — deliberately the simplest, most category-agnostic policy, not RE:WEAR's later
`category_confidence` winner from the M10 pilot, to match what an MVP-stage fix would plausibly
have looked like). For each embedding scheme, computed the full 2,160×2,160 cosine similarity
matrix (brute-force, exact — small enough not to need approximate nearest-neighbor search), took
each image's Top-10 neighbors (excluding itself), and tallied how often each catalog item appears
as *someone else's* Top-10 neighbor. An item with a count far above the expected average
(2,160 × 10 / 2,160 = 10, since every query contributes exactly 10 "votes" distributed across the
pool) is a hub. Inequality of this count distribution is summarized with the **Gini coefficient**
(0 = perfectly even, higher = more concentrated in a few items).

---

## 3. Results

### 3.1 Hubness barely changed

| | RAW | BBox (highest_confidence) |
|---|--:|--:|
| Gini coefficient | 0.5287 | 0.5267 |
| Max retrieval count | 133 | 143 |
| Mean retrieval count | 10.00 | 10.00 |

BBox did not reduce concentration — Gini is essentially unchanged, and the single most-retrieved
item actually has a *higher* count under BBox (143 vs. 133) than under RAW.

### 3.2 The hypothesis was wrong — hubs are not messy thumbnails

Of the top 15 RAW hubs, 14 had exactly **one** detection (a clean shot), and the messy rate among
the top 1% of RAW hubs was **9.1%** — *lower* than the pool's overall 24.8% messy rate. Messy
thumbnails are, if anything, slightly *under*-represented among hubs, not over-represented.

Manually inspecting the actual top hub images (박스 로고 티 화이트/"box logo white tee", 2010
Sports Hoodie Grey, WHITE TEES, BLUE TEES — see the dashboard's hub gallery) shows the real
pattern: **hubs are plain, minimal-design, single-color basic garments** (mostly white/grey
T-shirts and hoodies with little or no distinguishing print, texture, or silhouette detail). A
generic, low-information garment embeds close to the centroid of its category's embedding
distribution, making it a near-neighbor for many otherwise-unrelated queries. This is a
well-documented general phenomenon in high-dimensional embedding search ("hubness"), and it is
caused by *what the garment itself looks like*, not by *what else is visible in the photo*. BBox
cropping cannot fix this: a plain white T-shirt is still a plain white T-shirt after cropping to
its own bounding box.

### 3.3 Cross-checked against the M10 pilot's own result — a different, real effect

The M10 pilot (`milestone-10-pilot.md`, N=7 golden-set queries, 3 collections) found
`category_confidence` (E3) beating both RAW and category-agnostic bbox policies on NDCG@10
(0.738 vs. 0.712 RAW vs. ~0.60 for E1/E2). That result stands, but this hubness experiment shows
its real mechanism is **not** hub suppression — it is **avoiding wrong-collection detections in
multi-item photos** (e.g. cropping to visible jeans for a knit-sweater query, observed directly in
the M10 pilot's Q003/Q006). Two different real problems, two different (and only one correctly
matched) fixes:

| Problem | Cause | Does bbox cropping fix it? |
|---|---|---|
| Wrong-category retrieval in multi-item photos (M10 pilot) | Category-agnostic crop selects the wrong garment type | **Yes** — category-aware bbox (E3/E4) |
| Hub over-retrieval of specific items (this record) | Generic/plain garment design embeds near the category centroid | **No** — cropping doesn't change how plain the garment looks |

---

## 4. What This Means / Recommendations

1. **Hub suppression needs a different fix than bbox cropping** — e.g. re-ranking with a
   popularity/in-degree penalty for frequently-retrieved items, or a hybrid search that also
   weighs text/attribute similarity (a plain white tee and another plain white tee may look
   identical visually but differ in brand/price/material — signal bbox-cropped image embeddings
   alone cannot recover).
2. **Keep documenting `category_confidence` (E3)'s benefit by its real mechanism** — "avoids
   wrong-category detections in multi-item photos," not "reduces hub over-retrieval." Attributing
   it to the wrong mechanism would misdirect future work (e.g. someone reading only the M10 pilot
   might reasonably but incorrectly conclude bbox cropping also solves hubness).
3. **The golden set (N=7, M10 pilot) is still pilot-scale**; this hubness record's N=2,160 scan is
   a partial answer at a much larger scale, but for a different question (embedding-space
   concentration, not per-query relevance) — a real quantitative retrieval-quality validation at
   this scale is still future work.
4. **Category-specific messy-rate variance (청바지 79.3% vs. 후드티 10.0%) is itself worth
   acting on** — a single global detection threshold may not be equally well-tuned for every
   category's typical photography style.

---

## 5. Checklist: What a Production Team Would Likely Also Check

Raised explicitly by the user and worth recording as follow-up scope, not resolved here:

- **Query/catalog domain gap**: this experiment used catalog-style photos as both queries and
  catalog (leave-one-out). A live system's queries are user-submitted secondhand-clothing photos
  (different lighting, background, camera quality) against a catalog of professional product
  shots — that domain gap is a distinct, unmeasured source of potential hub/mismatch behavior.
- **Business impact of hubs**: a frequently-retrieved item isn't automatically a bug — it could
  be a genuinely popular, broadly-similar style. Offline embedding-space metrics alone can't
  distinguish "false positive" from "actually popular"; would need click-through/conversion data.
- **Threshold sensitivity**: how much does the messy/clean classification (and hub count) shift
  across `DETECTION_THRESHOLD` values (0.3/0.4/0.5, per Milestone 9's sensitivity scope)?
- **Scale sensitivity**: does Gini/hub-count behavior measured here (180-2,160 items) hold at
  real catalog scale (tens of thousands of items)? Hubness is known in the ML literature to often
  *worsen* with dimensionality/scale, not stay constant.
- **Reproducibility**: would re-running with a different random seed/batch order produce a
  materially different hub list, or is it stable?

---

## 6. Real Bugs/Frictions Found Along the Way

- The `retrieval/search.py` numpy float32 bug (see `milestone-10-pilot.md` section 3) — same fix
  applies here since this record reuses `embed_image`.
- No new bugs found in this record specifically; `detect_fashion_items`/`preprocess_image`/
  `embed_image` all behaved as documented across the full 2,160-image, ~4,320-embedding run.

---

## 7. Exact Validation Commands and Results

```bash
cd main && python evaluation/pilot/scan_messy_thumbnails.py
```
Result: 2,160 images scanned; 536 messy (24.8%), 1,497 clean (69.3%), 127 zero-detection (5.9%).

```bash
cd main && python evaluation/pilot/hubness_analysis.py
```
Result: RAW Gini=0.5287 (max=133), BBox Gini=0.5267 (max=143); top-1% RAW hubs are 9.1% messy vs.
24.8% pool-wide.

Both scripts' raw outputs (`messy_scan.json`, `hubness_results.json`, `hubness_meta.json`,
`raw_embeddings.npy`, `bbox_embeddings.npy`) are local working data under
`main/evaluation/pilot/_local/` (gitignored — large binary/derived files, and `messy_scan.json`
records absolute paths into the external, non-portable `itda` source tree). This document and the
portfolio dashboard are the durable, committed record of what those runs produced.

---

## 8. Known Limitations

- **Leave-one-out design mixes all categories in one pool.** A query's Top-10 can include items
  from any category; this is a reasonable proxy for "does this item get retrieved too often in
  general," but doesn't isolate hubness within a single collection (e.g. within `top` only).
- **`highest_confidence` was the only bbox policy tested here.** The M10 pilot's winning policy,
  `category_confidence`, was not re-tested for hubness specifically — plausible future work, given
  the mechanism identified in section 3.3 doesn't predict it would help, but that's an inference,
  not a measurement.
- **No dress_skirts coverage**, same root cause as the M10 pilot (no such images in the source
  data).
- **N=2,160 is real scale for the messy-thumbnail EDA, but the hubness metric (Gini, hub counts)
  has not been checked against a materially larger catalog** — see section 5's scale-sensitivity
  item.
- **This is an unsupervised embedding-geometry analysis, not a relevance-labeled retrieval-quality
  evaluation** — it answers "does this item get over-retrieved," not "are the results actually
  relevant." The M10 pilot's NDCG-based results remain the source for relevance-quality claims.
