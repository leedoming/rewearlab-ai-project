# Milestone 10 Follow-up — Real-Data Pilot: Evidence Record

**Branch:** `feat/m10-final-decision-regression` (same branch as PR #10; this pilot updates
ADR-001/ADR-002, which PR #10 introduced, with real numbers before merge)
**Repository:** fork `leedoming/rewearlab-ai-project` (upstream: `GeeYun086/rewearlab-ai-project`)

> **⚠️ Correction:** the E0–E4 numbers in this document (section 4 and ADR-001/ADR-002) were
> computed with a real embedding-pipeline bug present (`retrieval/models.py` fed every image
> through a random-crop training transform instead of the deterministic validation one). Fixed
> and re-run in `milestone-10-transform-bug.md`, which **reverses this document's conclusion**
> (RAW wins the corrected re-run, not `category_confidence`). This document is kept as the
> historical record of what was found and when; read the correction alongside it.

---

## 1. Objective

Every evidence doc from Milestone 4 through Milestone 10 documented the same honest limitation:
none of the ablation/failure/sensitivity infrastructure had ever been run against real data,
because `main/evaluation/dataset/labels.json` was an empty scaffold and
`torch`/`chromadb`/`transformers`/`open_clip` were not installed in this dev environment. After
PR #10 was opened, the user asked whether that gap should actually be closed. This pilot:

1. Installs the real ML dependencies.
2. Builds a real ChromaDB catalog from real Musinsa product photos (sourced from a separate,
   unrelated personal project — see section 2.1).
3. Curates 7 real query images, hand-labels their retrieval candidates, and commits this as
   `main/evaluation/dataset`'s real content — the first time this dataset has held anything but
   an empty scaffold.
4. Runs E0-E4 for real and updates ADR-001/ADR-002 with the actual numbers.

This is explicitly a **pilot**, not the full golden-set validation section 55 asks for — see
section 6 (Known Limitations) for exactly what's still missing.

---

## 2. Design Decisions

### 2.1 Source images: a real Musinsa crawl from an unrelated prior project

**Decision:** catalog and query images come from `C:/Users/smn07/Desktop/glacier-project/itda`'s
`musinsa_pants_1000`/`musinsa_upper_2000` folders — real product photos crawled for an earlier,
unrelated personal project, organized by Korean category folder name.

**Reason:** RE:WEAR Lab's own crawler (`main/crawler/musinsa_crawler.py`) was never run to
produce a real catalog for this specific project; building one from scratch (crawling Musinsa
live) was out of scope for what the user asked for. The `itda` project's photos are genuine
Musinsa product photos already organized by a category taxonomy close enough to RE:WEAR Lab's
four collections to map directly (see 2.2), making them a reasonable, real substitute without
needing a fresh crawl. A different, incompatible source in the same folder tree
(`itda-fashion-detect`, a pre-existing populated ChromaDB) was considered first and rejected —
it uses a different embedding model (`open_clip` generic vs. RE:WEAR's
`Marqo/marqo-fashionSigLIP`) and a 28-category taxonomy that doesn't map cleanly to RE:WEAR's
four collections.

### 2.2 Only 3 of 4 collections are populated — no dress_skirts

**Decision:** the pilot catalog and golden set cover `pants`/`top`/`outer` only.

**Reason:** neither source folder contained any 원피스 (dress) or 치마 (skirt) images with
usable file counts (both subfolders were empty in the actual crawl — see the folder listing in
this session's history). Fabricating a dress_skirts collection from nothing, or mislabeling an
unrelated garment as dress_skirts, would be worse than honestly excluding it. `dress_skirts`
remains untested by this pilot.

### 2.3 Folder-to-collection mapping

**Decision:**

| RE:WEAR collection | Source folders |
|---|---|
| `pants` | `musinsa_pants_1000/청바지`, `musinsa_pants_1000/바지` |
| `top` | `musinsa_upper_2000/{긴팔, 니트, 맨투맨, 반팔, 셔츠, 나시}` |
| `outer` | `musinsa_upper_2000/{가디건, 후드티}` |

**Reason:** these are the folders with unambiguous body-region correspondence to RE:WEAR's own
`CATEGORY_LABEL_MAPPING` (Milestone 1). Folders left out (기타/운동복/원피스/점프수트/치마/바지
duplicate under `musinsa_upper_2000`, 베스트, 코트) were either empty in the actual source data
or ambiguous enough (e.g. 운동복 mixing tops and bottoms) that guessing a mapping would risk
mislabeling the catalog, which would silently corrupt every metric computed against it.

### 2.4 Catalog ingestion matches production exactly

**Decision:** `main/evaluation/pilot/run_pilot.py` ingests catalog images with
`policy="category_confidence", fallback_policy="raw"` — byte-for-byte the same call
`main/embedding/musinsa_to_chromadb.py` (the production catalog builder) makes.

**Reason:** the ablation matrix's entire point is to vary *query-side* preprocessing (RAW vs.
four bbox policies) while everything else — including how the catalog itself was built — stays
fixed and realistic. Using a different (e.g. always-RAW) ingestion policy for the pilot catalog
would make it an unrealistic stand-in for the real production catalog and would confound the
query-side comparison this pilot exists to make.

### 2.5 Query selection: hand-picked, not randomly sampled

**Decision:** 7 query images were chosen by browsing the source folders, not drawn by random
sample.

**Reason:** section 55's golden-set requirement is explicitly about *deliberate* coverage (easy
case, hard case, multi-item, background-heavy, category boundary, known previous failure), not
representative random sampling — a random sample of mostly-clean product photos would likely
contain none of the harder cases these evaluations most need to exercise. Two of the seven
(Q003, Q006) were deliberately chosen because they show multiple garments/people in one photo
(a real multi-item/background-heavy case, found by inspection — see section 4), which turned out
to be exactly where the ablation experiments' results diverge most (ADR-002's Pilot Evidence).

### 2.6 Relevance labels: AI-assisted, single-rater, pooled

**Decision:** every relevance grade (0/1/2) in `main/evaluation/dataset/labels.json` was assigned
by the AI assistant (Claude) driving this session, viewing each query image and each pooled
candidate image directly, judging in the order `main/evaluation/dataset/README.md` specifies:
category → silhouette → design detail → pattern → color. Candidates were pooled from E0's and
all four bbox policies' top-5 results per query (`main/evaluation/pilot/run_pilot.py`), matching
the README's "pooled, non-exhaustive ground truth" design.

**Reason:** no professional or multi-rater human labeling process was available for this pilot.
Using the AI's own visual judgment, applied consistently and documented per-query (section 4),
was the only way to produce *some* real relevance signal rather than none — but this is a real
and significant limitation (section 6), not a substitute for actual human labeling, and every
grade in this pilot should be treated as provisional pending human review.

---

## 3. A Real Bug Found: `retrieval/search.py`'s numpy float32 embeddings

**Problem:** `search_collection()` called `collection.query(query_embeddings=[list(query_embedding)], ...)`.
`query_embedding` is a numpy array (returned by `retrieval.embedding.embed_image`), so
`list(query_embedding)` produces a list of numpy `float32` *scalar objects*, not native Python
floats.

**Evidence:** the very first real query against `chromadb==1.5.9` raised:
`Expected embeddings to be a list of floats or ints, a list of lists, a numpy array, or a list of
numpy arrays, got [np.float32(...), ...]`. Every unit test for `search.py` had passed because
they all construct `query_embedding` as a plain Python list (e.g. `[0.0]`), which never exercises
this conversion path at all.

**Fix:** changed to `[[float(value) for value in query_embedding]]` — one commit,
`main/retrieval/search.py`. Verified: `main/tests/unit/test_search.py` (18 tests, unchanged,
still passing) plus this pilot's own successful real ChromaDB queries.

**Why no prior milestone caught this:** every milestone's own known-limitations section already
disclosed "never run against a live detector/ChromaDB/dataset" — this bug could only ever surface
by actually doing that, which is exactly what this pilot is for.

---

## 4. Pilot Run Detail

### 4.1 Catalog

180 items (60 each for pants/top/outer), sampled (fixed seed `20260918`) from 169 pants, 1776
top, and 215 outer candidate images after excluding the 7 held-out query images.

### 4.2 Queries and per-query observations

| Query | Category | Difficulty | Scene | Notes |
|---|---|---|---|---|
| Q001 | pants | hard | person_wearing | Dark grey-black wash straight/tapered jeans |
| Q002 | pants | easy | clean_product | White straight-leg jeans, flat/clean shot |
| Q003 | top | hard | person_wearing | Cream angora boat-neck knit pullover, back view, jeans visible in frame |
| Q004 | top | easy | clean_product | Beige minimal-logo crewneck tee, flat lay |
| Q005 | top | medium | clean_product | Ivory long-sleeve tee with brown athletic stripe trim |
| Q006 | outer | hard | multi_item | Two people, each wearing a button-front knit cardigan (red / navy) |
| Q007 | outer | medium | clean_product | Light grey pullover hoodie, flat lay |

**Q003 and Q006 are this pilot's most informative cases.** For Q003 (knit sweater, jeans visible
in the same frame), `highest_confidence`/`largest`/`largest`-family selection picked up the
visible jeans instead of the sweater in nearly every pooled result — the retrieved candidates
were overwhelmingly from the **wrong collection** (`pants-*` items for a `top` query). The
category-aware policies (`category_confidence`/`category_largest`), constrained by
`retrieval.category.get_allowed_labels("top") == ["top", "outer"]`, never made this mistake. The
same pattern repeated for Q006 (cardigan query, category-agnostic policies again returned
`pants-*` results). This is a direct, real observation of the `wrong_object_rate` failure mode
`evaluation.sensitivity` (Milestone 9) was built to measure.

Full pooled candidates and every individual relevance judgment are recorded in
`main/evaluation/dataset/labels.json` (committed) and were produced from
`main/evaluation/pilot/_local/pooled_candidates.json` (not committed — see section 6).

### 4.3 Results

See ADR-001 and ADR-002's own "Pilot Evidence" sections for the full metrics table. Summary:
`category_confidence` (E3) had the best NDCG@10 (0.738) and tied for best MRR/recall; RAW (E0)
was a fairly close second (0.712 NDCG); the two category-agnostic bbox policies (E1/E2) both
underperformed RAW.

---

## 5. Exact Validation Commands and Results

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install transformers chromadb open-clip-torch
```
Installed: torch 2.14.0+cpu, transformers 5.17.0, chromadb 1.5.9, open-clip-torch 3.3.0.

```bash
cd main && python evaluation/pilot/run_pilot.py
```
Result: 180-item catalog built; 7 queries run through E0 + E1-E4; pooled candidates written.

```bash
cd main && python evaluation/pilot/run_experiments.py
```
Result: real E0-E4 summaries (see section 4.3 / ADR-001/002).

```bash
python -m compileall -q main/evaluation main/tests main/retrieval
cd main && python -m pytest tests/ -q
```
Result: `226 passed`. One previously-passing test
(`test_dataset_loader.py::test_empty_repository_template_is_valid_before_images_exist`) had to be
rewritten (`test_real_repository_dataset_loads_with_images_present`) since its premise — that the
committed dataset is an empty scaffold — is no longer true, by design, after this pilot.

---

## 6. Known Limitations

- **N=7 queries, not the 10-15 section 55 asks for.** Every number in this pilot should be read
  as indicative, not conclusive.
- **No dress_skirts coverage at all** — the source photos had none (section 2.2).
- **Relevance labels are AI-assisted, single-rater** (section 2.6) — not a validated human
  labeling process. Section 65's "Human relevance subjectivity" limitation applies here in a
  stronger form than the spec anticipated (a single AI rater's subjectivity, not a human team's).
- **No "known previous failure" case exists** — by definition, since this is the first real run
  ever; nothing has failed in production yet to designate as a known previous failure. Future
  golden-set versions can retroactively add one once this pilot (or a larger one) identifies a
  genuine failure worth tracking.
- **The pilot's local working data (ChromaDB, catalog manifest with absolute external paths,
  full pooled-candidates dump) is not committed** — `.gitignore`'s
  `main/evaluation/pilot/_local/` entry. It references a filesystem path
  (`C:/Users/smn07/Desktop/glacier-project/itda`) outside this repository that wouldn't be
  reproducible for another clone anyway. What's committed and reproducible: the pilot scripts
  themselves, the real dataset (queries + manifest + labels), and this evidence record.
- **Catalog size (180 items, 60/collection) is small** relative to a real product catalog —
  retrieval quality numbers here should not be read as representative of full-catalog-scale
  performance.
- No claim beyond what's in this record and ADR-001/002 is made — this pilot updates two ADRs
  from `TBD` to **provisional**, not to a confirmed final decision.
