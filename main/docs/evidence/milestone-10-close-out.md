# Milestone 10 Close-Out — Quantitative Validation Investigation

**Branch:** `feat/m10-final-decision-regression`
**Repository:** fork `leedoming/rewearlab-ai-project` (upstream: `GeeYun086/rewearlab-ai-project`)
**Status:** Closed. This record ties together every evidence doc this milestone produced, states
what's decided, and explicitly scopes what's deferred so the next person (or a future session)
doesn't have to re-derive it.

---

## 1. What this milestone set out to do

After Milestone 10's infrastructure-only refactor, the user asked directly whether accuracy had
actually been validated. It hadn't — this milestone's entire purpose was to fix that: validate the
pipeline against the published HF benchmark's own expectations, and separately validate real
retrieval quality against real, labeled data, with visualizable proof-of-work throughout.

## 2. The full evidence chain, in order

| # | Doc | Finding |
|---|---|---|
| 1 | `milestone-10-pilot.md` | Original N=7, 3-collection pilot — superseded, kept for history |
| 2 | `milestone-10-hubness.md` | Hub items are generic/plain-designed garments, not messy multi-item thumbnails (falsified the original hypothesis) |
| 3 | `milestone-10-transform-bug.md` | Real production bug: `open_clip`'s train/val transform tuple was unpacked wrong, causing run-to-run embedding non-determinism since before the refactor. Fixed; reversed the N=7 RAW-vs-BBox conclusion |
| 4 | `milestone-10-final-decision.md` | N=11, all 4 collections, first golden set to pass `validate_golden_set()`. RAW confirmed winner by a wider margin — then found `pants`/`outer` catalogs were built from a skewed random sample |
| 5 | `milestone-10-catalog-rebalance.md` | Stratified per-sub-style sampling fix; RAW's lead widened further (0.812 vs 0.650 NDCG@10) — the decision-grade result |
| 6 | `milestone-10-cluster-validation.md` | N=14, densified-relevant-set queries ruling out label scarcity; confirmed pants/denim fine-grained discrimination is a real, unresolved embedding weakness. Also: a real ChromaDB HNSW bug found and fixed |
| 7 | `milestone-10-letterbox.md` | Tested aspect-ratio-preserving padding as a fix for the pants anomaly — real mechanism, not the dominant cause. Not adopted |
| 8 | `milestone-10-padding-sweep.md` | Swept `padding_ratio` (ADR-004) — 10% helps `pants` specifically but hurts `top`; no single global value is right. Not adopted |
| 9 | `milestone-10-ensemble.md` | Tested RAW+bbox score/rank fusion — no fixed combination rule beats RAW alone, though the two views are provably complementary on some queries |

## 3. What's decided (won't be revisited without new evidence)

- **ADR-001: RAW preprocessing, decided.** `final_config.yaml`'s `preprocessing: raw`. Confirmed
  across three independent catalog builds (N=7 bug-fixed, N=11, N=11-rebalanced) and two additional
  query sets (N=14 cluster-validation) — the most heavily re-tested conclusion in this project.
- **ADR-002: if BBox were ever used, `category_confidence`/`category_largest` would be the policy** —
  moot given ADR-001, kept for the record.
- **The transform bug fix** (`retrieval/models.py`) — shipped, verified bit-identical determinism.
- **The stratified catalog sampling fix** (`evaluation/pilot/run_pilot.py::_stratified_quota`) —
  shipped, used by every catalog build since.

## 4. What's a real, open, quantified limitation (not silently smoothed over)

- **`pants`/denim fine-grained discrimination**: the embedding model (Marqo fashionSigLIP) reliably
  separates categories and broad styles but cannot reliably rank near-duplicate jeans by wash/cut —
  demonstrated on both sparse (`milestone-10-catalog-rebalance.md`) and deliberately dense
  (`milestone-10-cluster-validation.md`) relevant-item sets, and unaffected by three different
  preprocessing interventions (letterbox, padding, ensemble). This is very likely a property of
  generic contrastive image-text pretraining (captions describe "jeans," not wash-tone gradients),
  consistent with why classical fashion-retrieval work (e.g. in-shop clothes retrieval benchmarks)
  relies on metric-learning fine-tuning rather than off-the-shelf embeddings for instance-level
  discrimination — not something this project's evaluation methodology can be faulted for missing.
- **Q002's catalog coverage gap**: no white wide-leg denim exists anywhere in the sampled pants
  catalog. A real data-sourcing gap, not a retrieval defect.
- **`outer` doesn't recover under category-aware bbox** the way `top` does — root cause still open
  (cardigan scarcity was tested and ruled out in `milestone-10-catalog-rebalance.md`).
- **Single AI-assisted rater** throughout every labeling round this project has done. No independent
  human relabeling has occurred.
- **ChromaDB local persistence is flaky** under some not-fully-understood condition (HNSW segment
  reader failures that silently drop a whole collection from search results). Mitigated in pilot
  scripts with a post-build health check + rebuild, but the underlying cause in the ChromaDB version
  used here was never root-caused.

## 5. Explicitly deferred (out of scope for this milestone, not forgotten)

1. **Fine-tuning or metric-learning adaptation of the embedding model** on fashion similarity pairs —
   the most likely real fix for the pants/denim weakness, per the fashion-retrieval literature
   pattern (generic pretrained embeddings + task-specific metric-learning fine-tune). Substantial
   scope: needs labeled pair data, training infrastructure, and time this milestone didn't have.
2. **Hybrid re-ranking** (classical CV features — color histogram, texture — combined with or
   reranking the embedding-based top-N) — a lighter-weight alternative to fine-tuning, and the
   user's chosen next candidate *after* this milestone closes, not part of it.
3. **A learned/confidence-weighted RAW+bbox combiner** (per `milestone-10-ensemble.md`'s finding
   that a fixed max/mean/RRF rule can't tell which view to trust per query).
4. **Growing the golden set further / independent human relabeling** — every round in this project
   has used a single AI-assisted rater; a larger, human-verified set would be needed before treating
   any of these numbers as a production SLA rather than a regression/decision-support benchmark.
5. **`outer`'s bbox-recovery failure root cause** — cardigan scarcity was ruled out, nothing else
   has been tested.

## 6. Where the portfolio-facing material lives

A separate document (not part of this repo — a Claude Docs page) summarizes this milestone's real
achievements, debugging stories, and STAR-format talking points for resume/interview use. Ask the
user for that link if needed; it is derived entirely from the evidence docs listed above, nothing
fabricated beyond them.
