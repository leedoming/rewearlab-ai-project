# ADR-003: Category Filtering Policy

## Status

**Decision: TBD** — pending real ablation results. See "Decision" below.

## Context

Given a query's category (e.g. "outer"), the search step can restrict which ChromaDB collections
get queried at all, before any embedding similarity is computed. Restricting too aggressively
risks excluding genuinely relevant items (a coat mislabeled as "top"); not restricting at all
risks retrieving visually-similar-but-wrong-category items. `IMPLEMENTATION_SPEC.md` section 21
requires three policies to be supported and compared, not one assumed correct.

## Alternatives

`main/retrieval/category.py::get_filtered_collections` implements all three:

1. **none** — search every collection; category has no effect on which collections are queried.
2. **hard** — search only the one collection matching the category exactly
   (`HARD_CATEGORY_FILTER_MAPPING`).
3. **soft** — also search visually-adjacent collections (`SOFT_CATEGORY_FILTER_MAPPING`, e.g.
   top/outer overlap) — an initial hypothesis from section 21, not evidence-backed yet.

## Evidence

- **E5-E6** (Milestone 6) run hard and soft respectively, holding bbox policy fixed. This
  milestone also built the machinery to measure `relevant_exclusion_rate` honestly (via
  `evaluation/label_lookup.py`'s real ChromaDB metadata lookup, not a placeholder `0`) — so once
  real data exists, "did the filter throw away a relevant item" is directly measurable, not
  inferred.
- **E7** (Milestone 7) holds category filter policy at "Best" (still undetermined) while varying
  padding, and **E8** isolates soft-filtering with RAW preprocessing (no bbox at all).
- No real `relevant_exclusion_rate`, NDCG, or incompatible-category-rate numbers exist yet across
  none/hard/soft — same root cause as ADR-001/002.

## Decision

**TBD.** `docs/decisions/final_config.yaml`'s `category_filter.policy` field stays `TBD` until
E5/E6 (and E0-E4's own `incompatible_category_rate_at_10`, which already exists for the
`policy=none` baseline) can actually be compared.

## Reason

N/A until Decision is made.

## Trade-offs

- **none**: never excludes a relevant item (`relevant_exclusion_rate` is always 0 by
  construction), but may retrieve category-incompatible results
  (`incompatible_category_rate_at_10` — already measured for every experiment since Milestone 2).
- **hard**: eliminates category-incompatible results by construction, but risks excluding
  relevant items whose true collection doesn't match the query's stated category exactly
  (mislabeling, boundary garments).
- **soft**: a middle ground whose adjacency mapping (`SOFT_CATEGORY_FILTER_MAPPING`) is
  currently a documented *hypothesis* (section 21: "이는 최초 hypothesis이며... 조정 가능하다"),
  not itself validated — a soft-filter decision would also need to validate that specific mapping,
  not just "soft filtering in general."

## Rejected Alternatives

None rejected yet — all three remain candidates pending the ablation comparison.

## Future Work

Run E5-E6 for real; if soft filtering wins, separately validate `SOFT_CATEGORY_FILTER_MAPPING`'s
specific adjacency pairs against real confusion patterns (which categories actually get confused
in practice) rather than keeping the initial hypothesis unexamined.
