"""Evaluation core: retrieval quality metrics (Milestone 2 — Metric Foundation).

See IMPLEMENTATION_SPEC.md section 23 (Phase 6 — Retrieval Metrics). Pure,
dependency-free logic; no torch/chromadb/streamlit (section 26).
"""

from .metrics import (
    dcg_at_k,
    incompatible_category_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    relevant_exclusion_rate,
)

__all__ = [
    "dcg_at_k",
    "incompatible_category_rate_at_k",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "relevant_exclusion_rate",
]
