# -*- coding: utf-8 -*-
"""Does mean-centering (which fixed hubness in milestone-10-hubness.md)
regress retrieval quality on the M10 pilot's real, relevance-labeled
golden set (N=7)? Re-embeds the same 180-item catalog + 7 queries RAW,
computes NDCG@10/MRR/Precision/Recall with and without mean-centering
applied before cosine similarity, and compares.

Not part of the reusable evaluation/ package -- a one-off validation
script for docs/evidence/milestone-10-hubness.md.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image

from evaluation.dataset import load_dataset
from evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from retrieval.embedding import embed_image
from retrieval.models import load_embedding_model

BASE = Path(__file__).parent / "_local"
DATASET_DIR = Path(__file__).parent.parent / "dataset"
TOP_K = 10


def evaluate(query_embeddings, catalog_embeddings, catalog_ids, dataset):
    """query_embeddings: dict query_id -> vector. Returns per-query metrics."""
    records = []
    for query in dataset.queries:
        q_emb = query_embeddings[query.query_id]
        sims = catalog_embeddings @ q_emb
        order = np.argsort(-sims)[:TOP_K]
        ranked_ids = [catalog_ids[i] for i in order]

        relevances = [query.labels.get(pid, 0) for pid in ranked_ids]
        total_relevant = sum(1 for grade in query.labels.values() if grade >= 1)
        records.append({
            "query_id": query.query_id,
            "precision_at_5": precision_at_k(relevances, 5),
            "recall_at_10": recall_at_k(relevances, 10, total_relevant),
            "mrr": mrr(relevances),
            "ndcg_at_10": ndcg_at_k(relevances, 10, ideal_relevances=list(query.labels.values())),
        })
    return records


def summarize(records, field):
    values = [r[field] for r in records if r[field] is not None]
    return sum(values) / len(values) if values else None


def main():
    catalog_manifest = json.loads((BASE / "catalog_manifest.json").read_text(encoding="utf-8"))
    dataset = load_dataset(str(DATASET_DIR))

    print(f"Catalog: {len(catalog_manifest)} items. Queries: {len(dataset.queries)}.")
    print("Loading embedding model...")
    embed_model, preprocess_fn, embed_device = load_embedding_model()

    print("Embedding catalog (RAW)...")
    catalog_ids = []
    catalog_vecs = []
    for item in catalog_manifest:
        with Image.open(item["source_path"]) as raw_image:
            image = raw_image.convert("RGB")
            vec = embed_image(image, embed_model, preprocess_fn, embed_device)
        catalog_ids.append(item["product_id"])
        catalog_vecs.append(vec)
    catalog_matrix = np.stack(catalog_vecs)

    print("Embedding queries (RAW)...")
    query_embeddings = {}
    for query in dataset.queries:
        with Image.open(query.image_path) as raw_image:
            image = raw_image.convert("RGB")
            query_embeddings[query.query_id] = embed_image(image, embed_model, preprocess_fn, embed_device)

    print("\n=== Baseline (uncentered) ===")
    baseline_records = evaluate(query_embeddings, catalog_matrix, catalog_ids, dataset)
    for field in ["ndcg_at_10", "mrr", "precision_at_5", "recall_at_10"]:
        print(f"  {field}: {summarize(baseline_records, field):.4f}")

    print("\n=== Mean-centered ===")
    mean_vec = catalog_matrix.mean(axis=0)
    centered_catalog = catalog_matrix - mean_vec
    centered_catalog = centered_catalog / np.linalg.norm(centered_catalog, axis=1, keepdims=True)
    centered_queries = {
        qid: (vec - mean_vec) / np.linalg.norm(vec - mean_vec)
        for qid, vec in query_embeddings.items()
    }
    centered_records = evaluate(centered_queries, centered_catalog, catalog_ids, dataset)
    for field in ["ndcg_at_10", "mrr", "precision_at_5", "recall_at_10"]:
        print(f"  {field}: {summarize(centered_records, field):.4f}")

    print("\n=== Per-query comparison (NDCG@10) ===")
    for b, c in zip(baseline_records, centered_records):
        print(f"  {b['query_id']}: baseline={b['ndcg_at_10']:.4f}  centered={c['ndcg_at_10']:.4f}  "
              f"{'better' if c['ndcg_at_10']>b['ndcg_at_10'] else ('worse' if c['ndcg_at_10']<b['ndcg_at_10'] else 'same')}")

    results = {"baseline": baseline_records, "centered": centered_records}
    (BASE / "centering_validation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved to {BASE / 'centering_validation_results.json'}")


if __name__ == "__main__":
    main()
