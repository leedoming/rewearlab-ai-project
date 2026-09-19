# -*- coding: utf-8 -*-
"""Sweep a partial-centering blend factor alpha (0 = RAW, 1 = full
mean-centering) and report both the hub metric (Gini, on the 2160-image
pool) and retrieval quality (NDCG@10, on the M10 pilot's real 7-query
golden set) at each alpha, to find where hub suppression stops costing
relevance quality.

centered(v) = normalize(v - alpha * mean_vec)

Reuses cached embeddings (main/evaluation/pilot/_local/raw_embeddings.npy
for the hub pool; re-embeds the golden-set catalog/queries once and caches
them here too) -- no model re-inference needed for the sweep itself.

Not part of the reusable evaluation/ package -- a one-off follow-up
analysis for docs/evidence/milestone-10-hubness.md.
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
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def gini(counts):
    sorted_counts = np.sort(counts.astype(float))
    n = len(sorted_counts)
    cumulative = np.cumsum(sorted_counts)
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n


def hub_gini_at_alpha(matrix, mean_vec, alpha, top_k=TOP_K):
    centered = matrix - alpha * mean_vec
    centered = centered / np.linalg.norm(centered, axis=1, keepdims=True)
    similarity = centered @ centered.T
    n = similarity.shape[0]
    np.fill_diagonal(similarity, -np.inf)
    top_k_idx = np.argpartition(-similarity, top_k, axis=1)[:, :top_k]
    counts = np.zeros(n, dtype=int)
    for row in top_k_idx:
        for idx in row:
            counts[idx] += 1
    return gini(counts), int(counts.max())


def ndcg_at_alpha(catalog_matrix, catalog_ids, query_embeddings, mean_vec, alpha, dataset):
    centered_catalog = catalog_matrix - alpha * mean_vec
    centered_catalog = centered_catalog / np.linalg.norm(centered_catalog, axis=1, keepdims=True)
    centered_queries = {}
    for qid, vec in query_embeddings.items():
        c = vec - alpha * mean_vec
        centered_queries[qid] = c / np.linalg.norm(c)

    records = []
    for query in dataset.queries:
        q_emb = centered_queries[query.query_id]
        sims = centered_catalog @ q_emb
        order = np.argsort(-sims)[:TOP_K]
        ranked_ids = [catalog_ids[i] for i in order]
        relevances = [query.labels.get(pid, 0) for pid in ranked_ids]
        total_relevant = sum(1 for grade in query.labels.values() if grade >= 1)
        records.append({
            "precision_at_5": precision_at_k(relevances, 5),
            "recall_at_10": recall_at_k(relevances, 10, total_relevant),
            "mrr": mrr(relevances),
            "ndcg_at_10": ndcg_at_k(relevances, 10, ideal_relevances=list(query.labels.values())),
        })
    return {
        field: sum(r[field] for r in records if r[field] is not None) / len([r for r in records if r[field] is not None])
        for field in ["ndcg_at_10", "mrr", "precision_at_5", "recall_at_10"]
    }


def get_or_embed_golden_set():
    cache_path = BASE / "golden_set_raw_embeddings.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["catalog_matrix"], list(data["catalog_ids"]), data["query_ids"].tolist(), data["query_vecs"]

    catalog_manifest = json.loads((BASE / "catalog_manifest.json").read_text(encoding="utf-8"))
    dataset = load_dataset(str(DATASET_DIR))
    print("Loading embedding model for golden-set catalog/queries...")
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
    query_ids = []
    query_vecs = []
    for query in dataset.queries:
        with Image.open(query.image_path) as raw_image:
            image = raw_image.convert("RGB")
            query_ids.append(query.query_id)
            query_vecs.append(embed_image(image, embed_model, preprocess_fn, embed_device))
    query_vecs = np.stack(query_vecs)

    np.savez(cache_path, catalog_matrix=catalog_matrix, catalog_ids=np.array(catalog_ids, dtype=object),
              query_ids=np.array(query_ids, dtype=object), query_vecs=query_vecs)
    return catalog_matrix, catalog_ids, query_ids, query_vecs


def main():
    hub_matrix = np.load(BASE / "raw_embeddings.npy")
    hub_mean = hub_matrix.mean(axis=0)

    catalog_matrix, catalog_ids, query_ids, query_vecs = get_or_embed_golden_set()
    catalog_mean = catalog_matrix.mean(axis=0)
    query_embeddings = dict(zip(query_ids, query_vecs))
    dataset = load_dataset(str(DATASET_DIR))

    print(f"\n{'alpha':>6} | {'hub_gini':>9} | {'hub_max':>8} | {'ndcg@10':>8} | {'mrr':>6} | {'p@5':>6} | {'r@10':>6}")
    print("-" * 70)
    results = []
    for alpha in ALPHAS:
        hgini, hmax = hub_gini_at_alpha(hub_matrix, hub_mean, alpha)
        ndcg_metrics = ndcg_at_alpha(catalog_matrix, catalog_ids, query_embeddings, catalog_mean, alpha, dataset)
        print(f"{alpha:6.1f} | {hgini:9.4f} | {hmax:8d} | {ndcg_metrics['ndcg_at_10']:8.4f} | "
              f"{ndcg_metrics['mrr']:6.4f} | {ndcg_metrics['precision_at_5']:6.4f} | {ndcg_metrics['recall_at_10']:6.4f}")
        results.append({"alpha": alpha, "hub_gini": hgini, "hub_max": hmax, **ndcg_metrics})

    (BASE / "partial_centering_sweep.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved to {BASE / 'partial_centering_sweep.json'}")


if __name__ == "__main__":
    main()
