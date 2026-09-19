"""Ensemble idea #2 (following milestone-10-padding-sweep.md): Q001 wins with
RAW, Q012 wins with bbox (category_confidence) -- opposite directions, both
pants queries. A single blanket RAW-or-BBox policy can't be optimal for both.
This tests combining the two query-side representations at retrieval time,
searching the SAME (already-built, bbox-embedded) catalog with both a RAW
query embedding and a category_confidence-cropped query embedding, then
fusing the two similarity scores per catalog item.

Catalog embeddings are read back directly from the real catalog_db via
`collection.get(include=["embeddings"])` and compared with plain numpy
cosine similarity (brute-force, exact) rather than ChromaDB's `.query()` --
deliberately sidesteps the HNSW flakiness documented in
milestone-10-cluster-validation.md/milestone-10-padding-sweep.md, and is
exact rather than approximate for a catalog this small (237 items).

Two fusion rules are tested per query:
  - max:  fused_score(item) = max(sim_raw(item), sim_bbox(item))
  - mean: fused_score(item) = mean(sim_raw(item), sim_bbox(item))

See docs/evidence/milestone-10-ensemble.md for the writeup.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chromadb
import numpy as np
from PIL import Image

from evaluation.dataset import load_dataset
from evaluation.detection_cache import DetectionCache
from evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from retrieval.config import COLLECTION_NAMES, DETECTION_THRESHOLD, MIN_BBOX_AREA
from retrieval.detection import detect_fashion_items
from retrieval.embedding import embed_image
from retrieval.models import load_detection_model, load_embedding_model
from retrieval.preprocessing import preprocess_image

PILOT_DIR = Path(__file__).parent
LOCAL_DIR = PILOT_DIR / "_local"
DATASET_DIR = PILOT_DIR.parent / "dataset"
CATALOG_DB_DIR = LOCAL_DIR / "catalog_db"
POLICY = "category_confidence"
TOP_K = 10


def load_catalog():
    """Read every collection's (ids, embeddings, collection name) back via
    brute-force get(), not .query() -- exact, and avoids the ChromaDB HNSW
    flakiness this project has repeatedly hit."""
    client = chromadb.PersistentClient(path=str(CATALOG_DB_DIR))
    all_ids, all_embeddings, all_collections = [], [], []
    for name in COLLECTION_NAMES:
        coll = client.get_collection(name)
        data = coll.get(include=["embeddings"])
        all_ids.extend(data["ids"])
        all_embeddings.extend(data["embeddings"])
        all_collections.extend([name] * len(data["ids"]))
    embeddings = np.array(all_embeddings, dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    return all_ids, all_collections, normalized


def cosine_sims(query_embedding, catalog_normalized):
    q = np.asarray(query_embedding, dtype=np.float64)
    q = q / np.linalg.norm(q)
    return catalog_normalized @ q


def top_k_from_scores(scores, ids, collections, k=TOP_K):
    order = np.argsort(-scores)[:k]
    return [{"product_id": ids[i], "collection": collections[i]} for i in order]


def reciprocal_rank_fusion(sim_a, sim_b, ids, collections, k=TOP_K, rrf_constant=60):
    """Combine two independently-ranked views by RANK, not raw similarity
    score. Unlike max/mean score fusion, this doesn't assume the two views'
    similarity scores are on a comparable scale -- an item ranked #1 by one
    view counts the same regardless of what the other view's absolute
    scores look like. Standard IR technique; rrf_constant=60 is the
    conventional default (Cormack et al. 2009)."""
    rank_a = np.argsort(-sim_a)
    rank_b = np.argsort(-sim_b)
    rrf_score = np.zeros(len(ids))
    for rank_position, idx in enumerate(rank_a):
        rrf_score[idx] += 1.0 / (rrf_constant + rank_position + 1)
    for rank_position, idx in enumerate(rank_b):
        rrf_score[idx] += 1.0 / (rrf_constant + rank_position + 1)
    return top_k_from_scores(rrf_score, ids, collections, k=k)


def metrics_for(results, query):
    relevances = [query.labels.get(str(r["product_id"]), 0) for r in results]
    total_relevant = sum(1 for grade in query.labels.values() if grade >= 1)
    return {
        "ndcg_at_10": ndcg_at_k(relevances, 10, ideal_relevances=list(query.labels.values())),
        "precision_at_5": precision_at_k(relevances, 5),
        "recall_at_10": recall_at_k(relevances, 10, total_relevant),
        "mrr": mrr(relevances),
    }


def main():
    dataset = load_dataset(str(DATASET_DIR))

    print("Loading models...")
    image_processor, det_model, det_device = load_detection_model()
    embed_model, preprocess_fn, embed_device = load_embedding_model()
    detection_cache = DetectionCache(
        lambda image_path: detect_fashion_items(
            Image.open(image_path).convert("RGB"),
            image_processor=image_processor, model=det_model, device=det_device,
            threshold=DETECTION_THRESHOLD, min_area=MIN_BBOX_AREA,
        )
    )

    print("Reading catalog embeddings (brute-force, exact)...")
    ids, collections, catalog_normalized = load_catalog()
    print(f"  {len(ids)} catalog items across {len(set(collections))} collections")

    per_query = {"raw": [], "bbox": [], "fused_max": [], "fused_mean": [], "fused_rrf": []}
    for query in dataset.queries:
        detections = detection_cache.get(str(query.image_path))
        with Image.open(query.image_path) as raw_image:
            image = raw_image.convert("RGB")
            raw_embedding = embed_image(image, embed_model, preprocess_fn, embed_device)
            cropped, _ = preprocess_image(
                image, detections, policy=POLICY, category=query.category, fallback_policy="raw",
            )
            bbox_embedding = embed_image(cropped, embed_model, preprocess_fn, embed_device)

        sim_raw = cosine_sims(raw_embedding, catalog_normalized)
        sim_bbox = cosine_sims(bbox_embedding, catalog_normalized)
        sim_max = np.maximum(sim_raw, sim_bbox)
        sim_mean = (sim_raw + sim_bbox) / 2

        for mode, sims in [("raw", sim_raw), ("bbox", sim_bbox), ("fused_max", sim_max), ("fused_mean", sim_mean)]:
            results = top_k_from_scores(sims, ids, collections)
            m = metrics_for(results, query)
            m["query_id"] = query.query_id
            m["category"] = query.category
            per_query[mode].append(m)

        rrf_results = reciprocal_rank_fusion(sim_raw, sim_bbox, ids, collections)
        m = metrics_for(rrf_results, query)
        m["query_id"] = query.query_id
        m["category"] = query.category
        per_query["fused_rrf"].append(m)

    summary = {}
    for mode, records in per_query.items():
        ndcgs = [r["ndcg_at_10"] for r in records]
        summary[mode] = {"overall_ndcg_at_10": sum(ndcgs) / len(ndcgs), "per_query": records}
        print(f"{mode}: overall NDCG@10={summary[mode]['overall_ndcg_at_10']:.4f}")

    (LOCAL_DIR / "ensemble_summaries.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nWritten to", LOCAL_DIR / "ensemble_summaries.json")


if __name__ == "__main__":
    main()
