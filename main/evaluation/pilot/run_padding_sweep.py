"""ADR-004 (padding_ratio) was never tested against real data -- this sweeps
it directly, on the fixed-winning bbox_policy=category_confidence, to see
whether a small context margin around the bbox crop helps (or hurts)
compared to a razor-tight 0% crop, especially for the `pants` anomaly
(Q001/Q012, docs/evidence/milestone-10-cluster-validation.md).

For each padding_ratio, builds a SEPARATE ChromaDB catalog (catalog and
query embedded with the SAME padding_ratio -- an apples-to-apples ablation,
not a mismatched one) from the same source images already recorded in
catalog_manifest.json, then runs the category_confidence policy and
records metrics. RAW (E0, no crop at all) is not re-run here -- its
numbers are already known from run_experiments.py and serve as the
reference point.

See docs/evidence/milestone-10-padding-sweep.md for the writeup.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chromadb
from PIL import Image

from evaluation.dataset import load_dataset
from evaluation.detection_cache import DetectionCache
from evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from retrieval.config import COLLECTION_NAMES, DETECTION_THRESHOLD, MIN_BBOX_AREA
from retrieval.detection import detect_fashion_items
from retrieval.embedding import embed_image
from retrieval.models import load_detection_model, load_embedding_model
from retrieval.preprocessing import preprocess_image
from retrieval.search import search_collections

PILOT_DIR = Path(__file__).parent
LOCAL_DIR = PILOT_DIR / "_local"
DATASET_DIR = PILOT_DIR.parent / "dataset"

PADDING_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.5]
POLICY = "category_confidence"


def heal_flaky_collections(client):
    """Work around a real, reproducible local ChromaDB bug (see
    docs/evidence/milestone-10-cluster-validation.md section 3): a
    collection's on-disk HNSW segment can silently stop answering queries
    (raises 'Error creating hnsw segment reader: Nothing found on disk')
    after the first query in a process, or intermittently after a fresh
    build, for no consistent reason found so far. `search_collections`
    catches and logs this per collection and moves on -- which protects a
    run from crashing but also means a silently-empty collection can pass
    without anyone noticing.

    This checks every collection actually answers a real query before
    trusting the catalog, and rebuilds (delete + recreate + one bulk
    `add()`, reading back its own stored embeddings) any collection that
    doesn't. Cheap: it's a metadata-only fetch and a resubmit, no
    re-embedding.
    """
    for name in COLLECTION_NAMES:
        try:
            coll = client.get_collection(name)
        except Exception:
            continue
        data = coll.get(include=["embeddings", "metadatas"], limit=1)
        if not data["ids"]:
            continue
        probe_embedding = data["embeddings"][0]
        # The observed failure pattern is specifically "first query
        # succeeds, every query after it fails" -- so the check must query
        # at least twice and require ALL of them to succeed, not just one.
        healthy = True
        for _ in range(3):
            try:
                coll.query(query_embeddings=[probe_embedding], n_results=1)
            except Exception:
                healthy = False
        if healthy:
            continue
        print(f"    [heal] '{name}' failed a post-build health check -- rebuilding from stored embeddings")
        full = client.get_collection(name).get(include=["embeddings", "metadatas"])
        client.delete_collection(name)
        rebuilt = client.create_collection(name)
        rebuilt.add(ids=full["ids"], embeddings=full["embeddings"], metadatas=full["metadatas"])


def build_catalog_at_padding(client, manifest, padding_ratio, image_processor, det_model, det_device, embed_model, preprocess_fn, embed_device):
    by_collection = {}
    for entry in manifest:
        by_collection.setdefault(entry["collection"], []).append(entry)

    for collection, entries in by_collection.items():
        coll = client.get_or_create_collection(name=collection)
        ids, embeddings, metadatas = [], [], []
        for entry in entries:
            with Image.open(entry["source_path"]) as raw_image:
                image = raw_image.convert("RGB")
                detections = detect_fashion_items(
                    image, image_processor=image_processor, model=det_model, device=det_device,
                    threshold=DETECTION_THRESHOLD, min_area=MIN_BBOX_AREA,
                )
                cropped, _ = preprocess_image(
                    image, detections, policy=POLICY, category=collection,
                    padding_ratio=padding_ratio, fallback_policy="raw",
                )
                embedding = embed_image(cropped, embed_model, preprocess_fn, embed_device)
            ids.append(entry["product_id"])
            embeddings.append(embedding.tolist())
            metadatas.append({"id": entry["product_id"], "product_id": entry["product_id"]})
        coll.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        print(f"    {collection}: {len(ids)} items ingested (padding={padding_ratio})")


def main():
    dataset = load_dataset(str(DATASET_DIR))
    manifest = json.loads((LOCAL_DIR / "catalog_manifest.json").read_text(encoding="utf-8"))

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

    summaries = {}
    for padding_ratio in PADDING_RATIOS:
        tag = f"pad{int(padding_ratio * 100):03d}"
        db_dir = LOCAL_DIR / f"catalog_db_{tag}"
        results_dir = LOCAL_DIR / "results_padding_sweep" / tag
        already_built = db_dir.exists()
        client = chromadb.PersistentClient(path=str(db_dir))
        if not already_built:
            print(f"Building catalog at padding_ratio={padding_ratio}...")
            build_catalog_at_padding(client, manifest, padding_ratio, image_processor, det_model, det_device, embed_model, preprocess_fn, embed_device)
        else:
            print(f"Catalog for padding_ratio={padding_ratio} already exists, reusing.")

        heal_flaky_collections(client)

        # This is a pilot ablation over a value (padding_ratio != 0.0 on
        # E1-E4's fixed policy) that evaluation.evaluator's strict E1-E4/E7
        # runners deliberately reject (run_bbox_experiment requires
        # padding_ratio == 0.0; run_padding_experiment only accepts the
        # single spec-pinned value 0.1 under E7's full category-filter
        # contract). Metrics are computed directly here, the same pattern
        # hubness_mitigation.py / validate_centering_on_golden_set.py use
        # for other non-spec pilot sweeps -- not a reimplementation of
        # evaluate_query's *decisions*, just its formulas.
        per_query = []
        for query in dataset.queries:
            detections = detection_cache.get(str(query.image_path))
            with Image.open(query.image_path) as raw_image:
                image = raw_image.convert("RGB")
                cropped, _ = preprocess_image(
                    image, detections, policy=POLICY, category=query.category,
                    padding_ratio=padding_ratio, fallback_policy="raw",
                )
                embedding = embed_image(cropped, embed_model, preprocess_fn, embed_device)
            results = search_collections(client, COLLECTION_NAMES, query_embedding=embedding, top_k=10)
            relevances = [query.labels.get(str(r.get("product_id")), 0) for r in results]
            total_relevant = sum(1 for grade in query.labels.values() if grade >= 1)
            per_query.append({
                "query_id": query.query_id,
                "category": query.category,
                # ideal_relevances must be the query's FULL pooled label set
                # (evaluate_query's own convention, evaluation/evaluator.py),
                # not just the top-10 actually retrieved -- otherwise a query
                # with more known-relevant items than fit in the top 10 (e.g.
                # Q001 has 5 labeled relevant) gets an artificially weak
                # "ideal" ordering and an inflated NDCG.
                "ndcg_at_10": ndcg_at_k(relevances, 10, ideal_relevances=list(query.labels.values())),
                "precision_at_5": precision_at_k(relevances, 5),
                "recall_at_10": recall_at_k(relevances, 10, total_relevant),
                "mrr": mrr(relevances),
            })

        ndcgs = [r["ndcg_at_10"] for r in per_query]
        overall_ndcg = sum(ndcgs) / len(ndcgs)
        summaries[tag] = {"padding_ratio": padding_ratio, "overall_ndcg_at_10": overall_ndcg, "per_query": per_query}
        print(f"padding_ratio={padding_ratio} done: overall NDCG@10={overall_ndcg:.4f}")

        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "per_query.json").write_text(
            json.dumps(per_query, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    (LOCAL_DIR / "padding_sweep_summaries.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nAll padding sweep summaries written to", LOCAL_DIR / "padding_sweep_summaries.json")


if __name__ == "__main__":
    main()
