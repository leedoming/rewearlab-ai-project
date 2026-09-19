"""Ablation: does letterbox-padding to square (instead of the model's bare
non-aspect-preserving resize) change E0-E4 results, especially for the
`pants` anomaly found in docs/evidence/milestone-10-cluster-validation.md?

Builds a SEPARATE ChromaDB catalog (embeddings computed with
`embed_image(..., letterbox=True)`) from the same source images already
recorded in catalog_manifest.json -- no new image sourcing, just a
different preprocessing path -- then re-runs E0-E4 against it and compares
to the existing (non-letterboxed) results.

See docs/evidence/milestone-10-letterbox.md for the writeup.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chromadb
from PIL import Image

from evaluation.dataset import load_dataset
from evaluation.detection_cache import DetectionCache
from evaluation.evaluator import run_baseline, run_bbox_experiment
from retrieval.config import COLLECTION_NAMES, DETECTION_THRESHOLD, MIN_BBOX_AREA
from retrieval.detection import detect_fashion_items
from retrieval.embedding import embed_image
from retrieval.models import load_detection_model, load_embedding_model
from retrieval.preprocessing import preprocess_image
from retrieval.search import search_collections

PILOT_DIR = Path(__file__).parent
LOCAL_DIR = PILOT_DIR / "_local"
DATASET_DIR = PILOT_DIR.parent / "dataset"
CONFIGS_DIR = PILOT_DIR.parent / "configs"
LETTERBOX_DB_DIR = LOCAL_DIR / "catalog_db_letterbox"
RESULTS_DIR = LOCAL_DIR / "results_letterbox"

BBOX_CONFIGS = [
    "e1_bbox_confidence.json",
    "e2_bbox_largest.json",
    "e3_bbox_category_confidence.json",
    "e4_bbox_category_largest.json",
]


def build_letterboxed_catalog(client, manifest, image_processor, det_model, det_device, embed_model, preprocess_fn, embed_device):
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
                    image, detections, policy="category_confidence", category=collection, fallback_policy="raw",
                )
                embedding = embed_image(cropped, embed_model, preprocess_fn, embed_device, letterbox=True)
            ids.append(entry["product_id"])
            embeddings.append(embedding.tolist())
            metadatas.append({"id": entry["product_id"], "product_id": entry["product_id"]})
        coll.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        print(f"  {collection}: {len(ids)} items ingested (letterboxed)")


def main():
    dataset = load_dataset(str(DATASET_DIR))
    manifest = json.loads((LOCAL_DIR / "catalog_manifest.json").read_text(encoding="utf-8"))

    print("Loading models...")
    image_processor, det_model, det_device = load_detection_model()
    embed_model, preprocess_fn, embed_device = load_embedding_model()

    already_built = LETTERBOX_DB_DIR.exists()
    client = chromadb.PersistentClient(path=str(LETTERBOX_DB_DIR))
    if not already_built:
        print("Building letterboxed catalog (this is the slow part)...")
        build_letterboxed_catalog(client, manifest, image_processor, det_model, det_device, embed_model, preprocess_fn, embed_device)
    else:
        print("Letterboxed catalog already exists, reusing it.")

    detection_cache = DetectionCache(
        lambda image_path: detect_fashion_items(
            Image.open(image_path).convert("RGB"),
            image_processor=image_processor, model=det_model, device=det_device,
            threshold=DETECTION_THRESHOLD, min_area=MIN_BBOX_AREA,
        )
    )

    summaries = {}

    e0_config = json.loads((CONFIGS_DIR / "e0_raw.json").read_text(encoding="utf-8"))

    def raw_retrieve(query):
        with Image.open(query.image_path) as raw_image:
            image = raw_image.convert("RGB")
            embedding = embed_image(image, embed_model, preprocess_fn, embed_device, letterbox=True)
        return search_collections(client, COLLECTION_NAMES, query_embedding=embedding, top_k=e0_config["retrieval"]["top_k"])

    _, summary = run_baseline(dataset, raw_retrieve, RESULTS_DIR / "E0", e0_config)
    summaries["E0"] = summary
    print("E0 (letterbox) done:", summary["overall"]["ndcg_at_10"])

    for config_file in BBOX_CONFIGS:
        config = json.loads((CONFIGS_DIR / config_file).read_text(encoding="utf-8"))
        bbox_policy = config["preprocessing"]["bbox_policy"]
        top_k = config["retrieval"]["top_k"]

        def pipeline(query, bbox_policy=bbox_policy, top_k=top_k):
            detections = detection_cache.get(str(query.image_path))
            with Image.open(query.image_path) as raw_image:
                image = raw_image.convert("RGB")
                cropped, metadata = preprocess_image(
                    image, detections, policy=bbox_policy, category=query.category, fallback_policy="raw",
                )
                embedding = embed_image(cropped, embed_model, preprocess_fn, embed_device, letterbox=True)
            results = search_collections(client, COLLECTION_NAMES, query_embedding=embedding, top_k=top_k)
            return results, metadata

        _, summary = run_bbox_experiment(dataset, pipeline, RESULTS_DIR / config["experiment_id"], config)
        summaries[config["experiment_id"]] = summary
        print(f"{config['experiment_id']} (letterbox) done:", summary["overall"]["ndcg_at_10"])

    (LOCAL_DIR / "all_summaries_letterbox.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nAll letterbox summaries written to", LOCAL_DIR / "all_summaries_letterbox.json")


if __name__ == "__main__":
    main()
