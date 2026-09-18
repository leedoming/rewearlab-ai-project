"""Run the real E0-E4 experiments against the pilot's real ChromaDB catalog
and the real (now-populated) evaluation/dataset. Uses the actual, already-
tested `evaluation.evaluator.run_baseline`/`run_bbox_experiment` -- this
script only wires real models/DB into those functions' `pipeline` contract,
it does not reimplement any evaluation logic.

See docs/evidence/milestone-10-pilot.md for the full writeup of results.
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
RESULTS_DIR = LOCAL_DIR / "results"

BBOX_CONFIGS = [
    "e1_bbox_confidence.json",
    "e2_bbox_largest.json",
    "e3_bbox_category_confidence.json",
    "e4_bbox_category_largest.json",
]


def main():
    dataset = load_dataset(str(DATASET_DIR))
    client = chromadb.PersistentClient(path=str(LOCAL_DIR / "catalog_db"))

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

    # E0: RAW baseline
    e0_config = json.loads((CONFIGS_DIR / "e0_raw.json").read_text(encoding="utf-8"))

    def raw_retrieve(query):
        with Image.open(query.image_path) as raw_image:
            image = raw_image.convert("RGB")
            embedding = embed_image(image, embed_model, preprocess_fn, embed_device)
        return search_collections(client, COLLECTION_NAMES, query_embedding=embedding, top_k=e0_config["retrieval"]["top_k"])

    _, summary = run_baseline(dataset, raw_retrieve, RESULTS_DIR / "E0", e0_config)
    summaries["E0"] = summary
    print("E0 done:", summary["overall"]["ndcg_at_10"])

    # E1-E4: bbox policies
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
                embedding = embed_image(cropped, embed_model, preprocess_fn, embed_device)
            results = search_collections(client, COLLECTION_NAMES, query_embedding=embedding, top_k=top_k)
            return results, metadata

        _, summary = run_bbox_experiment(dataset, pipeline, RESULTS_DIR / config["experiment_id"], config)
        summaries[config["experiment_id"]] = summary
        print(f"{config['experiment_id']} done:", summary["overall"]["ndcg_at_10"])

    (LOCAL_DIR / "all_summaries.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nAll summaries written to", LOCAL_DIR / "all_summaries.json")


if __name__ == "__main__":
    main()
