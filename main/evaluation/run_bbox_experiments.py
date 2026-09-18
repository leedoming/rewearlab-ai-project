"""Command-line entry point for the E1-E4 bbox-policy ablation experiments.

IMPLEMENTATION_SPEC.md section 19 MUST: all four policies must be applied to
the *same* detection output -- the detector must not be re-run per policy.
Running each experiment config through a generic single-experiment CLI
(one process per config) would silently violate that, since each process
would call the detection model fresh. This script instead builds ONE
DetectionCache and reuses it across all four experiments in a single
process/run.
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from evaluation.dataset import load_dataset
from evaluation.detection_cache import DetectionCache
from evaluation.evaluator import run_bbox_experiment
from retrieval.config import (
    COLLECTION_NAMES,
    DEFAULT_CHROMADB_PORT,
    DEFAULT_LOCAL_DB_PATH,
    DETECTION_MODEL,
    DETECTION_THRESHOLD,
    EMBEDDING_MODEL,
    MIN_BBOX_AREA,
)
from retrieval.detection import detect_fashion_items
from retrieval.embedding import embed_image
from retrieval.models import load_detection_model, load_embedding_model
from retrieval.preprocessing import preprocess_image
from retrieval.search import get_chromadb_client, search_collections

EXPERIMENT_CONFIG_FILES = (
    "e1_bbox_confidence.json",
    "e2_bbox_largest.json",
    "e3_bbox_category_confidence.json",
    "e4_bbox_category_largest.json",
)


def _load_config(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def _build_detection_cache():
    image_processor, model, device = load_detection_model(DETECTION_MODEL)

    def detect(image_path):
        with Image.open(image_path) as image:
            return detect_fashion_items(
                image.convert("RGB"),
                image_processor=image_processor,
                model=model,
                device=device,
                threshold=DETECTION_THRESHOLD,
                min_area=MIN_BBOX_AREA,
            )

    return DetectionCache(detect)


def _make_pipeline(config, detection_cache, embed_model, preprocess_fn, embed_device, client):
    bbox_policy = config["preprocessing"]["bbox_policy"]
    top_k = config["retrieval"]["top_k"]
    dedupe = config["retrieval"].get("dedupe", True)

    def pipeline(query):
        detections = detection_cache.get(query.image_path)
        with Image.open(query.image_path) as image:
            image = image.convert("RGB")
            cropped, metadata = preprocess_image(
                image,
                detections,
                policy=bbox_policy,
                category=query.category,
                fallback_policy="raw",
            )
            embedding = embed_image(cropped, embed_model, preprocess_fn, embed_device)

        results = search_collections(
            client,
            COLLECTION_NAMES,
            query_embedding=embedding,
            top_k=top_k,
            dedupe=dedupe,
        )
        return results, metadata

    return pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run bbox-policy ablation experiments E1-E4")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument(
        "--output-dir", required=True, help="parent directory; each experiment writes its own subdirectory"
    )
    parser.add_argument("--configs-dir", default="evaluation/configs")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int, default=DEFAULT_CHROMADB_PORT)
    parser.add_argument("--local-db-path", default=DEFAULT_LOCAL_DB_PATH)
    args = parser.parse_args(argv)

    dataset = load_dataset(args.dataset_dir)
    detection_cache = _build_detection_cache()
    embed_model, preprocess_fn, embed_device = load_embedding_model(EMBEDDING_MODEL)
    client = get_chromadb_client(host=args.host, port=args.port, local_path=args.local_db_path)

    for config_file in EXPERIMENT_CONFIG_FILES:
        config = _load_config(Path(args.configs_dir) / config_file)
        pipeline = _make_pipeline(config, detection_cache, embed_model, preprocess_fn, embed_device, client)
        output_dir = Path(args.output_dir) / config["experiment_id"]
        run_bbox_experiment(dataset, pipeline, output_dir, config)

    print(
        f"Ran {len(EXPERIMENT_CONFIG_FILES)} bbox experiments; "
        f"detection ran for {len(detection_cache)} unique image(s), shared across all of them."
    )


if __name__ == "__main__":
    main()
