"""Command-line entry point for the E5-E6 category-filter ablation experiments.

Shares a DetectionCache across E5 and E6 for the same reason
run_bbox_experiments.py does for E1-E4 (IMPLEMENTATION_SPEC.md section 19).
Additionally resolves each query's labeled relevant items' true collection
once (see evaluation/label_lookup.py) so relevant_exclusion_rate reflects
what the category filter actually excluded, not a placeholder -- and, like
the detection cache, that resolution is memoized by query_id and shared
across E5 and E6 rather than repeated per experiment, since a query's
ground-truth labels don't change between them.
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from evaluation.dataset import load_dataset
from evaluation.detection_cache import DetectionCache
from evaluation.evaluator import run_category_filter_experiment
from evaluation.label_lookup import count_relevant_excluded_by_filter, resolve_label_collections
from retrieval.category import get_filtered_collections
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
    "e5_hard_category_filter.json",
    "e6_soft_category_filter.json",
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


def _make_pipeline(config, detection_cache, label_collections_cache, embed_model, preprocess_fn, embed_device, client):
    bbox_policy = config["preprocessing"]["bbox_policy"]
    filter_policy = config["category_filter"]["policy"]
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

        searched_collections = get_filtered_collections(query.category, filter_policy)
        results = search_collections(
            client,
            searched_collections,
            query_embedding=embedding,
            top_k=top_k,
            dedupe=dedupe,
        )

        if query.query_id not in label_collections_cache:
            label_collections_cache[query.query_id] = resolve_label_collections(
                client, list(query.labels.keys()), COLLECTION_NAMES
            )
        excluded_relevant_count = count_relevant_excluded_by_filter(
            query.labels, label_collections_cache[query.query_id], searched_collections
        )

        return results, metadata, excluded_relevant_count

    return pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run category-filter ablation experiments E5-E6")
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
    label_collections_cache = {}
    embed_model, preprocess_fn, embed_device = load_embedding_model(EMBEDDING_MODEL)
    client = get_chromadb_client(host=args.host, port=args.port, local_path=args.local_db_path)

    for config_file in EXPERIMENT_CONFIG_FILES:
        config = _load_config(Path(args.configs_dir) / config_file)
        pipeline = _make_pipeline(
            config, detection_cache, label_collections_cache, embed_model, preprocess_fn, embed_device, client
        )
        output_dir = Path(args.output_dir) / config["experiment_id"]
        run_category_filter_experiment(dataset, pipeline, output_dir, config)

    print(
        f"Ran {len(EXPERIMENT_CONFIG_FILES)} category-filter experiments; "
        f"detection ran for {len(detection_cache)} unique image(s), shared across all of them."
    )


if __name__ == "__main__":
    main()
