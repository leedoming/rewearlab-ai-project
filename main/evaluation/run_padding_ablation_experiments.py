"""Command-line entry point for the E7-E8 padding ablation experiments.

E7 (BBOX, padding_ratio=0.1) needs detection, so it shares a DetectionCache
the same way run_bbox_experiments.py / run_category_filter_experiments.py
do. E8 (RAW, no bbox at all) never calls the detector. Both experiments
filter collections by category, so both share the same
`label_collections_cache` for the real `relevant_exclusion_rate` count
(see evaluation/label_lookup.py), memoized by query_id exactly like
run_category_filter_experiments.py does for E5-E6.
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from evaluation.dataset import load_dataset
from evaluation.detection_cache import DetectionCache
from evaluation.evaluator import run_padding_experiment, run_raw_filter_experiment
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


def _resolve_exclusion_count(client, query, searched_collections, label_collections_cache):
    if query.query_id not in label_collections_cache:
        label_collections_cache[query.query_id] = resolve_label_collections(
            client, list(query.labels.keys()), COLLECTION_NAMES
        )
    return count_relevant_excluded_by_filter(
        query.labels, label_collections_cache[query.query_id], searched_collections
    )


def _make_e7_pipeline(config, detection_cache, label_collections_cache, embed_model, preprocess_fn, embed_device, client):
    bbox_policy = config["preprocessing"]["bbox_policy"]
    padding_ratio = config["preprocessing"]["padding_ratio"]
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
                padding_ratio=padding_ratio,
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
        excluded_relevant_count = _resolve_exclusion_count(client, query, searched_collections, label_collections_cache)
        return results, metadata, excluded_relevant_count

    return pipeline


def _make_e8_pipeline(config, label_collections_cache, embed_model, preprocess_fn, embed_device, client):
    filter_policy = config["category_filter"]["policy"]
    top_k = config["retrieval"]["top_k"]
    dedupe = config["retrieval"].get("dedupe", True)

    def pipeline(query):
        with Image.open(query.image_path) as image:
            image = image.convert("RGB")
            embedding = embed_image(image, embed_model, preprocess_fn, embed_device)

        searched_collections = get_filtered_collections(query.category, filter_policy)
        results = search_collections(
            client,
            searched_collections,
            query_embedding=embedding,
            top_k=top_k,
            dedupe=dedupe,
        )
        excluded_relevant_count = _resolve_exclusion_count(client, query, searched_collections, label_collections_cache)
        return results, excluded_relevant_count

    return pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run padding ablation experiments E7-E8")
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

    e7_config = _load_config(Path(args.configs_dir) / "e7_padding_ablation.json")
    e7_pipeline = _make_e7_pipeline(
        e7_config, detection_cache, label_collections_cache, embed_model, preprocess_fn, embed_device, client
    )
    run_padding_experiment(dataset, e7_pipeline, Path(args.output_dir) / "E7", e7_config)

    e8_config = _load_config(Path(args.configs_dir) / "e8_raw_soft_filter.json")
    e8_pipeline = _make_e8_pipeline(e8_config, label_collections_cache, embed_model, preprocess_fn, embed_device, client)
    run_raw_filter_experiment(dataset, e8_pipeline, Path(args.output_dir) / "E8", e8_config)

    print(
        f"Ran E7-E8 padding ablation experiments; "
        f"detection ran for {len(detection_cache)} unique image(s) (E7 only, E8 is RAW)."
    )


if __name__ == "__main__":
    main()
