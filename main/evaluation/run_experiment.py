"""Command-line entry point for the E0 RAW retrieval baseline."""

import argparse
import json
from pathlib import Path

from PIL import Image

from evaluation.dataset import load_dataset
from evaluation.evaluator import run_baseline
from retrieval.config import (
    COLLECTION_NAMES,
    DEFAULT_CHROMADB_PORT,
    DEFAULT_LOCAL_DB_PATH,
    EMBEDDING_MODEL,
)
from retrieval.embedding import embed_image
from retrieval.models import load_embedding_model
from retrieval.search import get_chromadb_client, search_collections


def _load_config(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def _distance_metadata(client):
    metrics = {}
    for name in COLLECTION_NAMES:
        collection = client.get_collection(name=name)
        metadata = collection.metadata or {}
        metrics[name] = metadata.get("hnsw:space", "not_explicitly_declared")
    return metrics


def _create_retriever(config, host, port, local_db_path):
    model, preprocess, device = load_embedding_model(EMBEDDING_MODEL)
    client = get_chromadb_client(host=host, port=port, local_path=local_db_path)
    config["embedding_model"] = EMBEDDING_MODEL
    config["collections"] = list(COLLECTION_NAMES)
    config["distance_metric_by_collection"] = _distance_metadata(client)
    top_k = config["retrieval"]["top_k"]
    dedupe = config["retrieval"].get("dedupe", True)

    def retrieve(query):
        with Image.open(query.image_path) as image:
            embedding = embed_image(image.convert("RGB"), model, preprocess, device)
        return search_collections(
            client,
            COLLECTION_NAMES,
            query_embedding=embedding,
            top_k=top_k,
            dedupe=dedupe,
        )

    return retrieve


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the E0 RAW retrieval baseline")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="evaluation/configs/e0_raw.json")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int, default=DEFAULT_CHROMADB_PORT)
    parser.add_argument("--local-db-path", default=DEFAULT_LOCAL_DB_PATH)
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    dataset = load_dataset(args.dataset_dir)
    retrieve = _create_retriever(config, args.host, args.port, args.local_db_path)
    run_baseline(dataset, retrieve, args.output_dir, config)


if __name__ == "__main__":
    main()
