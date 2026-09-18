"""ChromaDB retrieval.

Returns structured, raw results. Does not render UI and does not turn
`raw_distance` into a similarity score — IMPLEMENTATION_SPEC.md section 6
explicitly forbids using `1 / (1 + distance)` as an evaluation score in
core logic; any such display conversion belongs in the UI layer.
"""

import logging

from .config import DEFAULT_CHROMADB_PORT, DEFAULT_LOCAL_DB_PATH

logger = logging.getLogger(__name__)


def get_chromadb_client(host=None, port=DEFAULT_CHROMADB_PORT, local_path=DEFAULT_LOCAL_DB_PATH):
    """Get a ChromaDB client, matching the existing host-vs-local switch.

    If `host` is set, connects to a remote ChromaDB server (Azure
    deployment); otherwise uses a local persistent client.
    """
    import chromadb

    if host:
        return chromadb.HttpClient(host=host, port=port)
    return chromadb.PersistentClient(path=local_path)


def _structure_results(collection_name, raw_results):
    structured = []
    if not raw_results or not raw_results.get("metadatas"):
        return structured

    metadatas = raw_results["metadatas"][0]
    distances = raw_results["distances"][0]

    for rank, (metadata, distance) in enumerate(zip(metadatas, distances), start=1):
        product_id = metadata.get("id") or metadata.get("product_id")
        structured.append(
            {
                "product_id": product_id,
                "rank": rank,
                "raw_distance": float(distance),
                "collection": collection_name,
                "metadata": metadata,
            }
        )
    return structured


def search_collection(
    client,
    collection_name,
    query_embedding=None,
    query_image=None,
    top_k=10,
    embedding_function=None,
):
    """Query a single collection and return structured, unranked-globally results.

    Exactly one of `query_embedding` (a precomputed vector, as used by
    main-app/app.py) or `query_image` (a PIL Image, embedded by Chroma's
    own `embedding_function` — the path used by search-app/musinsa_detect.py
    via OpenCLIPEmbeddingFunction) must be provided.

    Each result dict has: product_id, rank (1-based, within this
    collection), raw_distance (the Chroma distance, untouched), collection,
    metadata.
    """
    import numpy as np

    if (query_embedding is None) == (query_image is None):
        raise ValueError("Exactly one of query_embedding or query_image must be provided")

    if embedding_function is not None:
        collection = client.get_collection(name=collection_name, embedding_function=embedding_function)
    else:
        collection = client.get_collection(name=collection_name)

    logger.info(f"컬렉션 '{collection_name}'에서 검색 중... (총 {collection.count()}개 아이템)")

    if query_image is not None:
        raw_results = collection.query(
            query_images=[np.array(query_image)],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
    else:
        raw_results = collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
            include=["metadatas", "distances"],
        )

    return _structure_results(collection_name, raw_results)


def search_collections(
    client,
    collection_names,
    query_embedding=None,
    query_image=None,
    top_k=10,
    embedding_function=None,
    dedupe=True,
):
    """Query several collections, merge by raw distance, and re-rank globally.

    Mirrors the existing "each collection top-K -> merge -> global top-K"
    behavior (IMPLEMENTATION_SPEC.md section 17), ranked by raw distance
    (ascending: smaller distance = more similar) rather than any derived
    score. A collection that raises is logged and skipped so the remaining
    collections still get searched, matching the existing per-collection
    try/except/log behavior.
    """
    all_results = []
    for collection_name in collection_names:
        try:
            all_results.extend(
                search_collection(
                    client,
                    collection_name,
                    query_embedding=query_embedding,
                    query_image=query_image,
                    top_k=top_k,
                    embedding_function=embedding_function,
                )
            )
        except Exception as exc:
            logger.error(f"컬렉션 '{collection_name}' 검색 중 오류: {exc}")
            continue

    all_results.sort(key=lambda r: r["raw_distance"])

    if dedupe:
        seen_ids = set()
        deduped = []
        for result in all_results:
            product_id = result["product_id"]
            if product_id in seen_ids:
                continue
            seen_ids.add(product_id)
            deduped.append(result)
            if len(deduped) >= top_k:
                break
        merged = deduped
    else:
        merged = all_results[:top_k]

    for rank, result in enumerate(merged, start=1):
        result["rank"] = rank

    return merged
