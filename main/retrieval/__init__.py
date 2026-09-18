"""Retrieval core: detection, bbox selection, preprocessing, embedding, search.

Pure, Streamlit-free logic shared by main/embedding/musinsa_to_chromadb.py,
main/search-app/musinsa_detect.py and main/main-app/app.py. See
IMPLEMENTATION_SPEC.md section 6 (Phase 1 — Retrieval Core Refactoring).
"""

from . import config
from .category import (
    get_allowed_labels,
    get_category_from_filename,
    get_collection_name,
    is_label_allowed_for_category,
)
from .detection import detect_fashion_items
from .embedding import embed_image
from .models import load_detection_model, load_embedding_model
from .preprocessing import (
    RetrievalFallbackError,
    crop_image,
    preprocess_image,
    select_bbox,
)
from .search import get_chromadb_client, search_collection, search_collections

__all__ = [
    "config",
    "get_allowed_labels",
    "get_category_from_filename",
    "get_collection_name",
    "is_label_allowed_for_category",
    "detect_fashion_items",
    "embed_image",
    "load_detection_model",
    "load_embedding_model",
    "RetrievalFallbackError",
    "crop_image",
    "preprocess_image",
    "select_bbox",
    "get_chromadb_client",
    "search_collection",
    "search_collections",
]
