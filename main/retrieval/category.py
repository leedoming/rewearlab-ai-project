"""Category <-> label / collection mapping helpers.

Extracted from main/embedding/musinsa_to_chromadb.py without changing
any mapping values.
"""

import logging
import os

from .config import CATEGORY_COLLECTION_MAPPING, CATEGORY_LABEL_MAPPING, COLLECTION_LABEL_MAPPING

logger = logging.getLogger(__name__)


def get_allowed_labels(category):
    """Detection labels considered compatible with `category` (possibly empty).

    Accepts either a Musinsa category string (e.g. "상의") or a ChromaDB
    collection name (e.g. "top", as used by the evaluation dataset's
    `category` field) — the two namespaces describe the same compatibility
    rule, just keyed differently.
    """
    if category in CATEGORY_LABEL_MAPPING:
        return CATEGORY_LABEL_MAPPING[category]
    return COLLECTION_LABEL_MAPPING.get(category, [])


def get_collection_name(category):
    """ChromaDB collection name for `category`, or 'unknown' if unmapped."""
    return CATEGORY_COLLECTION_MAPPING.get(category, "unknown")


def is_label_allowed_for_category(label, category):
    """Whether `label` is a compatible detection label for `category`."""
    return label in get_allowed_labels(category)


def get_category_from_filename(json_path):
    """Extract a Musinsa category from a `musinsa_<category>.json` filename.

    Returns None if the filename doesn't map to a known category.
    """
    filename = os.path.basename(json_path)
    category = filename.replace("musinsa_", "").replace(".json", "")

    if category in CATEGORY_COLLECTION_MAPPING:
        return category

    logger.warning(f"알 수 없는 카테고리: {category}")
    return None
