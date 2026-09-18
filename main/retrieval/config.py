"""Shared configuration for the retrieval core.

Centralizes values that were previously duplicated across
main/embedding/musinsa_to_chromadb.py, main/search-app/musinsa_detect.py
and main/main-app/app.py. Existing values are preserved as-is; this
module does not change any behavior, only removes duplication.
"""

# --- Models -----------------------------------------------------------

DETECTION_MODEL = "yainage90/fashion-object-detection"
EMBEDDING_MODEL = "hf-hub:Marqo/marqo-fashionSigLIP"

# --- Detection ----------------------------------------------------------

# Matches the `threshold` default used in musinsa_detect.py / app.py.
DETECTION_THRESHOLD = 0.4

# Matches the `min_size` default used across all three existing pipelines.
# Despite the name, existing code compares this against bbox *area*
# (width * height), not a side length.
MIN_BBOX_AREA = 100

# --- ChromaDB -------------------------------------------------------

DEFAULT_LOCAL_DB_PATH = "./musinsa_fashion_db_crop"
DEFAULT_CHROMADB_PORT = 8000

COLLECTION_NAMES = ["pants", "top", "outer", "dress_skirts"]

# Musinsa category (from crawled JSON / UI) -> ChromaDB collection name.
# From main/embedding/musinsa_to_chromadb.py CATEGORY_COLLECTION_MAPPING.
CATEGORY_COLLECTION_MAPPING = {
    "바지": "pants",
    "상의": "top",
    "아우터": "outer",
    "원피스_스커트": "dress_skirts",
}

# Musinsa category -> detection labels considered compatible with it.
# From main/embedding/musinsa_to_chromadb.py CATEGORY_LABEL_MAPPING.
CATEGORY_LABEL_MAPPING = {
    "바지": ["bottom"],
    "상의": ["top", "outer"],
    "아우터": ["top", "outer"],
    "원피스_스커트": ["bottom", "dress"],
}

# The same compatibility rule, keyed by ChromaDB collection name instead of
# the Musinsa category string. The evaluation dataset (main/evaluation/dataset)
# labels each query with a collection name (loader.py's CATEGORIES =
# frozenset(COLLECTION_NAMES)), so category-based bbox selection
# (category_confidence/category_largest) needs this namespace too. Derived
# from CATEGORY_LABEL_MAPPING/CATEGORY_COLLECTION_MAPPING rather than
# hardcoded again, so the two can never silently drift apart.
COLLECTION_LABEL_MAPPING = {
    CATEGORY_COLLECTION_MAPPING[category]: labels
    for category, labels in CATEGORY_LABEL_MAPPING.items()
}

# --- BBox selection / fallback policies --------------------------------

BBOX_SELECTION_POLICIES = (
    "highest_confidence",
    "largest",
    "category_confidence",
    "category_largest",
)

FALLBACK_POLICIES = ("raw", "largest", "fail")

# main/embedding/musinsa_to_chromadb.py falls back to the raw image when no
# category-compatible detection is found; this preserves that default.
DEFAULT_FALLBACK_POLICY = "raw"

DEFAULT_PADDING_RATIO = 0.0
