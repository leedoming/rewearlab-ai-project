"""Acceptance criterion: the retrieval core can be imported without Streamlit."""

import sys


def test_retrieval_package_does_not_import_streamlit():
    import retrieval  # noqa: F401

    assert "streamlit" not in sys.modules


def test_retrieval_exposes_expected_public_api():
    import retrieval

    for name in (
        "detect_fashion_items",
        "select_bbox",
        "preprocess_image",
        "crop_image",
        "embed_image",
        "search_collection",
        "search_collections",
        "get_chromadb_client",
    ):
        assert hasattr(retrieval, name)
