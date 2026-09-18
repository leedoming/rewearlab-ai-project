"""Regression test for search-app's no-detection fallback path.

musinsa_detect.py's main() must not hand-build a synthetic full-image
detection anymore; it must route through the shared retrieval fallback
mechanism (resolve_fallback_image -> retrieval.preprocessing.preprocess_image)
so fallback_used/fallback_reason are recorded. This imports the actual
Streamlit app module (with chromadb stubbed out, since it isn't installed
in this environment and search_collection/search_collections don't need
the real package for this test) to exercise the real call path rather than
re-testing the already-covered core function in isolation.
"""

import importlib
import sys
import types
from pathlib import Path

from PIL import Image

SEARCH_APP_DIR = Path(__file__).resolve().parents[2] / "search-app"


def _install_fake_chromadb(monkeypatch):
    chromadb_mod = types.ModuleType("chromadb")
    utils_mod = types.ModuleType("chromadb.utils")
    embedding_functions_mod = types.ModuleType("chromadb.utils.embedding_functions")

    class FakeOpenCLIPEmbeddingFunction:
        def __init__(self, *args, **kwargs):
            pass

    embedding_functions_mod.OpenCLIPEmbeddingFunction = FakeOpenCLIPEmbeddingFunction
    chromadb_mod.utils = utils_mod
    utils_mod.embedding_functions = embedding_functions_mod

    monkeypatch.setitem(sys.modules, "chromadb", chromadb_mod)
    monkeypatch.setitem(sys.modules, "chromadb.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "chromadb.utils.embedding_functions", embedding_functions_mod)


def _import_musinsa_detect(monkeypatch, tmp_path):
    _install_fake_chromadb(monkeypatch)
    # musinsa_detect.py configures logging.FileHandler('musinsa_search.log')
    # at import time; run from a throwaway directory so the test doesn't
    # leave a stray log file in the repo.
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(SEARCH_APP_DIR))
    sys.modules.pop("musinsa_detect", None)
    return importlib.import_module("musinsa_detect")


def test_resolve_fallback_image_records_fallback_metadata(monkeypatch, tmp_path):
    musinsa_detect = _import_musinsa_detect(monkeypatch, tmp_path)

    image = Image.new("RGB", (64, 48))
    cropped_image, metadata = musinsa_detect.resolve_fallback_image(image, detected_items=[])

    # No fake detection was fabricated: the raw original image comes back...
    assert cropped_image is image
    # ...and the fallback event is explicitly recorded, unlike the old
    # hand-built synthetic-detection behavior this replaces.
    assert metadata["mode"] == "raw"
    assert metadata["fallback_used"] is True
    assert metadata["fallback_reason"] == "no_detections"
    assert metadata["selected_bbox"] is None


def test_resolve_fallback_image_delegates_to_shared_preprocess_image(monkeypatch, tmp_path):
    musinsa_detect = _import_musinsa_detect(monkeypatch, tmp_path)

    calls = []

    def fake_preprocess_image(image, detections, policy, fallback_policy):
        calls.append((image, detections, policy, fallback_policy))
        return "sentinel-image", {"mode": "raw", "fallback_used": True, "fallback_reason": "no_detections"}

    monkeypatch.setattr(musinsa_detect, "preprocess_image", fake_preprocess_image)

    image = object()
    result_image, metadata = musinsa_detect.resolve_fallback_image(image, detected_items=[])

    assert result_image == "sentinel-image"
    assert calls == [(image, [], "largest", "raw")]
