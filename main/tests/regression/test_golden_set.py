"""Functional regression checks, per IMPLEMENTATION_SPEC.md section 54's
own list: BBox coordinates valid, Fallback works, Category mapping works,
Top-K result shape valid, Metric computation correct.

These are structural, code-behavior invariants -- deterministic given
synthetic inputs, independent of model/embedding quality -- not a claim
about retrieval *quality* on a real golden set (that requires the real
10-15 curated queries `evaluation.golden_set.validate_golden_set` expects,
which don't exist yet; see docs/evidence/milestone-10.md). Each test here
maps 1:1 to one bullet of section 54's "Functional" list, kept in this
dedicated file (matching section 58's `tests/regression/test_golden_set.py`)
so the mapping from spec requirement to test is direct, even though some of
the same code paths already have more granular coverage under `tests/unit/`.
"""

from retrieval.category import get_allowed_labels
from retrieval.preprocessing import crop_image, preprocess_image, select_bbox
from retrieval.search import search_collections


class FakeCollection:
    def __init__(self, ids_and_distances):
        metadatas = [{"id": product_id, "product_id": product_id} for product_id, _ in ids_and_distances]
        distances = [distance for _, distance in ids_and_distances]
        self._result = {"metadatas": [metadatas], "distances": [distances]}

    def count(self):
        return len(self._result["metadatas"][0])

    def query(self, **kwargs):
        return self._result


class FakeClient:
    def __init__(self, collections):
        self._collections = collections

    def get_collection(self, name, embedding_function=None):
        return self._collections[name]


def make_image(width=100, height=100):
    from PIL import Image

    return Image.new("RGB", (width, height))


# --- BBox coordinates valid ---------------------------------------------------

def test_bbox_coordinates_stay_within_image_bounds_even_with_padding():
    image = make_image(width=100, height=100)
    # A bbox near the edge, expanded by padding, must clamp to the image --
    # never produce negative coordinates or coordinates beyond the image size.
    cropped = crop_image(image, bbox=[90, 90, 100, 100], padding_ratio=0.5)
    assert 0 < cropped.width <= 100
    assert 0 < cropped.height <= 100


def test_bbox_coordinates_valid_with_zero_padding():
    image = make_image(width=200, height=150)
    cropped = crop_image(image, bbox=[10, 20, 60, 70], padding_ratio=0.0)
    assert cropped.size == (50, 50)


# --- Fallback works ------------------------------------------------------------

def test_fallback_returns_raw_image_when_no_detections_found():
    image = make_image()
    cropped, metadata = preprocess_image(image, detections=[], policy="highest_confidence", fallback_policy="raw")
    assert cropped is image
    assert metadata["fallback_used"] is True
    assert metadata["fallback_reason"] == "no_detections"
    assert metadata["selected_bbox"] is None


def test_fallback_to_largest_when_no_category_compatible_bbox():
    detections = [{"bbox": [0, 0, 10, 10], "label": "bottom", "score": 0.9, "area": 100}]
    image = make_image()
    cropped, metadata = preprocess_image(
        image, detections, policy="category_confidence", category="outer", fallback_policy="largest"
    )
    assert metadata["fallback_used"] is True
    assert metadata["fallback_reason"] == "no_category_compatible_bbox"
    assert metadata["selected_label"] == "bottom"


# --- Category mapping works -----------------------------------------------------

def test_category_mapping_resolves_known_collection_names():
    # Regression for the real cross-milestone bug found in Milestone 5:
    # dataset queries are labeled with collection names ("outer"), not
    # Musinsa category strings ("아우터") -- both must resolve.
    assert get_allowed_labels("outer") == ["top", "outer"]
    assert get_allowed_labels("아우터") == ["top", "outer"]


def test_category_mapping_unknown_category_returns_empty_not_an_error():
    assert get_allowed_labels("not_a_real_category") == []


# --- Top-K result shape valid ---------------------------------------------------

def test_top_k_result_shape_is_valid_and_bounded():
    collection = FakeCollection([("P1", 0.1), ("P2", 0.2), ("P3", 0.3)])
    client = FakeClient({"top": collection})

    results = search_collections(client, ["top"], query_embedding=[0.0], top_k=2)

    assert len(results) <= 2
    assert [r["rank"] for r in results] == list(range(1, len(results) + 1))
    for r in results:
        assert set(r) == {"product_id", "rank", "raw_distance", "collection", "metadata"}


def test_top_k_result_shape_valid_with_fewer_results_than_k():
    collection = FakeCollection([("P1", 0.1)])
    client = FakeClient({"top": collection})

    results = search_collections(client, ["top"], query_embedding=[0.0], top_k=10)

    assert len(results) == 1
    assert results[0]["rank"] == 1


# --- Metric computation correct -------------------------------------------------

def test_metric_computation_correct_for_a_known_ranking():
    from evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k

    # Relevance grades [0, 1, 2] at ranks 1, 2, 3 -- hand-computed expectations.
    relevances = [0, 1, 2]
    assert precision_at_k(relevances, k=3) == 2 / 3
    assert recall_at_k(relevances, k=3, total_relevant=2) == 1.0
    assert mrr(relevances) == 0.5  # first relevant item at rank 2

    ideal_order = [2, 1, 0]
    assert ndcg_at_k(ideal_order, k=3) == 1.0  # ranking already in ideal (descending) order
    assert ndcg_at_k(relevances, k=3) < 1.0  # worse than ideal order -- discounted


def test_metric_computation_correct_selects_the_right_bbox_policy():
    detections = [
        {"bbox": [0, 0, 10, 10], "label": "top", "score": 0.9, "area": 100},
        {"bbox": [0, 0, 5, 5], "label": "outer", "score": 0.5, "area": 25},
    ]
    assert select_bbox(detections, "highest_confidence")["label"] == "top"
    assert select_bbox(detections, "largest")["label"] == "top"
