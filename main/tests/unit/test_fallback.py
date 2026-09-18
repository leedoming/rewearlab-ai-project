import pytest
from PIL import Image

from retrieval.preprocessing import RetrievalFallbackError, preprocess_image


def make_detection(bbox, label, score):
    x1, y1, x2, y2 = bbox
    return {"bbox": bbox, "label": label, "score": score, "area": (x2 - x1) * (y2 - y1)}


def make_image(size=(200, 200)):
    return Image.new("RGB", size)


def test_successful_selection_records_no_fallback():
    detections = [make_detection([10, 10, 60, 60], "outer", 0.8)]
    image = make_image()

    processed, metadata = preprocess_image(image, detections, "highest_confidence")

    assert processed.size == (50, 50)
    assert metadata["mode"] == "bbox"
    assert metadata["fallback_used"] is False
    assert metadata["fallback_reason"] is None
    assert metadata["selected_bbox"] == [10, 10, 60, 60]
    assert metadata["selected_label"] == "outer"
    assert metadata["detection_score"] == 0.8
    assert metadata["bbox_area"] == 2500


def test_raw_fallback_on_no_detections():
    image = make_image()

    processed, metadata = preprocess_image(image, [], "highest_confidence", fallback_policy="raw")

    assert processed is image
    assert metadata["mode"] == "raw"
    assert metadata["fallback_used"] is True
    assert metadata["fallback_reason"] == "no_detections"
    # No selection occurred: metadata must not fabricate a bbox/label/score.
    assert metadata["selected_bbox"] is None
    assert metadata["selected_label"] is None
    assert metadata["detection_score"] is None
    assert metadata["bbox_area"] is None


def test_raw_fallback_on_no_category_compatible_bbox():
    detections = [make_detection([0, 0, 30, 30], "bottom", 0.9)]
    image = make_image()

    processed, metadata = preprocess_image(
        image, detections, "category_confidence", category="상의", fallback_policy="raw"
    )

    assert processed is image
    assert metadata["mode"] == "raw"
    assert metadata["fallback_used"] is True
    assert metadata["fallback_reason"] == "no_category_compatible_bbox"


def test_largest_fallback_uses_largest_overall_detection():
    detections = [
        make_detection([0, 0, 10, 10], "bottom", 0.9),  # not category-compatible
        make_detection([0, 0, 40, 40], "bottom", 0.1),  # largest overall, still not compatible
    ]
    image = make_image()

    processed, metadata = preprocess_image(
        image, detections, "category_confidence", category="상의", fallback_policy="largest"
    )

    assert processed.size == (40, 40)
    assert metadata["mode"] == "bbox"
    assert metadata["fallback_used"] is True
    assert metadata["fallback_reason"] == "no_category_compatible_bbox"
    assert metadata["selected_label"] == "bottom"
    assert metadata["bbox_area"] == 1600


def test_largest_fallback_with_zero_detections_degrades_to_raw():
    image = make_image()

    processed, metadata = preprocess_image(image, [], "highest_confidence", fallback_policy="largest")

    assert processed is image
    assert metadata["mode"] == "raw"
    assert metadata["fallback_used"] is True
    assert metadata["fallback_reason"] == "no_detections"


def test_fail_fallback_raises():
    image = make_image()

    with pytest.raises(RetrievalFallbackError):
        preprocess_image(image, [], "highest_confidence", fallback_policy="fail")


def test_unknown_fallback_policy_raises():
    image = make_image()

    with pytest.raises(ValueError):
        preprocess_image(image, [], "highest_confidence", fallback_policy="not_a_real_policy")
