from retrieval.preprocessing import select_bbox


def make_detection(bbox, label, score, area=None):
    x1, y1, x2, y2 = bbox
    return {
        "bbox": bbox,
        "label": label,
        "score": score,
        "area": area if area is not None else (x2 - x1) * (y2 - y1),
    }


DETECTIONS = [
    make_detection([0, 0, 10, 10], "top", 0.5),      # area 100, small, high-ish score
    make_detection([0, 0, 100, 100], "bottom", 0.3),  # area 10000, largest, low score
    make_detection([0, 0, 50, 50], "outer", 0.9),     # area 2500, highest confidence overall
]


def test_highest_confidence_selects_top_score():
    selected = select_bbox(DETECTIONS, "highest_confidence")
    assert selected["label"] == "outer"
    assert selected["score"] == 0.9


def test_largest_selects_top_area():
    selected = select_bbox(DETECTIONS, "largest")
    assert selected["label"] == "bottom"
    assert selected["area"] == 10000


def test_category_confidence_filters_then_picks_highest_score():
    # "상의" maps to ["top", "outer"] — excludes "bottom" (score 0.3, would
    # otherwise not matter here) and picks the higher-scoring of top/outer.
    selected = select_bbox(DETECTIONS, "category_confidence", category="상의")
    assert selected["label"] == "outer"
    assert selected["score"] == 0.9


def test_category_largest_filters_then_picks_largest_area():
    detections = [
        make_detection([0, 0, 10, 10], "top", 0.9),
        make_detection([0, 0, 40, 40], "outer", 0.1),
        make_detection([0, 0, 100, 100], "bottom", 0.99),
    ]
    # "상의" -> ["top", "outer"]; "bottom" is excluded despite highest score/area.
    selected = select_bbox(detections, "category_largest", category="상의")
    assert selected["label"] == "outer"
    assert selected["area"] == 1600


def test_category_policy_with_no_compatible_detection_returns_none():
    detections = [make_detection([0, 0, 10, 10], "bottom", 0.9)]
    assert select_bbox(detections, "category_confidence", category="상의") is None
    assert select_bbox(detections, "category_largest", category="상의") is None


def test_category_policy_with_no_category_returns_none():
    assert select_bbox(DETECTIONS, "category_confidence", category=None) is None


def test_empty_detections_returns_none_for_every_policy():
    for policy in ("highest_confidence", "largest", "category_confidence", "category_largest"):
        assert select_bbox([], policy, category="상의") is None


def test_unknown_policy_raises():
    import pytest

    with pytest.raises(ValueError):
        select_bbox(DETECTIONS, "not_a_real_policy")
