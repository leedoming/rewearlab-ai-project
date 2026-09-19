"""BBox selection, cropping and fallback policy.

Detection (`retrieval.detection.detect_fashion_items`) and selection are
kept as separate responsibilities per IMPLEMENTATION_SPEC.md section 6.
"""

from PIL import Image

from .category import get_allowed_labels
from .config import (
    BBOX_SELECTION_POLICIES,
    DEFAULT_FALLBACK_POLICY,
    DEFAULT_PADDING_RATIO,
    FALLBACK_POLICIES,
)


class RetrievalFallbackError(Exception):
    """Raised when fallback_policy="fail" and no bbox could be selected."""


def _filter_by_category(detections, category):
    allowed_labels = get_allowed_labels(category) if category else []
    if not allowed_labels:
        return []
    return [d for d in detections if d["label"] in allowed_labels]


def select_bbox(detections, policy, category=None):
    """Select a single detection from `detections` according to `policy`.

    Policies:
        highest_confidence: highest `score` among all detections.
        largest: largest `area` among all detections.
        category_confidence: highest `score` among detections whose label
            is compatible with `category` (see retrieval.category).
        category_largest: largest `area` among detections whose label is
            compatible with `category`.

    Returns the selected detection dict, or None if `detections` is empty,
    or if a category policy has no compatible candidate. This function
    does not decide what to do when nothing is selected — that is the
    fallback policy's responsibility (see `preprocess_image`).
    """
    if policy not in BBOX_SELECTION_POLICIES:
        raise ValueError(f"Unknown bbox selection policy: {policy!r}")

    if not detections:
        return None

    if policy == "highest_confidence":
        return max(detections, key=lambda d: d["score"])

    if policy == "largest":
        return max(detections, key=lambda d: d["area"])

    candidates = _filter_by_category(detections, category)
    if not candidates:
        return None

    if policy == "category_confidence":
        return max(candidates, key=lambda d: d["score"])

    # policy == "category_largest"
    return max(candidates, key=lambda d: d["area"])


def crop_image(image, bbox, padding_ratio=0.0):
    """Crop `image` to `bbox`, expanded by `padding_ratio` and clamped to bounds.

    `bbox` is [x1, y1, x2, y2]. Padding is applied as a fraction of the
    bbox's own width/height on each side, then the result is clamped to
    the image bounds (matches the clamping in the existing `crop_image`
    helpers in musinsa_to_chromadb.py / musinsa_detect.py / app.py, which
    had no padding support).
    """
    if padding_ratio < 0:
        raise ValueError("padding_ratio must be non-negative")

    width, height = image.size
    x1, y1, x2, y2 = [float(coord) for coord in bbox]

    if padding_ratio:
        pad_x = (x2 - x1) * padding_ratio
        pad_y = (y2 - y1) * padding_ratio
        x1 -= pad_x
        x2 += pad_x
        y1 -= pad_y
        y2 += pad_y

    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(width, int(x2)), min(height, int(y2))
    return image.crop((x1, y1, x2, y2))


def letterbox_to_square(image, fill=(128, 128, 128)):
    """Pad `image` to a square canvas, centered, preserving its aspect ratio.

    The embedding model's own preprocessing (`open_clip`'s val transform)
    does a bare `Resize((224, 224))` with no aspect-ratio handling -- a
    non-square crop gets unevenly stretched to fit the square. This is
    proportionally much worse for a tall, narrow bbox crop (e.g. a
    full-length pants crop, aspect ratio ~0.37) than for a roughly
    portrait-shaped raw photo (~0.83): see
    docs/evidence/milestone-10-letterbox.md for the investigation this
    came from. Padding to square first means the model's resize only
    scales, never distorts.

    `fill` defaults to a neutral mid-gray (matching common object-crop
    letterboxing practice, e.g. YOLO's own preprocessing) rather than
    white or black, so it doesn't bias the padded region toward either a
    light or dark product-photo background.
    """
    width, height = image.size
    side = max(width, height)
    canvas = Image.new(image.mode if image.mode in ("RGB", "L") else "RGB", (side, side), fill)
    offset = ((side - width) // 2, (side - height) // 2)
    canvas.paste(image, offset)
    return canvas


def _bbox_metadata(selected, padding_ratio, fallback_used, fallback_reason):
    return {
        "mode": "bbox",
        "selected_bbox": selected["bbox"],
        "selected_label": selected["label"],
        "detection_score": selected["score"],
        "bbox_area": selected["area"],
        "padding_ratio": padding_ratio,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
    }


def _raw_metadata(fallback_reason):
    return {
        "mode": "raw",
        "selected_bbox": None,
        "selected_label": None,
        "detection_score": None,
        "bbox_area": None,
        "padding_ratio": None,
        "fallback_used": True,
        "fallback_reason": fallback_reason,
    }


def preprocess_image(
    image,
    detections,
    policy,
    category=None,
    padding_ratio=DEFAULT_PADDING_RATIO,
    fallback_policy=DEFAULT_FALLBACK_POLICY,
):
    """Select a bbox, crop the image, and record what happened.

    Returns (processed_image, metadata). `metadata` never contains
    fabricated selection fields: when no bbox is selected and the
    fallback policy is "raw", selected_bbox/selected_label/etc. are None
    and only fallback_used/fallback_reason are recorded.

    Fallback policies (used when `select_bbox` returns None):
        raw: return the original, uncropped image.
        largest: fall back to the largest detection overall (ignoring
            `category`), if any exists.
        fail: raise RetrievalFallbackError.
    """
    if fallback_policy not in FALLBACK_POLICIES:
        raise ValueError(f"Unknown fallback policy: {fallback_policy!r}")

    selected = select_bbox(detections, policy, category=category)
    if selected is not None:
        cropped = crop_image(image, selected["bbox"], padding_ratio=padding_ratio)
        return cropped, _bbox_metadata(selected, padding_ratio, fallback_used=False, fallback_reason=None)

    reason = "no_detections" if not detections else "no_category_compatible_bbox"

    if fallback_policy == "fail":
        raise RetrievalFallbackError(reason)

    if fallback_policy == "raw":
        return image, _raw_metadata(reason)

    # fallback_policy == "largest"
    fallback_selected = select_bbox(detections, "largest")
    if fallback_selected is None:
        return image, _raw_metadata(reason)

    cropped = crop_image(image, fallback_selected["bbox"], padding_ratio=padding_ratio)
    return cropped, _bbox_metadata(fallback_selected, padding_ratio, fallback_used=True, fallback_reason=reason)
