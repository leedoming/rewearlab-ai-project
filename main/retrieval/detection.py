"""Fashion item detection.

`detect_fashion_items` returns every valid detection candidate. It does
not decide which one should be used for downstream cropping/embedding —
that is `retrieval.preprocessing.select_bbox`'s job (IMPLEMENTATION_SPEC.md
section 6: "Detection function은 bbox 선택까지 수행하지 않는다.").
"""

from .config import DETECTION_MODEL, DETECTION_THRESHOLD, MIN_BBOX_AREA
from .models import load_detection_model


def detect_fashion_items(
    image,
    image_processor=None,
    model=None,
    device=None,
    threshold=DETECTION_THRESHOLD,
    min_area=MIN_BBOX_AREA,
    model_name=DETECTION_MODEL,
):
    """Detect fashion items in `image` and return all valid candidates.

    Args:
        image: PIL Image.
        image_processor, model, device: pre-loaded detection model
            components (see `retrieval.models.load_detection_model`).
            If any is None, the model is loaded on demand using
            `model_name`.
        threshold: detection confidence threshold.
        min_area: minimum bbox area (width * height) in pixels; smaller
            detections are discarded.
        model_name: HF checkpoint to load when components aren't supplied.

    Returns:
        A list of dicts, each with keys: bbox ([x1, y1, x2, y2] ints),
        label (str), score (float), area (int). Unsorted, unfiltered by
        category — every detection that passes `threshold`/`min_area`.
    """
    import torch

    if image_processor is None or model is None:
        image_processor, model, device = load_detection_model(model_name, device)

    with torch.no_grad():
        inputs = image_processor(images=[image], return_tensors="pt")
        outputs = model(**inputs.to(device))
        target_sizes = torch.tensor([[image.size[1], image.size[0]]])
        results = image_processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
        )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = model.config.id2label[label.item()].lower()
            score_value = score.item()
            x1, y1, x2, y2 = [int(coord) for coord in box]
            area = (x2 - x1) * (y2 - y1)

            if area < min_area:
                continue

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "label": label_name,
                    "score": score_value,
                    "area": area,
                }
            )

        return detections
