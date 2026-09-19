# -*- coding: utf-8 -*-
"""Scan the 원피스/치마 (dress/skirt) folders from the itda-fashion-detect
source (a second, unrelated personal project) with real detection, to
separate usable single-product photos from multi-product collage banners
(a real problem in this specific source folder -- some filenames list
many garment types at once, e.g. promotional images showing 3 outfits
side by side)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image

from retrieval.config import DETECTION_THRESHOLD, MIN_BBOX_AREA
from retrieval.detection import detect_fashion_items
from retrieval.models import load_detection_model

SOURCE_ROOT = Path(r"C:\Users\smn07\Desktop\glacier-project\itda-fashion-detect\images\이미지")
FOLDERS = {"원피스": "dress_skirts", "치마": "dress_skirts"}
OUTPUT_DIR = Path(__file__).parent / "_local"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_processor, model, device = load_detection_model()

    rows = []
    for folder in FOLDERS:
        folder_path = SOURCE_ROOT / folder
        for path in sorted(folder_path.iterdir()):
            if not path.is_file():
                continue
            try:
                with Image.open(path) as raw_image:
                    image = raw_image.convert("RGB")
                    width, height = image.size
                    detections = detect_fashion_items(
                        image, image_processor=image_processor, model=model, device=device,
                        threshold=DETECTION_THRESHOLD, min_area=MIN_BBOX_AREA,
                    )
            except Exception as exc:
                print(f"ERROR on {path}: {exc}")
                continue
            labels = [d["label"] for d in detections]
            rows.append({
                "path": str(path), "folder": folder, "width": width, "height": height,
                "num_detections": len(detections), "labels": labels,
            })

    rows.sort(key=lambda r: r["num_detections"])
    for r in rows:
        print(f"{r['num_detections']}  {r['labels']}  {r['folder']}  {Path(r['path']).name}")

    (OUTPUT_DIR / "dress_skirt_scan.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
