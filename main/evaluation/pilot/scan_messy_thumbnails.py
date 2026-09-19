"""Scan every image in the itda source folders and run real fashion-object
detection over each one, to automatically flag "messy" thumbnails (multiple
garments/people visible) vs "clean" single-item product shots -- the exact
distinction that originally motivated adding detection/bbox cropping to this
project (see the user's own account: RAW embeddings over-retrieved certain
catalog items as false-positive Top-K results, traced back to their
thumbnails showing multiple garments or full-body shots instead of a single
product).

Heuristic: a thumbnail is "messy" if real detection (same threshold/min_area
as production, retrieval.config.DETECTION_THRESHOLD/MIN_BBOX_AREA) finds 2+
valid detections. A clean single-product photo should show exactly one.
This is deliberately a cheap, inspectable proxy, not a claim about ground
truth -- see docs/evidence/milestone-10-hubness.md for validation against a
manually-viewed sample.

Not part of the reusable evaluation/ package -- a one-off data-curation
script for building a hubness-analysis dataset from real (if unrelated-
project) Musinsa photos.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image

from retrieval.config import DETECTION_THRESHOLD, MIN_BBOX_AREA
from retrieval.detection import detect_fashion_items
from retrieval.models import load_detection_model

SOURCE_ROOT = Path(r"C:\Users\smn07\Desktop\glacier-project\itda")
FOLDERS = [
    "musinsa_pants_1000/바지",
    "musinsa_pants_1000/청바지",
    "musinsa_upper_2000/가디건",
    "musinsa_upper_2000/긴팔",
    "musinsa_upper_2000/나시",
    "musinsa_upper_2000/니트",
    "musinsa_upper_2000/맨투맨",
    "musinsa_upper_2000/반팔",
    "musinsa_upper_2000/셔츠",
    "musinsa_upper_2000/자켓",
    "musinsa_upper_2000/후드티",
]

OUTPUT_DIR = Path(__file__).parent / "_local"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_processor, model, device = load_detection_model()

    rows = []
    total = sum(1 for folder in FOLDERS for _ in (SOURCE_ROOT / folder).iterdir())
    done = 0
    for folder in FOLDERS:
        folder_path = SOURCE_ROOT / folder
        for path in sorted(folder_path.iterdir()):
            if not path.is_file():
                continue
            done += 1
            if done % 100 == 0:
                print(f"{done}/{total}...")
            try:
                with Image.open(path) as raw_image:
                    image = raw_image.convert("RGB")
                    width, height = image.size
                    detections = detect_fashion_items(
                        image, image_processor=image_processor, model=model, device=device,
                        threshold=DETECTION_THRESHOLD, min_area=MIN_BBOX_AREA,
                    )
            except Exception as exc:
                print(f"  ERROR on {path}: {exc}")
                continue

            labels = [d["label"] for d in detections]
            areas = [d["area"] for d in detections]
            image_area = width * height
            rows.append(
                {
                    "path": str(path),
                    "folder": folder,
                    "width": width,
                    "height": height,
                    "num_detections": len(detections),
                    "labels": labels,
                    "distinct_labels": sorted(set(labels)),
                    "max_area_ratio": (max(areas) / image_area) if areas else 0.0,
                }
            )

    (OUTPUT_DIR / "messy_scan.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT_DIR / "messy_scan.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "folder", "width", "height", "num_detections", "labels", "distinct_labels", "max_area_ratio"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "labels": "|".join(row["labels"]), "distinct_labels": "|".join(row["distinct_labels"])})

    messy = [r for r in rows if r["num_detections"] >= 2]
    clean = [r for r in rows if r["num_detections"] == 1]
    zero = [r for r in rows if r["num_detections"] == 0]
    print(f"\nScanned {len(rows)} images.")
    print(f"  messy (>=2 detections): {len(messy)} ({100*len(messy)/len(rows):.1f}%)")
    print(f"  clean (1 detection):    {len(clean)} ({100*len(clean)/len(rows):.1f}%)")
    print(f"  zero detections:        {len(zero)} ({100*len(zero)/len(rows):.1f}%)")


if __name__ == "__main__":
    main()
