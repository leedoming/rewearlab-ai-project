"""Quantify the exact phenomenon that originally motivated adding detection/
bbox cropping to this project: certain catalog items with messy thumbnails
(multiple garments/people visible) get over-retrieved as false-positive
Top-K neighbors for many unrelated queries ("hubness" in embedding space).

Method: embed every image in the real 2160-image pool (see
scan_messy_thumbnails.py) two ways -- RAW (whole image) and BBox
(highest_confidence detection, cropped, RAW fallback if nothing detected;
`highest_confidence` chosen deliberately as the simplest policy, matching
what an MVP fix would look like, not RE:WEAR's later category-aware
policies). For each embedding scheme, treat every image as a query against
every other image (leave-one-out, brute-force cosine similarity -- exact,
not approximate), take each query's Top-10 neighbors, and tally how many
times each catalog item appears as *someone else's* Top-10 neighbor. A
"hub" is an item with a disproportionately high tally relative to the
average.

Not part of the reusable evaluation/ package -- a one-off analysis script
for docs/evidence/milestone-10-hubness.md.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image

from retrieval.config import DETECTION_THRESHOLD, EMBEDDING_MODEL, MIN_BBOX_AREA
from retrieval.detection import detect_fashion_items
from retrieval.embedding import embed_image
from retrieval.models import load_detection_model, load_embedding_model
from retrieval.preprocessing import preprocess_image

TOP_K = 10
OUTPUT_DIR = Path(__file__).parent / "_local"


def main():
    scan = json.loads((OUTPUT_DIR / "messy_scan.json").read_text(encoding="utf-8"))
    print(f"{len(scan)} images to embed (RAW + BBox)")

    image_processor, det_model, det_device = load_detection_model()
    embed_model, preprocess_fn, embed_device = load_embedding_model(EMBEDDING_MODEL)

    raw_embeddings = []
    bbox_embeddings = []
    meta = []

    for i, row in enumerate(scan):
        if i % 100 == 0:
            print(f"{i}/{len(scan)}...")
        path = row["path"]
        try:
            with Image.open(path) as raw_image:
                image = raw_image.convert("RGB")
                raw_emb = embed_image(image, embed_model, preprocess_fn, embed_device)

                detections = detect_fashion_items(
                    image, image_processor=image_processor, model=det_model, device=det_device,
                    threshold=DETECTION_THRESHOLD, min_area=MIN_BBOX_AREA,
                )
                cropped, metadata = preprocess_image(
                    image, detections, policy="highest_confidence", fallback_policy="raw",
                )
                bbox_emb = embed_image(cropped, embed_model, preprocess_fn, embed_device)
        except Exception as exc:
            print(f"  ERROR on {path}: {exc}")
            continue

        raw_embeddings.append(raw_emb)
        bbox_embeddings.append(bbox_emb)
        meta.append({"path": path, "folder": row["folder"], "num_detections": row["num_detections"]})

    raw_matrix = np.stack(raw_embeddings)
    bbox_matrix = np.stack(bbox_embeddings)
    np.save(OUTPUT_DIR / "raw_embeddings.npy", raw_matrix)
    np.save(OUTPUT_DIR / "bbox_embeddings.npy", bbox_matrix)
    (OUTPUT_DIR / "hubness_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nEmbedded {len(meta)} images. Computing hubness...")

    def compute_hub_counts(matrix):
        similarity = matrix @ matrix.T
        n = similarity.shape[0]
        np.fill_diagonal(similarity, -np.inf)
        top_k_idx = np.argpartition(-similarity, TOP_K, axis=1)[:, :TOP_K]
        counts = np.zeros(n, dtype=int)
        for row in top_k_idx:
            for idx in row:
                counts[idx] += 1
        return counts

    raw_counts = compute_hub_counts(raw_matrix)
    bbox_counts = compute_hub_counts(bbox_matrix)

    def gini(counts):
        sorted_counts = np.sort(counts.astype(float))
        n = len(sorted_counts)
        cumulative = np.cumsum(sorted_counts)
        return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n

    results = {
        "n_images": len(meta),
        "top_k": TOP_K,
        "raw": {
            "counts": raw_counts.tolist(),
            "gini": gini(raw_counts),
            "max_count": int(raw_counts.max()),
            "mean_count": float(raw_counts.mean()),
        },
        "bbox_highest_confidence": {
            "counts": bbox_counts.tolist(),
            "gini": gini(bbox_counts),
            "max_count": int(bbox_counts.max()),
            "mean_count": float(bbox_counts.mean()),
        },
    }
    (OUTPUT_DIR / "hubness_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"RAW:  gini={results['raw']['gini']:.4f}  max_count={results['raw']['max_count']}  mean={results['raw']['mean_count']:.2f}")
    print(f"BBox: gini={results['bbox_highest_confidence']['gini']:.4f}  max_count={results['bbox_highest_confidence']['max_count']}  mean={results['bbox_highest_confidence']['mean_count']:.2f}")

    # Correlate hub-ness with messy thumbnails
    messy_flags = np.array([1 if m["num_detections"] >= 2 else 0 for m in meta])
    raw_hub_threshold = np.percentile(raw_counts, 99)
    raw_hubs = raw_counts >= raw_hub_threshold
    print(f"\nTop 1% RAW hubs: {raw_hubs.sum()} items; of those, {messy_flags[raw_hubs].sum()} ({100*messy_flags[raw_hubs].mean():.1f}%) are messy thumbnails")
    print(f"Overall messy rate in the pool: {100*messy_flags.mean():.1f}%")


if __name__ == "__main__":
    main()
