# -*- coding: utf-8 -*-
"""Test two candidate fixes for the hubness finding in
docs/evidence/milestone-10-hubness.md, reusing the already-computed
embeddings (no model inference needed):

1. Within-category search: the real production UI already does per-object
   detection + per-category Top-K (not one blended cross-category search).
   Does restricting the candidate pool to the same folder/category as the
   query change the hubness picture from the whole-pool analysis?
2. Mean-centering: subtract the pool's mean embedding vector before cosine
   similarity -- a standard, cheap hubness-reduction technique (hub points
   tend to sit close to the distribution's mean; centering pushes them
   away from every query equally, not just some).

Not part of the reusable evaluation/ package -- a one-off follow-up
analysis for docs/evidence/milestone-10-hubness.md.
"""

import json
import collections

import numpy as np

BASE = r"C:\Users\smn07\Desktop\clothing-resale-project\2025-AI-REWEARLab\main\evaluation\pilot\_local"
TOP_K = 10


def gini(counts):
    sorted_counts = np.sort(counts.astype(float))
    n = len(sorted_counts)
    cumulative = np.cumsum(sorted_counts)
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n


def hub_counts_global(matrix, top_k=TOP_K):
    similarity = matrix @ matrix.T
    n = similarity.shape[0]
    np.fill_diagonal(similarity, -np.inf)
    top_k_idx = np.argpartition(-similarity, top_k, axis=1)[:, :top_k]
    counts = np.zeros(n, dtype=int)
    for row in top_k_idx:
        for idx in row:
            counts[idx] += 1
    return counts


def hub_counts_within_category(matrix, folders, top_k=TOP_K):
    """Same tally, but each query's candidate pool is restricted to items
    in the same folder (category) -- matching the real UI's per-category
    Top-K search instead of one blended cross-category search."""
    n = matrix.shape[0]
    counts = np.zeros(n, dtype=int)
    by_folder = collections.defaultdict(list)
    for i, f in enumerate(folders):
        by_folder[f].append(i)

    for folder, idxs in by_folder.items():
        if len(idxs) <= 1:
            continue
        sub = matrix[idxs]
        similarity = sub @ sub.T
        np.fill_diagonal(similarity, -np.inf)
        k = min(top_k, len(idxs) - 1)
        top_k_idx = np.argpartition(-similarity, k, axis=1)[:, :k] if k < similarity.shape[1] - 1 else np.argsort(-similarity, axis=1)[:, :k]
        for row in top_k_idx:
            for local_idx in row:
                counts[idxs[local_idx]] += 1
    return counts


def main():
    raw = np.load(f"{BASE}\\raw_embeddings.npy")
    bbox = np.load(f"{BASE}\\bbox_embeddings.npy")
    with open(f"{BASE}\\hubness_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    folders = [m["folder"] for m in meta]

    print("=== 1. Within-category search (matches real UI behavior) ===")
    for name, matrix in [("RAW", raw), ("BBox", bbox)]:
        global_counts = hub_counts_global(matrix)
        within_counts = hub_counts_within_category(matrix, folders)
        print(f"{name}: global Gini={gini(global_counts):.4f} max={global_counts.max()}  |  "
              f"within-category Gini={gini(within_counts):.4f} max={within_counts.max()}")

    print("\n=== 2. Mean-centering (hub-reduction technique) ===")
    for name, matrix in [("RAW", raw), ("BBox", bbox)]:
        mean_vec = matrix.mean(axis=0, keepdims=True)
        centered = matrix - mean_vec
        centered = centered / np.linalg.norm(centered, axis=1, keepdims=True)
        centered_counts = hub_counts_global(centered)
        original_counts = hub_counts_global(matrix)
        print(f"{name}: original Gini={gini(original_counts):.4f} max={original_counts.max()}  |  "
              f"centered Gini={gini(centered_counts):.4f} max={centered_counts.max()}")

    # Save results for the dashboard/evidence doc
    mean_vec_raw = raw.mean(axis=0, keepdims=True)
    centered_raw = raw - mean_vec_raw
    centered_raw = centered_raw / np.linalg.norm(centered_raw, axis=1, keepdims=True)
    centered_raw_counts = hub_counts_global(centered_raw)

    within_raw_counts = hub_counts_within_category(raw, folders)
    global_raw_counts = hub_counts_global(raw)

    results = {
        "global_raw": {"gini": gini(global_raw_counts), "max": int(global_raw_counts.max()), "counts": global_raw_counts.tolist()},
        "within_category_raw": {"gini": gini(within_raw_counts), "max": int(within_raw_counts.max()), "counts": within_raw_counts.tolist()},
        "mean_centered_raw": {"gini": gini(centered_raw_counts), "max": int(centered_raw_counts.max()), "counts": centered_raw_counts.tolist()},
    }
    with open(f"{BASE}\\hubness_mitigation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to hubness_mitigation_results.json")


if __name__ == "__main__":
    main()
