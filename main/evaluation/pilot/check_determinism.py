# -*- coding: utf-8 -*-
"""Regression check for the preprocess_train/preprocess_val bug fixed in
retrieval/models.py (see docs/evidence/milestone-10-transform-bug.md):
embeds the same image 3 times in a single process and confirms the
results are bit-identical. Before the fix, this showed up to a 2.8e-2
max absolute difference between "identical" calls, traced to
open_clip.create_model_and_transforms() being unpacked so that the
random-crop training transform was used at inference time instead of the
deterministic validation one.

Not part of the reusable evaluation/ package -- a one-off diagnostic
script, kept because it's the exact reproduction case this bug needed.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image

from retrieval.embedding import embed_image
from retrieval.models import load_embedding_model

BASE = Path(__file__).parent / "_local"


def main():
    embed_model, preprocess_fn, embed_device = load_embedding_model()

    catalog_manifest = json.loads((BASE / "catalog_manifest.json").read_text(encoding="utf-8"))
    sample_path = catalog_manifest[0]["source_path"]

    vecs = []
    for _ in range(3):
        with Image.open(sample_path) as raw_image:
            image = raw_image.convert("RGB")
            vecs.append(embed_image(image, embed_model, preprocess_fn, embed_device))

    for i in range(1, 3):
        identical = np.array_equal(vecs[0], vecs[i])
        max_diff = np.abs(vecs[0] - vecs[i]).max()
        print(f"run 0 vs run {i}: identical={identical}, max_abs_diff={max_diff:.2e}")


if __name__ == "__main__":
    main()
