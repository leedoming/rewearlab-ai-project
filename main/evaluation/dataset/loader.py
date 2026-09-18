"""Load and validate the human-authored retrieval evaluation dataset.

Only the Python standard library is used. Validation intentionally focuses on
errors that would make experiment results misleading: duplicate query IDs,
invalid controlled values, missing query images, unknown label query IDs, and
relevance grades outside the documented 0/1/2 scale.
"""

from dataclasses import dataclass
import csv
import json
from pathlib import Path


DIFFICULTIES = {"easy", "medium", "hard"}
SCENE_TYPES = {
    "clean_product",
    "person_wearing",
    "multi_item",
    "complex_background",
    "occluded",
}
SPLITS = {"dev", "holdout"}
RELEVANCE_GRADES = {0, 1, 2}
REQUIRED_COLUMNS = {
    "query_id",
    "image_path",
    "category",
    "difficulty",
    "scene_type",
    "split",
    "num_visible_items",
    "background_complexity",
    "important_features",
}


class DatasetValidationError(ValueError):
    """Raised when dataset files violate the documented schema."""


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    image_path: Path
    category: str
    difficulty: str
    scene_type: str
    split: str
    num_visible_items: int
    background_complexity: str
    important_features: tuple[str, ...]
    labels: dict[str, int]


@dataclass(frozen=True)
class EvaluationDataset:
    version: str
    pooled_ground_truth: bool
    queries: tuple[QueryRecord, ...]

    def by_id(self, query_id):
        for query in self.queries:
            if query.query_id == query_id:
                return query
        raise KeyError(query_id)


def _require_text(row, field, query_id):
    value = (row.get(field) or "").strip()
    if not value:
        raise DatasetValidationError(f"{query_id}: {field} must not be empty")
    return value


def _parse_manifest(manifest_path, require_images):
    dataset_dir = manifest_path.parent
    with manifest_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing_columns:
            raise DatasetValidationError(
                f"query manifest is missing columns: {sorted(missing_columns)}"
            )

        rows = []
        seen_ids = set()
        for line_number, row in enumerate(reader, start=2):
            query_id = _require_text(row, "query_id", f"line {line_number}")
            if query_id in seen_ids:
                raise DatasetValidationError(f"duplicate query_id: {query_id}")
            seen_ids.add(query_id)

            difficulty = _require_text(row, "difficulty", query_id)
            if difficulty not in DIFFICULTIES:
                raise DatasetValidationError(f"{query_id}: invalid difficulty {difficulty!r}")
            scene_type = _require_text(row, "scene_type", query_id)
            if scene_type not in SCENE_TYPES:
                raise DatasetValidationError(f"{query_id}: invalid scene_type {scene_type!r}")
            split = _require_text(row, "split", query_id)
            if split not in SPLITS:
                raise DatasetValidationError(f"{query_id}: invalid split {split!r}")

            try:
                num_visible_items = int(_require_text(row, "num_visible_items", query_id))
            except ValueError as exc:
                raise DatasetValidationError(
                    f"{query_id}: num_visible_items must be an integer"
                ) from exc
            if num_visible_items < 1:
                raise DatasetValidationError(f"{query_id}: num_visible_items must be positive")

            relative_image_path = Path(_require_text(row, "image_path", query_id))
            if relative_image_path.is_absolute() or ".." in relative_image_path.parts:
                raise DatasetValidationError(
                    f"{query_id}: image_path must stay inside the dataset directory"
                )
            image_path = dataset_dir / relative_image_path
            if require_images and not image_path.is_file():
                raise DatasetValidationError(f"{query_id}: query image not found: {image_path}")

            features = tuple(
                feature.strip()
                for feature in (row.get("important_features") or "").split("|")
                if feature.strip()
            )
            rows.append(
                {
                    "query_id": query_id,
                    "image_path": image_path,
                    "category": _require_text(row, "category", query_id),
                    "difficulty": difficulty,
                    "scene_type": scene_type,
                    "split": split,
                    "num_visible_items": num_visible_items,
                    "background_complexity": _require_text(
                        row, "background_complexity", query_id
                    ),
                    "important_features": features,
                }
            )
    return rows


def _parse_labels(labels_path, manifest_ids):
    with labels_path.open(encoding="utf-8") as file:
        document = json.load(file)

    version = document.get("dataset_version")
    if not isinstance(version, str) or not version.strip():
        raise DatasetValidationError("labels.dataset_version must be a non-empty string")
    if document.get("pooled_ground_truth") is not True:
        raise DatasetValidationError("labels.pooled_ground_truth must be true")
    query_labels = document.get("queries")
    if not isinstance(query_labels, dict):
        raise DatasetValidationError("labels.queries must be an object")

    unknown_ids = set(query_labels) - manifest_ids
    if unknown_ids:
        raise DatasetValidationError(
            f"labels contain query IDs absent from manifest: {sorted(unknown_ids)}"
        )

    parsed = {}
    for query_id, entries in query_labels.items():
        if not isinstance(entries, list):
            raise DatasetValidationError(f"{query_id}: labels must be a list")
        products = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise DatasetValidationError(f"{query_id}: each label must be an object")
            product_id = entry.get("product_id")
            relevance = entry.get("relevance")
            if not isinstance(product_id, str) or not product_id.strip():
                raise DatasetValidationError(f"{query_id}: product_id must be non-empty")
            if product_id in products:
                raise DatasetValidationError(f"{query_id}: duplicate product_id {product_id!r}")
            if (
                isinstance(relevance, bool)
                or not isinstance(relevance, int)
                or relevance not in RELEVANCE_GRADES
            ):
                raise DatasetValidationError(
                    f"{query_id}/{product_id}: relevance must be 0, 1, or 2"
                )
            products[product_id] = relevance
        parsed[query_id] = products
    return version, parsed


def load_dataset(dataset_dir, require_images=True):
    """Load ``query_manifest.csv`` and ``labels.json`` from ``dataset_dir``.

    ``require_images=False`` is intended for authoring/validation before the
    private query images have been copied into place. Experiment runners should
    keep the default so a dataset cannot silently run with missing inputs.
    """
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "query_manifest.csv"
    labels_path = dataset_dir / "labels.json"
    if not manifest_path.is_file():
        raise DatasetValidationError(f"query manifest not found: {manifest_path}")
    if not labels_path.is_file():
        raise DatasetValidationError(f"labels file not found: {labels_path}")

    manifest_rows = _parse_manifest(manifest_path, require_images=require_images)
    manifest_ids = {row["query_id"] for row in manifest_rows}
    version, labels = _parse_labels(labels_path, manifest_ids)
    queries = tuple(
        QueryRecord(**row, labels=labels.get(row["query_id"], {}))
        for row in manifest_rows
    )
    return EvaluationDataset(version=version, pooled_ground_truth=True, queries=queries)
