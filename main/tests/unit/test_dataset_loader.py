import csv
import json

import pytest

from evaluation.dataset import DatasetValidationError, load_dataset


FIELDNAMES = [
    "query_id",
    "image_path",
    "category",
    "difficulty",
    "scene_type",
    "split",
    "num_visible_items",
    "background_complexity",
    "important_features",
]


def write_dataset(tmp_path, rows, queries=None):
    with (tmp_path / "query_manifest.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / "labels.json").write_text(
        json.dumps(
            {
                "dataset_version": "eval-v1",
                "pooled_ground_truth": True,
                "queries": queries or {},
            }
        ),
        encoding="utf-8",
    )


def valid_row(**overrides):
    row = {
        "query_id": "Q001",
        "image_path": "queries/Q001.jpg",
        "category": "outer",
        "difficulty": "hard",
        "scene_type": "person_wearing",
        "split": "dev",
        "num_visible_items": "2",
        "background_complexity": "high",
        "important_features": "black|leather|oversized collar",
    }
    row.update(overrides)
    return row


def test_load_dataset_combines_manifest_and_labels(tmp_path):
    image = tmp_path / "queries" / "Q001.jpg"
    image.parent.mkdir()
    image.write_bytes(b"fixture")
    write_dataset(tmp_path, [valid_row()], {"Q001": [{"product_id": "P10", "relevance": 2}]})

    dataset = load_dataset(tmp_path)

    query = dataset.by_id("Q001")
    assert dataset.version == "eval-v1"
    assert dataset.pooled_ground_truth is True
    assert query.image_path == image
    assert query.important_features == ("black", "leather", "oversized collar")
    assert query.labels == {"P10": 2}

    with pytest.raises(TypeError):
        query.labels["P10"] = 0


def test_real_repository_dataset_loads_with_images_present():
    # As of the Milestone 10 real-data pilot (docs/evidence/milestone-10-pilot.md),
    # evaluation/dataset is no longer the empty scaffold -- it holds 7 real,
    # pooled-labeled queries. This regression-tests that the committed
    # query_manifest.csv/labels.json/queries/ still form a valid, loadable
    # dataset (require_images=True, the experiment-runner default), not just
    # that the CSV/JSON shapes parse.
    from evaluation import dataset as dataset_package

    dataset_dir = dataset_package.__path__[0]
    dataset = load_dataset(dataset_dir)

    assert dataset.version == "pilot-real-v1"
    assert len(dataset.queries) == 7
    assert all(query.labels for query in dataset.queries)


@pytest.mark.parametrize(
    "override,error",
    [
        ({"difficulty": "extreme"}, "difficulty"),
        ({"category": "otuer"}, "category"),
        ({"scene_type": "studio-ish"}, "scene_type"),
        ({"split": "test"}, "split"),
        ({"num_visible_items": "0"}, "num_visible_items"),
        ({"image_path": "../outside.jpg"}, "image_path"),
    ],
)
def test_manifest_rejects_invalid_controlled_values(tmp_path, override, error):
    write_dataset(tmp_path, [valid_row(**override)])

    with pytest.raises(DatasetValidationError, match=error):
        load_dataset(tmp_path, require_images=False)


def test_manifest_rejects_duplicate_query_ids(tmp_path):
    write_dataset(tmp_path, [valid_row(), valid_row(image_path="queries/other.jpg")])

    with pytest.raises(DatasetValidationError, match="duplicate query_id"):
        load_dataset(tmp_path, require_images=False)


def test_loader_requires_image_by_default(tmp_path):
    write_dataset(tmp_path, [valid_row()])

    with pytest.raises(DatasetValidationError, match="query image not found"):
        load_dataset(tmp_path)


def test_labels_require_an_explicit_entry_for_every_manifest_query(tmp_path):
    write_dataset(tmp_path, [valid_row()], {})

    with pytest.raises(DatasetValidationError, match="missing from labels"):
        load_dataset(tmp_path, require_images=False)


def test_explicit_empty_label_list_is_allowed(tmp_path):
    write_dataset(tmp_path, [valid_row()], {"Q001": []})

    dataset = load_dataset(tmp_path, require_images=False)

    assert dataset.by_id("Q001").labels == {}


def test_labels_reject_unknown_query_and_invalid_relevance(tmp_path):
    write_dataset(tmp_path, [valid_row()], {"Q999": []})
    with pytest.raises(DatasetValidationError, match="absent from manifest"):
        load_dataset(tmp_path, require_images=False)

    write_dataset(tmp_path, [valid_row()], {"Q001": [{"product_id": "P10", "relevance": 3}]})
    with pytest.raises(DatasetValidationError, match="0, 1, or 2"):
        load_dataset(tmp_path, require_images=False)

    write_dataset(tmp_path, [valid_row()], {"Q001": [{"product_id": "P10", "relevance": 1.0}]})
    with pytest.raises(DatasetValidationError, match="0, 1, or 2"):
        load_dataset(tmp_path, require_images=False)


def test_labels_reject_duplicate_product_ids(tmp_path):
    labels = {
        "Q001": [
            {"product_id": "P10", "relevance": 1},
            {"product_id": "P10", "relevance": 2},
        ]
    }
    write_dataset(tmp_path, [valid_row()], labels)

    with pytest.raises(DatasetValidationError, match="duplicate product_id"):
        load_dataset(tmp_path, require_images=False)


def test_product_ids_are_trimmed_before_duplicate_detection_and_storage(tmp_path):
    labels = {"Q001": [{"product_id": " P10 ", "relevance": 2}]}
    write_dataset(tmp_path, [valid_row()], labels)

    dataset = load_dataset(tmp_path, require_images=False)

    assert dataset.by_id("Q001").labels == {"P10": 2}

    labels["Q001"].append({"product_id": "P10", "relevance": 1})
    write_dataset(tmp_path, [valid_row()], labels)
    with pytest.raises(DatasetValidationError, match="duplicate product_id"):
        load_dataset(tmp_path, require_images=False)


def test_labels_reject_non_object_root(tmp_path):
    write_dataset(tmp_path, [valid_row()], {"Q001": []})
    (tmp_path / "labels.json").write_text("[]", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="root must be an object"):
        load_dataset(tmp_path, require_images=False)
