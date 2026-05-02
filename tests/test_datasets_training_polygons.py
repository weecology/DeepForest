# Test polygon dataset model
import os
import numpy as np
import json
import pytest
import torch
from PIL import Image

from deepforest import get_data
from deepforest.datasets.training import PolygonDataset


@pytest.fixture()
def polygon_annotation_file():
    return get_data("coco_sample_file.json")


@pytest.fixture()
def polygon_root_dir():
    return os.path.dirname(get_data("coco_sample_file.json"))


def test_polygon_dataset_basic(polygon_annotation_file, polygon_root_dir):
    ds = PolygonDataset(
        annotation_file=polygon_annotation_file,
        root_dir=polygon_root_dir,
        label_dict={"tree": 0},
    )

    image, targets = ds[0]

    # Image checks
    assert torch.is_tensor(image)
    assert image.shape[0] == 3
    assert image.min() >= 0
    assert image.max() <= 1

    # Targets checks
    assert "boxes" in targets
    assert "labels" in targets
    assert "masks" in targets
    assert "image_id" in targets
    assert "iscrowd" in targets
    assert "area" in targets

    assert targets["boxes"].shape[-1] == 4
    assert targets["labels"].dtype == torch.int64
    assert targets["masks"].dtype == torch.uint8


def test_generate_mask_polygon(polygon_annotation_file, polygon_root_dir):
    ds = PolygonDataset(
        annotation_file=polygon_annotation_file,
        root_dir=polygon_root_dir,
        label_dict={"tree": 0},
    )

    row = ds.annotations.iloc[0]
    image_path = os.path.join(ds.root_dir, row["image_path"])

    with Image.open(image_path) as img:
        width, height = img.size
    mask = ds.generate_mask(row, width=width, height=height)

    assert mask.shape == (height, width)
    assert mask.dtype == np.uint8
    assert mask.sum() > 0


def test_annotations_for_path_tensor(polygon_annotation_file, polygon_root_dir):
    ds = PolygonDataset(
        annotation_file=polygon_annotation_file,
        root_dir=polygon_root_dir,
        label_dict={"tree": 0},
    )

    image_name = ds.image_names[0]
    image_path = os.path.join(polygon_root_dir, image_name)

    targets = ds.annotations_for_path(image_name, image_path, return_tensor=True)

    assert torch.is_tensor(targets["boxes"])
    assert torch.is_tensor(targets["labels"])
    assert torch.is_tensor(targets["masks"])

    assert targets["boxes"].shape[-1] == 4
    assert targets["masks"].ndim == 3


def test_polygon_oob_coordinates(tmp_path, polygon_root_dir):
    image_name = "5b90f92da0d7280005fab355_4310.tif"
    data = {
        "images": [{"id": 1, "file_name": image_name, "width": 2048, "height": 2048}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[2049, 10, 2500, 10, 2048.1, 50, 200, 50]],
                "bbox": [200, 10, 50, 40],
                "area": 2000,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "tree"}],
    }

    json_path = tmp_path / "oob.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    with pytest.raises(ValueError):
        PolygonDataset(
            annotation_file=str(json_path),
            root_dir=polygon_root_dir,
            label_dict={"tree": 0},
        )


def test_polygon_negative_coordinates(tmp_path, polygon_root_dir):
    image_name = "5b90f92da0d7280005fab355_4310.tif"
    data = {
        "images": [{"id": 1, "file_name": image_name, "width": 2048, "height": 2048}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[-10, 10, 50, 10, 50, 50, -10, 50]],
                "bbox": [-10, 10, 50, 50],
                "area": 1600,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "tree"}],
    }

    json_path = tmp_path / "neg.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    with pytest.raises(ValueError):
        PolygonDataset(
            annotation_file=str(json_path),
            root_dir=polygon_root_dir,
            label_dict={"tree": 0},
        )
