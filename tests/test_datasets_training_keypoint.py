"""Tests for KeypointDataset."""

import os

import numpy as np
import pandas as pd
import pytest
import torch

from deepforest import get_data
from deepforest.datasets.training import KeypointDataset


@pytest.fixture()
def keypoint_csv():
    return get_data("2019_BLAN_3_751000_4330000_image_crop_keypoints.csv")


@pytest.fixture()
def keypoint_root_dir():
    return os.path.dirname(
        get_data("2019_BLAN_3_751000_4330000_image_crop_keypoints.csv")
    )


@pytest.fixture()
def box_csv():
    """Bounding box CSV to test centroid conversion."""
    return get_data("example.csv")


@pytest.fixture()
def box_root_dir():
    return os.path.dirname(get_data("OSBS_029.png"))


def test_keypoint_dataset_centroid(keypoint_csv, keypoint_root_dir):
    """Basic construction, iteration, and output format."""
    ds = KeypointDataset(
        csv_file=keypoint_csv, root_dir=keypoint_root_dir, label_dict={"Tree": 0}, output="centroid"
    )
    raw = pd.read_csv(keypoint_csv)

    assert len(ds) == len(raw.image_path.unique())

    for i in range(len(ds)):
        image, targets, _path = ds[i]

        # Image: channels-first float tensor in [0, 1]
        assert torch.is_tensor(image)
        assert image.shape[0] == 3
        assert image.min() >= 0
        assert image.max() <= 1

        # Targets: correct shapes, types, and dtypes
        assert targets["points"].shape == (raw.shape[0], 2)
        assert targets["points"].dtype == torch.float32

        assert targets["labels"].shape == (raw.shape[0],)
        assert targets["labels"].dtype == torch.int64

def test_keypoint_dataset_density(keypoint_csv, keypoint_root_dir):
    """Density output mode should return a class-first tensor."""
    ds = KeypointDataset(
        csv_file=keypoint_csv,
        root_dir=keypoint_root_dir,
        label_dict={"Tree": 0},
        output="density",
    )

    image, targets, _ = ds[0]

    assert torch.is_tensor(image)
    assert "labels" in targets
    assert "points" not in targets
    assert targets["labels"].ndim == 3
    assert targets["labels"].shape[0] == 1
    assert targets["labels"].shape[1:] == image.shape[1:]
    assert targets["labels"].dtype == torch.float32
    assert targets["labels"].max() > 0


def test_keypoint_dataset_from_boxes(box_csv, box_root_dir):
    """When given bounding box geometry, annotations_for_path should extract centroids."""
    ds = KeypointDataset(
        csv_file=box_csv, root_dir=box_root_dir, label_dict={"Tree": 0}
    )
    raw = pd.read_csv(box_csv)

    _image, targets, _path = next(iter(ds))
    points = targets["points"]

    assert points.shape == (raw.shape[0], 2)

    # Verify raw annotations without augmentation
    targets = ds.annotations_for_path(ds.image_names[0])
    for i, (_, row) in enumerate(raw.iterrows()):
        expected_cx = (row["xmin"] + row["xmax"]) / 2
        expected_cy = (row["ymin"] + row["ymax"]) / 2
        np.testing.assert_allclose(
            targets["points"][i], [expected_cx, expected_cy], atol=0.01
        )


def test_keypoint_dataset_hflip(keypoint_csv, keypoint_root_dir):
    """Test that augmentation works by performing a horizontal flip augmentation,
    checking it correctly flips x coordinates and leaves y unchanged."""
    ds_orig = KeypointDataset(
        csv_file=keypoint_csv, root_dir=keypoint_root_dir,
    )
    ds_flip = KeypointDataset(
        csv_file=keypoint_csv, root_dir=keypoint_root_dir,
        augmentations=[{"HorizontalFlip": {"p": 1.0}}],
    )

    _, targets_orig, _ = ds_orig[0]
    _, targets_flip, _ = ds_flip[0]
    W = ds_orig.load_image(0).shape[1]

    # Flipped x should be approximately (W - original_x)
    torch.testing.assert_close(
        targets_flip["points"][:, 0],
        W - targets_orig["points"][:, 0],
        atol=1.0, rtol=0,
    )


def test_keypoint_dataset_validate_coordinates_oob(tmp_path, keypoint_root_dir):
    """Out-of-bounds points should raise ValueError."""
    image_name = "2019_BLAN_3_751000_4330000_image_crop.jpg"

    csv_path = str(tmp_path / "oob.csv")
    df = pd.DataFrame(
        {
            "image_path": [image_name],
            "x": [2000],  # image is 1024x1024
            "y": [500],
            "label": ["Tree"],
        }
    )
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="exceeds image dimensions"):
        KeypointDataset(
            csv_file=csv_path, root_dir=keypoint_root_dir, label_dict={"Tree": 0}
        )


def test_keypoint_dataset_validate_coordinates_negative(tmp_path, keypoint_root_dir):
    """Negative coordinates should raise ValueError."""
    image_name = "2019_BLAN_3_751000_4330000_image_crop.jpg"

    csv_path = str(tmp_path / "neg.csv")
    df = pd.DataFrame(
        {
            "image_path": [image_name],
            "x": [-10],
            "y": [500],
            "label": ["Tree"],
        }
    )
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="exceeds image dimensions"):
        KeypointDataset(
            csv_file=csv_path, root_dir=keypoint_root_dir, label_dict={"Tree": 0}
        )


def test_keypoint_dataset_empty_annotations(tmp_path, keypoint_root_dir):
    """Empty annotations (0,0) should produce empty targets."""
    image_name = "2019_BLAN_3_751000_4330000_image_crop.jpg"

    csv_path = str(tmp_path / "empty.csv")
    df = pd.DataFrame(
        {
            "image_path": [image_name],
            "x": [0],
            "y": [0],
            "label": ["Tree"],
        }
    )
    df.to_csv(csv_path, index=False)

    ds = KeypointDataset(
        csv_file=csv_path, root_dir=keypoint_root_dir, label_dict={"Tree": 0}
    )
    image, targets, path = ds[0]
    assert targets["points"].shape == (0, 2)
    assert targets["labels"].shape == (0,)


def test_keypoint_dataset_filter_points():
    """filter_points should remove out-of-bounds points."""
    ds_csv = get_data("2019_BLAN_3_751000_4330000_image_crop_keypoints.csv")
    root_dir = os.path.dirname(ds_csv)
    ds = KeypointDataset(csv_file=ds_csv, root_dir=root_dir, label_dict={"Tree": 0})

    points = torch.tensor([[10.0, 20.0], [-5.0, 30.0], [50.0, 60.0], [200.0, 300.0]])
    labels = torch.tensor([0, 0, 0, 0])
    image_shape = (3, 100, 100)  # H=100, W=100

    filtered_points, filtered_labels = ds.filter_points(points, labels, image_shape)

    assert filtered_points.shape[0] == 2  # only (10,20) and (50,60) are in bounds
    assert filtered_labels.shape[0] == 2
    torch.testing.assert_close(
        filtered_points, torch.tensor([[10.0, 20.0], [50.0, 60.0]])
    )


def test_keypoint_dataset_density_map(keypoint_csv, keypoint_root_dir):
    """Density map should place class-specific peaks at point locations."""
    ds = KeypointDataset(
        csv_file=keypoint_csv,
        root_dir=keypoint_root_dir,
        label_dict={"Tree": 0, "Shrub": 1},
        gaussian_radius=2,
        output="density",
    )

    points = torch.tensor([[25.0, 40.0], [70.0, 15.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.int64)
    density = ds.gaussian_density(points, labels, (3, 100, 100))

    assert density.shape == (2, 100, 100)
    assert density.dtype == torch.float32
    assert torch.argmax(density[0]).item() == (40 * 100 + 25)
    assert torch.argmax(density[1]).item() == (15 * 100 + 70)


def test_keypoint_dataset_density_ignores_oob_points(keypoint_csv, keypoint_root_dir):
    """Out-of-bounds points should not contribute to density map."""
    ds = KeypointDataset(
        csv_file=keypoint_csv,
        root_dir=keypoint_root_dir,
        label_dict={"Tree": 0},
        gaussian_radius=2,
        output="density",
    )

    points = torch.tensor([[-8.0, 40.0], [16.0, 16.0], [40.0, 160.0]], dtype=torch.float32)
    labels = torch.tensor([0, 0, 0], dtype=torch.int64)
    density = ds.gaussian_density(points, labels, (3, 64, 64))

    # Only the point [16.0, 16.0] is in bounds for a 64x64 image
    assert density.shape == (1, 64, 64)
    # Peak should be near position (16, 16) when flattened
    peak_pos = torch.argmax(density[0]).item()
    peak_y = peak_pos // 64
    peak_x = peak_pos % 64
    # Allow small tolerance due to Gaussian blur spreading
    assert abs(peak_x - 16.0) <= 1
    assert abs(peak_y - 16.0) <= 1
