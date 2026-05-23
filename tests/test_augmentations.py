"""Test the augmentations module."""

import io
import os

import kornia.augmentation as K
import pytest
import torch
from kornia.constants import DataKey

from deepforest import get_data, main
from deepforest.augmentations import RandomSizedBBoxSafeCrop
from deepforest.augmentations import _create_augmentation
from deepforest.augmentations import _parse_augmentations
from deepforest.augmentations import apply_augmentations
from deepforest.augmentations import get_transform
from deepforest.datasets.training import BoxDataset, PointDataset


@pytest.mark.parametrize(
    "data_keys,annotations,expected_shape",
    [
        (
            [DataKey.IMAGE, DataKey.BBOX_XYXY],
            [torch.tensor([[[10.0, 10.0, 50.0, 50.0]]])],
            (50, 80),
        ),
        (
            [DataKey.IMAGE, DataKey.KEYPOINTS],
            [torch.tensor([[[25.0, 30.0]]])],
            (50, 80),
        ),
    ],
)
def test_fixed_resize_transforms_annotations(data_keys, annotations, expected_shape):
    """FixedResize scales image and aligned boxes or points."""
    transform = get_transform(
        augmentations={"FixedResize": {"size": expected_shape, "p": 1.0}},
        data_keys=data_keys,
    )
    image = torch.rand(1, 3, 100, 100)
    out_image, out_ann = apply_augmentations(transform, image, *annotations, data_keys=data_keys)
    assert out_image.shape[-2:] == expected_shape
    assert out_ann.shape == annotations[0].shape
    if data_keys[1] == DataKey.BBOX_XYXY:
        assert out_ann[0, 0, 2] > out_ann[0, 0, 0]
    else:
        assert out_ann[0, 0, 0] < expected_shape[1]
        assert out_ann[0, 0, 1] < expected_shape[0]


def test_random_sized_bbox_safe_crop_preserves_boxes():
    """BBox-safe crop keeps every box inside the resized output."""
    transform = get_transform(
        augmentations={
            "RandomSizedBBoxSafeCrop": {
                "size": (200, 200),
                "context_scale_range": (1.0, 1.5),
                "p": 1.0,
            }
        }
    )
    image = torch.rand(1, 3, 400, 400)
    boxes = torch.tensor([[[50.0, 60.0, 120.0, 140.0], [200.0, 180.0, 280.0, 260.0]]])
    out_image, out_boxes = apply_augmentations(transform, image, boxes)
    h, w = out_image.shape[-2:]
    flat = out_boxes.reshape(-1, 4)
    assert (h, w) == (200, 200)
    assert torch.all(flat[:, 0] >= 0) and torch.all(flat[:, 1] >= 0)
    assert torch.all(flat[:, 2] <= w) and torch.all(flat[:, 3] <= h)


@pytest.mark.parametrize(
    "dataset_cls,augmentations,check",
    [
        (
            BoxDataset,
            {"FixedResize": {"size": (128, 128), "p": 1.0}},
            lambda image, targets: image.shape == (3, 128, 128)
            and torch.all(targets["boxes"][:, 2] <= 128),
        ),
        (
            PointDataset,
            {"FixedResize": {"size": (128, 128), "p": 1.0}},
            lambda image, targets: image.shape == (3, 128, 128)
            and torch.all(targets["points"][:, 0] <= 128),
        ),
        (
            BoxDataset,
            {
                "RandomSizedBBoxSafeCrop": {
                    "size": (180, 180),
                    "p": 1.0,
                }
            },
            lambda image, targets: image.shape == (3, 180, 180),
        ),
    ],
)
def test_datasets_with_resize_augmentations(dataset_cls, augmentations, check):
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    kwargs = {"csv_file": csv_file, "root_dir": root_dir, "augmentations": augmentations}
    if dataset_cls is PointDataset:
        kwargs["label_dict"] = {"Tree": 0}
    dataset = dataset_cls(**kwargs)
    image, targets, _ = dataset[0]
    assert check(image, targets)


def test_parse_augmentations_string():
    assert _parse_augmentations("HorizontalFlip") == {"HorizontalFlip": {}}


def test_parse_augmentations_list_of_dict():
    result = _parse_augmentations(
        [{"HorizontalFlip": {"p": 0.5}}, {"Rotate": {"degrees": 30, "p": 0.75}}]
    )
    assert result == {"HorizontalFlip": {"p": 0.5}, "Rotate": {"degrees": 30, "p": 0.75}}


def test_create_augmentation_unknown():
    with pytest.raises(ValueError, match="Unknown augmentation"):
        _create_augmentation("InvalidAugmentation", {})


def test_get_transform_reorders_pad_after_flip():
    """Pads are applied after flips regardless of config list order."""
    image = torch.zeros(1, 3, 100, 100)
    box = torch.tensor([[[10.0, 5.0, 40.0, 30.0]]])
    transform = get_transform(
        augmentations=[
            {"RandomPadTo": {"pad_range": (50, 50), "p": 1.0}},
            {"HorizontalFlip": {"p": 1.0}},
            {"VerticalFlip": {"p": 1.0}},
        ]
    )
    out_image, out_box = transform(image, box)
    assert out_image.shape == torch.Size([1, 3, 150, 150])
    expected_box = torch.tensor([[[59.0, 69.0, 89.0, 94.0]]])
    assert torch.allclose(out_box, expected_box, atol=1.0)


def test_override_transforms_via_load_dataset():
    m = main.deepforest()
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    train_ds = m.load_dataset(csv_file, root_dir=root_dir, augmentations=["VerticalFlip"])
    assert isinstance(train_ds.dataset.transform[0], K.RandomVerticalFlip)


def test_filter_boxes():
    boxes = torch.tensor(
        [
            [-50.0, -50.0, 0.0, 0.0],
            [50.0, 50.0, 150.0, 150.0],
            [-20.0, 80.0, 50.0, 150.0],
        ]
    )
    labels = torch.tensor([0, 2, 3])
    dataset = BoxDataset.__new__(BoxDataset)
    filtered_boxes, filtered_labels = dataset.filter_boxes(
        boxes, labels, width=200, height=200, min_size=1
    )
    assert filtered_boxes.shape[0] == 2
    assert torch.equal(filtered_labels, torch.tensor([2, 3]))
