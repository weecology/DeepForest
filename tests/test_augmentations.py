"""Test the new augmentations module."""
import io
import os

import torch
import kornia.augmentation as K
import pytest

from deepforest import main, get_data
from deepforest.augmentations import _create_augmentation
from deepforest.augmentations import _parse_augmentations
from deepforest.augmentations import get_available_augmentations
from deepforest.augmentations import get_transform
from deepforest.augmentations import _SUPPORTED_TRANSFORMS
from deepforest.datasets.training import BoxDataset

"""
Integration tests
"""

def test_override_transforms():
    """Test that augmentations can be overridden when calling load_dataset."""
    m = main.deepforest()

    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    train_ds = m.load_dataset(csv_file, root_dir=root_dir, augmentations=["VerticalFlip"])

    image, target, path = next(iter(train_ds))
    assert isinstance(train_ds.dataset.transform[0], K.RandomVerticalFlip)


def test_load_dataset_without_augmentations():
    """Test that dataset can be loaded with no augmentations."""
    m = main.deepforest()
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)

    train_ds = m.load_dataset(csv_file, root_dir=root_dir, augmentations=None)

    image, target, path = next(iter(train_ds))
    assert len(train_ds.dataset.transform) == 0

"""
Augmentation parsing tests:
"""

def test_parse_augmentations_string():
    result = _parse_augmentations("HorizontalFlip")
    assert result == {"HorizontalFlip": {}}


def test_parse_augmentations_dict():
    input_dict = {"HorizontalFlip": {"p": 0.5}, "VerticalFlip": {"p": 0.25}}
    result = _parse_augmentations(input_dict)
    assert result == input_dict


def test_parse_augmentations_string_list():
    result = _parse_augmentations(["HorizontalFlip", "VerticalFlip"])
    assert result == {"HorizontalFlip": {}, "VerticalFlip": {}}


def test_parse_augmentations_list_of_dict():
    # List of dicts format (for YAML support)
    list_of_dicts = [{"HorizontalFlip": {"p": 0.5}}, {"Rotate": {"degrees": 30, "p": 0.75}}]
    result = _parse_augmentations(list_of_dicts)
    expected = {"HorizontalFlip": {"p": 0.5}, "Rotate": {"degrees": 30, "p": 0.75}}
    assert result == expected


def test_parse_augmentations_string_and_dict():
    # Mixed list (strings and dicts)
    mixed_list = ["HorizontalFlip", {"Blur": {"kernel_size": (3, 3)}}, "VerticalFlip"]
    result = _parse_augmentations(mixed_list)
    expected = {"HorizontalFlip": {}, "Blur": {"kernel_size": (3, 3)}, "VerticalFlip": {}}
    assert result == expected


def test_parse_augmentations_empty():
    # Empty list
    result = _parse_augmentations([])
    assert result == {}


def test_parse_augmentations_invalid_multiple_key():
    # Invalid list with multiple keys in dict
    with pytest.raises(ValueError, match="one key"):
        _parse_augmentations([{"HorizontalFlip": {"p": 0.5}, "Rotate": {"degrees": 30}}])


def test_parse_augmentations_invalid_non_string():
    # Invalid list with non-string/non-dict elements
    with pytest.raises(ValueError, match="List elements must be strings or dicts"):
        _parse_augmentations([{"HorizontalFlip": {"p": 0.5}}, 123])


def test_parse_augmentations_omegaconf():
    # Test OmegaConf types (ListConfig and DictConfig)
    from omegaconf import OmegaConf

    # Test ListConfig
    yaml_config = """
    augmentations:
      - HorizontalFlip:
          p: 0.7
      - Blur:
          blur_limit: 3
    """
    config = OmegaConf.load(io.StringIO(yaml_config))
    result = _parse_augmentations(config.augmentations)
    expected = {"HorizontalFlip": {"p": 0.7}, "Blur": {"blur_limit": 3}}
    assert result == expected

    # Test DictConfig
    yaml_config2 = """
    augmentations:
      HorizontalFlip:
        p: 0.7
      Blur:
        blur_limit: 3
    """
    config2 = OmegaConf.load(io.StringIO(yaml_config2))
    result2 = _parse_augmentations(config2.augmentations)
    expected2 = {"HorizontalFlip": {"p": 0.7}, "Blur": {"blur_limit": 3}}
    assert result2 == expected2

"""
Higher level get_transform tests:
"""

def test_get_transform_default():
    """Test default behavior returns empty AugmentationSequential."""
    transform = get_transform()
    assert isinstance(transform, K.AugmentationSequential)
    assert len(transform) == 0


def test_get_transform_single_augmentation():
    """Test with single augmentation name."""
    transform = get_transform(augmentations="HorizontalFlip")
    assert isinstance(transform, K.AugmentationSequential)
    assert len(transform) == 1
    assert isinstance(transform[0], K.RandomHorizontalFlip)


def test_get_transform_multiple_augmentations():
    """Test augmentation list of strings."""
    transform = get_transform(augmentations=["HorizontalFlip", "VerticalFlip"])
    assert isinstance(transform, K.AugmentationSequential)
    assert len(transform) == 2
    assert isinstance(transform[0], K.RandomHorizontalFlip)
    assert isinstance(transform[1], K.RandomVerticalFlip)


def test_get_transform_with_parameters():
    """Test with overidden parameters."""
    augmentations = {
        "HorizontalFlip": {"p": 0.8},
        "VerticalFlip": {"p": 0.3}
    }
    transform = get_transform(augmentations=augmentations)
    assert transform[0].p == 0.8
    assert transform[1].p == 0.3


def test_create_augmentation():
    """Test _create_augmentation function."""
    # Valid augmentation
    aug = _create_augmentation("HorizontalFlip", {"p": 0.7})
    assert isinstance(aug, K.RandomHorizontalFlip)
    assert aug.p == 0.7

    # Invalid augmentation should raise ValueError
    with pytest.raises(ValueError, match="Unknown augmentation 'InvalidAugmentation'"):
        _create_augmentation("InvalidAugmentation", {})


def test_blur_augmentations():
    """Test that all blur augmentations can be created successfully."""
    blur_augmentations = ["Blur", "GaussianBlur", "MotionBlur"]
    expected_types = [K.RandomBoxBlur, K.RandomGaussianBlur, K.RandomMotionBlur]

    for blur_aug, expected_type in zip(blur_augmentations, expected_types):
        transform = get_transform(augmentations=[{blur_aug: {}}])
        assert isinstance(transform, K.AugmentationSequential)
        assert len(transform) == 1
        assert isinstance(transform[0], expected_type)


def test_blur_augmentations_with_parameters():
    """Test blur augmentations with custom parameters."""
    blur_configs = {
        "GaussianBlur": {"kernel_size": (5, 5), "p": 0.8},
        "MotionBlur": {"kernel_size": 7, "p": 0.6},
    }

    transform = get_transform(augmentations=blur_configs)
    assert isinstance(transform, K.AugmentationSequential)
    assert len(transform) == 2


def test_mixed_blur_and_other_augmentations():
    """Test combining blur augmentations with other augmentations using mixed format."""
    mixed_augmentations = ["HorizontalFlip", {"GaussianBlur": {"kernel_size": (3, 3)}}, "VerticalFlip", {"MotionBlur": {"kernel_size": 5}}]

    transform = get_transform(augmentations=mixed_augmentations)
    assert isinstance(transform, K.AugmentationSequential)
    assert len(transform) == 4
    assert isinstance(transform[0], K.RandomHorizontalFlip)
    assert isinstance(transform[1], K.RandomGaussianBlur)
    assert isinstance(transform[2], K.RandomVerticalFlip)
    assert isinstance(transform[3], K.RandomMotionBlur)


def test_unknown_augmentation_error():
    """Test that unknown augmentations raise ValueError."""
    with pytest.raises(ValueError, match="Unknown augmentation 'UnknownAugmentation'"):
        get_transform(augmentations="UnknownAugmentation")


def test_unsupported_augmentations_raise_error():
    """Test that completely unknown augmentations raise ValueError."""
    with pytest.raises(ValueError, match="Unknown augmentation 'RandomSizedBBoxSafeCrop'"):
        get_transform(augmentations="RandomSizedBBoxSafeCrop")


def test_no_op_augmentation_pipeline():
    """Test that empty augmentation pipeline works correctly as no-op."""
    for augs in [None, []]:
        transform = get_transform(augmentations=augs)

        image = torch.randn(1, 3, 100, 100)
        bboxes = torch.tensor([[[10., 10., 50., 50.], [20., 20., 60., 60.]]])

        output = transform(image, bboxes)

        assert len(output) == 2
        assert torch.equal(output[0], image)
        assert torch.equal(output[1], bboxes)


def test_box_augmentation():
    """Test that augmentations correctly transform images and bboxes."""
    transform = get_transform(augmentations={"HorizontalFlip": {"p": 1.0}})

    image = torch.randn(1, 3, 100, 100)
    bboxes = torch.tensor([[[10., 10., 50., 50.]]])

    output_image, output_bboxes = transform(image, bboxes)

    assert output_image.shape == image.shape
    assert output_bboxes.shape == bboxes.shape
    assert not torch.equal(output_bboxes, bboxes)





def test_zoom_blur():
    """Test ZoomBlur augmentation works correctly."""
    transform = get_transform(augmentations={"ZoomBlur": {"p": 1.0}})

    image = torch.randn(1, 3, 100, 100)
    bboxes = torch.tensor([[[10., 10., 50., 50.]]])

    output_image, output_bboxes = transform(image, bboxes)

    assert output_image.shape == image.shape
    assert output_bboxes.shape == bboxes.shape
    assert torch.equal(output_bboxes, bboxes)


def test_random_pad_to():
    """Test RandomPadTo augmentation adds random padding."""
    transform = get_transform(augmentations={"RandomPadTo": {"pad_range": (5, 5), "p": 1.0}})

    image = torch.randn(1, 3, 100, 100)
    bboxes = torch.tensor([[[10., 10., 50., 50.]]])

    output_image, output_bboxes = transform(image, bboxes)

    assert output_image.shape == torch.Size([1, 3, 105, 105])
    assert torch.equal(output_bboxes, bboxes)


def test_filter_boxes():
    """Test box filtering after augmentation."""
    boxes = torch.tensor([
        [-50., -50., 0., 0.],       # Out of bounds, becomes zero area
        [250., 250., 290., 290.],   # Out of bounds
        [50., 50., 150., 150.],     # Valid
        [-20., 80., 50., 150.],     # Partial, valid after clamp
        [150., 80., 220., 150.],    # Partial, valid after clamp
        [0., 0., 0., 10.],         # Too small
    ])
    labels = torch.tensor([0, 1, 2, 3, 4, 5])

    dataset = BoxDataset.__new__(BoxDataset)
    filtered_boxes, filtered_labels = dataset.filter_boxes(boxes, labels, width=200, height=200, min_size=1)

    assert filtered_boxes.shape[0] == 3
    assert torch.equal(filtered_labels, torch.tensor([2, 3, 4]))
    assert torch.all((filtered_boxes[:, 2] - filtered_boxes[:, 0]) >= 1)
    assert torch.all((filtered_boxes[:, 3] - filtered_boxes[:, 1]) >= 1)


def test_geometric_augmentation_filters_boxes():
    """Test that geometric augmentations filter out-of-bounds boxes in dataset."""


    m = main.deepforest()
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)

    # Use random crop which will place edge boxes out of bounds
    train_loader = m.load_dataset(
        csv_file,
        root_dir=root_dir,
        augmentations={"RandomCrop": {"size": (200, 200), "p": 1.0}}
    )
    dataset = train_loader.dataset

    # Get a few samples and verify boxes are valid
    for i in range(min(5, len(dataset))):
        image, targets, path = dataset[i]
        boxes = targets["boxes"]
        labels = targets["labels"]

        # Image shape is (C, H, W) after augmentation
        C, H, W = image.shape
        assert H == 200 and W == 200, f"Expected 200x200 crop, got {H}x{W}"

        # All boxes should be within bounds
        assert torch.all(boxes[:, 0] >= 0), f"Box x1 out of bounds: {boxes}"
        assert torch.all(boxes[:, 1] >= 0), f"Box y1 out of bounds: {boxes}"
        assert torch.all(boxes[:, 2] <= W), f"Box x2 out of bounds: {boxes}"
        assert torch.all(boxes[:, 3] <= H), f"Box y2 out of bounds: {boxes}"

        # All boxes should meet minimum size
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        assert torch.all(widths >= 1), f"Box width too small: {widths}"
        assert torch.all(heights >= 1), f"Box height too small: {heights}"

        # Labels should match box count
        assert len(labels) == len(boxes), f"Label count mismatch: {len(labels)} labels, {len(boxes)} boxes"


if __name__ == "__main__":
    pytest.main([__file__])
