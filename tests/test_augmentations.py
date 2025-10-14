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


def test_get_transform_default():
    """Test default behavior (backward compatibility)."""
    # Test without augmentations
    transform = get_transform()
    assert isinstance(transform, torch.nn.Identity)


def test_get_transform_single_augmentation():
    """Test with single augmentation name."""
    transform = get_transform(augmentations="Downscale")
    assert isinstance(transform, torch.nn.Sequential)
    assert len(transform) == 1  # Just Downscale
    assert isinstance(transform[0], K.RandomResizedCrop)


def test_get_transform_multiple_augmentations():
    """Test with list of augmentation names (strings)."""
    transform = get_transform(augmentations=["HorizontalFlip", "Downscale"])
    assert isinstance(transform, torch.nn.Sequential)
    assert len(transform) == 2  # HorizontalFlip + Downscale
    assert isinstance(transform[0], K.RandomHorizontalFlip)
    assert isinstance(transform[1], K.RandomResizedCrop)


def test_get_transform_with_parameters():
    """Test with augmentation parameters."""
    augmentations = {
        "HorizontalFlip": {"p": 0.8},
        "Downscale": {"scale": (0.5, 0.9), "p": 0.3}
    }
    transform = get_transform(augmentations=augmentations)
    assert isinstance(transform, torch.nn.Sequential)
    assert len(transform) == 2  # HorizontalFlip + Downscale

    # Check parameters were applied
    assert transform[0].p == 0.8  # HorizontalFlip
    assert transform[1].scale == (0.5, 0.9)  # Downscale
    assert transform[1].p == 0.3


def test_parse_augmentations_string():
    """Test _parse_augmentations function."""
    # String input
    result = _parse_augmentations("HorizontalFlip")
    assert result == {"HorizontalFlip": {}}


def test_parse_augmentations_dict():
    # Dict input
    input_dict = {"HorizontalFlip": {"p": 0.5}, "Downscale": {"scale_min": 0.25}}
    result = _parse_augmentations(input_dict)
    assert result == input_dict


def test_parse_augmentations_string_list():
    # List of strings (simple augmentations)
    result = _parse_augmentations(["HorizontalFlip", "Downscale"])
    assert result == {"HorizontalFlip": {}, "Downscale": {}}


def test_parse_augmentations_list_of_dict():
    # List of dicts format (for YAML support)
    list_of_dicts = [{"HorizontalFlip": {"p": 0.5}}, {"Downscale": {"scale_min": 0.25, "scale_max": 0.75}}]
    result = _parse_augmentations(list_of_dicts)
    expected = {"HorizontalFlip": {"p": 0.5}, "Downscale": {"scale_min": 0.25, "scale_max": 0.75}}
    assert result == expected


def test_parse_augmentations_string_and_dict():
    # Mixed list (strings and dicts)
    mixed_list = ["HorizontalFlip", {"Blur": {"blur_limit": 3}}, "Downscale"]
    result = _parse_augmentations(mixed_list)
    expected = {"HorizontalFlip": {}, "Blur": {"blur_limit": 3}, "Downscale": {}}
    assert result == expected


def test_parse_augmentations_empty():
    # Empty list
    result = _parse_augmentations([])
    assert result == {}


def test_parse_augmentations_invalid_multiple_key():
    # Invalid list with multiple keys in dict
    with pytest.raises(ValueError, match="one key"):
        _parse_augmentations([{"HorizontalFlip": {"p": 0.5}, "Downscale": {"scale_min": 0.25}}])


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


def test_create_augmentation():
    """Test _create_augmentation function."""
    # Valid augmentation
    aug = _create_augmentation("HorizontalFlip", {"p": 0.7})
    assert isinstance(aug, K.RandomHorizontalFlip)
    assert aug.p == 0.7

    # Invalid augmentation should raise ValueError
    with pytest.raises(ValueError, match="Unknown augmentation 'InvalidAugmentation'"):
        _create_augmentation("InvalidAugmentation", {})


def test_get_available_augmentations():
    """Test get_available_augmentations function."""
    augs = get_available_augmentations()
    assert isinstance(augs, list)
    assert "HorizontalFlip" in augs
    assert "Downscale" in augs
    assert "RandomSizedBBoxSafeCrop" in augs
    assert "PadIfNeeded" in augs


def test_bbox_params():
    """Test that transforms are properly created."""
    transform = get_transform(augmentations="HorizontalFlip")

    # Check that transform is properly created
    assert isinstance(transform, torch.nn.Sequential)
    assert len(transform) == 1
    assert isinstance(transform[0], K.RandomHorizontalFlip)


def test_blur_augmentations():
    """Test that all blur augmentations can be created successfully."""
    blur_augmentations = ["Blur", "GaussianBlur"]

    for blur_aug in blur_augmentations:
        transform = get_transform(augmentations=[{blur_aug: {}}])
        assert isinstance(transform, torch.nn.Sequential)
        assert len(transform) == 1  # Just blur augmentation
        assert isinstance(transform[0], K.RandomGaussianBlur)


def test_blur_augmentations_with_parameters():
    """Test blur augmentations with custom parameters."""
    blur_configs = {
        "GaussianBlur": {"kernel_size": (5, 5), "p": 0.8},
        "MotionBlur": {"kernel_size": 7, "p": 0.6},
        "ZoomBlur": {"scale": (1.0, 1.3), "p": 0.4}
    }

    transform = get_transform(augmentations=blur_configs)
    assert isinstance(transform, torch.nn.Sequential)
    assert len(transform) == 3  # 3 blur augmentations


def test_mixed_blur_and_other_augmentations():
    """Test combining blur augmentations with other augmentations using mixed format."""
    mixed_augmentations = ["HorizontalFlip", {"GaussianBlur": {"kernel_size": (3, 3)}}, "Downscale", {"MotionBlur": {"kernel_size": 5}}]

    transform = get_transform(augmentations=mixed_augmentations)
    assert isinstance(transform, torch.nn.Sequential)
    assert len(transform) == 4  # 4 augmentations
    assert isinstance(transform[0], K.RandomHorizontalFlip)
    assert isinstance(transform[1], K.RandomGaussianBlur)
    assert isinstance(transform[2], K.RandomResizedCrop)
    assert isinstance(transform[3], K.RandomMotionBlur)


def test_unknown_augmentation_error():
    """Test that unknown augmentations raise ValueError."""
    with pytest.raises(ValueError, match="Unknown augmentation 'UnknownAugmentation'"):
        get_transform(augmentations="UnknownAugmentation")


def test_override_transforms():

    def get_transform(augment):
        """This is the new transform"""
        if augment:
            print("I'm a new augmentation!")
            transform = torch.nn.Sequential(
                K.RandomHorizontalFlip(p=0.5)
            )
        else:
            transform = torch.nn.Identity()
        return transform

    m = main.deepforest(transforms=get_transform)

    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    train_ds = m.load_dataset(csv_file, root_dir=root_dir, augmentations=["HorizontalFlip"])

    image, target, path = next(iter(train_ds))
    assert m.transforms.__doc__ == "This is the new transform"

def test_config_augmentations():
    """Test that augmentations can be configured via config."""
    # Test with config args containing augmentations
    config_args = {
        "train": {
            "augmentations": ["HorizontalFlip", "Downscale"]
        }
    }

    m = main.deepforest(config_args=config_args)
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)

    # Load dataset with config-based augmentations
    train_ds = m.load_dataset(csv_file, root_dir=root_dir, augmentations=["HorizontalFlip"])

    # Check that we can iterate over the dataset
    image, target, path = next(iter(train_ds))
    assert image is not None


def test_config_augmentations_with_params():
    """Test that augmentations with parameters can be configured via config."""
    # Test with config args containing augmentations with parameters
    config_args = {
        "train": {
            "augmentations": [
                {"HorizontalFlip": {"p": 0.8}},
                {"Downscale": {"scale": (0.5, 0.9), "p": 0.3}}
            ]
        },
        "validation": {
            "augmentations": [
                {"VerticalFlip": {"p": 0.8}},
            ]
        }
    }

    m = main.deepforest(config_args=config_args)
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)

    # Load dataset with config-based augmentations
    train_ds = m.load_dataset(csv_file, root_dir=root_dir, augmentations=["HorizontalFlip"])

    # Check that we can iterate over the dataset
    image, target, path = next(iter(train_ds))
    assert image is not None


def test_config_no_augmentations():
    """Test that default behavior works when no augmentations are specified in config."""
    # Test with no augmentations in config (should use defaults)
    m = main.deepforest()
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)

    # Load dataset - should use default augmentations
    train_ds = m.load_dataset(csv_file, root_dir=root_dir, augmentations=["HorizontalFlip"])

    # Check that we can iterate over the dataset
    image, target, path = next(iter(train_ds))
    assert image is not None


if __name__ == "__main__":
    pytest.main([__file__])
