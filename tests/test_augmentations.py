"""Test the new augmentations module."""
import pytest
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from deepforest.augmentations import get_transform, get_available_augmentations, _parse_augmentations, _create_augmentation


def test_get_transform_default():
    """Test default behavior (backward compatibility)."""
    # Test without augmentations
    transform = get_transform(augment=False)
    assert isinstance(transform, A.Compose)
    assert len(transform.transforms) == 1
    assert isinstance(transform.transforms[0], ToTensorV2)
    
    # Test with default augmentations
    transform = get_transform(augment=True)
    assert isinstance(transform, A.Compose)
    assert len(transform.transforms) == 2  # HorizontalFlip + ToTensorV2
    assert isinstance(transform.transforms[0], A.HorizontalFlip)
    assert isinstance(transform.transforms[1], ToTensorV2)


def test_get_transform_single_augmentation():
    """Test with single augmentation name."""
    transform = get_transform(augment=True, augmentations="Downscale")
    assert isinstance(transform, A.Compose)
    assert len(transform.transforms) == 2  # Downscale + ToTensorV2
    assert isinstance(transform.transforms[0], A.Downscale)
    assert isinstance(transform.transforms[1], ToTensorV2)


def test_get_transform_multiple_augmentations():
    """Test with list of augmentation names."""
    transform = get_transform(augment=True, augmentations=["HorizontalFlip", "Downscale"])
    assert isinstance(transform, A.Compose)
    assert len(transform.transforms) == 3  # HorizontalFlip + Downscale + ToTensorV2
    assert isinstance(transform.transforms[0], A.HorizontalFlip)
    assert isinstance(transform.transforms[1], A.Downscale)
    assert isinstance(transform.transforms[2], ToTensorV2)


def test_get_transform_with_parameters():
    """Test with augmentation parameters."""
    augmentations = {
        "HorizontalFlip": {"p": 0.8},
        "Downscale": {"scale_range": (0.5, 0.9), "p": 0.3}
    }
    transform = get_transform(augment=True, augmentations=augmentations)
    assert isinstance(transform, A.Compose)
    assert len(transform.transforms) == 3  # HorizontalFlip + Downscale + ToTensorV2
    
    # Check parameters were applied
    assert transform.transforms[0].p == 0.8  # HorizontalFlip
    assert transform.transforms[1].scale_range == (0.5, 0.9)  # Downscale
    assert transform.transforms[1].p == 0.3


def test_parse_augmentations():
    """Test _parse_augmentations function."""
    # String input
    result = _parse_augmentations("HorizontalFlip")
    assert result == {"HorizontalFlip": {}}
    
    # List input
    result = _parse_augmentations(["HorizontalFlip", "Downscale"])
    assert result == {"HorizontalFlip": {}, "Downscale": {}}
    
    # Dict input
    input_dict = {"HorizontalFlip": {"p": 0.5}, "Downscale": {"scale_min": 0.25}}
    result = _parse_augmentations(input_dict)
    assert result == input_dict


def test_create_augmentation():
    """Test _create_augmentation function."""
    # Valid augmentation
    aug = _create_augmentation("HorizontalFlip", {"p": 0.7})
    assert isinstance(aug, A.HorizontalFlip)
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
    """Test that bbox_params are properly set."""
    transform = get_transform(augment=True, augmentations="HorizontalFlip")
    
    # Check that bbox_params is configured in the transform repr
    transform_repr = repr(transform)
    assert "bbox_params" in transform_repr
    assert "'format': 'pascal_voc'" in transform_repr
    assert "'label_fields': ['category_ids']" in transform_repr


def test_blur_augmentations():
    """Test that all blur augmentations can be created successfully."""
    blur_augmentations = ["Blur", "GaussianBlur"]
    
    for blur_aug in blur_augmentations:
        transform = get_transform(augment=True, augmentations=[blur_aug])
        assert isinstance(transform, A.Compose)
        assert len(transform.transforms) == 2  # Blur augmentation + ToTensorV2
        assert isinstance(transform.transforms[1], ToTensorV2)


def test_blur_augmentations_with_parameters():
    """Test blur augmentations with custom parameters."""
    blur_configs = {
        "GaussianBlur": {"blur_limit": 5, "p": 0.8},
        "MotionBlur": {"blur_limit": 7, "p": 0.6},
        "ZoomBlur": {"max_factor": 1.3, "p": 0.4}
    }
    
    transform = get_transform(augment=True, augmentations=blur_configs)
    assert isinstance(transform, A.Compose)
    assert len(transform.transforms) == 4  # 3 blur augmentations + ToTensorV2
    assert isinstance(transform.transforms[3], ToTensorV2)


def test_mixed_blur_and_other_augmentations():
    """Test combining blur augmentations with other augmentations."""
    mixed_augmentations = ["HorizontalFlip", "GaussianBlur", "Downscale", "MotionBlur"]
    
    transform = get_transform(augment=True, augmentations=mixed_augmentations)
    assert isinstance(transform, A.Compose)
    assert len(transform.transforms) == 5  # 4 augmentations + ToTensorV2
    assert isinstance(transform.transforms[0], A.HorizontalFlip)
    assert isinstance(transform.transforms[1], A.GaussianBlur)
    assert isinstance(transform.transforms[2], A.Downscale)
    assert isinstance(transform.transforms[3], A.MotionBlur)
    assert isinstance(transform.transforms[4], ToTensorV2)


def test_unknown_augmentation_error():
    """Test that unknown augmentations raise ValueError."""
    with pytest.raises(ValueError, match="Unknown augmentation 'UnknownAugmentation'"):
        get_transform(augment=True, augmentations="UnknownAugmentation")


if __name__ == "__main__":
    pytest.main([__file__])