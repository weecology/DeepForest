"""Augmentation module for DeepForest using albumentations.

This module provides configurable augmentations for training and
validation that can be specified through configuration files or direct
parameters.
"""

from typing import List, Optional, Union, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf.listconfig import ListConfig


def get_transform(
        augment: bool = True,
        augmentations: Optional[Union[str, List[str], Dict[str,
                                                           Any]]] = None) -> A.Compose:
    """Create Albumentations transformation for bounding boxes.

    Args:
        augment (bool): Whether to apply augmentations. If False, only ToTensorV2 is applied.
        augmentations (str, list, dict, optional): Augmentation configuration.
            - If str: Single augmentation name (e.g., "HorizontalFlip")
            - If list: List of augmentation names
            - If dict: Dict with augmentation names as keys and parameters as values
            - If None: Uses default augmentations when augment=True

    Returns:
        A.Compose: Composed albumentations transform with bbox parameters

    Examples:
        >>> # Default behavior (backward compatible)
        >>> transform = get_transform(augment=True)

        >>> # Single augmentation
        >>> transform = get_transform(augment=True, augmentations="Downscale")

        >>> # Multiple augmentations
        >>> transform = get_transform(augment=True,
        ...                          augmentations=["HorizontalFlip", "Downscale"])

        >>> # Augmentations with parameters
        >>> transform = get_transform(augment=True,
        ...                          augmentations={
        ...                              "HorizontalFlip": {"p": 0.5},
        ...                              "Downscale": {"scale_min": 0.25, "scale_max": 0.75}
        ...                          })
    """
    if not augment:
        return A.Compose([ToTensorV2()])

    # Build list of transforms
    transforms_list = []

    if augmentations is None:
        # Default augmentations for backward compatibility
        transforms_list.append(A.HorizontalFlip(p=0.5))
    else:
        # Parse augmentations parameter
        augment_configs = _parse_augmentations(augmentations)

        for aug_name, aug_params in augment_configs.items():
            aug_transform = _create_augmentation(aug_name, aug_params)
            transforms_list.append(aug_transform)

    # Always add ToTensorV2 at the end
    transforms_list.append(ToTensorV2())

    bbox_params = A.BboxParams(format='pascal_voc', label_fields=["category_ids"])
    return A.Compose(transforms_list, bbox_params=bbox_params)


def _parse_augmentations(
        augmentations: Union[str, List[str], Dict[str,
                                                  Any]]) -> Dict[str, Dict[str, Any]]:
    """Parse augmentations parameter into a standardized dict format.

    Args:
        augmentations: Augmentation specification in various formats

    Returns:
        Dict mapping augmentation names to their parameters
    """
    if isinstance(augmentations, str):
        return {augmentations: {}}
    elif isinstance(augmentations, list) or isinstance(augmentations, ListConfig):
        return {aug: {} for aug in augmentations}
    elif isinstance(augmentations, dict):
        return augmentations
    else:
        raise ValueError(f"Unsupported augmentations type: {type(augmentations)}")


def _create_augmentation(name: str, params: Dict[str, Any]) -> Optional[A.BasicTransform]:
    """Create an albumentations transform by name with given parameters.

    Args:
        name: Name of the augmentation
        params: Parameters to pass to the augmentation

    Returns:
        Albumentations transform or None if name not recognized
    """
    # Default parameters for each augmentation
    default_params = {
        "HorizontalFlip": {
            "p": 0.5
        },
        "VerticalFlip": {
            "p": 0.5
        },
        "Downscale": {
            "scale_range": (0.25, 0.5),
            "p": 0.5
        },
        "RandomCrop": {
            "height": 200,
            "width": 200,
            "p": 0.5
        },
        "RandomSizedBBoxSafeCrop": {
            "height": 200,
            "width": 200,
            "p": 0.5
        },
        "PadIfNeeded": {
            "min_height": 800,
            "min_width": 800,
            "p": 1.0
        },
        "Rotate": {
            "limit": 15,
            "p": 0.5
        },
        "RandomBrightnessContrast": {
            "brightness_limit": 0.2,
            "contrast_limit": 0.2,
            "p": 0.5
        },
        "HueSaturationValue": {
            "hue_shift_limit": 10,
            "sat_shift_limit": 10,
            "val_shift_limit": 10,
            "p": 0.5
        },
        "GaussNoise": {
            "var_limit": (5.0, 20.0),
            "p": 0.3
        },
        "Blur": {
            "blur_limit": 2,
            "p": 0.3
        },
        "GaussianBlur": {
            "blur_limit": 2,
            "p": 0.3
        },
        "MotionBlur": {
            "blur_limit": 2,
            "p": 0.3
        },
        "ZoomBlur": {
            "max_factor": 1.05,
            "p": 0.3
        },
    }

    # Available augmentations mapping
    augmentation_classes = {
        "HorizontalFlip": A.HorizontalFlip,
        "VerticalFlip": A.VerticalFlip,
        "Downscale": A.Downscale,
        "RandomCrop": A.RandomCrop,
        "RandomSizedBBoxSafeCrop": A.RandomSizedBBoxSafeCrop,
        "PadIfNeeded": A.PadIfNeeded,
        "Rotate": A.Rotate,
        "RandomBrightnessContrast": A.RandomBrightnessContrast,
        "HueSaturationValue": A.HueSaturationValue,
        "GaussNoise": A.GaussNoise,
        "Blur": A.Blur,
        "GaussianBlur": A.GaussianBlur,
        "MotionBlur": A.MotionBlur,
        "ZoomBlur": A.ZoomBlur,
    }

    if name not in augmentation_classes:
        raise ValueError(
            f"Unknown augmentation '{name}'. Available augmentations: {list(augmentation_classes.keys())}"
        )

    # Merge default params with user params
    final_params = default_params.get(name, {}).copy()
    final_params.update(params)

    try:
        return augmentation_classes[name](**final_params)
    except Exception as e:
        raise ValueError(
            f"Failed to create augmentation '{name}' with params {final_params}: {e}")


def get_available_augmentations() -> List[str]:
    """Get list of available augmentation names.

    Returns:
        List of available augmentation names
    """
    return [
        "HorizontalFlip",
        "VerticalFlip",
        "Downscale",
        "RandomCrop",
        "RandomSizedBBoxSafeCrop",
        "PadIfNeeded",
        "Rotate",
        "RandomBrightnessContrast",
        "HueSaturationValue",
        "GaussNoise",
        "Blur",
        "GaussianBlur",
        "MotionBlur",
        "ZoomBlur",
    ]
