"""Augmentation module for DeepForest using albumentations.

This module provides configurable augmentations for training and
validation that can be specified through configuration files or direct
parameters.
"""

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

_SUPPORTED_TRANSFORMS = {
    "HorizontalFlip": (A.HorizontalFlip, {"p": 0.5}),
    "VerticalFlip": (A.VerticalFlip, {"p": 0.5}),
    "Downscale": (A.Downscale, {"scale_range": (0.25, 0.5), "p": 0.5}),
    "RandomCrop": (A.RandomCrop, {"height": 200, "width": 200, "p": 0.5}),
    "RandomSizedBBoxSafeCrop": (
        A.RandomSizedBBoxSafeCrop,
        {"height": 200, "width": 200, "p": 0.5},
    ),
    "PadIfNeeded": (A.PadIfNeeded, {"min_height": 800, "min_width": 800, "p": 1.0}),
    "Rotate": (A.Rotate, {"limit": 15, "p": 0.5}),
    "RandomBrightnessContrast": (
        A.RandomBrightnessContrast,
        {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.5},
    ),
    "HueSaturationValue": (
        A.HueSaturationValue,
        {"hue_shift_limit": 10, "sat_shift_limit": 10, "val_shift_limit": 10, "p": 0.5},
    ),
    "GaussNoise": (A.GaussNoise, {"var_limit": (5.0, 20.0), "p": 0.3}),
    "Blur": (A.Blur, {"blur_limit": 2, "p": 0.3}),
    "GaussianBlur": (A.GaussianBlur, {"blur_limit": 2, "p": 0.3}),
    "MotionBlur": (A.MotionBlur, {"blur_limit": 2, "p": 0.3}),
    "ZoomBlur": (A.ZoomBlur, {"max_factor": 1.05, "p": 0.3}),
}


def get_available_augmentations() -> list[str]:
    """Get list of available augmentation names.

    Returns:
        List of available augmentation names
    """
    return sorted(_SUPPORTED_TRANSFORMS.keys())


def get_transform(
    augmentations: str | list[str] | dict[str, Any] | None = None,
) -> A.Compose:
    """Create Albumentations transform for bounding boxes.

    Args:
        augmentations: Augmentation configuration:
            - str: Single augmentation name
            - list: List of augmentation names
            - dict: Dict with names as keys and params as values
            - None: No augmentations

    Returns:
        Composed albumentations transform

    Examples:
        >>> # Default behavior, returns a ToTensorV2 transform
        >>> transform = get_transform()

        >>> # Single augmentation
        >>> transform = get_transform(augmentations="Downscale")

        >>> # Multiple augmentations
        >>> transform = get_transform(augmentations=["HorizontalFlip", "Downscale"])

        >>> # Augmentations with parameters
        >>> transform = get_transform(augmentations={
        ...                              "HorizontalFlip": {"p": 0.5},
        ...                              "Downscale": {"scale_min": 0.25, "scale_max": 0.75}
        ...                          })
    """
    transforms_list = []
    bbox_params = None

    if augmentations is not None:
        augment_configs = _parse_augmentations(augmentations)

        for aug_name, aug_params in augment_configs.items():
            aug_transform = _create_augmentation(aug_name, aug_params)
            transforms_list.append(aug_transform)

        bbox_params = A.BboxParams(format="pascal_voc", label_fields=["category_ids"])

    # Always add ToTensorV2 at the end
    transforms_list.append(ToTensorV2())

    return A.Compose(transforms_list, bbox_params=bbox_params)


def _parse_augmentations(
    augmentations: str | list | dict | ListConfig | DictConfig,
) -> dict[str, dict[str, Any]]:
    """Parse augmentations parameter into a standardized dict format.

    Examples:
        - "HorizontalFlip" -> {"HorizontalFlip": {}}
        - ["HorizontalFlip", "Downscale"] -> {"HorizontalFlip": {}, "Downscale": {}}
        - {"HorizontalFlip": {"p": 0.5}}
        - [{"HorizontalFlip": {"p": 0.5}}, {"Downscale": {"scale_min": 0.25}}] -> {"HorizontalFlip": {"p": 0.5}, "Downscale": {"scale_min": 0.25}}

    Args:
        augmentations: Augmentation specification in various formats:
            - str: Single augmentation name
            - List: List of strings or dicts with augmentation configs
            - Dict: Dict with augmentation names as keys and parameters as values

    Returns:
        Dict mapping augmentation names to their parameters
    """

    # Convert OmegaConf to primitives
    if isinstance(augmentations, (DictConfig, ListConfig)):
        augmentations = OmegaConf.to_container(augmentations, resolve=True)

    if isinstance(augmentations, str):
        return {augmentations: {}}

    if isinstance(augmentations, dict):
        return augmentations
    elif isinstance(augmentations, list):
        result = {}
        for augmentation in augmentations:
            if isinstance(augmentation, str):
                result[augmentation] = {}
            elif isinstance(augmentation, dict):
                if len(augmentation) != 1:
                    raise ValueError(
                        f"Each augmentation dict must have exactly "
                        f"one key (corresponding to a single operation), "
                        f"got {len(augmentation)} for {augmentation}."
                    )
                name, params = next(iter(augmentation.items()))
                result[name] = params
            else:
                raise ValueError(
                    f"List elements must be strings or dicts, got {type(augmentation)}"
                )
        return result
    else:
        raise ValueError(f"Unable to parse augmentation parameters: {augmentations}")


def _create_augmentation(name: str, params: dict[str, Any]) -> A.BasicTransform | None:
    """Create an albumentations transform by name with given parameters.

    Args:
        name: Name of the augmentation
        params: Parameters to pass to the augmentation

    Returns:
        Albumentations transform or None if name not recognized
    """

    if name not in get_available_augmentations():
        raise ValueError(
            f"Unknown augmentation '{name}'. Available augmentations: {get_available_augmentations()}"
        )

    # Retrieve factory and defaults, merge with user-provided params
    transform, base_params = _SUPPORTED_TRANSFORMS[name]
    final_params = base_params.copy()
    final_params.update(params)

    try:
        return transform(**final_params)
    except Exception as e:
        raise ValueError(
            f"Failed to create augmentation '{name}' with params {final_params}: {e}"
        ) from e
