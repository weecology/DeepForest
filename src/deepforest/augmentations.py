"""Augmentation module for DeepForest using kornia.

This module provides configurable augmentations for training and
validation that can be specified through configuration files or direct
parameters.
"""

from typing import Any

import kornia.augmentation as K
import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

_SUPPORTED_TRANSFORMS = {
    "HorizontalFlip": (K.RandomHorizontalFlip, {"p": 0.5}),
    "VerticalFlip": (K.RandomVerticalFlip, {"p": 0.5}),
    "Downscale": (
        K.RandomResizedCrop,
        {"size": (200, 200), "scale": (0.25, 0.5), "p": 0.5},
    ),
    "RandomCrop": (K.RandomCrop, {"size": (200, 200), "p": 0.5}),
    "RandomSizedBBoxSafeCrop": (
        K.RandomResizedCrop,
        {"size": (200, 200), "scale": (0.5, 1.0), "p": 0.5},
    ),
    "PadIfNeeded": (K.PadTo, {"size": (800, 800), "p": 1.0}),
    "Rotate": (K.RandomRotation, {"degrees": 15, "p": 0.5}),
    "RandomBrightnessContrast": (
        K.ColorJitter,
        {"brightness": 0.2, "contrast": 0.2, "p": 0.5},
    ),
    "HueSaturationValue": (
        K.ColorJitter,
        {"hue": 0.1, "saturation": 0.1, "p": 0.5},
    ),
    "GaussNoise": (K.RandomGaussianNoise, {"std": 0.1, "p": 0.3}),
    "Blur": (
        K.RandomGaussianBlur,
        {"kernel_size": (3, 3), "sigma": (0.1, 2.0), "p": 0.3},
    ),
    "GaussianBlur": (
        K.RandomGaussianBlur,
        {"kernel_size": (3, 3), "sigma": (0.1, 2.0), "p": 0.3},
    ),
    "MotionBlur": (
        K.RandomMotionBlur,
        {"kernel_size": 3, "angle": 45, "direction": 0.0, "p": 0.3},
    ),
    "ZoomBlur": (K.RandomAffine, {"degrees": 0, "scale": (1.0, 1.05), "p": 0.3}),
}


def get_available_augmentations() -> list[str]:
    """Get list of available augmentation names.

    Returns:
        List of available augmentation names
    """
    return sorted(_SUPPORTED_TRANSFORMS.keys())


def get_transform(
    augmentations: str | list[str] | dict[str, Any] | None = None,
) -> torch.nn.Module:
    """Create Kornia transform for bounding boxes.

    Args:
        augmentations: Augmentation configuration:
            - str: Single augmentation name
            - list: List of augmentation names
            - dict: Dict with names as keys and params as values
            - None: No augmentations

    Returns:
        Composed kornia transform

    Examples:
        >>> # Default behavior, returns a basic transform
        >>> transform = get_transform()

        >>> # Single augmentation
        >>> transform = get_transform(augmentations="Downscale")

        >>> # Multiple augmentations
        >>> transform = get_transform(augmentations=["HorizontalFlip", "Downscale"])

        >>> # Augmentations with parameters
        >>> transform = get_transform(augmentations={
        ...                              "HorizontalFlip": {"p": 0.5},
        ...                              "Downscale": {"scale": (0.25, 0.75)}
        ...                          })
    """
    transforms_list = []

    if augmentations is not None:
        augment_configs = _parse_augmentations(augmentations)

        for aug_name, aug_params in augment_configs.items():
            aug_transform = _create_augmentation(aug_name, aug_params)
            transforms_list.append(aug_transform)

    # Create a sequential container for all transforms
    if transforms_list:
        return torch.nn.Sequential(*transforms_list)
    else:
        # Return identity transform if no augmentations
        return torch.nn.Identity()


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


def _create_augmentation(name: str, params: dict[str, Any]) -> torch.nn.Module | None:
    """Create a kornia transform by name with given parameters.

    Args:
        name: Name of the augmentation
        params: Parameters to pass to the augmentation

    Returns:
        Kornia transform or None if name not recognized
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
