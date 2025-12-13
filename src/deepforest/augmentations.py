"""Augmentation module for DeepForest using kornia.

This module provides configurable augmentations for training and
validation that can be specified through configuration files or direct
parameters.
"""

import math
from typing import Any

import kornia.augmentation as K
import torch
import torch.nn.functional as F
from kornia.augmentation import IntensityAugmentationBase2D
from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.constants import DataKey
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch import Tensor


class RandomPadTo(GeometricAugmentationBase2D):
    r"""Pad the given sample by a random amount. This function is copied from
    kornia, but allows p to be changed.

    Args:
        pad_range: Range of padding to apply as (min, max) in pixels.
        pad_mode: Padding mode (constant, reflect, replicate, circular).
        pad_value: Fill value for constant padding mode.
        p: Probability of applying the transform.
        same_on_batch: Apply same transformation to all batch elements.
        keepdim: Maintain shape.
    """

    def __init__(
        self,
        pad_range: tuple[int, int] = (0, 10),
        pad_mode: str = "constant",
        pad_value: float = 0,
        p: float = 0.5,
        same_on_batch: bool = False,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self.flags = {"pad_mode": pad_mode, "pad_value": pad_value}
        self._param_generator = rg.PlainUniformGenerator(
            (pad_range, "pad_height", None, None),
            (pad_range, "pad_width", None, None),
        )

    def compute_transformation(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any]
    ) -> Tensor:
        # This is probably OK as the padding always extends right/down,
        # so applying the identity to objects/labels is fine. Kornia docs
        # say that we must implement the inverse, but this is auto-computed in
        # the base class.
        return self.identity_matrix(input)

    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        height_pad = int(params["pad_height"].item())
        width_pad = int(params["pad_width"].item())
        return torch.nn.functional.pad(
            input,
            [0, width_pad, 0, height_pad],
            mode=flags["pad_mode"],
            value=flags["pad_value"],
        )

    def inverse_transform(
        self,
        input: Tensor,
        flags: dict[str, Any],
        transform: Tensor | None = None,
        size: tuple[int, int] | None = None,
    ) -> Tensor:
        if size is None:
            raise RuntimeError("`size` has to be a tuple. Got None.")
        return input[..., : size[0], : size[1]]


class ZoomBlur(IntensityAugmentationBase2D):
    """Apply zoom blur effect by averaging multiple zoomed versions of the
    image.

    Simulates the effect of zooming during camera exposure by creating
    multiple center-cropped and zoomed versions at different scales and averaging them.

    Args:
        max_factor: Maximum zoom factor range (min, max). Default: (1.0, 1.03)
        step_factor: Step size for zoom progression range (min, max). Default: (0.01, 0.02)
        p: Probability of applying the transform. Default: 0.5
        same_on_batch: Apply same transformation to all batch elements. Default: False
        keepdim: Maintain shape. Default: False
    """

    def __init__(
        self,
        max_factor: tuple[float, float] = (1.0, 1.03),
        step_factor: tuple[float, float] = (0.01, 0.02),
        p: float = 0.5,
        same_on_batch: bool = False,
        keepdim: bool = False,
    ):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = rg.PlainUniformGenerator(
            (max_factor, "max_factor", None, None),
            (step_factor, "step_factor", None, None),
        )

    def apply_transform(self, input, params, flags, transform=None):
        """Apply zoom blur to image tensor.

        Args:
            input: Input tensor of shape (B, C, H, W)
            params: Dict with 'max_factor' and 'step_factor'
            flags: Additional flags
            transform: Optional transformation matrix (unused)

        Returns:
            Blurred image tensor of same shape
        """
        max_f = params["max_factor"].item()
        step = params["step_factor"].item()
        max_f = max(1.0 + step, max_f)
        steps = torch.arange(1.0, max_f, step)

        B, C, H, W = input.shape
        out = input.clone()

        for zoom_factor in steps:
            zf = zoom_factor.item()
            h_crop = math.ceil(H / zf)
            w_crop = math.ceil(W / zf)
            h_start = (H - h_crop) // 2
            w_start = (W - w_crop) // 2

            cropped = input[:, :, h_start : h_start + h_crop, w_start : w_start + w_crop]
            zoomed = F.interpolate(
                cropped, size=(H, W), mode="bilinear", align_corners=False
            )
            out += zoomed

        return out / (len(steps) + 1)


_SUPPORTED_TRANSFORMS = {
    "HorizontalFlip": (K.RandomHorizontalFlip, {"p": 0.5}),
    "VerticalFlip": (K.RandomVerticalFlip, {"p": 0.5}),
    "Resize": (K.LongestMaxSize, {"max_size": 400}),
    "RandomCrop": (K.RandomCrop, {"size": (200, 200), "p": 0.5}),
    "RandomResizedCrop": (
        K.RandomResizedCrop,
        {"size": (400, 400), "scale": (0.5, 1.0), "ratio": (1.0, 1.0), "p": 0.5},
    ),
    "RandomPadTo": (
        RandomPadTo,
        {"pad_range": (0, 10), "pad_mode": "constant", "pad_value": 0, "p": 0.5},
    ),
    "PadIfNeeded": (K.PadTo, {"size": (800, 800)}),
    "Rotate": (K.RandomRotation, {"degrees": 15, "p": 0.5}),
    "RandomBrightnessContrast": (
        K.ColorJiggle,
        {"brightness": 0.2, "contrast": 0.2, "p": 0.5},
    ),
    "HueSaturationValue": (
        K.ColorJiggle,
        {"hue": 0.1, "saturation": 0.1, "p": 0.5},
    ),
    "GaussNoise": (K.RandomGaussianNoise, {"std": 0.1, "p": 0.3}),
    "Blur": (
        K.RandomBoxBlur,
        {"kernel_size": (3, 3), "p": 0.3},
    ),
    "GaussianBlur": (
        K.RandomGaussianBlur,
        {"kernel_size": (3, 3), "sigma": (0.1, 2.0), "p": 0.3},
    ),
    "MotionBlur": (
        K.RandomMotionBlur,
        {"kernel_size": 3, "angle": 45, "direction": 0.0, "p": 0.3},
    ),
    "ZoomBlur": (
        ZoomBlur,
        {"max_factor": (1.0, 1.03), "step_factor": (0.01, 0.02), "p": 0.5},
    ),
}


def get_available_augmentations() -> list[str]:
    """Get list of available augmentation names.

    Returns:
        List of available augmentation names
    """
    return sorted(_SUPPORTED_TRANSFORMS.keys())


def get_transform(
    augmentations: str | list[str] | dict[str, Any] | None = None,
) -> K.AugmentationSequential:
    """Create Kornia transform for bounding boxes.

    Args:
        augmentations: Augmentation configuration:
            - str: Single augmentation name
            - list: List of augmentation names
            - dict: Dict with names as keys and params as values
            - None: No augmentations

    Returns:
        Kornia AugmentationSequential

    Examples:
        >>> # Default behavior, returns a basic transform
        >>> transform = get_transform()

        >>> # Single augmentation
        >>> transform = get_transform(augmentations="VerticalFlip")

        >>> # Multiple augmentations
        >>> transform = get_transform(augmentations=["HorizontalFlip", "VerticalFlip"])

        >>> # Augmentations with parameters
        >>> transform = get_transform(augmentations={
        ...                              "HorizontalFlip": {"p": 0.5},
        ...                              "VerticalFlip"
        ...                          })
    """
    transforms_list = []

    if augmentations is not None:
        augment_configs = _parse_augmentations(augmentations)

        for aug_name, aug_params in augment_configs.items():
            aug_transform = _create_augmentation(aug_name, aug_params)
            transforms_list.append(aug_transform)

    # Create a sequential container for all transforms
    return K.AugmentationSequential(
        *transforms_list, data_keys=[DataKey.IMAGE, DataKey.BBOX_XYXY]
    )


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


def _create_augmentation(
    name: str, params: dict[str, Any]
) -> K.AugmentationSequential | None:
    """Create a kornia transform by name with given parameters.

    Args:
        name: Name of the augmentation
        params: Parameters to pass to the augmentation

    Returns:
        Kornia AugmentationSequential or None if name not recognized
    """

    if name not in get_available_augmentations():
        raise ValueError(
            f"Unknown augmentation '{name}'. Available augmentations: {get_available_augmentations()}"
        )

    # Retrieve factory and defaults, merge with user-provided params
    transform, base_params = _SUPPORTED_TRANSFORMS[name]

    # Check if augmentation is unsupported
    if transform is None:
        raise NotImplementedError(f"Augmentation '{name}' is not currently supported.")

    final_params = base_params.copy()
    final_params.update(params)

    try:
        return transform(**final_params)
    except Exception as e:
        raise ValueError(
            f"Failed to create augmentation '{name}' with params {final_params}: {e}"
        ) from e
