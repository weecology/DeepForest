"""Augmentation module for DeepForest using kornia.

This module provides configurable augmentations for training and
validation that can be specified through configuration files or direct
parameters.
"""

import contextvars
import math
from contextlib import contextmanager
from typing import Any

import kornia.augmentation as K
import torch
import torch.nn.functional as F
from kornia.augmentation import IntensityAugmentationBase2D
from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.constants import DataKey, Resample
from kornia.geometry.bbox import bbox_generator
from kornia.geometry.transform import crop_by_transform_mat, get_perspective_transform
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch import Tensor

_bbox_augmentation_context: contextvars.ContextVar[Tensor | None] = (
    contextvars.ContextVar("deepforest_bbox_augmentation_context", default=None)
)


@contextmanager
def bbox_augmentation_context(boxes: Tensor | None):
    """Provide bounding boxes to bbox-aware augmentations during a transform
    call."""
    token = _bbox_augmentation_context.set(boxes)
    try:
        yield
    finally:
        _bbox_augmentation_context.reset(token)


def _get_bbox_augmentation_context() -> Tensor | None:
    return _bbox_augmentation_context.get()


def _needs_bbox_context(transform: K.AugmentationSequential) -> bool:
    return any(isinstance(module, RandomSizedBBoxSafeCrop) for module in transform)


def apply_augmentations(
    transform: K.AugmentationSequential,
    image: Tensor,
    *annotations: Tensor,
    data_keys: list[DataKey] | None = None,
) -> tuple[Tensor, ...]:
    """Run a Kornia pipeline on an image and aligned annotations.

    Uses ``data_keys`` on the sequential container (for example image + bbox or
    image + keypoints). Enables bbox-aware crops via context when needed.
    """
    keys = data_keys if data_keys is not None else list(transform.data_keys)
    annotation_keys = [key for key in keys if key not in (DataKey.IMAGE, DataKey.INPUT)]
    if _needs_bbox_context(transform) and DataKey.BBOX_XYXY in annotation_keys:
        box_idx = annotation_keys.index(DataKey.BBOX_XYXY)
        with bbox_augmentation_context(annotations[box_idx]):
            return transform(image, *annotations)
    return transform(image, *annotations)


def _union_of_boxes_xyxy(boxes: Tensor, erosion_rate: float) -> Tensor | None:
    """Return axis-aligned union of xyxy boxes, optionally eroded."""
    if boxes.numel() == 0:
        return None

    flat = boxes.reshape(-1, 4)
    x_min = flat[:, 0].min()
    y_min = flat[:, 1].min()
    x_max = flat[:, 2].max()
    y_max = flat[:, 3].max()

    if erosion_rate > 0:
        if erosion_rate >= 1:
            return None
        width = x_max - x_min
        height = y_max - y_min
        erosion_x = width * erosion_rate * 0.5
        erosion_y = height * erosion_rate * 0.5
        x_min = x_min + erosion_x
        y_min = y_min + erosion_y
        x_max = x_max - erosion_x
        y_max = y_max - erosion_y
        if (x_max - x_min) < 1e-6 or (y_max - y_min) < 1e-6:
            return None

    return torch.stack([x_min, y_min, x_max, y_max])


def _random_crop_coords_without_boxes(
    image_height: int,
    image_width: int,
    erosion_rate: float,
    generator: torch.Generator,
) -> tuple[int, int, int, int]:
    erosive_h = int(image_height * (1.0 - erosion_rate))
    crop_height = (
        image_height
        if erosive_h >= image_height
        else int(
            torch.randint(erosive_h, image_height + 1, (1,), generator=generator).item()
        )
    )
    crop_width = int(crop_height * image_width / image_height)

    max_x = max(image_width - crop_width, 0)
    max_y = max(image_height - crop_height, 0)
    crop_x_min = int(torch.randint(0, max_x + 1, (1,), generator=generator).item())
    crop_y_min = int(torch.randint(0, max_y + 1, (1,), generator=generator).item())
    return crop_x_min, crop_y_min, crop_x_min + crop_width, crop_y_min + crop_height


def _sample_bbox_safe_crop_coords(
    boxes: Tensor | None,
    image_height: int,
    image_width: int,
    erosion_rate: float,
    context_scale_range: tuple[float, float],
    generator: torch.Generator,
) -> tuple[int, int, int, int]:
    """Sample a crop that contains all boxes, with random context around
    detections."""
    min_h = max(int(image_height * (1.0 - erosion_rate)), 1)
    min_w = max(int(image_width * (1.0 - erosion_rate)), 1)

    if boxes is None or boxes.numel() == 0:
        return _random_crop_coords_without_boxes(
            image_height, image_width, erosion_rate, generator
        )

    union = _union_of_boxes_xyxy(boxes, erosion_rate=0.0)
    if union is None:
        return _random_crop_coords_without_boxes(
            image_height, image_width, erosion_rate, generator
        )

    ux0, uy0, ux1, uy1 = union.tolist()
    union_w = max(ux1 - ux0, 1.0)
    union_h = max(uy1 - uy0, 1.0)
    scale_lo, scale_hi = context_scale_range
    scale = scale_lo + (scale_hi - scale_lo) * torch.rand(1, generator=generator).item()
    crop_w = min(max(int(union_w * scale), min_w), image_width)
    crop_h = min(max(int(union_h * scale), min_h), image_height)

    max_x0 = max(min(int(ux0), image_width - crop_w), 0)
    max_y0 = max(min(int(uy0), image_height - crop_h), 0)
    min_x0 = max(int(ux1) - crop_w, 0)
    min_y0 = max(int(uy1) - crop_h, 0)

    if min_x0 > max_x0:
        crop_x_min = max(0, min(int(ux0 - crop_w / 2), image_width - crop_w))
    else:
        crop_x_min = int(
            torch.randint(min_x0, max_x0 + 1, (1,), generator=generator).item()
        )

    if min_y0 > max_y0:
        crop_y_min = max(0, min(int(uy0 - crop_h / 2), image_height - crop_h))
    else:
        crop_y_min = int(
            torch.randint(min_y0, max_y0 + 1, (1,), generator=generator).item()
        )

    crop_x_max = crop_x_min + crop_w
    crop_y_max = crop_y_min + crop_h
    return crop_x_min, crop_y_min, crop_x_max, crop_y_max


class RandomSizedBBoxSafeCrop(GeometricAugmentationBase2D):
    r"""Crop a random region that contains all boxes, then resize to a fixed
    size.

    Similar to
    `Albumentations RandomSizedBBoxSafeCrop <https://albumentations.ai/explore/transform/RandomSizedBBoxSafeCrop/docs/>`_.
    The crop is sampled so every bounding box remains inside the crop window, then the
    crop is resized to ``size``. Use ``context_scale_range`` to control how much extra
    context is included around the union of all boxes.

    Args:
        size: Output ``(height, width)`` after cropping and resizing.
        erosion_rate: Minimum crop size as a fraction of the original image. A value of
            ``0.2`` means the crop is at least 80% of the image height and width. Also
            controls how tightly the crop can hug the box union when sampling placement.
        context_scale_range: Random multiplier applied to the union of all boxes to set
            crop extent around detections. For example, ``(1.0, 2.0)`` crops between 1x
            and 2x the union width and height (clipped to the image bounds).
        resample: Interpolation mode passed to Kornia.
        align_corners: Interpolation flag.
        p: Probability of applying the transform.
        same_on_batch: Apply the same transformation to all batch elements.
        keepdim: Maintain shape.
    """

    def __init__(
        self,
        size: tuple[int, int] = (400, 400),
        erosion_rate: float = 0.0,
        context_scale_range: tuple[float, float] = (1.0, 2.0),
        resample: str | int | Resample = Resample.BILINEAR.name,
        align_corners: bool = True,
        p: float = 0.5,
        same_on_batch: bool = False,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=1.0, same_on_batch=same_on_batch, p_batch=p, keepdim=keepdim)
        if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
            raise ValueError(
                f"`size` must be a tuple of two positive integers. Got {size}."
            )
        if not 0.0 <= erosion_rate <= 1.0:
            raise ValueError(f"`erosion_rate` must be in [0, 1]. Got {erosion_rate}.")
        scale_lo, scale_hi = context_scale_range
        if scale_lo <= 0 or scale_hi < scale_lo:
            raise ValueError(
                f"`context_scale_range` must satisfy 0 < min <= max. Got {context_scale_range}."
            )
        self.size = size
        self.erosion_rate = erosion_rate
        self.context_scale_range = context_scale_range
        self.flags = {
            "size": size,
            "resample": Resample.get(resample),
            "align_corners": align_corners,
            "cropping_mode": "resample",
            "padding_mode": "zeros",
        }

    def generate_parameters(self, batch_shape: torch.Size) -> dict[str, Tensor]:
        return {}

    def compute_transformation(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any]
    ) -> Tensor:
        batch_size, _, img_h, img_w = input.shape
        out_h, out_w = flags["size"]
        device, dtype = input.device, input.dtype
        context_boxes = _get_bbox_augmentation_context()

        src_corners = []
        dst_corners = []
        for batch_idx in range(batch_size):
            if context_boxes is not None and context_boxes.ndim == 3:
                boxes = context_boxes[batch_idx]
            else:
                boxes = None

            generator = torch.Generator(device="cpu")
            generator.manual_seed(torch.randint(0, 2**31 - 1, (1,)).item())
            crop_x0, crop_y0, crop_x1, crop_y1 = _sample_bbox_safe_crop_coords(
                boxes,
                img_h,
                img_w,
                self.erosion_rate,
                self.context_scale_range,
                generator,
            )
            crop_w = crop_x1 - crop_x0
            crop_h = crop_y1 - crop_y0
            src = bbox_generator(
                torch.tensor(crop_x0, device=device, dtype=dtype),
                torch.tensor(crop_y0, device=device, dtype=dtype),
                torch.tensor(crop_w, device=device, dtype=dtype),
                torch.tensor(crop_h, device=device, dtype=dtype),
            )
            dst = torch.tensor(
                [
                    [
                        [0.0, 0.0],
                        [out_w - 1, 0.0],
                        [out_w - 1, out_h - 1],
                        [0.0, out_h - 1],
                    ]
                ],
                device=device,
                dtype=dtype,
            )
            src_corners.append(src)
            dst_corners.append(dst)

        src_batch = torch.cat(src_corners, dim=0)
        dst_batch = torch.cat(dst_corners, dim=0)
        transform = get_perspective_transform(src_batch, dst_batch)
        return transform.expand(batch_size, -1, -1)

    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        if transform is None:
            raise TypeError("Expected `transform` to be a Tensor.")
        return crop_by_transform_mat(
            input,
            transform,
            flags["size"],
            mode=flags["resample"].name.lower(),
            padding_mode=flags["padding_mode"],
            align_corners=flags["align_corners"],
        )


class RandomPadTo(GeometricAugmentationBase2D):
    r"""Pad the given sample by a random amount.

    This function is copied from kornia, but allows p to be changed.

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
    "FixedResize": (K.Resize, {"size": (400, 400), "p": 1.0}),
    "RandomCrop": (K.RandomCrop, {"size": (200, 200), "p": 0.5}),
    "RandomResizedCrop": (
        K.RandomResizedCrop,
        {"size": (400, 400), "scale": (0.5, 1.0), "ratio": (1.0, 1.0), "p": 0.5},
    ),
    "RandomSizedBBoxSafeCrop": (
        RandomSizedBBoxSafeCrop,
        {
            "size": (400, 400),
            "erosion_rate": 0.0,
            "context_scale_range": (1.0, 2.0),
            "p": 0.5,
        },
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
    data_keys: list[DataKey] | None = None,
) -> K.AugmentationSequential:
    """Create Kornia transform pipeline.

    Args:
        augmentations: Augmentation configuration:
            - str: Single augmentation name
            - list: List of augmentation names
            - dict: Dict with names as keys and params as values
            - None: No augmentations
        data_keys: Kornia DataKey list describing the inputs.
            Defaults to [DataKey.IMAGE, DataKey.BBOX_XYXY].

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
    if data_keys is None:
        data_keys = [DataKey.IMAGE, DataKey.BBOX_XYXY]

    transforms_list = []

    if augmentations is not None:
        augment_configs = _parse_augmentations(augmentations)

        for aug_name, aug_params in augment_configs.items():
            aug_transform = _create_augmentation(aug_name, aug_params)
            transforms_list.append(aug_transform)

    # Enforce ordering to avoid issues with padding and cropping
    _crop_types = (K.RandomCrop, K.RandomResizedCrop, RandomSizedBBoxSafeCrop)
    _pad_types = (K.PadTo, RandomPadTo)
    standard = [t for t in transforms_list if not isinstance(t, _crop_types + _pad_types)]
    crops = [t for t in transforms_list if isinstance(t, _crop_types)]
    pads = [t for t in transforms_list if isinstance(t, _pad_types)]
    transforms_list = standard + crops + pads

    # Create a sequential container for all transforms
    return K.AugmentationSequential(*transforms_list, data_keys=data_keys)


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
