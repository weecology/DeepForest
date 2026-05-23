"""Generate documentation figure for RandomSizedBBoxSafeCrop."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

from deepforest import get_data
from deepforest.augmentations import apply_augmentations
from deepforest.augmentations import get_transform
from deepforest.datasets.training import BoxDataset


def _draw_boxes(ax, image: np.ndarray, boxes: torch.Tensor, title: str) -> None:
    ax.imshow(image)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    for box in boxes.reshape(-1, 4):
        x0, y0, x1, y1 = box.tolist()
        rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)


def main() -> None:
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    dataset = BoxDataset(csv_file=csv_file, root_dir=root_dir, augmentations=None)
    image, targets, _ = dataset[0]
    boxes = targets["boxes"].unsqueeze(0)
    image_tensor = image.unsqueeze(0)
    original = image.permute(1, 2, 0).numpy().astype(np.uint8)

    scales = [(1.0, 1.1), (1.0, 1.5), (1.0, 2.0)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    _draw_boxes(axes[0], original, targets["boxes"], "Original")

    for ax, scale in zip(axes[1:], scales):
        transform = get_transform(
            augmentations={
                "RandomSizedBBoxSafeCrop": {
                    "size": (256, 256),
                    "context_scale_range": scale,
                    "p": 1.0,
                }
            }
        )
        aug_image, aug_boxes = apply_augmentations(
            transform, image_tensor.clone(), boxes.clone()
        )
        aug_np = (
            aug_image.squeeze(0).permute(1, 2, 0).clamp(0, 255).numpy().astype(np.uint8)
        )
        _draw_boxes(ax, aug_np, aug_boxes.squeeze(0), f"context_scale_range={scale}")

    fig.suptitle(
        "RandomSizedBBoxSafeCrop keeps every box, varies context, then resizes",
        fontsize=12,
    )
    fig.tight_layout()
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    out_path = os.path.join(repo_root, "www", "bbox_safe_crop_example.jpg")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
