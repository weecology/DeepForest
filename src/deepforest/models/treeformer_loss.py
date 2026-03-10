import torch
from torch import nn
from torchvision.ops import sigmoid_focal_loss


def cardinality_loss(
    probs: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_background: bool = False,
) -> torch.Tensor:
    """Count-consistency loss via density integration.

    Compares class-wise summed prediction mass and target mass in log-space
    using SmoothL1. This keeps count-matching pressure while reducing the
    dominance of very large absolute mismatches early in training.

    Args:
        probs: Tensor of shape (B, C, H, W) with predicted probabilities.
        labels: Tensor of shape (B, C, H, W) with density targets.
        ignore_background: If True and C > 1, excludes class channel 0 from
            the cardinality term.

    Returns:
        Scalar tensor representing the cardinality loss.
    """
    if probs.ndim != 4:
        raise ValueError(f"Expected probs with shape (B, C, H, W), got {probs.shape}.")
    if labels.ndim != 4:
        raise ValueError(f"Expected labels with shape (B, C, H, W), got {labels.shape}.")
    if labels.shape != probs.shape:
        raise ValueError(
            "Dense labels must match probs shape (B, C, H, W), "
            f"got labels={labels.shape}, probs={probs.shape}."
        )

    pred_counts = probs.sum(dim=(-2, -1))
    gt_counts = labels.to(dtype=probs.dtype).sum(dim=(-2, -1))

    if ignore_background and probs.shape[1] > 1:
        pred_counts = pred_counts[:, 1:]
        gt_counts = gt_counts[:, 1:]

    pred_counts = torch.clamp(pred_counts, min=0.0)
    gt_counts = torch.clamp(gt_counts, min=0.0)

    pred_log_counts = torch.log1p(pred_counts)
    gt_log_counts = torch.log1p(gt_counts)

    return nn.functional.smooth_l1_loss(pred_log_counts, gt_log_counts)


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
    ignore_background: bool = False,
    normalize_by_target_mass: bool = False,
) -> torch.Tensor:
    """Focal loss for dense prediction maps (Lin et al., 2017).

    Uses ``torchvision.ops.sigmoid_focal_loss`` to avoid maintaining
    a custom implementation.

    Args:
        logits: Tensor of shape (B, C, H, W) with raw logits.
        labels: Tensor of shape (B, C, H, W) with dense targets in [0, 1].
        alpha: Weighting factor for the focal loss.
        gamma: Focusing parameter that reduces loss for well-classified examples.
        reduction: ``"mean"`` | ``"sum"`` | ``"none"``.
        ignore_background: If True and ``C > 1``, excludes class channel 0
            from the focal term.
        normalize_by_target_mass: If True and ``reduction='mean'``, scales
            the focal map by foreground target mass (sum of targets) instead
            of averaging over all pixels.
    """
    if logits.ndim != 4:
        raise ValueError(f"Expected logits with shape (B, C, H, W), got {logits.shape}.")

    if labels.ndim != 4:
        raise ValueError(f"Expected labels with shape (B, C, H, W), got {labels.shape}.")
    if labels.shape != logits.shape:
        raise ValueError(
            "Dense labels must match logits shape (B, C, H, W), "
            f"got labels={labels.shape}, logits={logits.shape}."
        )

    targets = torch.clamp(labels.to(dtype=logits.dtype), min=0.0, max=1.0)

    loss = sigmoid_focal_loss(
        inputs=logits,
        targets=targets,
        alpha=alpha,
        gamma=gamma,
        reduction="none",
    )

    if ignore_background and logits.shape[1] > 1:
        loss = loss[:, 1:, ...]
        targets = targets[:, 1:, ...]

    if normalize_by_target_mass:
        if reduction != "mean":
            raise ValueError(
                "normalize_by_target_mass is only supported with reduction='mean'."
            )
        numerator = loss.sum()
        denominator = torch.clamp(targets.sum(), min=torch.finfo(loss.dtype).eps)
        return numerator / denominator

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
