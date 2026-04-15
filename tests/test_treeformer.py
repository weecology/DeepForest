"""Unit tests for TreeFormerModel.

Tests are scoped to the model class itself; deepforest.main is not used
except where explicitly noted.
"""

import os

import pytest
import torch

from deepforest import get_data, utilities
from deepforest.datasets.training import KeypointDataset
from deepforest.models.treeformer import TreeFormerModel
from deepforest.utilities import density_to_points

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Use the smallest PvT-V2 variant to keep tests fast.
_BACKBONE = "OpenGVLab/pvt_v2_b0"

_TREEFORMER_KEYPOINT_FIELDS = {
    "density_sigma",
    "mae_weight",
    "ot_weight",
    "density_l1_weight",
    "count_cls_weight",
    "losses",
    "norm_cood",
    "enforce_count",
    "sinkhorn_reg",
    "num_of_iter_in_ot",
}


@pytest.fixture(scope="module")
def model():
    """Untrained TreeFormerModel with the smallest available backbone."""
    m = TreeFormerModel(backbone=_BACKBONE, pretrained=False, enforce_count=True)
    return m


def _make_targets(batch_size: int, n_points: int = 5) -> list[dict]:
    """Create dummy centroid targets for a batch."""
    return [{"points": torch.rand(n_points, 2) * 64} for _ in range(batch_size)]


def _build_treeformer_from_config(tmp_path, keypoint_overrides: dict | None = None):
    """Create a config object for TreeFormer factory tests."""
    return utilities.load_config(
        config_name="treeformer",
        overrides={
            "architecture": "treeformer",
            "model": {"name": None},
            "num_classes": 1,
            "label_dict": {"Tree": 0},
            "log_root": str(tmp_path),
            "keypoint": keypoint_overrides or {},
        },
    )


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_create_model_num_classes():
    """num_classes is stored on the model."""
    m = TreeFormerModel(backbone=_BACKBONE, pretrained=False, num_classes=2)
    assert m.num_classes == 2


def test_treeformer_warns_for_unsafe_enforce_count_without_count_cls():
    """Unsafe enforce_count configurations should warn without mutating losses."""
    with pytest.warns(UserWarning, match="count_cls is not active in losses"):
        model = TreeFormerModel(
            backbone=_BACKBONE,
            pretrained=False,
            losses=["count", "ot", "density_l1"],
            enforce_count=True,
        )

    assert model.losses == ["count", "ot", "density_l1"]
    assert "count_cls" not in model.active_losses


def test_treeformer_density_mode_scales_counts_by_image_area():
    """Density-mode CLS outputs should scale to absolute counts per image."""
    model = TreeFormerModel(
        backbone=_BACKBONE,
        pretrained=False,
    )

    counts = model._cls_outputs_to_count(
        torch.tensor([0.25, 0.25], dtype=torch.float32),
        [(32, 32), (64, 64)],
    )

    torch.testing.assert_close(counts, torch.tensor([256.0, 1024.0]))


def test_treeformer_density_mode_masks_padded_regions_when_enforcing_count():
    """Enforce-count coupling should ignore padded decoder regions."""
    model = TreeFormerModel(
        backbone=_BACKBONE,
        pretrained=False,
        enforce_count=True,
    )

    output_mask = model._build_output_mask([(16, 16), (32, 32)], 8, 8)
    score_map = torch.ones(2, 1, 8, 8) * output_mask
    counts = model._cls_outputs_to_count(
        torch.tensor([0.25, 0.25], dtype=torch.float32),
        [(16, 16), (32, 32)],
    )
    density_map, _ = model._normalize_density(score_map, counts)

    torch.testing.assert_close(
        density_map.view(2, -1).sum(dim=1),
        torch.tensor([64.0, 256.0]),
        atol=1e-3,
        rtol=1e-5,
    )
    assert density_map[0, :, 4:, :].abs().sum().item() == 0.0
    assert density_map[0, :, :, 4:].abs().sum().item() == 0.0


def test_treeformer_cls_bias_initializes_positive():
    """The CLS head should retain its small positive density prior."""
    model = TreeFormerModel(
        backbone=_BACKBONE,
        pretrained=False,
    )

    torch.testing.assert_close(
        model.regression.cls_lin4.bias.detach(),
        torch.full_like(model.regression.cls_lin4.bias, 1e-4),
    )


# ---------------------------------------------------------------------------
# Training mode — various batch sizes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_forward_train_uniform_batch(model, batch_size):
    """Training mode returns a loss dict with positive total loss for B=1,2,3."""
    model.train()
    images = [torch.rand(3, 64, 64) for _ in range(batch_size)]
    targets = _make_targets(batch_size, n_points=4)
    loss_dict = model(images, targets)

    assert "loss" in loss_dict
    assert loss_dict["loss"].item() > 0
    assert all(
        k in loss_dict
        for k in [
            "count_mae",
            "count_loss",
            "ot_loss",
            "density_l1_loss",
            "count_cls_mae",
            "count_cls_loss",
        ]
    )


def test_forward_train_dict_targets(model):
    """Targets may be dicts with a 'points' key."""
    model.train()
    images = [torch.rand(3, 64, 64)]
    targets = [{"points": torch.tensor([[10.0, 20.0], [30.0, 40.0]])}]
    loss_dict = model(images, targets)
    assert loss_dict["loss"].item() >= 0


def test_forward_train_empty_targets(model):
    """Training mode handles images with zero annotations (empty point sets)."""
    model.train()
    images = [torch.rand(3, 64, 64)]
    targets = [{"points": torch.zeros(0, 2)}]
    loss_dict = model(images, targets)
    # Total loss should be finite even with no points.
    assert torch.isfinite(loss_dict["loss"])


def test_forward_train_requires_targets(model):
    """Training mode raises ValueError when no targets are provided."""
    model.train()
    images = [torch.rand(3, 64, 64)]
    with pytest.raises(ValueError, match="targets"):
        model(images, targets=None)


def test_forward_train_mixed_sizes(model):
    """Training mode handles a list of images with different spatial sizes."""
    model.train()
    images = [torch.rand(3, 64, 64), torch.rand(3, 64, 48)]
    targets = _make_targets(2, n_points=3)
    loss_dict = model(images, targets)
    assert loss_dict["loss"].item() > 0


# ---------------------------------------------------------------------------
# Eval mode
# ---------------------------------------------------------------------------


def test_forward_eval_mixed_sizes(model):
    """With a mixed-size list, each output is cropped to its own input size."""
    model.eval()
    shapes = [(128, 128), (128, 96)]
    images = [torch.rand(3, h, w) for h, w in shapes]
    with torch.no_grad():
        density_maps, normed_densities = model(images)

    assert len(density_maps) == 2
    dr = model.downsample_ratio
    for (h, w), dm, nd in zip(shapes, density_maps, normed_densities, strict=True):
        assert dm.shape == (1, 1, h // dr, w // dr)
        assert nd.shape == (1, 1, h // dr, w // dr)


def test_forward_eval_non_multiple_of_32(model):
    """Images whose dimensions are not multiples of 32 are handled correctly."""
    model.eval()
    H, W = 100, 70  # neither is a multiple of 32
    images = [torch.rand(3, H, W)]
    with torch.no_grad():
        density_maps, _ = model(images)
    # Output must correspond to the original input size, not the padded size.
    assert density_maps[0].shape == (
        1,
        1,
        H // model.downsample_ratio,
        W // model.downsample_ratio,
    )


# ---------------------------------------------------------------------------
# Dataset integration
# ---------------------------------------------------------------------------


def test_forward_train_from_keypoint_dataset(model):
    """End-to-end: KeypointDataset -> model training forward pass."""
    csv_file = get_data("2019_BLAN_3_751000_4330000_image_crop_keypoints.csv")
    root_dir = os.path.dirname(csv_file)
    ds = KeypointDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        label_dict={"Tree": 0},
    )
    image, targets, _ = ds[0]

    model.train()
    loss_dict = model([image], [targets])
    assert "loss" in loss_dict
    assert torch.isfinite(loss_dict["loss"])


def test_density_to_points_roundtrip(model):
    """Round-trip: GT points -> GT density map -> peak extraction recovers points.

    Verifies that density_to_points finds exactly one peak per ground-truth
    tree and that each recovered location is within the integration radius of
    the original point.
    """
    # Use a 128x128 output map (= 512x512 image / downsample_ratio 4).
    # Points must be spaced > 2*sigma apart to produce distinct, non-overlapping bumps.
    out_h, out_w = 128, 128
    # Points spaced ~40+ px apart with sigma=5
    gt_points = torch.tensor([[15.0, 15.0], [60.0, 40.0], [100.0, 100.0]])  # (x, y)

    sigma = model.density_sigma
    score_integration_radius = 5

    # Build GT density the same way the model does during training.
    gt_density = model._make_gt_density(
        [gt_points],
        out_h=out_h,
        out_w=out_w,
    )  # (1, 1, out_h, out_w)

    preds = density_to_points(
        gt_density,
        score_thresh=0.01,
        score_integration_radius=score_integration_radius,
    )
    pred = preds[0]

    assert len(pred["points"]) == len(gt_points), (
        f"Expected {len(gt_points)} peaks, got {len(pred['points'])}. "
        f"GT sigma={sigma}, integration_radius={score_integration_radius}"
    )

    # Each predicted point should be within score_integration_radius pixels of
    # its nearest GT point (centroid of the Gaussian bump is close to GT).
    pred_xy = pred["points"]  # (N, 2) in (x, y)
    for gt_pt in gt_points:
        dists = torch.norm(pred_xy - gt_pt.unsqueeze(0), dim=1)
        assert dists.min().item() <= score_integration_radius, (
            f"No pred within {score_integration_radius}px of GT point {gt_pt.tolist()}"
        )


def test_ot_loss_all_empty_points_has_grad():
    """OT_Loss must return a differentiable loss even when every image in the
    batch has zero annotation points (n_active == 0).

    This reproduces the DDP training failure reported when all ranks receive
    a batch with no annotations: the returned loss tensor must have a grad_fn
    so that loss.backward() does not raise
    ``RuntimeError: element 0 of tensors does not require grad``.
    """
    from deepforest.losses.ot_loss import OT_Loss

    device = torch.device("cpu")
    loss_fn = OT_Loss(norm_coord=True, device=device, num_of_iter_in_ot=10, reg=1.0)

    # Simulate a batch of 2 images with no annotation points.
    unnormed = torch.rand(2, 1, 8, 8, requires_grad=True)
    normed = unnormed / (unnormed.sum(dim=(2, 3), keepdim=True) + 1e-8)
    empty_points = [torch.zeros(0, 2), torch.zeros(0, 2)]

    loss, wd, ot_obj, avg_its, K_min, beta_abs_max, sinkhorn_err = loss_fn(
        normed, unnormed, empty_points
    )

    assert loss.requires_grad, "loss must require grad when all point lists are empty"
    # Should not raise.
    loss.backward()
