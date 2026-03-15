"""Unit tests for TreeFormerModel.

Tests are scoped to the model class itself; deepforest.main is not used.
"""

import os

import pytest
import torch

from deepforest import get_data
from deepforest.datasets.training import KeypointDataset
from deepforest.models.treeformer import TreeFormerModel
from deepforest.utilities import density_to_points

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Use the smallest PvT-V2 variant to keep tests fast.
_BACKBONE = "OpenGVLab/pvt_v2_b0"


@pytest.fixture(scope="module")
def model():
    """Untrained TreeFormerModel with the smallest available backbone."""
    m = TreeFormerModel(backbone=_BACKBONE, pretrained=False, enforce_count=True)
    return m


def _make_targets(batch_size: int, n_points: int = 5) -> list[dict]:
    """Create dummy centroid targets for a batch."""
    return [{"points": torch.rand(n_points, 2) * 64} for _ in range(batch_size)]


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_create_model_pretrained_false():
    """Model can be instantiated without downloading backbone weights."""
    m = TreeFormerModel(backbone=_BACKBONE, pretrained=False)
    assert m.downsample_ratio == 4


def test_create_model_num_classes():
    """num_classes is stored on the model."""
    m = TreeFormerModel(backbone=_BACKBONE, pretrained=False, num_classes=2)
    assert m.num_classes == 2


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
        k in loss_dict for k in ["count_loss", "ot_loss", "density_l1_loss", "count_cls_loss"]
    )


def test_forward_train_tensor_input(model):
    """Training mode accepts a pre-stacked (B, C, H, W) tensor."""
    model.train()
    images = torch.rand(2, 3, 64, 64)
    targets = _make_targets(2, n_points=3)
    loss_dict = model(images, targets)
    assert loss_dict["loss"].item() > 0


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
# Eval mode — output shapes and semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 2])
def test_forward_eval_output_shape(model, batch_size):
    """Eval mode returns one tensor per image, each sized input // downsample_ratio."""
    model.eval()
    H, W = 64, 96
    images = [torch.rand(3, H, W) for _ in range(batch_size)]
    with torch.no_grad():
        density_maps, normed_densities = model(images)

    assert len(density_maps) == batch_size
    assert len(normed_densities) == batch_size
    expected_h = H // model.downsample_ratio
    expected_w = W // model.downsample_ratio
    for dm, nd in zip(density_maps, normed_densities, strict=True):
        assert dm.shape == (1, 1, expected_h, expected_w)
        assert nd.shape == (1, 1, expected_h, expected_w)


def test_forward_eval_tensor_input(model):
    """Eval mode with a pre-stacked tensor returns a stacked tensor pair."""
    model.eval()
    images = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        density_map, _ = model(images)
    # Tensor input → Tensor output
    assert isinstance(density_map, torch.Tensor)
    assert density_map.shape[0] == 2


def test_normed_density_sums_to_one(model):
    """normed_density should sum to approximately 1 per image."""
    model.eval()
    images = [torch.rand(3, 64, 64), torch.rand(3, 64, 64)]
    with torch.no_grad():
        _, normed_list = model(images)
    for nd in normed_list:
        assert abs(nd.sum().item() - 1.0) < 1e-3


def test_forward_eval_mixed_sizes(model):
    """With a mixed-size list, each output is cropped to its own input size."""
    model.eval()
    shapes = [(64, 64), (64, 48)]
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
# Dataset integration (no deepforest.main)
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
    # Points spaced ~40+ px apart with sigma=5 → well-separated Gaussians.
    gt_points = torch.tensor([[15.0, 15.0], [60.0, 40.0], [100.0, 100.0]])  # (x, y)

    sigma = model.density_sigma
    score_integration_radius = 5

    # Build GT density the same way the model does during training.
    gt_density = model._make_gt_density(
        [gt_points],
        out_h=out_h,
        out_w=out_w,
        image_h=out_h * model.downsample_ratio,
        image_w=out_w * model.downsample_ratio,
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
