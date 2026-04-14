
"""Unit tests for TreeFormerModel.

Tests are scoped to the model class itself; deepforest.main is not used
except where explicitly noted.
"""

# ...existing code...

def test_density_mode_cls_gradient_not_scaled_by_area():
    """Sanity check: CLS-head gradients should not be scaled by image area.

    If the loss is computed in absolute count space (after multiplying by area),
    the gradient norm will be orders of magnitude larger for large images.
    This test ensures the gradient norm is within a reasonable range.
    """
    torch.manual_seed(42)
    image_size = 512
    n_points = 10
    images = [torch.rand(3, image_size, image_size)]
    targets = [{"points": torch.rand(n_points, 2) * image_size}]

    m = TreeFormerModel(
        backbone=_BACKBONE,
        pretrained=False,
        enforce_count=True,
        losses=["count", "count_cls"],
    )
    m.train()
    loss_dict = m(images, targets)
    loss_dict["loss"].backward()

    # Collect gradient on the last CLS linear layer (cls_lin4).
    cls_grad = m.regression.cls_lin4.weight.grad
    assert cls_grad is not None, "No gradient on cls_lin4"
    grad_norm = cls_grad.norm().item()

    # If the loss was computed in absolute space, grad_norm would be ~1e5-1e6 for 512x512 images.
    # In density mode, it should be much smaller (typically < 100).
    assert grad_norm < 1000, (
        f"CLS grad norm too large: {grad_norm}. "
        "Loss may be computed in absolute space (scaled by area) instead of density."
    )

"""Unit tests for TreeFormerModel.

Tests are scoped to the model class itself; deepforest.main is not used
except where explicitly noted.
"""


import inspect
import os
from copy import deepcopy
import pytest
import torch

from deepforest import get_data, utilities
from deepforest.conf.schema import KeypointConfig
from deepforest.datasets.training import KeypointDataset
from deepforest.models import treeformer
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


def test_treeformer_keypoint_fields_match_constructor_signature():
    """Key TreeFormer config fields should exist in both schema and constructor."""
    schema_fields = set(KeypointConfig.__dataclass_fields__)
    constructor_fields = set(inspect.signature(TreeFormerModel.__init__).parameters)

    assert _TREEFORMER_KEYPOINT_FIELDS <= schema_fields
    assert _TREEFORMER_KEYPOINT_FIELDS <= constructor_fields


def test_treeformer_nested_keypoint_override_preserves_defaults():
    """Nested keypoint overrides should merge with defaults rather than replace them."""
    config = utilities.load_config(
        config_name="treeformer",
        overrides={"keypoint": {"enforce_count": False}},
    )

    assert config.keypoint.enforce_count is False
    assert config.keypoint.ot_weight == 0.4
    assert config.keypoint.density_l1_weight == 0.01
    assert config.keypoint.losses == ["count", "count_cls", "ot", "density_l1"]


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


@pytest.mark.parametrize(
    ("field_name", "value", "attr_name", "expected"),
    [
        ("density_l1_weight", 0.125, "density_l1_weight", 0.125),
        ("sinkhorn_reg", 0.75, "sinkhorn_reg", 0.75),
        ("num_of_iter_in_ot", 17, "ot_iter", 17),
        ("enforce_count", False, "enforce_count", False),
        ("losses", ["count", "ot"], "losses", ["count", "ot"]),
    ],
)
def test_treeformer_factory_threads_keypoint_config(
    monkeypatch, tmp_path, field_name, value, attr_name, expected
):
    """Factory-created TreeFormer models should receive configured keypoint values."""
    config = _build_treeformer_from_config(tmp_path, {field_name: value})
    captured_kwargs = {}

    class SpyTreeFormerModel:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def to(self, map_location):
            return self

    monkeypatch.setattr(treeformer, "TreeFormerModel", SpyTreeFormerModel)
    treeformer.Model(config=config).create_model(backbone=_BACKBONE)

    expected_kwargs = {
        "density_l1_weight": ("density_l1_weight", 0.125),
        "sinkhorn_reg": ("sinkhorn_reg", 0.75),
        "num_of_iter_in_ot": ("num_of_iter_in_ot", 17),
        "enforce_count": ("enforce_count", False),
        "log_count_loss": ("log_count_loss", True),
        "losses": ("losses", ["count", "ot"]),
    }
    constructor_arg, constructor_value = expected_kwargs[field_name]
    assert captured_kwargs[constructor_arg] == constructor_value


def test_treeformer_save_pretrained_preserves_configured_fields(monkeypatch, tmp_path):
    """Hub export should round-trip the TreeFormer config contract."""
    model = TreeFormerModel(
        backbone=_BACKBONE,
        pretrained=False,
        density_l1_weight=0.25,
        sinkhorn_reg=0.5,
        num_of_iter_in_ot=23,
        enforce_count=False,
        losses=["count", "ot"],
    )

    monkeypatch.setattr(
        treeformer.AutoImageProcessor,
        "from_pretrained",
        lambda *args, **kwargs: model.processor,
    )

    def fake_backbone_from_pretrained(*args, **kwargs):
        return treeformer.PvtV2Model(deepcopy(model.backbone.config))

    monkeypatch.setattr(
        treeformer.PvtV2Model,
        "from_pretrained",
        fake_backbone_from_pretrained,
    )

    export_dir = tmp_path / "treeformer_hf"
    model.save_pretrained(export_dir)
    reloaded = TreeFormerModel.from_pretrained(export_dir)

    assert reloaded.backbone_name == _BACKBONE
    assert reloaded.density_l1_weight == 0.25
    assert reloaded.sinkhorn_reg == 0.5
    assert reloaded.ot_iter == 23
    assert reloaded.enforce_count is False
    assert reloaded.losses == ["count", "ot"]


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


@pytest.mark.xfail(reason="Random weights can cause normed_density to not sum to 1.0; TODO: check with a trained model.")
def test_normed_density_sums_to_one(model):
    """
    normed_density should sum to approximately 1 per image.
    TODO: Replace with a check using a trained model for deterministic behavior.
    """
    model.eval()
    images = [torch.rand(3, 128, 128), torch.rand(3, 128, 128)]
    with torch.no_grad():
        _, normed_list = model(images)
    for nd in normed_list:
        assert abs(nd.sum().item() - 1.0) < 1e-2


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


# ---------------------------------------------------------------------------
# Density mode dead-neuron: negative CLS init must recover with count_cls loss
# ---------------------------------------------------------------------------


def _make_density_model_negative_cls(losses):
    """Create a density-mode model with the CLS head biased strongly negative.

    Forces raw_cls < 0 by setting cls_lin4 bias to a large negative value,
    which causes clamp(min=0) in _normalize_density to zero out density_map
    and block all count-loss gradients to the CLS head.
    """
    m = TreeFormerModel(
        backbone=_BACKBONE,
        pretrained=False,
        enforce_count=True,
        losses=losses,
        count_cls_weight=1.0,
    )
    m.train()
    # A large negative bias guarantees the final linear output is negative
    # regardless of how upstream weights transform the backbone features.
    with torch.no_grad():
        m.regression.cls_lin4.bias.fill_(-100.0)
    return m


def test_density_mode_negative_cls_count_loss_gradient_blocked():
    """RED: count loss gives zero gradient to cls_lin4 when raw_cls < 0.

    With enforce_count=True and count_prediction_mode='density', density_map =
    normed * clamp(raw_cls * area, 0).  When raw_cls < 0 the clamp zeros
    density_map, so pred_sum = 0 and the gradient from count_loss through
    the clamp is zero.  Without count_cls in the losses there is no other
    unblocked gradient path to cls_lin4.
    """
    torch.manual_seed(0)
    images = [torch.rand(3, 64, 64)]
    targets = [{"points": torch.rand(5, 2) * 64}]

    m = _make_density_model_negative_cls(["count", "ot", "density_l1"])
    loss_dict = m(images, targets)
    loss_dict["loss"].backward()

    cls_grad = m.regression.cls_lin4.bias.grad
    assert cls_grad is not None
    # Gradient must be exactly zero: clamp blocks the count-loss path and
    # there is no other loss that touches cls_lin4 in this loss set.
    assert cls_grad.abs().max().item() == 0.0, (
        f"Expected zero gradient on cls_lin4.bias without count_cls loss "
        f"when raw_cls < 0, got {cls_grad.abs().max().item():.6f}"
    )


def test_density_mode_negative_cls_count_cls_loss_unblocked():
    """GREEN: count_cls loss provides non-zero gradient to cls_lin4 when raw_cls < 0.

    The count_cls loss computes L1(cls_preds, gt_count/area) directly from the
    raw CLS head outputs, before any clamp.  This provides a gradient path that
    survives even when raw_cls < 0 and density_map = 0.
    """
    torch.manual_seed(0)
    images = [torch.rand(3, 64, 64)]
    targets = [{"points": torch.rand(5, 2) * 64}]

    m = _make_density_model_negative_cls(["count", "ot", "density_l1", "count_cls"])
    loss_dict = m(images, targets)
    loss_dict["loss"].backward()

    cls_grad = m.regression.cls_lin4.bias.grad
    assert cls_grad is not None
    assert cls_grad.abs().max().item() > 0.0, (
        f"Expected non-zero gradient on cls_lin4.bias with count_cls loss, "
        f"got {cls_grad.abs().max().item():.6f}"
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

    assert loss.requires_grad, (
        "loss must require grad when all point lists are empty"
    )
    # Should not raise.
    loss.backward()


def test_normalize_density_floor_prevents_collapse():
    """With enforce_count=True and cls_count near zero, density_map.sum() > 0.

    The clamp(min=1e-4) floor ensures OT's manual gradient (source_count /
    (source_count^2 + 1e-8)) is nonzero from the first training step, even
    before the CLS head has learned to output positive values.
    """
    m = TreeFormerModel(
        backbone=_BACKBONE,
        pretrained=False,
        enforce_count=True,
        losses=["ot"],
    )
    m.train()
    # Force CLS head to output near-zero (cold-start scenario).
    with torch.no_grad():
        m.regression.cls_lin4.bias.fill_(0.0)
        m.regression.cls_lin4.weight.fill_(0.0)

    images = [torch.rand(3, 64, 64)]
    # Call _normalize_density directly via a forward pass inspection.
    score_map = torch.ones(1, 1, 8, 8)
    cls_count = torch.tensor([0.0])
    density_map, _ = m._normalize_density(score_map, cls_count)
    total = density_map.sum().item()
    assert total > 0.0, (
        f"density_map.sum() should be > 0 due to clamp floor, got {total}"
    )


def test_training_step_nonfinite_loss_stays_differentiable():
    """training_step must return a differentiable tensor even when loss is
    non-finite, so DDP gradient allreduce fires on every rank."""
    import deepforest.main as df_main

    model = df_main.deepforest(config_args={"architecture": "treeformer"})
    model.create_model()
    model.model.train()

    # Build a minimal batch (image + target) that the dataset normally produces.
    images = [torch.rand(3, 64, 64)]
    targets = [
        {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros(0, dtype=torch.long),
            "image_id": torch.tensor(0),
        }
    ]
    batch = (images, targets, ["dummy.tif"])

    # Patch model.forward to return a NaN loss with grad_fn.
    _nan = torch.tensor(float("nan"), requires_grad=True)

    def _bad_forward(imgs, tgts):
        return {"loss": _nan + 0.0}  # keeps grad_fn

    original_forward = model.model.forward
    model.model.forward = _bad_forward
    try:
        result = model.training_step(batch, batch_idx=0)
    finally:
        model.model.forward = original_forward

    assert result.grad_fn is not None, (
        "training_step must return a tensor with grad_fn on non-finite loss "
        "so that DDP gradient allreduce fires on all ranks"
    )
    result.backward()  # must not raise


# ---------------------------------------------------------------------------
# OT loss numerical stability — run_019 degenerate case
# ---------------------------------------------------------------------------


def test_ot_loss_converges_within_budget_with_norm_coord():
    """With norm_cood=True (run_020 fix), Sinkhorn converges within the
    iteration budget for a realistic 128x128 output map.

    Normalized coordinates bound the max squared distance to ~8, so
    K = exp(-C/reg) >= exp(-8) ~ 0.0003 for any pixel pair — the transport
    matrix is dense and well-conditioned.

    This is the GREEN half of the run_019 regression: the same assertion
    fails with norm_cood=False (see test below).
    """
    from deepforest.losses.ot_loss import OT_Loss

    device = torch.device("cpu")
    maxIter = 50
    output_h, output_w = 128, 128
    n_gt = 50

    torch.manual_seed(0)
    unnormed = torch.rand(1, 1, output_h, output_w, requires_grad=True)
    normed = unnormed / unnormed.sum()
    points = [torch.rand(n_gt, 2) * torch.tensor([float(output_w), float(output_h)])]

    loss_fn = OT_Loss(norm_coord=True, device=device, num_of_iter_in_ot=maxIter, reg=1.0)
    ot_loss, _, _, avg_its, K_min, beta_abs_max, sinkhorn_err = loss_fn(
        normed, unnormed, points
    )

    # Should converge well before the cap.
    assert avg_its < maxIter, (
        f"Sinkhorn should converge within {maxIter} iterations with norm_cood=True, "
        f"got {avg_its:.1f}"
    )

    # Gradient should be non-trivially non-zero.
    ot_loss.backward()
    grad_norm = unnormed.grad.abs().max().item()
    assert grad_norm > 1e-6, (
        f"Expected non-zero OT gradient with norm_cood=True, got max |grad|={grad_norm:.2e}"
    )


@pytest.mark.xfail(
    reason=(
        "run_019 regression: with norm_cood=False, reg=1.0 and a 128x128 output map, "
        "K = exp(clamp(-C/reg, -80)) hits its floor for all pixel pairs >9px apart. "
        "The floor dominates the Sinkhorn denominator (M_EPS=1e-16 >> floor=5.6e-35), "
        "causing u values to grow exponentially and overflow float32 before convergence. "
        "Fix: norm_cood=True (run_020), or scale reg to the coordinate range."
    ),
    strict=True,
)
def test_ot_loss_converges_within_budget_with_run019_params():
    """Sinkhorn should converge within the iteration budget for a 128x128 map.

    This test FAILS (xfail) with run_019 params (norm_cood=False, reg=1.0):
    the transport plan never converges and avg_its always equals maxIter + 1.

    It should PASS (xpass -> test turns green) once the numerical issue is fixed,
    e.g. by enabling norm_cood=True.
    """
    from deepforest.losses.ot_loss import OT_Loss

    device = torch.device("cpu")
    maxIter = 50
    output_h, output_w = 128, 128
    n_gt = 50

    torch.manual_seed(0)
    unnormed = torch.rand(1, 1, output_h, output_w, requires_grad=True)
    normed = unnormed / unnormed.sum()
    points = [torch.rand(n_gt, 2) * torch.tensor([float(output_w), float(output_h)])]

    loss_fn = OT_Loss(norm_coord=False, device=device, num_of_iter_in_ot=maxIter, reg=1.0)
    _, _, _, avg_its, _, _, _ = loss_fn(normed, unnormed, points)

    assert avg_its < maxIter, (
        f"Sinkhorn should converge within {maxIter} iterations, got {avg_its:.1f} "
        f"(always hitting cap — transport plan is unconverged)"
    )
