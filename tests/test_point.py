import os

import pandas as pd
import pytest
import torch

from deepforest import evaluate, get_data
from deepforest.main import deepforest
from deepforest.models.treeformer import TreeFormerModel

@pytest.fixture()
def point_model(tmp_path_factory):
    csv_file = get_data("2019_BLAN_3_751000_4330000_image_crop_keypoints.csv")
    root_dir = os.path.dirname(csv_file)
    m = deepforest(config="point", config_args={"train":
                                                   {"csv_file": csv_file,
                                                    "root_dir": root_dir,
                                                    "fast_dev_run": False},
                                                   "validation":
                                                   {"csv_file": csv_file,
                                                    "root_dir": root_dir},
                                                   "log_root": str(tmp_path_factory.mktemp("logs"))})
    return m


def test_point_prediction(point_model):
    path = get_data("2019_BLAN_3_751000_4330000_image_crop.jpg")
    prediction = point_model.predict_image(path=path)

    assert isinstance(prediction, pd.DataFrame)
    assert not prediction.empty
    assert set(prediction.columns) == {"x", "y", "label", "score", "image_path", "geometry"}
    assert (prediction["score"] >= 0).all() and (prediction["score"] <= 1).all()
    assert (prediction["label"] == "Tree").all()


def test_point_evaluation(point_model):
    """Predict the sample image and check precision/recall against bundled ground truth."""
    path = get_data("2019_BLAN_3_751000_4330000_image_crop.jpg")
    prediction = point_model.predict_image(path=path)
    assert prediction is not None, "Model returned no predictions for the sample image"

    ground_df = pd.read_csv(
        get_data("2019_BLAN_3_751000_4330000_image_crop_keypoints.csv")
    )

    results = evaluate.evaluate_geometry(
        predictions=prediction,
        ground_df=ground_df,
        geometry_type="point",
        distance_threshold=40,
    )

    assert results["point_recall"] >= 0.7, (
        f"point_recall {results['point_recall']:.2f} is below 0.7"
    )
    assert results["point_precision"] >= 0.5, (
        f"point_precision {results['point_precision']:.2f} is below 0.5"
    )

# Test train
def test_train_single(point_model):
    point_model.create_trainer(limit_train_batches=1)
    point_model.trainer.fit(point_model)

def test_eval_single(point_model):
    point_model.config.validation.val_accuracy_interval = 1
    point_model.create_trainer(limit_train_batches=1)
    results = point_model.trainer.validate(point_model)

    assert len(results) == 1
    metrics = results[0]
    assert "val_mae" in metrics
    assert "point_precision" in metrics
    assert "point_recall" in metrics
    assert metrics["val_mae"] <= 5.0, f"Expected val_mae to be <= 5.0, got {metrics['val_mae']:.2f}"
    assert 0.3 <= metrics["point_precision"] <= 1.0, f"Expected point_precision to be in [0.3, 1.0], got {metrics['point_precision']:.2f}"
    assert 0.3 <= metrics["point_recall"] <= 1.0, f"Expected point_recall to be in [0.3, 1.0], got {metrics['point_recall']:.2f}"

def test_treeformer_forward_pass_train():
    """Training forward pass returns the full loss dict; the loss is a scalar,
    differentiable, and gradients reach the backbone and regression head."""
    # norm_cood=True gives non-degenerate (global) OT, matching the NEON config.
    model = TreeFormerModel(backbone="pvt_v2_b0", num_classes=1, norm_cood=True)
    model.train()

    images = torch.rand(2, 3, 128, 128)
    targets = [
        {"points": torch.rand(5, 2) * 128, "labels": torch.zeros(5, dtype=torch.int64)},
        {"points": torch.rand(3, 2) * 128, "labels": torch.zeros(3, dtype=torch.int64)},
    ]

    output = model(images, targets)

    assert isinstance(output, dict)
    for key in ("loss", "count_loss", "ot_loss", "density_l1_loss", "count_cls_loss"):
        assert key in output, f"missing loss term: {key}"

    loss = output["loss"]
    assert loss.ndim == 0
    assert loss.requires_grad
    assert torch.isfinite(loss)

    # With norm_cood=True the OT term is non-degenerate: finite Wasserstein
    # distance and Sinkhorn converges within the iteration budget.
    assert torch.isfinite(output["ot_wd"])
    assert output["sinkhorn_its"] < model.ot_iter

    loss.backward()
    backbone_grads = regression_grads = 0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        assert torch.isfinite(param.grad).all(), f"non-finite grad in {name}"
        backbone_grads += name.startswith("backbone")
        regression_grads += name.startswith("regression")
    assert backbone_grads > 0, "no gradient reached the backbone"
    assert regression_grads > 0, "no gradient reached the regression head"


def test_treeformer_forward_pass_val():
    """Eval forward pass returns one prediction dict per image."""
    model = TreeFormerModel(backbone="pvt_v2_b0", num_classes=1)
    model.eval()

    images = torch.rand(2, 3, 128, 128)

    with torch.no_grad():
        output = model(images)

    assert isinstance(output, list)
    assert len(output) == 2

    for pred in output:
        assert "points" in pred
        assert "scores" in pred
        assert "labels" in pred
        n = pred["points"].shape[0]
        assert pred["points"].shape == (n, 2)
        assert pred["scores"].shape == (n,)
        assert pred["labels"].shape == (n,)


def test_treeformer_loss_terms_toggle():
    """active_losses gating: disabled terms are zero, count_cls is omitted, and
    the total equals the single active term."""
    model = TreeFormerModel(
        backbone="pvt_v2_b0", num_classes=1, losses=["count"], enforce_count=False
    )
    model.train()

    images = torch.rand(2, 3, 96, 96)
    targets = [
        {"points": torch.rand(4, 2) * 96, "labels": torch.zeros(4, dtype=torch.int64)},
        {"points": torch.rand(2, 2) * 96, "labels": torch.zeros(2, dtype=torch.int64)},
    ]

    output = model(images, targets)

    assert output["ot_loss"].item() == 0.0
    assert output["density_l1_loss"].item() == 0.0
    assert "count_cls_loss" not in output
    assert torch.allclose(output["loss"], output["count_loss"])
    assert output["loss"].requires_grad
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
