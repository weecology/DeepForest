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
    """Training forward pass returns a loss dict with a scalar, differentiable loss."""
    model = TreeFormerModel(backbone="pvt_v2_b0", num_classes=1)
    model.train()

    images = torch.rand(2, 3, 128, 128)
    targets = [
        {"points": torch.rand(5, 2) * 128, "labels": torch.zeros(5, dtype=torch.int64)},
        {"points": torch.rand(3, 2) * 128, "labels": torch.zeros(3, dtype=torch.int64)},
    ]

    output = model(images, targets)

    assert isinstance(output, dict)
    assert "loss" in output
    assert output["loss"].ndim == 0
    assert output["loss"].requires_grad


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
