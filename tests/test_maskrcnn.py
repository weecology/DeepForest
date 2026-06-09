# Test Mask R-CNN polygon (instance segmentation) workflow
import os

import numpy as np
import pytest
import torch

from deepforest import get_data, main
from deepforest.models import maskrcnn


@pytest.fixture()
def polygon_annotation_file():
    return get_data("coco_sample_file.json")


@pytest.fixture()
def polygon_root_dir():
    return os.path.dirname(get_data("coco_sample_file.json"))


def build_model(score_thresh=0.5):
    """Random-init Mask R-CNN (no weight download) with a small input size."""
    return maskrcnn.MaskRCNN(
        backbone_weights=None,
        num_classes=1,
        label_dict={"tree": 0},
        score_thresh=score_thresh,
        min_size=400,
        max_size=512,
    )


def test_task_is_polygon():
    assert maskrcnn.MaskRCNN.task == "polygon"


def test_predict_outputs_masks():
    model = build_model(score_thresh=0.0)
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 400, 300)]
    predictions = model(x)

    assert len(predictions) == 2
    assert sorted(predictions[0].keys()) == ["boxes", "labels", "masks", "scores"]
    # Labels are shifted back to the zero-indexed DeepForest convention
    if len(predictions[0]["labels"]) > 0:
        assert predictions[0]["labels"].min() >= 0


def test_training_label_shift_does_not_mutate_targets():
    model = build_model()
    model.train()
    images = [torch.rand(3, 200, 200)]
    targets = [
        {
            "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.int64),
            "masks": torch.zeros((1, 200, 200), dtype=torch.uint8),
        }
    ]
    targets[0]["masks"][0, 10:50, 10:50] = 1

    loss_dict = model(images, targets)

    assert "loss_mask" in loss_dict
    # The caller's target labels must remain zero-indexed
    assert targets[0]["labels"].tolist() == [0]


def test_check_model():
    model = maskrcnn.MaskRCNN(backbone_weights=None, num_classes=1)
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)
    assert sorted(predictions[1].keys()) == ["boxes", "labels", "masks", "scores"]


def _make_polygon_model(tmp_path, polygon_annotation_file, polygon_root_dir):
    model = build_model(score_thresh=0.0)
    m = main.deepforest(
        model=model,
        config_args={"num_classes": 1, "label_dict": {"tree": 0}},
    )
    m.set_labels({"tree": 0})
    m.config.train.csv_file = polygon_annotation_file
    m.config.train.root_dir = polygon_root_dir
    m.config.validation.csv_file = polygon_annotation_file
    m.config.validation.root_dir = polygon_root_dir
    m.config.train.fast_dev_run = True
    m.config.validation.val_accuracy_interval = 1
    m.config.log_root = str(tmp_path)
    m.create_trainer()
    return m


def test_polygon_train_and_validate(
    tmp_path, polygon_annotation_file, polygon_root_dir
):
    m = _make_polygon_model(tmp_path, polygon_annotation_file, polygon_root_dir)
    assert m.model.task == "polygon"

    m.trainer.fit(m)
    m.trainer.validate(m)

    # Segmentation mAP and polygon recall/precision are logged
    logged = m.trainer.logged_metrics
    assert any("polygon" in key for key in logged) or "map" in logged


def test_polygon_predict_image(polygon_root_dir):
    model = build_model(score_thresh=0.0)
    m = main.deepforest(
        model=model,
        config_args={"num_classes": 1, "label_dict": {"tree": 0}},
    )
    m.set_labels({"tree": 0})

    image = np.array(
        torch.randint(0, 255, (200, 200, 3), dtype=torch.uint8)
    ).astype("float32")
    result = m.predict_image(image=image)

    # Random weights may or may not produce detections; if they do, they must
    # be polygons in the standard results schema.
    if result is not None:
        assert "geometry" in result.columns
        assert "label" in result.columns
        assert result.geometry.iloc[0].geom_type in ("Polygon", "MultiPolygon")


def test_polygon_predict_tile():
    """predict_tile tiles the image and routes each window through the model,
    mosaicking polygon outputs across windows."""
    model = build_model(score_thresh=0.0)
    m = main.deepforest(
        model=model,
        config_args={"num_classes": 1, "label_dict": {"tree": 0}},
    )
    m.set_labels({"tree": 0})

    image = np.array(
        torch.randint(0, 255, (150, 150, 3), dtype=torch.uint8)
    ).astype("float32")
    result = m.predict_tile(
        image=image, patch_size=100, patch_overlap=0.25, dataloader_strategy="single"
    )

    if result is not None:
        assert "geometry" in result.columns
        assert result.geometry.iloc[0].geom_type in ("Polygon", "MultiPolygon")
