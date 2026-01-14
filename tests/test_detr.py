# test Transformers/detr
import os

import numpy as np
import pytest
import torch
from PIL import Image

from deepforest import get_data
from deepforest import utilities
from deepforest.datasets.training import BoxDataset
from deepforest.models import DeformableDetr


@pytest.fixture()
def config(tmp_path_factory):
    config = utilities.load_config()
    config.model.name = "joshvm/milliontrees-detr"
    config.architecture = "DeformableDetr"
    config.train.fast_dev_run = True
    config.batch_size = 1
    config.score_thresh = 0.5
    config.log_root = str(tmp_path_factory.mktemp("logs"))
    return config


@pytest.fixture()
def coco_sample():
    """
    Dummy sample that conforms to the MS-COCO format
    """
    images = [torch.rand((3, 100, 100), dtype=torch.float32)]
    target = {
        "labels": torch.zeros(0, dtype=torch.int64),
        "image_id": 4,
        "annotations": [{
            "id": 0,
            "image_id": 4,
            "category_id": 0,
            "bbox": [0, 0, 10, 10],
            "area": 10,
            "iscrowd": 0,
        }]
    }

    targets = [target]
    return images, targets


def test_check_model(config):
    model = DeformableDetr.Model(config)
    model.check_model()


# The test case "2" currently fails due to a bug in transformers
# which is fixed in transformers-4.53.0, related to the
# from_pretrained logic.
@pytest.mark.parametrize("num_classes", [1, 5, 10])
def test_create_model(config, num_classes):
    """
    Test that we can instantiate a model with differing numbers
    of classes and that we can pass images through.
    """
    config.num_classes = num_classes
    config.label_dict = {f"{i}": i for i in range(num_classes)}
    detr_model = DeformableDetr.Model(config).create_model()
    detr_model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    _ = detr_model(x)

    assert detr_model.label_dict == config.label_dict


def test_boxes_in_output(config):
    """
    Test that a reference input image yields predictions that
    include boxes, scores and labels. This model should have
    trained weights.
    """
    detr_model = DeformableDetr.Model(config).create_model(config.model.name, revision=config.model.revision)
    detr_model.eval()

    image_path = get_data("OSBS_029.png")

    # Passing a numpy array (or tensor) should work:
    result = detr_model(np.array(Image.open(image_path)))

    for r in result:
        assert "boxes" in r
        assert "scores" in r
        assert "labels" in r

    # Passing a list is also allowed:
    result = detr_model([np.array(Image.open(image_path))])

    for r in result:
        assert "boxes" in r
        assert "scores" in r
        assert "labels" in r


def test_forward_sample_dummy(config, coco_sample):
    """
    Test that in training mode, we get a loss dict and it's
    non-zero.
    """
    detr_model = DeformableDetr.Model(config).create_model()
    detr_model.train()

    image, targets = coco_sample
    loss_dict = detr_model(image, targets, prepare_targets=False)

    # Assert non-zero loss
    assert sum([loss for loss in loss_dict.values()]) > 0


def test_training_sample(config):
    """
    Confirm integration between the training BoxDataset and
    the model.
    """
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    ds = BoxDataset(csv_file=csv_file, root_dir=root_dir)

    image, targets, _ = next(iter(ds))

    detr_model = DeformableDetr.Model(config).create_model()
    detr_model.train()

    loss_dict = detr_model(image, targets)

    # Assert non-zero loss
    assert sum([loss for loss in loss_dict.values()]) > 0


def test_prepare_targets_coco_format(config):
    """
    Test conversion to COCO format for targets.
    """
    detr_model = DeformableDetr.Model(config).create_model()

    x1, y1, w1, h1 = 10.0, 20.0, 40.0, 60.0
    x2, y2, w2, h2 = 100.0, 150.0, 100.0, 150.0

    target = {
        "labels": torch.tensor([0, 1]),
        "boxes": torch.tensor([
            [x1, y1, x1 + w1, y1 + h1],
            [x2, y2, x2 + w2, y2 + h2],
        ]),
    }

    coco_targets = detr_model._prepare_targets([target])

    # Expect 1 group with 2 annotations, IDs shouldn't matter
    assert len(coco_targets) == 1
    assert coco_targets[0]["image_id"] == 0
    assert "annotations" in coco_targets[0]

    annotations = coco_targets[0]["annotations"]
    assert len(annotations) == 2

    # First annotation
    assert annotations[0]["id"] == 0
    assert annotations[0]["image_id"] == 0
    assert annotations[0]["category_id"] == 0
    assert annotations[0]["bbox"] == [x1, y1, w1, h1]
    assert annotations[0]["area"] == w1 * h1
    assert annotations[0]["iscrowd"] == 0

    # Second annotation
    assert annotations[1]["id"] == 1
    assert annotations[1]["image_id"] == 1
    assert annotations[1]["category_id"] == 1
    assert annotations[1]["bbox"] == [x2, y2, w2, h2]
    assert annotations[1]["area"] == w2 * h2
    assert annotations[1]["iscrowd"] == 0
