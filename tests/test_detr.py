# test Transformers/detr
from deepforest.models import detr
from deepforest import utilities
from deepforest import get_data
import pytest
import torch
from PIL import Image
import numpy as np

@pytest.fixture()
def config():
    config = utilities.load_config()
    config.model.name = "joshvm/milliontrees-detr"
    config.architecture = "detr"
    config.train.fast_dev_run = True
    config.batch_size = 1 #True Why is this true?
    return config

@pytest.fixture()
def coco_sample():
    images = [torch.rand((3, 100, 100), dtype=torch.float32)]
    negative_target = {
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

    targets = [negative_target]
    return images, targets

def test_check_model(config):
    r = detr.Model(config)
    r.check_model()

# The test case "2" currently fails due to a bug in transformers
# which is fixed in transformers-4.53.0, related to the
# from_pretrained logic.
@pytest.mark.parametrize("num_classes", [1, 5, 10])
def test_create_model(config, num_classes):
    config.num_classes = num_classes
    detr_model = detr.Model(config).create_model()
    detr_model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    _ = detr_model(x)

def test_boxes_in_output(config):
    detr_model = detr.Model(config).create_model()
    detr_model.eval()

    image_path = get_data("OSBS_029.png")

    # Passing a numpy array (or tensor) should work:
    result = detr_model(np.array(Image.open(image_path)))

    assert "boxes" in result[0]
    assert "scores" in result[0]
    assert "labels" in result[0]

    # Passing a list is also allowed:
    result = detr_model([np.array(Image.open(image_path))])

    assert "boxes" in result[0]
    assert "scores" in result[0]
    assert "labels" in result[0]

# Test
def test_forward_sample(config, coco_sample):
    r = detr.Model(config)
    model = r.create_model()
    image, targets = coco_sample
    loss_dict = model(image, targets)

    # Assert non-zero loss
    assert sum([loss for loss in loss_dict.values()]) > 0
