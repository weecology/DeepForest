# test retinanet
from deepforest.models import retinanet
from deepforest import get_data
import pytest
import numpy as np
import torch
import torchvision
import os
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Empty tester from https://github.com/datumbox/vision/blob/06ebee1a9f10c76d8ac5768fd578362dd5ace6e9/test/test_models_detection_negative_samples.py#L14
def _make_empty_sample():
    images = [torch.rand((3, 100, 100), dtype=torch.float32)]
    boxes = torch.zeros((0, 4), dtype=torch.float32)
    negative_target = {
        "boxes": boxes,
        "labels": torch.zeros(0, dtype=torch.int64),
        "image_id": 4,
        "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
        "iscrowd": torch.zeros((0,), dtype=torch.int64)
    }

    targets = [negative_target]
    return images, targets


def test_retinanet(config):
    r = retinanet.Model(config)
    assert r


def retinanet_check_model(config):
    r = retinanet.Model(config)
    r.check_model()


# This test still fails, do we want a way to pass kwargs directly to method,
# instead of being limited by config structure?
# Need to create issue when I get online.
@pytest.mark.parametrize("num_classes", [1, 2, 10])
def test_create_model(config, num_classes):
    config.num_classes = num_classes
    config.label_dict = {f"{i}": i for i in range(num_classes)}
    retinanet_model = retinanet.Model(config).create_model()
    retinanet_model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = retinanet_model(x)


def test_forward_empty(config):
    r = retinanet.Model(config)
    model = r.create_model()
    image, targets = _make_empty_sample()
    loss = model(image, targets)
    assert torch.equal(loss["bbox_regression"], torch.tensor(0.))


# Can we update parameters after training
def test_maintain_parameters(config):
    config.score_thresh = 0.4
    retinanet_model = retinanet.Model(config).create_model()
    assert retinanet_model.score_thresh == config.score_thresh
    retinanet_model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = retinanet_model(x)
    assert retinanet_model.score_thresh == config.score_thresh

    retinanet_model.score_thresh = 0.9
    retinanet_model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = retinanet_model(x)
    assert retinanet_model.score_thresh == 0.9

    #TODO: Check that updating the score threshold affects prediction count.
