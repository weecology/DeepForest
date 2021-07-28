#test model
from deepforest import model
import pytest
import numpy as np
import torch
import torchvision
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#Empty tester from https://github.com/datumbox/vision/blob/06ebee1a9f10c76d8ac5768fd578362dd5ace6e9/test/test_models_detection_negative_samples.py#L14
def _make_empty_sample():
    images = [torch.rand((3, 100, 100), dtype=torch.float32)]
    boxes = torch.zeros((0, 4), dtype=torch.float32)
    negative_target = {"boxes": boxes,
                       "labels": torch.zeros(0, dtype=torch.int64),
                       "image_id": 4,
                       "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                       "iscrowd": torch.zeros((0,), dtype=torch.int64)}

    targets = [negative_target]
    return images, targets

def test_load_backbone():
    retinanet = model.load_backbone()
    retinanet.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    prediction = retinanet(x)    

@pytest.mark.parametrize("num_classes",[1,2,10])
def test_create_model(num_classes):
    retinanet_model = model.create_model(num_classes=2,nms_thresh=0.1, score_thresh=0.2)
    
    retinanet_model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = retinanet_model(x)    

def test_forward_empty():
    retinanet_model = model.create_model(num_classes=2,nms_thresh=0.1, score_thresh=0.2)
    image, targets = _make_empty_sample()
    loss = retinanet_model(image, targets)
    assert torch.equal(loss["bbox_regression"], torch.tensor(0.))
    
def test_forward_negative_sample_retinanet():
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        num_classes=2, min_size=100, max_size=100)

    images, targets = _make_empty_sample()
    loss_dict = model(images, targets)

    assert torch.equal(loss_dict["bbox_regression"], torch.tensor(0.))