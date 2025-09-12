# test retinanet
import os

import pytest
import torch

from deepforest.models import retinanet

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

@pytest.mark.parametrize("model_name", [None, "dinov3"])
def test_retinanet_inference(config, model_name):
    config.model.name = model_name
    r = retinanet.Model(config)
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    retinanet_model = retinanet.Model(config).create_model()
    retinanet_model.eval()

    # Expect output to be a list for batched input, each
    # output should have a box, score and label key.
    with torch.no_grad():
        predictions = retinanet_model(x)
        assert isinstance(predictions, list)
        for pred in predictions:
            assert "boxes" in pred
            assert "scores" in pred
            assert "labels" in pred

@pytest.mark.parametrize("model_name", [None, "dinov3"])
def test_retinanet_train(config, model_name):
    config.model.name = model_name
    r = retinanet.Model(config)
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    targets = [{"boxes": torch.tensor([[0,0,50,50], [25,25,90,90]]),
               "labels": torch.tensor([0,0]).long()},
               {"boxes": torch.tensor([[10,50,30,80], [100,100, 200,200]]),
               "labels": torch.tensor([0,0]).long()}]

    retinanet_model = retinanet.Model(config).create_model()
    retinanet_model.train()

    # Expect output to be a dictionary of loss values for the batch
    # for bbox regression and classification
    loss_dict = retinanet_model(x, targets)
    assert isinstance(loss_dict, dict)
    assert "bbox_regression" in loss_dict
    assert "classification" in loss_dict
    assert loss_dict["bbox_regression"] > 0
    assert loss_dict["classification"] > 0


def retinanet_check_model(config):
    r = retinanet.Model(config)
    r.check_model()


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
