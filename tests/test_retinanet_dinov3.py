import os

import pytest
import torch
from torchvision.io import read_image

from deepforest import get_data, main
from deepforest.models import retinanet_dinov3


@pytest.mark.parametrize("freeze_backbone", [True, False])
def test_train_from_config(dinov3_encoder, tmp_path, freeze_backbone):
    csv_file = get_data("OSBS_029.csv")
    m = main.deepforest(
        config="dinov3",
        config_args={
            "model": {"name": dinov3_encoder},
            "train": {
                "csv_file": csv_file,
                "root_dir": os.path.dirname(csv_file),
                "fast_dev_run": True,
                "freeze_backbone": freeze_backbone,
            },
            "accelerator": "cpu",
            "devices": 1,
            "log_root": str(tmp_path),
        },
    )
    assert isinstance(m.model, retinanet_dinov3.RetinaNetDinov3Hub)

    encoder_weight = m.model.backbone.model.embeddings.patch_embeddings.weight
    pyramid_weight = m.model.backbone.fpn_1x.weight
    encoder_before = encoder_weight.detach().clone()
    pyramid_before = pyramid_weight.detach().clone()

    m.create_trainer()
    m.trainer.fit(m)

    assert torch.isfinite(m.trainer.callback_metrics["train_loss"])
    # Check pyramid weights changed, but backbone only does if unfrozen.
    assert not torch.equal(pyramid_before, pyramid_weight)
    assert torch.equal(encoder_before, encoder_weight) == freeze_backbone


def test_retinanet_dinov3_check_model(config):
    r = retinanet_dinov3.Model(config)
    r.check_model()


@pytest.mark.parametrize("num_classes", [1, 2, 10])
def test_create_model(config, num_classes):
    config.num_classes = num_classes
    config.label_dict = {f"{i}": i for i in range(num_classes)}
    retinanet_model = retinanet_dinov3.Model(config).create_model()
    retinanet_model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = retinanet_model(x)


def test_from_encoder_needs_labels(dinov3_encoder):
    with pytest.raises(ValueError, match="Set num_classes and label_dict"):
        retinanet_dinov3.RetinaNetDinov3Hub.from_pretrained(dinov3_encoder)

