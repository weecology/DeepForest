"""Tests for spatial-temporal metadata embeddings in CropModel."""

import os

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from deepforest import get_data
from deepforest.datasets.cropmodel import BoundingBoxDataset
from deepforest.datasets.training import MetadataImageFolder
from deepforest.model import CropModel, SpatialTemporalEncoder

def test_crop_model_metadata_forward():
    cm = CropModel(config_args={"use_metadata": True, "metadata_dim": 32})
    cm.create_model(num_classes=5)
    x = torch.rand(4, 3, 224, 224)
    meta = torch.tensor([[35.0, -120.0, 145.0]] * 4)
    out = cm.forward(x, metadata=meta)
    assert out.shape == (4, 5)

def test_crop_model_metadata_none():
    """When use_metadata=True but metadata=None, model should still predict."""
    cm = CropModel(config_args={"use_metadata": True})
    cm.create_model(num_classes=5)
    x = torch.rand(4, 3, 224, 224)
    out = cm.forward(x, metadata=None)
    assert out.shape == (4, 5)


def test_crop_model_no_metadata_backward_compat():
    cm = CropModel()
    cm.create_model(num_classes=2)
    x = torch.rand(4, 3, 224, 224)
    out = cm.forward(x)
    assert out.shape == (4, 2)
    assert cm.backbone is None
    assert cm.metadata_encoder is None
    assert cm.classifier is None


def test_training_step_with_metadata():
    cm = CropModel(config_args={"use_metadata": True})
    cm.create_model(num_classes=3)
    x = torch.rand(4, 3, 224, 224)
    y = torch.tensor([0, 1, 2, 0])
    meta = torch.rand(4, 3)
    batch = (x, y, meta)
    loss = cm.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


@pytest.fixture()
def bbox_df():
    df = pd.read_csv(get_data("testfile_multi.csv"))
    single_image = df.image_path.unique()[0]
    return df[df.image_path == single_image].reset_index(drop=True)


def test_bounding_box_dataset_with_metadata(bbox_df):
    root_dir = os.path.dirname(get_data("SOAP_061.png"))
    n = len(bbox_df)
    metadata = dict.fromkeys(range(n), (35.0, -120.0, 145.0))
    ds = BoundingBoxDataset(bbox_df, root_dir=root_dir, metadata=metadata)
    item = ds[0]
    assert isinstance(item, tuple)
    assert len(item) == 2
    assert item[0].shape[0] == 3
    assert item[1].shape == (3,)
    assert item[1][0] == 35.0
    assert item[1][1] == -120.0
    assert item[1][2] == 145.0
