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


def test_spatial_temporal_encoder_output_shape():
    enc = SpatialTemporalEncoder(embed_dim=32, dropout=0.0)
    meta = torch.tensor([[35.0, -120.0, 145.0], [0.0, 0.0, 1.0]])
    out = enc(meta)
    assert out.shape == (2, 32)


def test_crop_model_metadata_forward():
    cm = CropModel(config_args={"use_metadata": True, "metadata_dim": 32})
    cm.create_model(num_classes=5)
    x = torch.rand(4, 3, 224, 224)
    meta = torch.tensor([[35.0, -120.0, 145.0]] * 4)
    out = cm.forward(x, metadata=meta)
    assert out.shape == (4, 5)


def test_crop_model_metadata_none_graceful_degradation():
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


def test_metadata_image_folder(tmp_path):
    """Test MetadataImageFolder wrapping an ImageFolder."""
    for cls in ["A", "B"]:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8))
            img.save(cls_dir / f"{cls}_{i}.png")

    rows = []
    for cls in ["A", "B"]:
        for i in range(3):
            rows.append({
                "filename": f"{cls}_{i}.png",
                "lat": 35.0 + i,
                "lon": -120.0 + i,
                "date": "2024-06-15",
            })
    metadata_csv = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_csv, index=False)

    base_ds = ImageFolder(str(tmp_path))
    meta_ds = MetadataImageFolder(base_ds, str(metadata_csv))

    assert len(meta_ds) == 6
    image, label, metadata = meta_ds[0]
    assert isinstance(image, (torch.Tensor, np.ndarray, Image.Image))
    assert isinstance(label, int)
    assert metadata.shape == (3,)

    found = False
    for i in range(len(meta_ds)):
        _, _, meta = meta_ds[i]
        if meta[2].item() == 167.0:
            found = True
            break
    assert found, "day_of_year should be 167 for 2024-06-15"


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


def test_checkpoint_save_load_metadata(tmp_path):
    """Test that metadata models can be saved and loaded from checkpoint."""
    for cls in ["A", "B"]:
        cls_dir = tmp_path / "data" / cls
        cls_dir.mkdir(parents=True)
        for i in range(3):
            img = Image.fromarray(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            )
            img.save(cls_dir / f"{cls}_{i}.png")

    rows = []
    for cls in ["A", "B"]:
        for i in range(3):
            rows.append({
                "filename": f"{cls}_{i}.png",
                "lat": 40.0,
                "lon": -100.0,
                "date": "2024-01-15",
            })
    metadata_csv = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_csv, index=False)

    data_dir = str(tmp_path / "data")

    cm = CropModel(config_args={"use_metadata": True, "metadata_dim": 16})
    cm.create_trainer(
        fast_dev_run=False,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        default_root_dir=str(tmp_path / "logs"),
    )
    cm.load_from_disk(
        train_dir=data_dir, val_dir=data_dir, metadata_csv=str(metadata_csv)
    )
    cm.create_model(num_classes=len(cm.label_dict))
    cm.trainer.fit(cm)

    checkpoint_path = str(tmp_path / "test.ckpt")
    cm.trainer.save_checkpoint(checkpoint_path)

    loaded = CropModel.load_from_checkpoint(checkpoint_path)
    assert loaded.backbone is not None
    assert loaded.metadata_encoder is not None
    assert loaded.classifier is not None
    assert loaded.label_dict == cm.label_dict

    x = torch.rand(2, 3, 224, 224)
    meta = torch.tensor([[40.0, -100.0, 15.0]] * 2)
    out = loaded(x, metadata=meta)
    assert out.shape == (2, len(cm.label_dict))
