# test dataset model
import os

import numpy as np
from PIL import Image
import pytest

from deepforest import get_data
from deepforest.datasets.prediction import TiledRaster, SingleImage, MultiImage, FromCSVFile, PredictionDataset


def test_TiledRaster():
    tile_path = get_data("test_tiled.tif")
    ds = TiledRaster(path=tile_path,
                             patch_size=300,
                             patch_overlap=0)
    assert len(ds) == 16

    # assert crop shape
    assert ds[1].shape == (3, 300, 300)

def test_SingleImage_path():
    ds = SingleImage(
        path=get_data("OSBS_029.png"),
        patch_size=300,
        patch_overlap=0)

    assert len(ds) == 4
    assert ds[0].shape == (3, 300, 300)

    for i in range(len(ds)):
        assert ds.get_crop(i).shape == (3, 300, 300)

def test_invalid_image_shape():
    # Not 3 channels
    test_data = (np.random.rand(300, 300, 4) * 255).astype(np.uint8)
    with pytest.raises(ValueError):
        SingleImage(image=Image.fromarray(test_data))

def test_valid_image():
    # 8-bit, HWC
    test_data = np.random.randint(0, 256, (300,300,3)).astype(np.uint8)
    SingleImage(image=Image.fromarray(test_data))

def test_valid_array():
    # 8-bit, HWC
    test_data = np.random.randint(0, 256, (300,300,3)).astype(np.uint8)
    SingleImage(image=test_data)

def test_MultiImage():
    ds = MultiImage(paths=[get_data("OSBS_029.png"), get_data("OSBS_029.png")],
                    patch_size=300,
                    patch_overlap=0)
    # 2 windows each image 2 * 2 = 4
    assert len(ds) == 2
    assert ds[0][0].shape == (3, 300, 300)

def test_FromCSVFile():
    ds = FromCSVFile(csv_file=get_data("example.csv"),
                     root_dir=os.path.dirname(get_data("example.csv")))
    assert len(ds) == 1


def test_SingleImage_metadata_batch():
    path = get_data("OSBS_029.png")
    ds = SingleImage(path=path, patch_size=300, patch_overlap=0, return_metadata=True)

    sample = ds[0]
    assert sample["image"].shape == (3, 300, 300)
    assert sample["metadata"]["image_path"] == os.path.basename(path)
    assert sample["metadata"]["window_bounds"] == ds.get_crop_bounds(0)

    batch = ds.collate_fn([ds[0], ds[1]])
    assert len(batch["images"]) == 2
    assert len(batch["metadata"]) == 2


def test_MultiImage_metadata_batch():
    path = get_data("OSBS_029.png")
    ds = MultiImage(
        paths=[path, path],
        patch_size=300,
        patch_overlap=0,
        return_metadata=True,
    )

    batch = ds.collate_fn([ds[0]])
    assert len(batch["images"]) == 4
    assert len(batch["metadata"]) == 4
    assert all(item["image_path"] == os.path.basename(path) for item in batch["metadata"])
