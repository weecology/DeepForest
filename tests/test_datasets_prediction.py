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


def test_single_image_float32_0_255_consistent_normalization():
    """Float32 [0, 255] crops must be normalized uniformly from full-image decision.

    A dark crop (all pixels <= 1.0) would be misclassified as [0, 1] by the
    per-crop heuristic; with the fix, we use the full-image max to decide once.
    """
    # Image: left half dark (0.5), right half bright (128). Full max > 1.
    h, w = 200, 400
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:, : w // 2, :] = 0.5  # Dark region
    img[:, w // 2 :, :] = 128.0  # Bright region
    # CHW for preprocess.compute_windows
    img = np.moveaxis(img, -1, 0)

    ds = SingleImage(image=img, patch_size=100, patch_overlap=0)
    assert len(ds) >= 2

    # First crop(s) from dark region: max=0.5. Should be divided by 255 -> max ~0.002
    dark_crop = ds[0]
    assert dark_crop.shape == (3, 100, 100)
    assert dark_crop.max().item() < 0.01, "Dark crop should be /255, not left as [0,1]"

    # Crop from bright region: max=128. Should be divided by 255 -> max ~0.5
    bright_idx = len(ds) - 1
    bright_crop = ds[bright_idx]
    assert bright_crop.max().item() > 0.4

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
