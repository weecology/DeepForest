# test dataset model
import os

import numpy as np
import pytest
from PIL import Image

from deepforest import get_data
from deepforest.datasets.prediction import (
    FromCSVFile,
    MultiImage,
    SingleImage,
    TiledRaster,
)


def test_TiledRaster():
    tile_path = get_data("test_tiled.tif")
    ds = TiledRaster(path=tile_path, patch_size=300, patch_overlap=0)
    assert len(ds) == 16

    # assert crop shape
    assert ds[1].shape == (3, 300, 300)


def test_TiledRaster_non_square(tmp_path):
    import rasterio as rio
    from rasterio.transform import from_origin

    transform = from_origin(0, 0, 1, 1)
    raster_path = str(tmp_path / "non_square.tif")
    # create a 3 band, 500 height, 800 width raster
    with rio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=500,
        width=800,
        count=3,
        dtype=np.uint8,
        crs="+proj=latlong",
        transform=transform,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(np.zeros((3, 500, 800), dtype=np.uint8))

    ds = TiledRaster(path=raster_path, patch_size=400, patch_overlap=0)

    # width 800 -> 2 patches. height 500 -> 2 patches. Total = 4 patches
    assert len(ds) == 4


def test_SingleImage_path():
    ds = SingleImage(path=get_data("OSBS_029.png"), patch_size=300, patch_overlap=0)

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
    test_data = np.random.randint(0, 256, (300, 300, 3)).astype(np.uint8)
    SingleImage(image=Image.fromarray(test_data))


def test_valid_array():
    # 8-bit, HWC
    test_data = np.random.randint(0, 256, (300, 300, 3)).astype(np.uint8)
    SingleImage(image=test_data)


def test_MultiImage():
    ds = MultiImage(
        paths=[get_data("OSBS_029.png"), get_data("OSBS_029.png")],
        patch_size=300,
        patch_overlap=0,
    )
    # 2 windows each image 2 * 2 = 4
    assert len(ds) == 2
    assert ds[0][0].shape == (3, 300, 300)


def test_FromCSVFile():
    ds = FromCSVFile(
        csv_file=get_data("example.csv"),
        root_dir=os.path.dirname(get_data("example.csv")),
    )
    assert len(ds) == 1
