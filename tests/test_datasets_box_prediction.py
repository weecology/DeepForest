# test dataset model
from deepforest import get_data
import pytest
import numpy as np
import rasterio as rio
from deepforest.datasets.box.prediction import TiledRaster

@pytest.mark.parametrize("preload_images", [True, False])
def test_tile_dataset(preload_images):
    tile_path = get_data("2019_YELL_2_528000_4978000_image_crop2.png")
    tile = rio.open(tile_path).read()
    tile = np.moveaxis(tile, 0, 2)
    ds = TiledRaster(tile=tile,
                             preload_images=preload_images,
                             patch_size=100,
                             patch_overlap=0)
    assert len(ds) > 0

    # assert crop shape
    assert ds[1].shape == (3, 100, 100)
