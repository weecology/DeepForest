# test dataset model
import os

import numpy as np
from PIL import Image
import pytest

from deepforest import get_data
from deepforest import predict
from shapely import geometry
import pandas as pd
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


def test_translate_predictions_boxes():
    predictions = pd.DataFrame(
        {
            "xmin": [1, 5],
            "ymin": [2, 6],
            "xmax": [3, 9],
            "ymax": [4, 10],
            "label": [0, 1],
            "score": [0.9, 0.8],
            "window_xmin": [10, 100],
            "window_ymin": [20, 200],
            "geometry": [
                geometry.box(1, 2, 3, 4),
                geometry.box(5, 6, 9, 10),
            ],
        }
    )

    translated = predict.translate_predictions(predictions)

    assert "window_xmin" not in translated.columns
    assert "window_ymin" not in translated.columns
    assert translated[["xmin", "ymin", "xmax", "ymax"]].values.tolist() == [
        [11, 22, 13, 24],
        [105, 206, 109, 210],
    ]
    assert translated.geometry.iloc[0].bounds == (11.0, 22.0, 13.0, 24.0)
    assert translated.geometry.iloc[1].bounds == (105.0, 206.0, 109.0, 210.0)


def test_translate_predictions_points():
    predictions = pd.DataFrame(
        {
            "x": [5.0, 10.0],
            "y": [6.0, 12.0],
            "label": [0, 0],
            "score": [0.9, 0.8],
            "patch_id": ["a", "b"],
            "window_xmin": [100, 200],
            "window_ymin": [300, 400],
            "geometry": [geometry.Point(5.0, 6.0), geometry.Point(10.0, 12.0)],
        }
    )

    translated = predict.translate_predictions(predictions)

    assert translated[["x", "y"]].values.tolist() == [[105.0, 306.0], [210.0, 412.0]]
    assert translated["patch_id"].tolist() == ["a", "b"]
    assert translated.geometry.iloc[0].x == pytest.approx(105.0)
    assert translated.geometry.iloc[0].y == pytest.approx(306.0)
    assert translated.geometry.iloc[1].x == pytest.approx(210.0)
    assert translated.geometry.iloc[1].y == pytest.approx(412.0)
