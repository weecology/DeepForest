# test preprocessing
import glob
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from deepforest import get_data
from deepforest import preprocess
from deepforest import utilities
from deepforest import visualize

import rasterio

@pytest.fixture()
def config():
    config = utilities.read_config("deepforest_config.yml")
    config["patch_size"] = 200
    config["patch_overlap"] = 0.25
    config["annotations_xml"] = get_data("OSBS_029.xml")
    config["rgb_dir"] = "data"
    config["annotations_file"] = "tests/data/OSBS_029.csv"
    config["path_to_raster"] = get_data("OSBS_029.tif")

    # Create a clean config test data
    annotations = utilities.xml_to_annotations(
        xml_path=config["annotations_xml"])
    annotations.to_csv("tests/data/OSBS_029.csv", index=False)

    return config


@pytest.fixture()
def image(config):
    raster = Image.open(config["path_to_raster"])
    return np.array(raster)


def test_compute_windows(config, image):
    windows = preprocess.compute_windows(image, config["patch_size"],
                                         config["patch_overlap"])
    assert len(windows) == 9


def test_select_annotations(config, image):
    windows = preprocess.compute_windows(image, config["patch_size"],
                                         config["patch_overlap"])
    image_annotations = pd.read_csv("tests/data/OSBS_029.csv")
    selected_annotations = preprocess.select_annotations(image_annotations,
                                                         windows,
                                                         index=7)

    # Returns a 5 column matrix
    assert selected_annotations.shape[0] == 17

    # image name should be name of image plus the index .tif
    assert selected_annotations.image_path.unique()[0] == "OSBS_029_7.png"


def test_select_annotations_tile(config, image):
    config["patch_size"] = 50
    windows = preprocess.compute_windows(image, config["patch_size"],
                                         config["patch_overlap"])
    image_annotations = pd.read_csv("tests/data/OSBS_029.csv")
    selected_annotations = preprocess.select_annotations(image_annotations,
                                                         windows,
                                                         index=10)

    # The largest box cannot be off the edge of the window
    assert selected_annotations.xmin.min() >= 0
    assert selected_annotations.ymin.min() >= 0
    assert selected_annotations.xmax.max() <= config["patch_size"]
    assert selected_annotations.ymax.max() <= config["patch_size"]

def test_split_raster(config, tmpdir):
    """Split raster into crops with overlaps to maintain all annotations"""
    raster = get_data("2019_YELL_2_528000_4978000_image_crop2.png")
    annotations = utilities.xml_to_annotations(get_data("2019_YELL_2_528000_4978000_image_crop2.xml"))
    annotations.to_csv("{}/example.csv".format(tmpdir), index=False)
    #annotations.label = 0
    #visualize.plot_prediction_dataframe(df=annotations, root_dir=os.path.dirname(get_data(".")), show=True)
    
    annotations_file = preprocess.split_raster(path_to_raster=raster,
                                               annotations_file="{}/example.csv".format(tmpdir),
                                               base_dir=tmpdir,
                                               patch_size=500,
                                               patch_overlap=0)

    # Returns a 6 column pandas array
    assert annotations_file.shape[1] == 6
    
    #Assert that all boxes that exists are the same as original size
    #annotations_file["label"] = 0 
    #visualize.plot_prediction_dataframe(df=annotations_file, root_dir=tmpdir, show=True)

def test_split_raster_from_image(config):
    r = rasterio.open(config["path_to_raster"]).read()
    r = np.rollaxis(r,0,3)
    annotations_file = preprocess.split_raster(numpy_image=r,
                                               annotations_file=config["annotations_file"],
                                               base_dir="tests/data/",
                                               patch_size=config["patch_size"],
                                               patch_overlap=config["patch_overlap"],
                                               image_name="OSBS_029.tif")

    # Returns a 6 column pandas array
    assert annotations_file.shape[1] == 6

def test_split_raster_empty(config):
    # Clean output folder
    for f in glob.glob("tests/output/empty/*"):
        os.remove(f)

    # Blank annotations file
    blank_annotations = pd.DataFrame({
        "image_path": "OSBS_029.tif",
        "xmin": [""],
        "ymin": [""],
        "xmax": [""],
        "ymax": [""],
        "label": [""]
    })
    blank_annotations.to_csv("tests/data/blank_annotations.csv", index=False)

    # Ignore blanks
    with pytest.raises(ValueError):
        annotations_file = preprocess.split_raster(
            path_to_raster=config["path_to_raster"],
            annotations_file="tests/data/blank_annotations.csv",
            base_dir="tests/output/empty/",
            patch_size=config["patch_size"],
            patch_overlap=config["patch_overlap"],
            allow_empty=False)
        assert annotations_file.shape[0] == 0
    assert not os.path.exists("tests/output/empty/OSBS_029_1.png")

    # Include blanks
    annotations_file = preprocess.split_raster(
        path_to_raster=config["path_to_raster"],
        annotations_file="tests/data/blank_annotations.csv",
        base_dir="tests/output/empty/",
        patch_size=config["patch_size"],
        patch_overlap=config["patch_overlap"],
        allow_empty=True)
    assert annotations_file.shape[0] > 0
    assert os.path.exists("tests/output/empty/OSBS_029_1.png")


def test_split_size_error(config):
    with pytest.raises(ValueError):
        annotations_file = preprocess.split_raster(path_to_raster=config["path_to_raster"],
                                                   annotations_file=config["annotations_file"],
                                                   base_dir="tests/data/",
                                                   patch_size=2000,
                                                   patch_overlap=config["patch_overlap"])
    