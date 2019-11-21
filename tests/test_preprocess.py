#test preprocessing
from PIL import Image
import numpy as np
import pytest
import pandas as pd
import os

from deepforest import preprocess
from deepforest import utilities


@pytest.fixture("module")
def config():
    config = {}
    config["patch_size"] = 200
    config["patch_overlap"] = 0.25
    config["annotations_xml"] = "tests/data/OSBS_029.xml"
    config["rgb_dir"] = "tests/data"
    config["annotations_file"] = "tests/data/OSBS_029.csv"
    config["path_to_raster"] ="tests/data/OSBS_029.tif"
    
    #Create a clean config test data
    annotations = utilities.xml_to_annotations(xml_path = config["annotations_xml"],rgb_dir =  config["rgb_dir"])
    annotations.to_csv("tests/data/OSBS_029.csv",index=False)
    
    return config

@pytest.fixture()
def numpy_image(config):
    raster = Image.open(config["path_to_raster"])
    return np.array(raster)
    
def test_compute_windows(config, numpy_image):
    windows = preprocess.compute_windows(numpy_image, config["patch_size"], config["patch_overlap"])
    assert len(windows) == 9

def test_select_annotations(config, numpy_image):      
    windows = preprocess.compute_windows(numpy_image, config["patch_size"], config["patch_overlap"])
    image_annotations = pd.read_csv("tests/data/OSBS_029.csv")
    selected_annotations = preprocess.select_annotations(image_annotations, windows, index=7)
    
    #Returns a 5 column matrix
    assert selected_annotations.shape[0] == 17
    
    #image name should be name of image plus the index .tif
    assert selected_annotations.image_path.unique()[0] == "OSBS_029_7.jpg"

def test_select_annotations_tile(config, numpy_image):
    config["patch_size"] = 50
    windows = preprocess.compute_windows(numpy_image, config["patch_size"], config["patch_overlap"])
    image_annotations = pd.read_csv("tests/data/OSBS_029.csv")    
    selected_annotations = preprocess.select_annotations(image_annotations, windows, index=10)
    
    #The largest box cannot be off the edge of the window
    assert selected_annotations.xmin.min() >= 0
    assert selected_annotations.ymin.min() >= 0
    assert selected_annotations.xmax.max() <= config["patch_size"]
    assert selected_annotations.ymax.max() <= config["patch_size"]
    
def test_split_training_raster(config):
    annotations_file = preprocess.split_training_raster(config["path_to_raster"], config["annotations_file"], "tests/data/",config["patch_size"], config["patch_overlap"])
    
    #Returns a 6 column pandas array
    assert annotations_file.shape[1] == 6