#test preprocessing
from deepforest import preprocess
from deepforest import utilities
from PIL import Image
import numpy as np
import pytest

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
    annotations.to_csv("tests/data/OSBS_029.csv",index=False, header=False)
    
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
    selected_annotations = preprocess.select_annotations("OSBS_029.tif", config["annotations_file"], windows, index=7)
    
    #Returns a 5 column matrix
    assert selected_annotations.shape[0] == 9 

def test_split_training_raster(config):
    annotations_file = preprocess.split_training_raster(config["path_to_raster"], config["annotations_file"], "tests/data/",config["patch_size"], config["patch_overlap"])
    
    #There should be one image name for each window crop
    assert len(annotations_file.image_path.unique()) == 9
    
