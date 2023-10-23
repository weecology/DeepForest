# test data locations and existance
import os
import deepforest
from deepforest.utilities import read_config

# Make sure package data is present
def test_get_data():
    assert os.path.exists(deepforest.get_data("testfile_deepforest.csv"))
    assert os.path.exists(deepforest.get_data("testfile_multi.csv"))    
    assert os.path.exists(deepforest.get_data("example.csv")) 
    assert os.path.exists(deepforest.get_data("2019_YELL_2_541000_4977000_image_crop.png"))
    assert os.path.exists(deepforest.get_data("OSBS_029.png"))
    assert os.path.exists(deepforest.get_data("OSBS_029.tif"))
    assert os.path.exists(deepforest.get_data("SOAP_061.png"))    
    assert os.path.exists(deepforest.get_data("classes.csv"))

# Assert that the included config file matches the front of the repo.
def test_matching_config(ROOT):
    config = read_config("{}/deepforest_config.yml".format(os.path.dirname(ROOT)))
    config_from_data_dir = read_config("{}/data/deepforest_config.yml".format(ROOT))
    assert config == config_from_data_dir