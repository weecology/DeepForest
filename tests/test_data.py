# test data locations and existance
import os

import deepforest


# Make sure package data is present
def test_get_data():
    assert os.path.exists(deepforest.get_data("testfile_deepforest.csv"))
    assert os.path.exists(deepforest.get_data("OSBS_029.png"))
    assert os.path.exists(deepforest.get_data("OSBS_029.tif"))
    assert os.path.exists(deepforest.get_data("classes.csv"))
