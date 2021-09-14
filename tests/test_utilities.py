# test_utilities
import numpy as np
import os
import pytest
import pandas as pd
import rasterio as rio

from deepforest import get_data
from deepforest import utilities
from deepforest import main

#import general model fixture
from .conftest import download_release

@pytest.fixture()
def config():
    config = utilities.read_config("deepforest_config.yml")
    return config


def test_xml_to_annotations():
    annotations = utilities.xml_to_annotations(
        xml_path=get_data("OSBS_029.xml"))
    print(annotations.shape)
    assert annotations.shape == (61, 6)

    # bounding box extents should be int
    assert annotations["xmin"].dtype == np.int64

def test_use_release(download_release):
    # Download latest model from github release
    release_tag, state_dict = utilities.use_release(check_release=False)

def test_use_bird_release(download_release):
    # Download latest model from github release
    release_tag, state_dict = utilities.use_bird_release()
    assert os.path.exists(get_data("bird.pt"))
    
def test_float_warning(config):
    """Users should get a rounding warning when adding annotations with floats"""
    float_annotations = "tests/data/float_annotations.txt"
    annotations = utilities.xml_to_annotations(float_annotations)
    assert annotations.xmin.dtype is np.dtype('int64')

def test_project_boxes():
    csv_file = get_data("OSBS_029.csv")
    df = pd.read_csv(csv_file)
    gdf = utilities.project_boxes(df, root_dir=os.path.dirname(csv_file))
    
    assert df.shape[0] == gdf.shape[0]
    

def test_annotations_to_shapefile(download_release):
    img = get_data("OSBS_029.tif")
    r = rio.open(img)
    transform = r.transform 
    crs = r.crs
    m = main.deepforest()
    m.use_release(check_release=False)
    df = m.predict_image(path=img)
    gdf = utilities.annotations_to_shapefile(df, transform=transform, crs=crs)
    assert df.shape[0] == gdf.shape[0]