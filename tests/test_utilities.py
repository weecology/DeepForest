# test_utilities
from deepforest import utilities
import pytest
import os
import pandas as pd

@pytest.fixture()
def annotations():
    annotations = utilities.xml_to_annotations("tests/data/OSBS_029.xml",rgb_dir="tests/data")
    annotations.to_csv("tests/data/OSBS_029.csv",index=False)

def test_xml_to_annotations():
    annotations = utilities.xml_to_annotations(xml_path = "tests/data/OSBS_029.xml",rgb_dir = "tests/data")
    print(annotations.shape)
    assert annotations.shape == (60,6)
    
    #bounding box extents should be int
    assert annotations["xmin"].dtype == "int"
    
def test_create_classes(annotations):
    classes_file = utilities.create_classes(annotations_file="tests/data/OSBS_029.csv")
    assert os.path.exists(classes_file)

def test_number_of_images(annotations):
    n = utilities.number_of_images(annotations_file="tests/data/OSBS_029.csv")
    assert n == 1
    
def format_args():
    config  = utilities.read_config()
    arg_list = utilities.format_args(annotations, config)
    assert isinstance(arg_list, list)