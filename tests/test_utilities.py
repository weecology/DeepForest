# test_utilities
from deepforest import utilities
import pytest
import os
import pandas as pd
import numpy as np

@pytest.fixture()
def annotations():
    annotations = utilities.xml_to_annotations("tests/data/OSBS_029.xml",rgb_dir="tests/data")
    annotations_file = "tests/data/OSBS_029.csv"
    annotations.to_csv("tests/data/OSBS_029.csv",index=False)
    return annotations_file

@pytest.fixture()
def config():
    config  = utilities.read_config()
    return config
    
def test_xml_to_annotations():
    annotations = utilities.xml_to_annotations(xml_path = "tests/data/OSBS_029.xml",rgb_dir = "tests/data")
    print(annotations.shape)
    assert annotations.shape == (60,6)
    
    #bounding box extents should be int
    assert annotations["xmin"].dtype == "int"
    
def test_create_classes(annotations):
    classes_file = utilities.create_classes(annotations_file=annotations)
    assert os.path.exists(classes_file)

def test_number_of_images(annotations):
    n = utilities.number_of_images(annotations_file=annotations)
    assert n == 2
    
def test_format_args(annotations, config):
    arg_list = utilities.format_args(annotations, config)
    assert isinstance(arg_list, list)

def test_format_args_steps(annotations, config):
    arg_list = utilities.format_args(annotations, config, steps_per_epoch=100)
    assert isinstance(arg_list, list)
    
    #A bit ugly, but since its a list, what is the argument after --steps to assert
    steps_position = np.where(["--steps" in x for x in arg_list])[0][0] + 1
    assert arg_list[steps_position] == '100'