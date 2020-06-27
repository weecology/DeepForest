# test_utilities
import os

import numpy as np
import pytest

from deepforest import get_data
from deepforest import utilities

#import general model fixture
from .conftest import download_release

@pytest.fixture()
def annotations():
    annotations = utilities.xml_to_annotations(get_data("OSBS_029.xml"))
    annotations_file = "tests/data/OSBS_029.csv"
    annotations.to_csv(annotations_file, index=False, header=False)
    return annotations_file


@pytest.fixture()
def config():
    config = utilities.read_config(get_data("deepforest_config.yml"))
    return config


def test_xml_to_annotations():
    annotations = utilities.xml_to_annotations(xml_path=get_data("OSBS_029.xml"))
    print(annotations.shape)
    assert annotations.shape == (61, 6)

    # bounding box extents should be int
    assert annotations["xmin"].dtype == np.int64


def test_create_classes(annotations):
    classes_file = utilities.create_classes(annotations_file=annotations)
    assert os.path.exists(classes_file)


def test_number_of_images(annotations):
    n = utilities.number_of_images(annotations_file=annotations)
    assert n == 1


def test_format_args(annotations, config):
    classes_file = utilities.create_classes(annotations)
    arg_list = utilities.format_args(annotations, classes_file, config)
    assert isinstance(arg_list, list)


def test_format_args_steps(annotations, config):
    classes_file = utilities.create_classes(annotations)
    arg_list = utilities.format_args(annotations,
                                     classes_file,
                                     config,
                                     images_per_epoch=2)
    assert isinstance(arg_list, list)

    # A bit ugly, but since its a list, what is the argument after --steps to assert
    steps_position = np.where(["--steps" in x for x in arg_list])[0][0] + 1
    assert arg_list[steps_position] == '2'


def test_use_release():
    # Download latest model from github release
    release_tag, weights = utilities.use_release()
    assert os.path.exists(get_data("NEON.h5"))


def test_float_warning(config):
    """Users should get a rounding warning when adding annotations with floats"""
    float_annotations = "tests/data/float_annotations.txt"
    annotations = utilities.xml_to_annotations(float_annotations)
    assert annotations.xmin.dtype is np.dtype('int64')
