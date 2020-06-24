#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `deepforest` package."""
# matplotlib.use("MacOSX")

import os
import cv2

import keras
import numpy as np
import pytest

from deepforest import deepforest
from deepforest import get_data
from deepforest import tfrecords
from deepforest import utilities

#import general model fixture
from .conftest import download_release

@pytest.fixture()
def annotations():
    annotations = utilities.xml_to_annotations(get_data("OSBS_029.xml"))
    # Point at the png version for tfrecords
    annotations.image_path = annotations.image_path.str.replace(".tif", ".png")

    annotations_file = get_data("testfile_deepforest.csv")
    annotations.to_csv(annotations_file, index=False, header=False)

    return annotations_file


@pytest.fixture()
def multi_annotations():
    annotations = utilities.xml_to_annotations(get_data("SOAP_061.xml"))
    annotations.image_path = annotations.image_path.str.replace(".tif", ".png")
    annotations_file = get_data("testfile_multi.csv")
    annotations.to_csv(annotations_file, index=False, header=False)

    return annotations_file


@pytest.fixture()
def prepare_tfdataset(annotations):
    records_created = tfrecords.create_tfrecords(annotations_file=annotations,
                                                 class_file="tests/data/classes.csv",
                                                 image_min_side=800,
                                                 backbone_model="resnet50",
                                                 size=100,
                                                 savedir="tests/data/")
    assert os.path.exists("tests/data/testfile_deepforest_0.tfrecord")
    return records_created


def test_deepforest():
    model = deepforest.deepforest(weights=None)
    assert model.weights is None


def test_use_release(download_release):
    test_model = deepforest.deepforest()
    test_model.use_release()

    # Check for release tag
    assert isinstance(test_model.__release_version__, str)

    # Assert is model instance
    assert isinstance(test_model.model, keras.models.Model)

    assert test_model.config["weights"] == test_model.weights
    assert test_model.config["weights"] is not "None"


@pytest.fixture()
def release_model(download_release):
    test_model = deepforest.deepforest()
    test_model.use_release()

    # Check for release tag
    assert isinstance(test_model.__release_version__, str)

    # Assert is model instance
    assert isinstance(test_model.model, keras.models.Model)

    return test_model


def test_predict_image(download_release):
    # Load model
    test_model = deepforest.deepforest(weights=get_data("NEON.h5"))
    assert isinstance(test_model.model, keras.models.Model)

    # Predict test image and return boxes
    boxes = test_model.predict_image(image_path=get_data("OSBS_029.tif"),
                                     show=False,
                                     return_plot=False,
                                     score_threshold=0.1)

    # Returns a 6 column numpy array, xmin, ymin, xmax, ymax, score, label
    assert boxes.shape[1] == 6

    assert boxes.score.min() > 0.1

def test_predict_image_raise_error(download_release):
    #Predict test image and return boxes
    test_model = deepforest.deepforest(weights=get_data("NEON.h5"))
    with pytest.raises(ValueError):
        boxes = test_model.predict_image()

@pytest.fixture()
def test_train(annotations):
    test_model = deepforest.deepforest()
    test_model.config["epochs"] = 1
    test_model.config["save-snapshot"] = False
    test_model.config["steps"] = 1
    test_model.train(annotations=annotations, input_type="fit_generator")

    return test_model


def test_predict_generator(release_model, annotations):
    release_model.config["save_path"] = "tests/output/"
    boxes = release_model.predict_generator(annotations=annotations, return_plot=False)
    assert boxes.shape[1] == 7

    release_model.predict_generator(annotations=annotations, return_plot=True)
    assert os.path.exists("tests/output/OSBS_029.png")


def test_evaluate_generator(release_model, annotations):
    mAP = release_model.evaluate_generator(annotations=annotations)
    assert isinstance(mAP, float)


# Test random transform
def test_random_transform(annotations):
    test_model = deepforest.deepforest()
    test_model.config["random_transform"] = True
    classes_file = utilities.create_classes(annotations)
    arg_list = utilities.format_args(annotations, classes_file, test_model.config)
    assert "--random-transform" in arg_list


def test_predict_tile(release_model):
    raster_path = get_data("OSBS_029.tif")
    image = release_model.predict_tile(raster_path,
                                       patch_size=300,
                                       patch_overlap=0.5,
                                       return_plot=True)

    # Test no non-max suppression
    boxes = release_model.predict_tile(raster_path,
                                       patch_size=100,
                                       patch_overlap=0,
                                       return_plot=False)
    assert not boxes.empty

    # Test numpy_image read
    numpy_array = cv2.imread(raster_path)
    boxes = release_model.predict_tile(numpy_image =numpy_array,
                                       patch_size=100,
                                       patch_overlap=0,
                                       return_plot=False)
    assert not boxes.empty


def test_retrain_release(annotations, release_model):
    release_model.config["epochs"] = 1
    release_model.config["save-snapshot"] = False
    release_model.config["steps"] = 1

    assert release_model.config["weights"] == release_model.weights

    # test that it gets passed to retinanet
    classes_file = utilities.create_classes(annotations)
    arg_list = utilities.format_args(annotations,
                                     classes_file,
                                     release_model.config,
                                     images_per_epoch=1)
    strs = ["--weights" == x for x in arg_list]
    index = np.where(strs)[0][0] + 1
    assert arg_list[index] == release_model.weights

    release_model.train(annotations=annotations, input_type="fit_generator")


def test_multi_train(multi_annotations):
    test_model = deepforest.deepforest()
    test_model.config["epochs"] = 1
    test_model.config["save-snapshot"] = False
    test_model.config["steps"] = 1
    test_model.train(annotations=multi_annotations, input_type="fit_generator")

    # Test labels
    labels = list(test_model.labels.values())
    labels.sort()
    target_labels = ["Dead", "Alive"]
    target_labels.sort()

    assert labels == target_labels


def test_reload_model(release_model):
    release_model.model.save("tests/output/example_saved_model.h5")
    reloaded = deepforest.deepforest(saved_model="tests/output/example_saved_model.h5")
    assert reloaded.prediction_model

    # Predict test image and return boxes
    boxes = reloaded.predict_image(image_path=get_data("OSBS_029.tif"),
                                   show=False,
                                   return_plot=False,
                                   score_threshold=0.1)

    # Returns a 6 column numpy array, xmin, ymin, xmax, ymax, score, label
    assert boxes.shape[1] == 6

    assert boxes.score.min() > 0.1


def test_reload_weights(release_model):
    release_model.model.save_weights("tests/output/example_saved_weights.h5")
    reloaded = deepforest.deepforest(weights="tests/output/example_saved_weights.h5")
    assert reloaded.prediction_model

    # Predict test image and return boxes
    boxes = reloaded.predict_image(image_path=get_data("OSBS_029.tif"),
                                   show=False,
                                   return_plot=False,
                                   score_threshold=0.1)

    # Returns a 6 column numpy array, xmin, ymin, xmax, ymax, score, label
    assert boxes.shape[1] == 6
