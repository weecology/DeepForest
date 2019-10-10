#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `deepforest` package."""
import os
import sys
import pytest
import keras
from deepforest import deepforest
from deepforest import utilities
import numpy as np

#download latest release
@pytest.fixture()
def download_release(scope="session"):
    print("running fixtures")
    utilities.use_release()    
    
@pytest.fixture()
def test_model(download_release):
    test_model = deepforest.deepforest(weights="tests/data/universal_model_july30.h5")
    return test_model

@pytest.fixture()
def annotations():
    annotations = utilities.xml_to_annotations("tests/data/OSBS_029.xml",rgb_dir="tests/data")
    annotations.to_csv("tests/data/OSBS_029.csv",index=False, header=False)

def test_deepforest():
    model = deepforest.deepforest(weights=None)
    assert model.weights is None

def test_use_release(download_release):
    test_model = deepforest.deepforest() 
    test_model.use_release()
    #Assert is model instance    
    assert isinstance(test_model.model,keras.models.Model)

def test_predict_image(test_model):
    assert isinstance(test_model.model,keras.models.Model)
    image = test_model.predict_image(image_path = "tests/data/OSBS_029.tif",show=False)
    
    #Returns a 4 column numpy array
    assert isinstance(image,np.ndarray)
    assert image.shape[1] == 4

def test_train(annotations):
    test_model = deepforest.deepforest()
    test_model.config["epochs"] = 1
    test_model.config["save-snapshot"] = False
    test_model.train(annotations="tests/data/OSBS_029.csv")