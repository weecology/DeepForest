#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `deepforest` package."""
import os
import sys
import pytest
import keras
import glob
import numpy as np

from deepforest import deepforest
from deepforest import utilities
from deepforest import tfrecords

#download latest release
@pytest.fixture()
def download_release():
    print("running fixtures")
    utilities.use_release(save_dir = "tests/data")    
    
@pytest.fixture()
def annotations():
    annotations = utilities.xml_to_annotations("tests/data/OSBS_029.xml",rgb_dir="tests/data")
    annotations.to_csv("tests/data/testfile_deepforest.csv",index=False,header=False)

@pytest.fixture(annotations)
def prepare_tfdataset():    
    tfrecords.create_tfrecords(annotations_file="tests/data/testfile_deepforest.csv", class_file="tests/data/classes.csv", image_min_side=800, backbone_model="resnet50", size=10, savedir="tests/data/")
    assert os.path.exists("tests/data/0.tfrecord")

def test_deepforest():
    model = deepforest.deepforest(weights=None)
    assert model.weights is None

def test_use_release(download_release):
    test_model = deepforest.deepforest() 
    test_model.use_release()
    #Assert is model instance    
    assert isinstance(test_model.model,keras.models.Model)

def test_predict_image(download_release):
    test_model = deepforest.deepforest(weights="tests/data/universal_model_july30.h5")    
    assert isinstance(test_model.model,keras.models.Model)
    boxes = test_model.predict_image(image_path="tests/data/OSBS_029.tif", show=False, return_plot = False)
    
    #Returns a 4 column numpy array
    assert isinstance(boxes,np.ndarray)
    assert boxes.shape[1] == 4

def test_train(annotations):
    test_model = deepforest.deepforest()
    test_model.config["epochs"] = 1
    test_model.config["save-snapshot"] = False
    test_model.train(annotations="tests/data/testfile_deepforest.csv")
    
#Training    
def test_tfrecord_train(prepare_tfdataset):
    test_model = deepforest.deepforest()
    test_model.config["epochs"] = 1
    test_model.config["save-snapshot"] = False
    list_of_tfrecords = glob.glob("tests/data/*.tfrecord")    
    
    print("Found {} tfrecords to train".format(len(list_of_tfrecords)))
    test_model.train(annotations="tests/data/testfile_deepforest.csv",input_type="tfrecord", list_of_tfrecords=list_of_tfrecords)
