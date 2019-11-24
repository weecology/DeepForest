#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `deepforest` package."""
import os
import sys
import pytest
import keras
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

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
    #Point at the jpg version for tfrecords
    annotations.image_path = annotations.image_path.str.replace(".tif",".jpg")
    annotations.to_csv("tests/data/testfile_deepforest.csv",index=False,header=False)
    return "tests/data/testfile_deepforest.csv"

@pytest.fixture()
def prepare_tfdataset(annotations):    
    records_created = tfrecords.create_tfrecords(annotations_file=annotations, class_file="tests/data/classes.csv", image_min_side=800, backbone_model="resnet50", size=100, savedir="tests/data/")
    assert os.path.exists("tests/data/testfile_deepforest_0.tfrecord")
    return records_created

def test_deepforest():
    model = deepforest.deepforest(weights=None)
    assert model.weights is None

@pytest.fixture()
def test_use_release(download_release):
    test_model = deepforest.deepforest() 
    test_model.use_release()
    #Assert is model instance    
    assert isinstance(test_model.model,keras.models.Model)
    
    return test_model

def test_predict_image(download_release):
    
    #Load model
    test_model = deepforest.deepforest(weights="tests/data/universal_model_july30.h5")    
    assert isinstance(test_model.model,keras.models.Model)
    
    #Predict test image and return boxes
    boxes = test_model.predict_image(image_path="tests/data/OSBS_029.tif", show=False, return_plot = False)
    
    #Returns a 6 column numpy array, xmin, ymin, xmax, ymax, score, label
    assert boxes.shape[1] == 6

@pytest.fixture()
def test_train(annotations):
    test_model = deepforest.deepforest()
    test_model.config["epochs"] = 1
    test_model.config["save-snapshot"] = False
    test_model.config["steps"] = 1
    test_model.train(annotations=annotations, input_type="fit_generator")
    
    return test_model

#Check that weights have changed, following https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
def test_train_no_freeze(annotations, test_train):
    test_train.config["freeze_layers"] = 0
    
    #Get initial weights to compare from one layer
    before = test_train.training_model.layers[100].get_weights()
        
    #retrain with no freezing
    test_train.train(annotations=annotations, input_type="fit_generator")
    
    #Get updated weights to compare
    after = test_train.training_model.layers[100].get_weights()

    assert test_train.training_model.layers[100].trainable
    assert not np.array_equal(before, after)
    
def test_freeze_train(annotations, test_train):
    test_train.config["freeze_resnet"] = True
    
    #Get initial weights to compare from one layer, it should start in trainable mode
    print("Trainable layers before freezing: {}".format(len(test_train.training_model.trainable_weights)))
    assert test_train.training_model.layers[6].trainable    
    before = test_train.training_model.layers[6].get_weights()
        
    #retrain with no freezing
    test_train.train(annotations=annotations, input_type="fit_generator")
    
    #Get updated weights to compare
    after = test_train.training_model.layers[10].get_weights()
    print("Trainable layers after freezing: {}".format(len(test_train.training_model.trainable_weights)))

    assert not test_train.training_model.layers[10].trainable
    
    #Because the network uses batch norm layers, there is a slight drift in weights. It should be very small.
    #assert np.array_equal(before, after)

def test_predict_generator(test_use_release, annotations):
    boxes = test_use_release.predict_generator(annotations=annotations)
    assert boxes.shape[1] == 7
    
def test_evaluate(test_use_release, annotations):
    mAP = test_use_release.evaluate_generator(annotations=annotations)
    #Assert that function returns a float numpy value
    assert mAP.dtype == float

#Training    
def test_tfrecord_train(prepare_tfdataset, annotations):
    test_model = deepforest.deepforest()
    test_model.config["epochs"] = 1
    test_model.config["save-snapshot"] = False
    
    print("Found {} tfrecords to train".format(len(prepare_tfdataset)))
    test_model.train(annotations=annotations,input_type="tfrecord", list_of_tfrecords=prepare_tfdataset, images_per_epoch=1)

#Test random transform
def test_random_transform(annotations):
    test_model = deepforest.deepforest()
    test_model.config["random_transform"]
    arg_list = utilities.format_args(annotations, test_model.config)
    assert "--random-transform" in arg_list

def test_predict_tile(test_use_release):
    raster_path = "tests/data/OSBS_029.tif"
    original_raster = Image.open(raster_path)
    original_raster = np.array(original_raster)  
    
    #This should make the same prediction?
    #predicted_raster = test_use_release.predict_tile(raster_path, return_plot = False, patch_size=400,patch_overlap=0)
    #predicted_image = test_use_release.predict_image(raster_path, return_plot = False)
    
    #assert predicted_raster.shape == predicted_image.shape
    
    #Debug
    predicted_raster = test_use_release.predict_tile(raster_path, return_plot = True, patch_size=300,patch_overlap=0.5)
    predicted_image = test_use_release.predict_image(raster_path, return_plot = True)
    
    assert original_raster.shape == predicted_raster.shape
    
    fig=plt.figure()
    fig.add_subplot(2,1,1)
    plt.imshow(predicted_raster)
    fig.add_subplot(2,1,2)    
    plt.imshow(predicted_image[...,::-1])
    plt.show()
    