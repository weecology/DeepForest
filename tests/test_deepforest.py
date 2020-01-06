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
from deepforest import preprocess
from deepforest import utilities
from deepforest import tfrecords
from deepforest import get_data

#download latest release
@pytest.fixture()
def download_release():
    print("running fixtures")
    utilities.use_release()    
    
@pytest.fixture()
def annotations():
    annotations = utilities.xml_to_annotations(get_data("OSBS_029.xml"))
    #Point at the jpg version for tfrecords
    annotations.image_path = annotations.image_path.str.replace(".tif",".jpg")
    
    annotations_file = get_data("testfile_deepforest.csv")
    annotations.to_csv(annotations_file,index=False,header=False)
    
    return annotations_file

@pytest.fixture()
def prepare_tfdataset(annotations):    
    records_created = tfrecords.create_tfrecords(annotations_file=annotations, class_file="tests/data/classes.csv", image_min_side=800, backbone_model="resnet50", size=100, savedir="tests/data/")
    assert os.path.exists("tests/data/testfile_deepforest_0.tfrecord")
    return records_created

def test_deepforest():
    model = deepforest.deepforest(weights=None)
    assert model.weights is None

def test_use_release(download_release):
    test_model = deepforest.deepforest() 
    test_model.use_release()
    
    #Check for release tag
    assert isinstance(test_model.__release_version__, str)
    
    #Assert is model instance    
    assert isinstance(test_model.model,keras.models.Model)
    
    assert test_model.config["weights"] == test_model.weights 
    assert test_model.config["weights"] is not "None"
    
@pytest.fixture()
def release_model(download_release):
    test_model = deepforest.deepforest() 
    test_model.use_release()
    
    #Check for release tag
    assert isinstance(test_model.__release_version__, str)
    
    #Assert is model instance    
    assert isinstance(test_model.model,keras.models.Model)
    
    return test_model

def test_predict_image(download_release):
    #Load model
    test_model = deepforest.deepforest(weights=get_data("NEON.h5"))    
    assert isinstance(test_model.model,keras.models.Model)
    
    #Predict test image and return boxes
    boxes = test_model.predict_image(image_path=get_data("OSBS_029.tif"), show=False, return_plot = False)
    
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

#def test_train_no_freeze(annotations, test_train):
    #test_train.config["freeze_layers"] = 0
    
    ##Get initial weights to compare from one layer
    #before = test_train.training_model.layers[100].get_weights()
        
    ##retrain with no freezing
    #test_train.train(annotations=annotations, input_type="fit_generator")
    
    ##Get updated weights to compare
    #after = test_train.training_model.layers[100].get_weights()

    #assert test_train.training_model.layers[100].trainable
    #assert not np.array_equal(before, after)
    
#def test_freeze_train(annotations, test_train):
    #test_train.config["freeze_resnet"] = True
    
    ##Get initial weights to compare from one layer, it should start in trainable mode
    #print("Trainable layers before freezing: {}".format(len(test_train.training_model.trainable_weights)))
    #assert test_train.training_model.layers[6].trainable    
    #before = test_train.training_model.layers[6].get_weights()
        
    ##retrain with no freezing
    #test_train.train(annotations=annotations, input_type="fit_generator")
    
    ##Get updated weights to compare
    #after = test_train.training_model.layers[10].get_weights()
    #print("Trainable layers after freezing: {}".format(len(test_train.training_model.trainable_weights)))

    #assert not test_train.training_model.layers[10].trainable
    
    ##Because the network uses batch norm layers, there is a slight drift in weights. It should be very small.
    ##assert np.array_equal(before, after)

def test_predict_generator(release_model, annotations):
    boxes = release_model.predict_generator(annotations=annotations)
    assert boxes.shape[1] == 7
    
def test_evaluate(release_model, annotations):
    mAP = release_model.evaluate_generator(annotations=annotations)
    #Assert that function returns a float numpy value
    assert mAP.dtype == float

#Training    

#@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", reason="Skipping this test on Travis CI.")
#def test_tfrecord_train(prepare_tfdataset, annotations):
    #test_model = deepforest.deepforest()
    #test_model.config["epochs"] = 1
    #test_model.config["save-snapshot"] = False
    
    #print("Found {} tfrecords to train".format(len(prepare_tfdataset)))
    #test_model.train(annotations=annotations,input_type="tfrecord", list_of_tfrecords=prepare_tfdataset, images_per_epoch=1)

#Test random transform
def test_random_transform(annotations):
    test_model = deepforest.deepforest()
    test_model.config["random_transform"] = True
    arg_list = utilities.format_args(annotations, test_model.config)
    assert "--random-transform" in arg_list

def test_predict_tile(release_model):
    raster_path = get_data("OSBS_029.tif")
    image = release_model.predict_tile(raster_path,patch_size=300,patch_overlap=0.5,return_plot=True)
    plt.imshow(image)
    plt.show()
        
def test_retrain_release(annotations, release_model):
    release_model.config["epochs"] = 1
    release_model.config["save-snapshot"] = False
    release_model.config["steps"] = 1
    
    assert release_model.config["weights"] == release_model.weights
    
    #test that it gets passed to retinanet
    arg_list = utilities.format_args(annotations, release_model.config, images_per_epoch=1)
    strs = ["--weights" == x for x in arg_list]
    index = np.where(strs)[0][0] + 1
    arg_list[index] == release_model.weights