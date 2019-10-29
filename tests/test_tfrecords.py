#test module for tfrecords
from deepforest import tfrecords
from deepforest import utilities
from deepforest import preprocess

import pytest
import os
import glob

@pytest.fixture()
def config():
    config = {}
    config["patch_size"] = 200
    config["patch_overlap"] = 0.25
    config["annotations_xml"] = "tests/data/OSBS_029.xml"
    config["rgb_dir"] = "tests/data"
    config["annotations_file"] = "tests/data/OSBS_029.csv"
    config["path_to_raster"] ="tests/data/OSBS_029.tif"
    config["image-min-side"] = 800
    config["backbone"] = "resnet50"
    
    #Create a clean config test data
    annotations = utilities.xml_to_annotations(xml_path=config["annotations_xml"],rgb_dir= config["rgb_dir"])
    annotations.to_csv("tests/data/testtfrecords_OSBS_029.csv",index=False)
    
    annotations_file = preprocess.split_training_raster(path_to_raster=config["path_to_raster"],
                                                        annotations_file="tests/data/testtfrecords_OSBS_029.csv",
                                                        base_dir= "tests/data/",
                                                        patch_size=config["patch_size"],
                                                        patch_overlap=config["patch_overlap"])
    
    annotations_file.to_csv("tests/data/testfile_tfrecords.csv", index=False,header=False)
    return config

@pytest.fixture()
def prepare_dataset(config):    
    tfrecords.create_tfrecords(annotations_file="tests/data/testfile_tfrecords.csv", class_file="tests/data/classes.csv", image_min_side=config["image-min-side"], backbone_model=config["backbone"], size=10, savedir="tests/data/")
    assert os.path.exists("tests/data/testfile_tfrecords_0.tfrecord")

#Writing
def test_create_tfrecords(config):
    tfrecords.create_tfrecords(annotations_file="tests/data/testfile_tfrecords.csv",
                               class_file="tests/data/classes.csv",
                               image_min_side=config["image-min-side"], 
                               backbone_model=config["backbone"],
                               size=100,
                               savedir="tests/data/")
    assert os.path.exists("tests/data/testfile_tfrecords_0.tfrecord")
#Reading
def test_create_dataset(prepare_dataset):
    dataset = tfrecords.create_dataset("tests/data/testfile_tfrecords_0.tfrecord")        
        
#Training  of cropped records
def test_train(prepare_dataset, config):
    list_of_tfrecords = glob.glob("tests/data/*.tfrecord")
    print("Found {} tfrecords".format(len(list_of_tfrecords)))
    tfrecords.train(list_of_tfrecords=list_of_tfrecords, steps_per_epoch=1, backbone_name=config["backbone"])

