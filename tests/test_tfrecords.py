#test module for tfrecords
from deepforest import tfrecords
import pytest

@pytest.fixture("module")
def config():
    config = {}
    config["patch_size"] = 200
    config["patch_overlap"] = 0.25
    config["annotations_xml"] = "tests/data/OSBS_029.xml"
    config["rgb_dir"] = "tests/data"
    config["annotations_file"] = "tests/data/OSBS_029.csv"
    config["path_to_raster"] ="tests/data/OSBS_029.tif"
    
    #Create a clean config test data
    annotations = utilities.xml_to_annotations(xml_path=config["annotations_xml"],rgb_dir= config["rgb_dir"])
    annotations.to_csv("tests/data/OSBS_029.csv",index=False)
    annotations_file = preprocess.split_training_raster(config["path_to_raster"], config["annotations_file"], "tests/data/",config["patch_size"], config["patch_overlap"])
    return config

@pytest.fixture()
def prepare_dataset(config):
    tfrecords.create_tfrecords("tests/data/OSBS_029.csv", "tests/data/classes.csv", config["image_min_side"], config["backbone"], size=1)

#Writing
def test_create_tfrecords(config):
    tfrecords.create_tfrecords(annotations_file, class_file, image_min_side, backbone, size)

#Reading
def test_create_dataset():
    dataset = tfrecords.create_dataset("tests/data/OSBS_029.tfrecords")

#Training    
def test_train(prepare_dataset):
    training_model, prediction_model = tfrecords.train("tests/data/OSBS_029.tfrecords")
    
    
