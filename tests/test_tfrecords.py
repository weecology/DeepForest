#test module for tfrecords
from deepforest import tfrecords
from deepforest import utilities
from deepforest import preprocess

import pytest
import os
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras_retinanet.preprocessing import csv_generator
from keras_retinanet import models

@pytest.fixture()
def config():
    config = {}
    config["patch_size"] = 200
    config["patch_overlap"] = 0.05
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
    """This test is in flux due to the fact that tensorflow and cv2 resize methods are not identical: https://jricheimer.github.io/tensorflow/2019/02/11/resize-confusion/ """
    created_records = tfrecords.create_tfrecords(annotations_file="tests/data/testfile_tfrecords.csv",
                               class_file="tests/data/classes.csv",
                               image_min_side=config["image-min-side"], 
                               backbone_model=config["backbone"],
                               size=100,
                               savedir="tests/data/")
    assert os.path.exists("tests/data/testfile_tfrecords_0.tfrecord")
    
    #the image going in to tensorflow should be equivalent to the image from the fit_generator
    backbone = models.backbone(config["backbone"])
        
    #CSV generator
    generator = csv_generator.CSVGenerator(
        csv_data_file="tests/data/testfile_tfrecords.csv",
        csv_class_file="tests/data/classes.csv",
        image_min_side=config["image-min-side"],
        preprocess_image=backbone.preprocess_image,
    )
    
    #find file in randomize generator group
    first_file = generator.groups[0][0]
    gen_filename = os.path.join(generator.base_dir, generator.image_names[first_file])
    original_image = generator.load_image(first_file)
    inputs, targets = generator.__getitem__(0)
    
    image = inputs[0,...]
    targets = targets[0][0,...]
    
    tf_inputs, tf_targets = tfrecords.create_tensors(created_records, shuffle = False)
    
    with tf.Session() as sess:
        #seek the randomized image to match
        tf_inputs, tf_targets = sess.run([tf_inputs,tf_targets])

    #assert filename is the same as generator
    #assert gen_filename == filename
    #tf_image = tf_image[0,...]
    tf_inputs = tf_inputs[0,...]
    tf_targets = tf_targets[0][0,...]
    
    #Same shape
    #assert tf_image.shape == image.shape
    assert tf_inputs.shape == image.shape
    assert tf_targets.shape == targets.shape
    
    #Same values, slightly duplicitious with above, but useful for debugging.
    #Saved array is the same as generator image
    #np.testing.assert_array_equal(image, tf_image)
    
    #Loaded array is the same as generator, this is not true currently, the opencv and the tensorflow interpolation method is slightly different, waiting for tf. 2.0
   #np.testing.assert_array_equal(tf_loaded, tf_image)
    
    ##Useful for debug to plot
    #fig = plt.figure()
    #ax1 = fig.add_subplot(1,4,1)
    #ax1.title.set_text('Fit Gen Original')    
    #plt.imshow(original_image[...,::-1])
    #ax1 = fig.add_subplot(1,4,2)
    #ax1.title.set_text('Fit Generator')    
    #plt.imshow(image)
    #ax2 = fig.add_subplot(1,4,3)
    #ax2 = fig.add_subplot(1,4,4)
    #ax2.title.set_text('Loaded Image')        
    #plt.imshow(tf_inputs)
    #plt.show()    
    
#Reading
def test_create_dataset(prepare_dataset):
    dataset = tfrecords.create_dataset("tests/data/testfile_tfrecords_0.tfrecord")