import tensorflow as tf
import os
import numpy as np
from math import ceil
import keras 
import cv2

from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet import models
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet import losses

import matplotlib.pyplot as plt
import psutil
import gc

def create_tf_example(image, regression_target, class_target):
    
    #Encode Image data to reduce size
    success, encoded_image = cv2.imencode(".jpg", image)
    
    #Class labels can be stored as int list
    class_target = class_target.astype(int)
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded':  tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image.tostring()])),
        'image/object/regression_target': tf.train.Feature(float_list=tf.train.FloatList(value=regression_target.flatten())),
        'image/object/class_target': tf.train.Feature(int64_list=tf.train.Int64List(value=class_target.flatten()))
    }))
    
    # Serialize to string and write to file
    return example

#TODO create classes file from annotations_file
def create_tfrecords(annotations_file, class_file, backbone_model="resnet50", image_min_side=800, size=1, savedir="./"):
    """
    Args:
        annotations_file: path to 5 column data in form image_path, xmin, ymin, xmax, ymax, label
        backbone_model: A keras retinanet backbone
        image_min_side: resized image object minimum size
        size: Number of images per tfrecord
        savedir: dir path to save tfrecords files
    Returns:
        NULL -> side effect writes tfrecords
    """
    memory_used = []
    
    #Image preprocess function
    backbone = models.backbone(backbone_model)
    
    #filebase name
    image_basename = os.path.splitext(os.path.basename(annotations_file))[0]
    
    #Create generator - because of how retinanet yields data, this should always be 1. Shape problems in the future?
    train_generator = CSVGenerator(
        annotations_file,
        class_file,
         batch_size = 1,
         image_min_side = image_min_side,
         preprocess_image = backbone.preprocess_image
         )
    
    #chunk size 
    indices = np.arange(train_generator.size())
    chunks = [
        indices[i * size:(i * size) + size]
        for i in range(ceil(len(indices) / size))
    ]
    
    for chunk in chunks:
        #Create tfrecord dataset
        writer = tf.io.TFRecordWriter(savedir + "{}_{}.tfrecord".format(image_basename, chunk[0]))
        images = []
        regression_targets = []
        class_targets = []
        
        for i in chunk:
            batch = train_generator.__getitem__(i),
            
            #split into images and targets
            inputs, targets =  batch[0]
           
            #grab image, asssume batch size of 1, squeeze
            images.append(inputs[0,...])
            
            #Grab anchor targets
            regression_batch, labels_batch = targets
        
            #grab regression anchors
            #regression_batch: batch that contains bounding-box regression targets for an image & anchor states (np.array of shape (batch_size, N, 4 + 1),
            #where N is the number of anchors for an image, the first 4 columns define regression targets for (x1, y1, x2, y2) and the
            #last column defines anchor states (-1 for ignore, 0 for bg, 1 for fg).       
            regression_anchors = regression_batch[0,...] 
            regression_targets.append(regression_anchors)
            
            #grab class labels - squeeze out batch size
            #From retinanet: labels_batch: batch that contains labels & anchor states (np.array of shape (batch_size, N, num_classes + 1),
           #where N is the number of anchors for an image and the last column defines the anchor state (-1 for ignore, 0 for bg, 1 for fg).        
            labels = labels_batch[0,...]
            print("Label shape is: {}".format(labels.shape))
            class_targets.append(labels)
            
            i +=1
                
        for image, regression_target, class_target in zip(images,regression_targets, class_targets):
            tf_example = create_tf_example(image, regression_target, class_target)
            writer.write(tf_example.SerializeToString())        
        
        memory_used.append(psutil.virtual_memory().used / 2 ** 30)

    plt.plot(memory_used)
    plt.title('Evolution of memory')
    plt.xlabel('iteration')
    plt.ylabel('memory used (GB)')   
    plt.savefig(os.path.join(savedir,"memory.png"))
        

#Reading
def _parse_fn(example):
    
    #Define features
    features = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),       
        "image/object/regression_target": tf.FixedLenFeature([120087, 5], tf.float32),
        "image/object/class_target": tf.FixedLenFeature([120087, 2], tf.int64)
                        }
    
    # Load one example and parse
    example = tf.io.parse_single_example(example, features)
    image = tf.image.decode_image(example['image/encoded'],3)
    regression_target = tf.cast(example['image/object/regression_target'], tf.float32)
    class_target = tf.cast(example['image/object/class_target'], tf.float32)
    
    #TODO allow this vary from config? Or read during sess?    
    image = tf.reshape(image, [800, 800, 3],name="cast_image")            
    regression_target = tf.reshape(regression_target, [120087, 5], name="cast_regression")
    class_target = tf.reshape(class_target, [120087, 2], name="cast_class_label")
    
    return image, regression_target, class_target

def create_dataset(filepath, batch_size=1):
    """
    Args:
        filepath: list of tfrecord files
        batch_size: number of images per batch
    Returns:
        dataset: a tensorflow dataset object for model training or prediction
    """
    
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)
    
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_fn)
    
    ## This dataset will go on forever
    dataset = dataset.repeat()
    
    ## Set the number of datapoints you want to load and shuffle 
    #dataset = dataset.shuffle(1000)
    
    ## Set the batchsize
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    
    ## Create an iterator
    iterator = dataset.make_one_shot_iterator()
    
    return iterator

def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.

    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model

def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0,
                  freeze_backbone=False, lr=1e-5, config=None, targets=None):
    """ Creates three models (model, training_model, prediction_model).

    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.
        config             : Config parameters, None indicates the default configuration.
        targets            : tf.dataset object tensor


    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """

    modifier = freeze_model if freeze_backbone else None

    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors   = None
    if config and 'anchor_parameters' in config:
        anchor_params = parse_anchor_parameters(config)
        num_anchors   = anchor_params.num_anchors()

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model          = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = model

    # make prediction model
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)

    # compile model
    training_model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
            },
        optimizer=keras.optimizers.adam(lr=lr, clipnorm=0.001),
        target_tensors=targets
    )

    return model, training_model, prediction_model

def train(list_of_tfrecords, backbone_name, steps_per_epoch=None):
    """
    Train a retinanet model using tfrecords input
    
    Args:
        list_of_tfrecords: a path or wildcard glob of tfrecords
        backbone_model: A keras retinanet backbone name
        steps_per_epoch: How often should validation data be evaluated?
    
    Returns:
        training_model: The retinanet training model
        prediction_model: The retinanet prediction model with nms for bbox
    """
    
    #Create tensorflow iterator
    iterator = create_dataset(list_of_tfrecords)
    next_element = iterator.get_next()
    
    #Split into inputs and targets 
    inputs = next_element[0]
    targets = [next_element[1], next_element[2]]
    
    backbone = models.backbone(backbone_name)
    model, training_model, prediction_model = create_models(backbone_retinanet=backbone.retinanet, weights=None, targets=targets, num_classes=1)
    
    #Train model
    if steps_per_epoch is None:
        raise ValueError("Unknown steps for a tfrecord")
    
    training_model.fit(inputs, steps_per_epoch=steps_per_epoch)
    
    return training_model, prediction_model    