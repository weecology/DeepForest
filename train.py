#!/usr/bin/env python

### Retina-net Training ###
#https://github.com/bw4sz/keras-retinanet

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#Import logger.
from comet_ml import Experiment

#keras-retinanet imports

import argparse
import functools
import os
import sys
import warnings

import keras
import keras.preprocessing.image
from keras.utils import multi_gpu_model
import tensorflow as tf

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'keras-retinanet ', 'keras-retinanet '))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_retinanet  import layers  # noqa: F401
from keras_retinanet  import losses
from keras_retinanet  import models
from keras_retinanet .callbacks import RedirectModel
from keras_retinanet .callbacks.eval import Evaluate
from keras_retinanet .models.retinanet import retinanet_bbox
from keras_retinanet .utils.anchors import make_shapes_callback, anchor_targets_bbox
from keras_retinanet .utils.keras_version import check_keras_version
from keras_retinanet .utils.model import freeze as freeze_model
from keras_retinanet .utils.transform import random_transform_generator

#Custom Generator
from DeepForest.h5_generator import H5Generator
from DeepForest.onthefly_generator import OnTheFlyGenerator

#Custom Callbacks
from DeepForest.callbacks import recallCallback, NEONmAP

def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


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


def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0, freeze_backbone=False,nms_threshold=None):
    """ Creates three models (model, training_model, prediction_model).

    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.

    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """
    modifier = freeze_model if freeze_backbone else None

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model          = model_with_weights(backbone_retinanet(num_classes, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = model

    # make prediction model
    print("Making prediction model with nms = %.2f" % nms_threshold )
    prediction_model = retinanet_bbox(model=model,nms_threshold=nms_threshold)

    # compile model
    training_model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer = keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )

    return model, training_model, prediction_model

def create_NEON_generator(args,site,DeepForest_config):
    """ Create generators for training and validation.
    """
    annotations, windows = preprocess.NEON_annotations(site, DeepForest_config)
    
    #Training Generator
    generator =  OnTheFlyGenerator(
        annotations,
        windows,
        batch_size=args.batch_size,
        DeepForest_config=DeepForest_config,
        group_method="none",
        base_dir=os.path.join("data",site),
        name="NEON_validation"
    )
    
    #set lidar and rgb path
    generator.lidar_path = "data/" + site + "/"
    
    return(generator)

def create_callbacks(model, training_model, prediction_model, train_generator, validation_generator, args, experiment,DeepForest_config):
    """ Creates the callbacks to use during training.

    Args
        model: The base model.
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None

    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = args.tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = args.batch_size,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        
        evaluation = Evaluate(validation_generator, 
                              tensorboard=tensorboard_callback,
                              experiment=experiment,
                              save_path=args.save_path,
                              score_threshold=args.score_threshold,
                              DeepForest_config=DeepForest_config)
        
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{backbone}_{{epoch:02d}}.h5'.format(backbone=args.backbone)
            ),
            verbose=1,
            save_best_only=True,
             monitor="mAP",
             mode='max'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.1,
        patience = 2,
        verbose  = 1,
        mode     = 'auto',
        epsilon  = 0.0001,
        cooldown = 0,
        min_lr   = 0
    ))
    
    #Neon Callbacks
    site=DeepForest_config["evaluation_site"]
    
    NEON_recall_generator = create_NEON_generator(args, site, DeepForest_config)
    
    recall=recallCallback(site=site,
                          generator=NEON_recall_generator,
                          save_path=args.save_path,
                          DeepForest_config=DeepForest_config,
                          score_threshold=args.score_threshold,
                          experiment=experiment)
    
    recall = RedirectModel(recall, prediction_model)
    
    callbacks.append(recall)
    
    #create the NEON generator 
    NEON_generator = create_NEON_generator(args, site, DeepForest_config)
    
    neon_evaluation = NEONmAP(NEON_generator, 
                              experiment=experiment,
                              save_path=args.save_path,
                              score_threshold=args.score_threshold,
                              DeepForest_config=DeepForest_config)

    neon_evaluation = RedirectModel(neon_evaluation, prediction_model)
    callbacks.append(neon_evaluation)  
        
    return callbacks


def create_generators(args, data, DeepForest_config):
    """ Create generators for training and validation.
    """
    # create random transform generator for augmenting training data
    if args.random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)

    #Split training and test data
    train, test = preprocess.split_training(data, DeepForest_config, experiment=None)

    #Write out for debug
    if args.save_path:
        train.to_csv(os.path.join(args.save_path,'training_dict.csv'), header=False)
           
    #Training Generator
    train_generator = H5Generator(train, batch_size = args.batch_size, DeepForest_config = DeepForest_config, group_method="none", name = "training")

    #Validation Generator        
    validation_generator = H5Generator(test, batch_size = args.batch_size, DeepForest_config = DeepForest_config, group_method = "none", name = "validation")

    return train_generator, validation_generator


def check_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    Args
        parsed_args: parser.parse_args()

    Returns
        parsed_args
    """

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError("Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    if 'resnet' not in parsed_args.backbone:
        warnings.warn('Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))

    return parsed_args


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')


    def csv_list(string):
        return string.split(',')
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)
    
    parser.add_argument('--backbone',        help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size',      help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',             help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu',       help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force', help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--epochs',          help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--snapshot-path',   help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',    help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',   help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int, default=400)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    
    #Comet ml image viewer
    parser.add_argument('--save-path',       help='Path for saving eval images with detections (doesn\'t work for COCO).')
    parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.3).', default=0.05, type=float)

    return check_args(parser.parse_args(args))


def main(args=None, data=None, DeepForest_config=None, experiment=None):
    # parse arguments
    print("parsing arguments")
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
         
    # create object that stores backbone information
    backbone = models.backbone(args.backbone)

    # make sure keras is the minimum required version
    print("Get keras version")
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        
    print("Get session")
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the generators
    print("Creating generators")
    train_generator, validation_generator = create_generators(args, data, DeepForest_config=DeepForest_config)
    
    #Log number of trees trained on
    if experiment:
        experiment.log_parameter("Number of Training Trees", train_generator.total_trees)    
       
    # create the model
    if args.snapshot is not None:
        print('Loading model, this may take a secondkeras-retinanet.\n')
        model            = models.load_model(args.snapshot, backbone_name=args.backbone)
        training_model   = model
        prediction_model = retinanet_bbox(model=model,nms_threshold=DeepForest_config["nms_threshold"])
    else:
        weights = args.weights
        # default to imagenet if nothing else is specified
        if weights is None and args.imagenet_weights:
            weights = backbone.download_imagenet()

        print('Creating model, this may take a secondkeras-retinanet .')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            num_classes=train_generator.num_classes(),
            weights=weights,
            multi_gpu=args.multi_gpu,
            freeze_backbone=args.freeze_backbone,
            nms_threshold=DeepForest_config["nms_threshold"]
        )

    # print model summary
    #print(model.summary())

    # this lets the generator compute backbone layer shapes using the actual backbone model
    if 'vgg' in args.backbone or 'densenet' in args.backbone:
        compute_anchor_targets = functools.partial(anchor_targets_bbox, shapes_callback=make_shapes_callback(model))
        train_generator.compute_anchor_targets = compute_anchor_targets
        if validation_generator is not None:
            validation_generator.compute_anchor_targets = compute_anchor_targets

    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        train_generator,        
        validation_generator,
        args,
        experiment,
        DeepForest_config
    )
    
    matched=[]
    for entry in validation_generator.image_data.values():
        test=entry in train_generator.image_data.values() 
        matched.append(test)
    if sum(matched) > 0:
        raise Exception("%.2f percent of validation windows are in training data" % (100 * sum(matched)/train_generator.size()))
    else:
        print("Test passed: No overlapping data in training and validation")
    
    #start training
    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.size()/DeepForest_config["batch_size"],
        epochs=args.epochs,
        verbose=1,
        shuffle=False,
        callbacks=callbacks,
        workers=DeepForest_config["workers"],
        use_multiprocessing=DeepForest_config["use_multiprocessing"],
        max_queue_size=DeepForest_config["max_queue_size"]
    )
     
    
if __name__ == '__main__':
    
    import argparse
    
    #Set training or training
    mode_parser     = argparse.ArgumentParser(description='Retinanet training or finetuning?')
    mode_parser.add_argument('--mode', help='train or retrain?' )
    mode_parser.add_argument('--dir', help='destination dir on HPC' )
    
    mode=mode_parser.parse_args()
    
    import os
    import pandas as pd
    import glob
    import numpy as np
    from datetime import datetime
    from DeepForest.config import load_config
    from DeepForest import preprocess

    #set experiment and log configs
    experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2", project_name='deeplidar', log_code=False)

    #save time for logging
    if mode.dir:
        dirname = os.path.split(mode.dir)[-1]
    else:
        dirname = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    #Load DeepForest_config and data file based on training or retraining mode
    if mode.mode == "train":
        DeepForest_config = load_config("train")
        data = preprocess.load_csvs(DeepForest_config["h5_dir"])
        
    if mode.mode == "retrain":
        #TODO needs annotations to find lidar path
        DeepForest_config = load_config("retrain")        
        data = preprocess.load_xml(DeepForest_config["hand_annotations"], DeepForest_config["rgb_res"])

    #Log site
    site = DeepForest_config["evaluation_site"]

    #pass an args object instead of using command line    
    args = [
        "--epochs",str(DeepForest_config["epochs"]),
        "--batch-size",str(DeepForest_config['batch_size']),
        "--backbone",str(DeepForest_config["backbone"]),
        "--score-threshold",str(DeepForest_config["score_threshold"])
    ]

    #Create log directory if saving snapshots
    if not DeepForest_config["save_snapshot_path"] == "None":
        snappath=DeepForest_config["save_snapshot_path"]+ dirname
        os.mkdir(snappath)  

    #if no snapshots, add arg to front, will ignore path above
    if DeepForest_config["save_snapshot_path"] == "None":
        args = ["--no-snapshots"] + args
    else:
        args = [snappath] + args
        args = ["--snapshot-path"] + args

    #Restart from a preview snapshot?
    if not DeepForest_config["snapshot"] == "None":
        args = [DeepForest_config["snapshot"]] + args
        args = ["--snapshot"] + args

    #Create log directory if saving eval images, add to arguments
    if not DeepForest_config["save_image_path"]=="None":
        save_image_path=DeepForest_config["save_image_path"]+ dirname
        if not os.path.exists(save_image_path):
            os.mkdir(save_image_path)

        args= [save_image_path] + args
        args=["--save-path"] + args        

    #log params
    experiment.log_parameters(DeepForest_config)    
    experiment.log_parameter("Start Time", dirname)    
    experiment.log_parameter("Training Mode", mode.mode)
    experiment.log_parameter("Site", site)
    
    #Run training, and pass comet experiment   
    main(args, data, DeepForest_config, experiment=None)
