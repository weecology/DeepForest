#!/usr/bin/env python

### Retina-net Training ###
#https://github.com/bw4sz/keras-retinanet

#Import logger.
if __name__ == "__main__":
    from comet_ml import Experiment
    
#keras-retinanet imports
import keras
import keras.preprocessing.image
from keras.utils import multi_gpu_model
import argparse
import functools
import os
import sys
import warnings
import pandas as pd
import glob
import numpy as np
import tensorflow as tf
from datetime import datetime

#supress warnings
import warnings
warnings.simplefilter("ignore")

from keras_retinanet  import layers 
from keras_retinanet  import losses
from keras_retinanet  import models
from keras_retinanet .callbacks import RedirectModel
from keras_retinanet .models.retinanet import retinanet_bbox
from keras_retinanet .utils.anchors import make_shapes_callback, anchor_targets_bbox
from keras_retinanet .utils.keras_version import check_keras_version
from keras_retinanet .utils.model import freeze as freeze_model

#Custom Generator
from DeepForest.h5_generator import H5Generator
from DeepForest.config import load_config
from DeepForest import preprocess
from DeepForest.utils.generators import create_NEON_generator, load_training_data, load_retraining_data
from DeepForest.utils import image_utils

#Custom Callbacks
from DeepForest.callbacks import recallCallback, NEONmAP, Evaluate

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


def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0, freeze_backbone=False, nms_threshold=None, input_channels=3):
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
            model = model_with_weights(backbone_retinanet(num_classes, modifier=modifier, input_channels=input_channels), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model          = model_with_weights(backbone_retinanet(num_classes, modifier=modifier, input_channels=input_channels), weights=weights, skip_mismatch=True)
        training_model = model

    # make prediction model
    print("Making prediction model with nms = %.2f" % nms_threshold )
    prediction_model = retinanet_bbox(model=model, nms_threshold=nms_threshold)

    # compile model
    training_model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer = keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )

    return model, training_model, prediction_model

def create_callbacks(model, training_model, prediction_model, train_generator, validation_generator, args, experiment, DeepForest_config):
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

    if args.evaluation and validation_generator:
        
        evaluation = Evaluate(validation_generator, 
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
            verbose=1
            ,
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
    NEON_recall_generator = create_NEON_generator(args.batch_size, DeepForest_config)
    recall = recallCallback(
        generator=NEON_recall_generator,
        save_path=args.save_path,
        score_threshold=args.score_threshold,
        experiment=experiment,
        sites=DeepForest_config["evaluation_site"]    )
    
    recall = RedirectModel(recall, prediction_model)
    callbacks.append(recall)
    
    #create the NEON generator 
    NEON_generator = create_NEON_generator(args.batch_size, DeepForest_config)
    
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
    #Split training and test data
    train, test = preprocess.split_training(data, DeepForest_config, experiment=None)

    #Write out for debug
    if args.save_path:
        train.to_csv(os.path.join(args.save_path,'training_dict.csv'), header=False)         
        
    #Training Generator
    train_generator = H5Generator(train, 
                                  batch_size = args.batch_size, 
                                  DeepForest_config = DeepForest_config, 
                                  name = "training")

    #Validation Generator, check that it exists
    if test is not None:
        validation_generator = H5Generator(test, 
                                           batch_size = args.batch_size, 
                                           DeepForest_config = DeepForest_config, 
                                           name = "training")
    else:
        validation_generator = None
        
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
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

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
        prediction_model = retinanet_bbox(model=model, nms_threshold=DeepForest_config["nms_threshold"])
    else:
        weights = args.weights
        # default to imagenet if nothing else is specified
        if weights is None and args.imagenet_weights:
            
            print("Loading imagenet weights")
            weights = backbone.download_imagenet()

        print('Creating model, this may take a secondkeras-retinanet .')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            num_classes=train_generator.num_classes(),
            weights=weights,
            multi_gpu=args.multi_gpu,
            freeze_backbone=args.freeze_backbone,
            nms_threshold=DeepForest_config["nms_threshold"],
            input_channels=DeepForest_config["input_channels"]
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
    
    #Make sure no overlapping data
    if validation_generator:
        matched=[]
        for entry in validation_generator.image_data.values():
            test = entry in train_generator.image_data.values() 
            matched.append(test)
        if sum(matched) > 0:
            raise Exception("%.2f percent of validation windows are in training data" % (100 * sum(matched)/train_generator.size()))
        else:
            print("Test passed: No overlapping data in training and validation")
    
    #start training
    history = training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.size()/DeepForest_config["batch_size"],
        epochs=args.epochs,
        verbose=2,
        shuffle=False,
        callbacks=callbacks,
        workers=DeepForest_config["workers"],
        use_multiprocessing=DeepForest_config["use_multiprocessing"],
        max_queue_size=DeepForest_config["max_queue_size"]
    )
    
    #return path snapshot of final epoch
    saved_models = glob.glob(os.path.join(args.snapshot_path,"*.h5"))
    saved_models.sort()
    
    #Return model if found
    if len(saved_models) > 0:
        return saved_models[-1]
     
if __name__ == '__main__':
    
    import argparse
    
    #Set training or training
    mode_parser     = argparse.ArgumentParser(description='Retinanet training or finetuning?')
    mode_parser.add_argument('--mode', help='train, retrain or final')
    mode_parser.add_argument('--dir', help='destination dir on HPC' )
    mode=mode_parser.parse_args()
    
    #load config
    DeepForest_config = load_config()

    #set experiment and log configs
    experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2", project_name='deeplidar', log_code=True)

    #save time for logging
    if mode.dir:
        dirname = os.path.split(mode.dir)[-1]
    else:
        dirname = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    #Load DeepForest_config and data file based on training or retraining mode. Final mode firsts run training then retrains on top.
    DeepForest_config["mode"] = mode.mode
    
    if mode.mode in ["train","final"]:
        data = load_training_data(DeepForest_config)

    if mode.mode == "retrain":
        data = load_retraining_data(DeepForest_config)   
        for site in DeepForest_config["hand_annotation_site"]:
            DeepForest_config[site]["h5"] = os.path.join(DeepForest_config[site]["h5"],"hand_annotations")
        
    #log params
    experiment.log_parameters(DeepForest_config)    
    experiment.log_parameter("Start Time", dirname)    
    experiment.log_parameter("Training Mode", mode.mode)
    
    #Run training, and pass comet experiment   
    if mode.mode in ["train","retrain","final"]:
        #pass an args object instead of using command line    
        args = [
            "--epochs", str(DeepForest_config["epochs"]),
            "--batch-size", str(DeepForest_config['batch_size']),
            "--backbone", str(DeepForest_config["backbone"]),
            "--score-threshold", str(DeepForest_config["score_threshold"])
        ]
    
        #Create log directory if saving snapshots
        if not DeepForest_config["save_snapshot_path"] == "None":
            snappath=DeepForest_config["save_snapshot_path"]+ dirname
            os.mkdir(snappath)  
    
        #if no snapshots, add arg to front, will ignore path above
        if DeepForest_config["save_snapshot_path"] == "None":
            args = ["--no-snapshots"] + args
        else:
            args = ["--snapshot-path", snappath] + args
    
        #Restart from a preview snapshot?
        if not DeepForest_config["weights"] == "None":            
            args = ["--weights", DeepForest_config["weights"] ] + args
    
        #Use imagenet weights?
        if not DeepForest_config["imagenet_weights"] and DeepForest_config["weights"] == "None":
            print("Turning off imagenet weights")
            args = ["--no-weights"] + args
      
        #Create log directory if saving eval images, add to arguments
        if not DeepForest_config["save_image_path"]=="None":
            save_image_path=DeepForest_config["save_image_path"]+ dirname
            if not os.path.exists(save_image_path):
                os.mkdir(save_image_path)
    
            args= ["--save-path", save_image_path] + args
    
        #Use imagenet weights?
        if DeepForest_config["num_GPUs"] > 1:
            args = ["--multi-gpu-force", "--multi-gpu", str(DeepForest_config["num_GPUs"])] + args 
            
        output_model = main(args, data, DeepForest_config, experiment=experiment)
    
    #Allow to build from the main training process
    if mode.mode == "final":
        
        #Make a new dir and reformat args
        dirname = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_snapshot_path=DeepForest_config["save_snapshot_path"] + dirname            
        save_image_path=DeepForest_config["save_image_path"] + dirname
        os.mkdir(save_snapshot_path)        
        
        if not os.path.exists(save_image_path):
            os.mkdir(save_image_path)        
        
        #Load retraining data
        data = load_retraining_data(DeepForest_config)     
        for site in DeepForest_config["hand_annotation_site"]:
            DeepForest_config[site]["h5"] = os.path.join(DeepForest_config[site]["h5"],"hand_annotations")
            
        #pass an args object instead of using command line    
        args = [
            "--epochs", str(40),
            "--batch-size", str(DeepForest_config['batch_size']),
            "--backbone", str(DeepForest_config["backbone"]),
            "--score-threshold", str(DeepForest_config["score_threshold"]),
            "--save-path", save_image_path,
            "--snapshot-path", save_snapshot_path,
            "--weights", output_model
        ]
    
        DeepForest_config["evaluation_images"] = 0 
        experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2", project_name='deeplidar',log_code=True)
        experiment.log_parameters(DeepForest_config)    
        experiment.log_parameter("Start Time", dirname)    
        experiment.log_parameter("mode", "final_hand_annotation")
        
        #Run training, and pass comet experiment class
        main(args, data, DeepForest_config, experiment=experiment)  