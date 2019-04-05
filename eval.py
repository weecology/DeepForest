#!/usr/bin/env python

#Log training
from comet_ml import Experiment

import argparse
import os
import sys
from datetime import datetime

import warnings
warnings.simplefilter("ignore")

import keras
import tensorflow as tf
from keras_retinanet import models
from keras_retinanet.utils.keras_version import check_keras_version

#Custom Generators and callbacks
from DeepForest.onthefly_generator import OnTheFlyGenerator
from DeepForest.h5_generator import H5Generator
from DeepForest.evaluation import neonRecall
from DeepForest.evalmAP import evaluate
from DeepForest.utils.generators import create_NEON_generator, load_training_data, load_retraining_data
from DeepForest import preprocess

def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_generator(args, data, DeepForest_config):
    """ Create generators for training and validation.
    """
    #Split training and test data - hardcoded paths set below.
    _ , test = preprocess.split_training(data, DeepForest_config, experiment=None)

    #Training Generator
    generator =  H5Generator(
        test,
        batch_size=args.batch_size,
        DeepForest_config=DeepForest_config,
        group_method="none",
        name = "training"
    )
        
    return(generator)


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    parser.add_argument('--batch-size',      help='Size of the batches.', default=1, type=int)
    parser.add_argument('--model',             help='Path to RetinaNet model.')
    parser.add_argument('--convert-model',   help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
    parser.add_argument('--backbone',        help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',             help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
    parser.add_argument('--iou-threshold',   help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--max-detections',  help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--suppression-threshold',  help='Permitted overlap among predictions', default=0.2, type=float)
    parser.add_argument('--save-path',       help='Path for saving images with detections (doesn\'t work for COCO).')
    parser.add_argument('--image-min-side',  help='Rescale the image so the smallest side is min_side.', type=int, default=400)
    parser.add_argument('--image-max-side',  help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)

    return parser.parse_args(args)


def main(data, DeepForest_config, experiment, args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    #Add seperate dir
    #save time for logging
    dirname=datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment.log_parameter("Start Time", dirname)
    
    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path + dirname):
        os.makedirs(args.save_path + dirname)

    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone, convert=args.convert_model, nms_threshold=DeepForest_config["nms_threshold"])

    #print(model.summary())

    # create the testing generators
    if DeepForest_config["evaluation_images"] > 0:
        generator = create_generator(args, data, DeepForest_config)
        
        average_precisions = evaluate(
            generator,
            model,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
            save_path=args.save_path + dirname
        )
    
        ## print evaluation
        present_classes = 0
        precision = 0
        for label, (average_precision, num_annotations) in average_precisions.items():
            print('{:.0f} instances of class'.format(num_annotations),
                  generator.label_to_name(label), 'with average precision: {:.3f}'.format(average_precision))
            if num_annotations > 0:
                present_classes += 1
                precision       += average_precision
        print('mAP: {:.3f}'.format(precision / present_classes))
        experiment.log_metric("mAP", precision / present_classes)                 

    #Evaluation metrics
    sites=DeepForest_config["evaluation_site"]
    
    #create the NEON mAP generator 
    NEON_generator = create_NEON_generator(args.batch_size, DeepForest_config)
    NEON_recall_generator = create_NEON_generator(args.batch_size, DeepForest_config)

    recall = neonRecall(
        sites,
        NEON_recall_generator,
        model,            
        score_threshold=args.score_threshold,
        save_path=args.save_path + dirname,
        max_detections=args.max_detections
    )
    
    print("Recall is {:0.3f}".format(recall))
    
    experiment.log_metric("Recall", recall)               
        
    #NEON plot mAP
    average_precisions = evaluate(
        NEON_generator,
        model,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        save_path=args.save_path + dirname,
        experiment=experiment
    )

    # print evaluation
    present_classes = 0
    precision = 0
    for label, (average_precision, num_annotations) in average_precisions.items():
        print('{:.0f} instances of class'.format(num_annotations),
              NEON_generator.label_to_name(label), 'with average precision: {:.3f}'.format(average_precision))
        if num_annotations > 0:
            present_classes += 1
            precision       += average_precision
    NEON_map = round(precision / present_classes,3)
    print('Neon mAP: {:.3f}'.format(NEON_map))
    experiment.log_metric("Neon mAP", NEON_map)        
    
    return [recall, NEON_map]
    
if __name__ == '__main__':
    
    import argparse
    import os
    import pandas as pd
    import glob
    import numpy as np
    from datetime import datetime
    from DeepForest.config import load_config
    from DeepForest import preprocess, Generate    
    
    #Set training or training
    mode_parser     = argparse.ArgumentParser(description='Retinanet training or finetuning?')
    mode_parser.add_argument('--mode', help='train or retrain?' )
    mode_parser.add_argument('--dir', help='destination dir' )    
    mode_parser.add_argument('--saved_model', help='train or retrain?' )    
    
    mode = mode_parser.parse_args()

    #set experiment and log configs
    experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",project_name='deeplidar',log_code=False)

    DeepForest_config = load_config()

    #Log parameters
    experiment.log_parameter("Start Time", mode.dir)
    experiment.log_parameter("Training Mode", mode.mode)
    experiment.log_parameters(DeepForest_config)
    
    DeepForest_config["mode"] = mode.mode
    if mode.mode == "train":
        data = load_training_data(DeepForest_config)

    if mode.mode == "retrain":
        data = load_retraining_data(DeepForest_config)
        for site in DeepForest_config["hand_annotation_site"]:
            DeepForest_config[site]["h5"] = os.path.join(DeepForest_config[site]["h5"],"hand_annotations")        
    
    #pass an args object instead of using command line        
    args = [
        "--batch-size", str(DeepForest_config['batch_size']),
        '--score-threshold', str(DeepForest_config['score_threshold']),
        '--suppression-threshold', '0.1', 
        '--save-path', 'snapshots/images/', 
        '--model', mode.saved_model, 
        '--convert-model'
    ]
       
    #Run training, and pass comet experiment 
    main(data, DeepForest_config, experiment, args)