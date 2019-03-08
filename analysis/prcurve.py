#!/usr/bin/env python


import argparse
import os
import sys
from datetime import datetime
sys.path.insert(0, os.path.abspath('..'))

import warnings
warnings.simplefilter("ignore")

import keras
import tensorflow as tf

from keras_retinanet import models
from keras_retinanet.utils.keras_version import check_keras_version

#Custom Generators and callbacks
from DeepForest.onthefly_generator import OnTheFlyGenerator
from DeepForest.evaluation import neonRecall
from DeepForest.evalmAP import evaluate_pr
from DeepForest import preprocess
from DeepForest.utils.generators import create_NEON_generator

def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

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

def main(DeepForest_config, args=None):
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
    
    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path + dirname):
        os.makedirs(args.save_path + dirname)
        
    #Evaluation metrics
    site=DeepForest_config["evaluation_site"]
    
    #create the NEON mAP generator 
    NEON_generator = create_NEON_generator(args, site, DeepForest_config, dir="..")
    
    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone, convert=args.convert_model, nms_threshold=DeepForest_config["nms_threshold"])

    #print(model.summary())
        
    #NEON plot mAP
    recall, precision = evaluate_pr(
        NEON_generator,
        model,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        save_path=args.save_path + dirname,
        experiment=None
    )
    
    return [recall, precision]
    
if __name__ == '__main__':
    
    import argparse
    
    #Set training or training
    mode_parser     = argparse.ArgumentParser(description='Retinanet training or finetuning?')
    mode_parser.add_argument('--saved_model', help='train or retrain?' )    
    mode = mode_parser.parse_args()
    
    import pandas as pd
    import numpy as np
    from DeepForest.config import load_config

    DeepForest_config = load_config("..")

    results = []
    for score_threshold in np.arange(0, 0.8, 0.1):
        #pass an args object instead of using command line        
        args = [
            "--batch-size", str(DeepForest_config['batch_size']),
            '--score-threshold', str(score_threshold),
            '--suppression-threshold','0.1', 
            '--save-path', 'snapshots/images/', 
            '--model', mode.saved_model, 
            '--convert-model'
        ]
           
        #Run eval
        recall, precision = main(DeepForest_config, args)
        results.append({"Threshold": score_threshold, "Recall": recall, "Precision": precision})
        
    results = pd.DataFrame(results)
    #model name
    model_name = os.path.splitext(os.path.basename(mode.saved_model))[0]
    
    results.to_csv("/Users/Ben/Dropbox/Weecology/RGB/prcurve_data" + model_name + ".csv")