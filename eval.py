#!/usr/bin/env python

"""
FCopyright 2017-2018 Fizyr (https://fizyr.com)

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

#Log training
from comet_ml import Experiment

import argparse
import os
import sys
from datetime import datetime

import keras
import tensorflow as tf

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

from keras_retinanet import models
from keras_retinanet.utils.eval import evaluate
from keras_retinanet.utils.keras_version import check_keras_version

#Custom Generator
from DeepForest.onthefly_generator import OnTheFlyGenerator

#Custom callback
from DeepForest.evaluation import neonRecall, Jaccard

def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def create_NEON_generator(args,site,DeepForest_config):
    """ Create generators for training and validation.
    """

    annotations,windows=preprocess.NEON_annotations(site,DeepForest_config)

    #Training Generator
    generator =  OnTheFlyGenerator(
        annotations,
        windows,
        batch_size=args.batch_size,
        DeepForest_config=DeepForest_config,
        group_method="none")
    
    return(generator)

def create_generator(args,data,config):
    """ Create generators for training and validation.
    """

    #Split training and test data - hardcoded paths set below.
    _,test=preprocess.split_training(data,DeepForest_config,single_tile=DeepForest_config["single_tile"],experiment=None)

    #Training Generator
    generator =  OnTheFlyGenerator(
        data,
        test,
        batch_size=args.batch_size,
        DeepForest_config=DeepForest_config,
        group_method="none")
    
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
    parser.add_argument('--image-min-side',  help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side',  help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)

    return parser.parse_args(args)


def main(data,DeepForest_config,experiment,args=None):
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

    # create the testing generators
    #generator = create_generator(args,data,DeepForest_config)

    #create the NEON mAP generator 
    NEON_generator = create_NEON_generator(args,site,DeepForest_config)
    
    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone, convert=args.convert_model,nms_threshold=DeepForest_config["nms_threshold"])

    #print(model.summary())

    #average_precisions = evaluate(
        #generator,
        #model,
        #iou_threshold=args.iou_threshold,
        #score_threshold=args.score_threshold,
        #max_detections=args.max_detections,
        #save_path=args.save_path + dirname
    #)

    ## print evaluation
    #present_classes = 0
    #precision = 0
    #for label, (average_precision, num_annotations) in average_precisions.items():
        #print('{:.0f} instances of class'.format(num_annotations),
              #generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
        #if num_annotations > 0:
            #present_classes += 1
            #precision       += average_precision
    #print('mAP: {:.4f}'.format(precision / present_classes))
    #experiment.log_metric("mAP", precision / present_classes)    

    #Use field collected polygons only for Florida site
    #if site == "OSBS":

        ##Ground truth scores
        #jaccard=Jaccard(
            #generator=generator,
            #model=model,            
            #score_threshold=args.score_threshold,
            #save_path=args.save_path,
            #experiment=experiment,
            #DeepForest_config=DeepForest_config
        #)
        #print(f" Mean IoU: {jaccard:.2f}")
        
        #experiment.log_metric("Mean IoU", jaccard)               
        
    #Neon plot recall rate
    #recall=neonRecall(
        #site,
        #generator,
        #model,            
        #score_threshold=args.score_threshold,
        #save_path=args.save_path,
        #experiment=experiment,
        #DeepForest_config=DeepForest_config
    #)
    
    #experiment.log_metric("Recall", recall)       
    
    #print(f" Recall: {recall:.2f}")
        
    #NEON plot mAP
    average_precisions = evaluate(
        NEON_generator,
        model,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        save_path=args.save_path + dirname,
        experiment=experiment,
    )

    # print evaluation
    present_classes = 0
    precision = 0
    for label, (average_precision, num_annotations) in average_precisions.items():
        print('{:.0f} instances of class'.format(num_annotations),
              NEON_generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
        if num_annotations > 0:
            present_classes += 1
            precision       += average_precision
    print('NEON mAP: {:.4f}'.format(precision / present_classes))
    experiment.log_metric("NEON_mAP", precision / present_classes)        
    
    
if __name__ == '__main__':
    
    import argparse
    
    #Set training or training
    mode_parser     = argparse.ArgumentParser(description='Retinanet training or finetuning?')
    mode_parser.add_argument('--mode', help='train or retrain?' )
    
    mode=mode_parser.parse_args()
    
    import os
    import pandas as pd
    import glob
    import numpy as np
    from datetime import datetime
    from DeepForest.config import load_config
    from DeepForest import preprocess

    #set experiment and log configs
    experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",project_name='deeplidar')

    #save time for logging
    dirname=datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment.log_parameter("Start Time", dirname)

    #log training mode
    experiment.log_parameter("Training Mode",mode.mode)
    
    #Load DeepForest_config and data file based on training or retraining mode
    
    if mode.mode == "train":
        DeepForest_config=load_config("train")
        data=preprocess.load_data(DeepForest_config["training_csvs"],DeepForest_config["rgb_res"])
        
    if mode.mode == "retrain":
        DeepForest_config=load_config("retrain")        
        data=preprocess.load_xml(DeepForest_config["hand_annotations"],DeepForest_config["rgb_res"])

    experiment.log_multiple_params(DeepForest_config)

    #Log site
    site=DeepForest_config["evaluation_site"]
    experiment.log_parameter("Site", site)
    
    ##Preprocess Filters##
    if DeepForest_config['preprocess']['zero_area']:
        data=preprocess.zero_area(data)

    #pass an args object instead of using command line        
    args = [
        "--batch-size",str(DeepForest_config['batch_size']),
        '--score-threshold', str(DeepForest_config['score_threshold']),
        '--suppression-threshold','0.1', 
        '--save-path', 'snapshots/images/', 
        '--model', '/Users/ben/Documents/DeepForest/snapshots/handannotatedonly.h5', 
        '--convert-model'
    ]
       
    #Run training, and pass comet experiment   
    main(data,DeepForest_config,experiment,args)