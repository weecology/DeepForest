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

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_retinanet import models
from keras_retinanet.utils.eval import evaluate
from keras_retinanet.utils.eval import JaccardEvaluate, neonRecall
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.preprocessing import onthefly

def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def create_generator(args,config):
    """ Create generators for evaluation.
    """
    if  args.dataset_type == 'onthefly':
            
        #Replace config subsample with validation subsample. Not the best, or the worst, way to do this.
        config["subsample"]=config["validation_subsample"]
    
        validation_generator=onthefly.OnTheFlyGenerator(
                    args.annotations,
                batch_size=args.batch_size,
                base_dir=config["evaluation_tile_dir"],
                config=config,
                group_method="none",
                shuffle_groups=False,
                shuffle_tiles=config["shuffle_eval"])   
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return validation_generator


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    parser.add_argument('--batch-size',      help='Size of the batches.', default=1, type=int)
    
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')

    #On the fly parser
    otf_parser = subparsers.add_parser('onthefly')
    otf_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    
    parser.add_argument('model',             help='Path to RetinaNet model.')
    parser.add_argument('--convert-model',   help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
    parser.add_argument('--backbone',        help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',             help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
    parser.add_argument('--iou-threshold',   help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--max-detections',  help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--save-path',       help='Path for saving images with detections (doesn\'t work for COCO).')
    parser.add_argument('--image-min-side',  help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side',  help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)

    return parser.parse_args(args)


def main(config,experiment,args=None):
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

    # create the generator
    generator = create_generator(args,config)

    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone, convert=args.convert_model)

    print(model.summary())

    average_precisions = evaluate(
        generator,
        model,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        save_path=args.save_path + dirname
    )

    # print evaluation
    present_classes = 0
    precision = 0
    for label, (average_precision, num_annotations) in average_precisions.items():
        print('{:.0f} instances of class'.format(num_annotations),
              generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
        if num_annotations > 0:
            present_classes += 1
            precision       += average_precision
    print('mAP: {:.4f}'.format(precision / present_classes))
    experiment.log_metric("mAP", precision / present_classes)    

    #Use field collected polygons only for Florida site
    if site == "OSBS":

        #Ground truth scores
        jaccard_scores = JaccardEvaluate(
            generator,
            model,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
            save_path=args.save_path + dirname,
            experiment=experiment,
            config=config
        )
        
        print("Mean IoU = %.3f" % (jaccard_scores))
        #Log results
        experiment.log_metric("Mean IoU", jaccard_scores)    
        
    ##Neon plot recall rate
    recall = neonRecall(
        site,
        generator,
        model,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        save_path=args.save_path + dirname,
        experiment=experiment,
        config=config
    )    
    
    experiment.log_metric("Recall", recall)    
    
    #Logs the number of train and eval "trees"
    ntrees=[len(x) for x in generator.annotation_dict.values()]
    experiment.log_parameter("Number of Trees", ntrees)    

if __name__ == '__main__':
    
    import numpy as np
    
    np.random.seed(2)
    from DeepForest.config import config    
    from DeepForest import preprocess
    
    #set experiment and log configs
    experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",project_name='deepforest-retinanet')
    experiment.log_multiple_params(config)
    
    #Log site
    site=os.path.split(os.path.normpath(config["training_csvs"]))[1]
    experiment.log_parameter("Site", site)
    
    #Prepare Evaluation
    evaluation=preprocess.load_data(data_dir=config['evaluation_csvs'])
    
    ##Preprocess Filters##
    if config['preprocess']['zero_area']:
        evaluation=preprocess.zero_area(evaluation)
        
    #Write training and evaluation data to file for annotations
    evaluation.to_csv("data/training/evaluation.csv") 
    
    main(config,experiment)