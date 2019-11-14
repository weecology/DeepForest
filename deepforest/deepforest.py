"""
.. module:: deepforest
   :platform: Unix, Windows
   :synopsis: A module for individual tree crown detection using deep learning neural networks. see Weinstein et al. Remote Sensing. 2019

.. moduleauthor:: Ben Weinstein <ben.weinstein@weecology.org>

"""
import os
import pandas as pd
from matplotlib import pyplot as plt

from deepforest import utilities
from deepforest import predict
from deepforest.retinanet_train import main as retinanet_train
from deepforest.retinanet_train import parse_args

from keras_retinanet import models
from keras_retinanet.models import convert_model
from keras_retinanet.bin.train import create_models
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.utils.eval import evaluate
from keras_retinanet.utils.eval import _get_detections

class deepforest:
    ''' Class for training and predicting tree crowns in RGB images
    
    Args:
        weights (str): Path to model saved on disk from keras.model.save_weights(). A new model is created and weights are copied. Default is None. 
        saved_model: Path to a saved model from disk using keras.model.save(). No new model is created.
    
    Attributes:
        model: A keras training model from keras-retinanet
    '''
    
    def __init__(self, weights=None, saved_model=None):
        self.weights = weights
        self.saved_model = saved_model
        
        #Read config file
        self.config = utilities.read_config()
        
        #Load saved model if needed
        if self.saved_model:
            print("Loading saved model")
            self.model = utilities.load_model(saved_model)
            self.prediction_model = convert_model(self.model)
            
        if self.weights is not None:
            print("Creating model from weights")
            backbone = models.backbone(self.config["backbone"])            
            self.model, self.training_model, self.prediction_model = create_models(backbone.retinanet, num_classes=1, weights=self.weights)
        else:
            print("No model initialized, either train or load an existing retinanet model")
            self.model = None
            
    def train(self, annotations, input_type="fit_generator", list_of_tfrecords=None, comet_experiment=None, images_per_epoch=None):
        '''Train a deep learning tree detection model using keras-retinanet
        This is the main entry point for training a new model based on either existing weights or scratch
        
        Args:
            annotations (str): Path to csv label file, labels are in the format -> path/to/image.jpg,x1,y1,x2,y2,class_name
            comet_experiment: A comet ml object to log images. Optional.
            list_of_tfrecords: Ignored if input_type != "tfrecord", list of tf records to process
            input_type: "fit_generator" or "tfrecord"
            images_per_epoch: number of images to override default config of # images in annotations file / batch size. Useful for debug
        Returns:
            model (object): A trained keras model
            prediction model: with bbox nms
            trained model: without nms
        '''
        arg_list = utilities.format_args(annotations, self.config, images_per_epoch)
        
        print("Training retinanet with the following args {}".format(arg_list))
        
        #Train model
        self.model, self.prediction_model, self.training_model = retinanet_train(args=arg_list, input_type = input_type, list_of_tfrecords = list_of_tfrecords, comet_experiment = comet_experiment)
    
    def use_release(self):
        '''Use the latest DeepForest model release from github and load model. Optionally download if release doesn't exist
        
        Returns:
            model (object): A trained keras model
        '''        
        #Download latest model from github release
        weights = utilities.use_release()  
        
        #load saved model
        self.weights = weights
        self.model = utilities.read_model(self.weights, self.config)
        self.prediction_model = convert_model(self.model)
    
    def predict_generator(self, annotations, comet_experiment = None, iou_threshold=0.5, score_threshold=0.05, max_detections=200):
        """
         Predict bounding boxes for a model using a csv fit_generator
        Args:
            annotations (str): Path to csv label file, labels are in the format -> path/to/image.jpg,x1,y1,x2,y2,class_name
            iou_threshold(float): IoU Threshold to count for a positive detection (defaults to 0.5)
            score_threshold (float): Eliminate bounding boxes under this threshold
            max_detections (int): Maximum number of bounding box predictions
            comet_experiment(object): A comet experiment class objects to track
        Return:
            boxes_output: a pandas dataframe of bounding boxes for each image in the annotations file
        """
        #Format args for CSV generator 
        arg_list = utilities.format_args(annotations, self.config)
        args = parse_args(arg_list)
        
        #create generator
        generator = CSVGenerator(
            args.annotations,
            args.classes,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config,
            shuffle_groups=False,
        )
        
        if self.prediction_model:
            boxes_output = [ ]
            #For each image, gather predictions
            for i in range(generator.size()):
                #pass image as path
                plot_name = generator.image_names[i]
                image_path = os.path.join(generator.base_dir,plot_name)
                boxes = self.predict_image(image_path, return_plot=False)
                
                #Turn to pandas frame and save output
                box_df = pd.DataFrame(boxes)
                box_df["plot_name"] = plot_name
                boxes_output.append(box_df)
        else:
                raise ValueError("No prediction model loaded. Either load a retinanet from file, download the latest release or train a new model")
    
        #name columns and return box data
        boxes_output = pd.concat(boxes_output)
        boxes_output.columns = ["xmin","ymin","xmax","ymax","plot_name"]
        boxes_output = boxes_output.reindex(columns= ["plot_name","xmin","ymin","xmax","ymax"])    
        
        return boxes_output
    
    def evaluate_generator(self, annotations, comet_experiment = None, iou_threshold=0.5, score_threshold=0.05, max_detections=200):
        """
        Evaluate prediction model using a csv fit_generator
        Args:
            annotations (str): Path to csv label file, labels are in the format -> path/to/image.jpg,x1,y1,x2,y2,class_name
            iou_threshold(float): IoU Threshold to count for a positive detection (defaults to 0.5)
            score_threshold (float): Eliminate bounding boxes under this threshold
            max_detections (int): Maximum number of bounding box predictions
            comet_experiment(object): A comet experiment class objects to track
        Return:
            mAP: Mean average precision of the evaluated data
        """
        #Format args for CSV generator 
        arg_list = utilities.format_args(annotations, self.config)
        args = parse_args(arg_list)
        
        #create generator
        validation_generator = CSVGenerator(
            args.annotations,
            args.classes,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config,
            shuffle_groups=False,
        )
        
        average_precisions = evaluate(
            validation_generator,
            self.prediction_model,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            max_detections=max_detections,
            save_path=args.save_path,
            comet_experiment=comet_experiment
        )
        
        # print evaluation
        total_instances = []
        precisions = []
        for label, (average_precision, num_annotations) in average_precisions.items():
            print('{:.0f} instances of class'.format(num_annotations),
                  validation_generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)

        if sum(total_instances) == 0:
            print('No test instances found.')
            return

        print('mAP using the weighted average of precisions among classes: {:.4f}'.format(sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)))
        
        mAP = sum(precisions) / sum(x > 0 for x in total_instances)
        print('mAP: {:.4f}'.format(mAP))   
        return mAP

    def predict_image(self, image_path=None, raw_image=None, return_plot=True, show=False):
        '''Predict tree crowns based on loaded (or trained) model
        
        Args:
            image_path (str): Path to image on disk
            raw_image (array): Numpy image array in BGR channel order following openCV convention
            show (bool): Plot the predicted image with bounding boxes. Ignored if return_plot=False
            return_plot: Whether to return image with annotations overlaid, or just a numpy array of boxes
        Returns:
            predictions (array): if return_plot, an image. Otherwise a numpy array of predicted bounding boxes
        '''     
        #Check for model save
        
        if(self.prediction_model is None):
            raise ValueError("Model currently has no prediction weights, either train a new model using deepforest.train, loading existing model, or use prebuilt model (see deepforest.use_release()")
                
        if return_plot:
            image = predict.predict_image(self.prediction_model, image_path, raw_image, return_plot=return_plot)            
            #cv2 channel order
            if show:
                plt.imshow(image[:,:,::-1])
                plt.show()             
            return image
        else:
            boxes = predict.predict_image(self.prediction_model, image_path, return_plot=return_plot)            
            return boxes            

