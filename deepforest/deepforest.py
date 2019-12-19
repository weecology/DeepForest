"""
.. module:: deepforest
   :platform: Unix, Windows
   :synopsis: A module for individual tree crown detection using deep learning neural networks. see Weinstein et al. Remote Sensing. 2019

.. moduleauthor:: Ben Weinstein <ben.weinstein@weecology.org>
"""
import os
import warnings

from PIL import Image
with warnings.catch_warnings():
    #Suppress some of the verbose tensorboard warnings, compromise to avoid numpy version errors    
    warnings.filterwarnings("ignore",category=FutureWarning)    
    import tensorflow as tf
    
import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt

from deepforest import get_data
from deepforest import utilities
from deepforest import predict
from deepforest import preprocess
from deepforest.retinanet_train import main as retinanet_train
from deepforest.retinanet_train import parse_args

from keras_retinanet import models
from keras_retinanet.models import convert_model
from keras_retinanet.bin.train import create_models
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.utils.eval import evaluate
from keras_retinanet.utils.eval import _get_detections
from keras_retinanet.utils.visualization import draw_box

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
        
        #Read config file - if a config file exists in local dir use it, if not use installed.
        if os.path.exists("deepforest_config.yml"):
            config_path = "deepforest_config.yml"
        else:
            try:
                config_path = get_data("deepforest_config.yml")
            except Exception as e:
                raise ValueError("No deepforest_config.yml found either in local directory or in installed package location. {}".format(e))
        
        print("Reading config file: {}".format(config_path))                    
        self.config = utilities.read_config(config_path)
        
        #release version id to flag if release is being used
        self.__release_version__ = None
        
        #Load saved model if needed
        if self.saved_model:
            print("Loading saved model")
            #Capture user warning, not relevant here
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",category=UserWarning)                    
                self.model = utilities.load_model(saved_model)
            self.prediction_model = convert_model(self.model)
            
        if self.weights is not None:
            print("Creating model from weights")
            backbone = models.backbone(self.config["backbone"])            
            self.model, self.training_model, self.prediction_model = create_models(backbone.retinanet, num_classes=1, weights=self.weights)
        else:
            print("A blank deepforest object created. To perform prediction, either train or load an existing model.")
            self.model = None
            
    def train(self, annotations, input_type="fit_generator", list_of_tfrecords=None, comet_experiment=None, images_per_epoch=None):
        '''Train a deep learning tree detection model using keras-retinanet.
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
        release_tag, self.weights = utilities.use_release()  
        
        #load saved model and tag release
        self.__release_version__ = release_tag
        print("Loading pre-built model: {}".format(release_tag))
        
        with warnings.catch_warnings():
            #Suppress compilte warning, not relevant here
            warnings.filterwarnings("ignore",category=UserWarning)          
            self.model = utilities.read_model(self.weights, self.config)
        
        #Convert model
        self.prediction_model = convert_model(self.model)
        
        #add to config
        self.config["weights"] = self.weights        
    
    def predict_generator(self, annotations, comet_experiment = None, iou_threshold=0.5, score_threshold=0.05, max_detections=200):
        """Predict bounding boxes for a model using a csv fit_generator
        
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
                boxes = self.predict_image(image_path, return_plot=False, score_threshold=args.score_threshold)
                
                #Turn to pandas frame and save output
                box_df = pd.DataFrame(boxes)
                #use only plot name, not extension
                box_df["plot_name"] = os.path.splitext(plot_name)[0]
                boxes_output.append(box_df)
        else:
                raise ValueError("No prediction model loaded. Either load a retinanet from file, download the latest release or train a new model")
    
        #name columns and return box data
        boxes_output = pd.concat(boxes_output)
        boxes_output.columns = ["xmin","ymin","xmax","ymax","score","label","plot_name"]
        boxes_output = boxes_output.reindex(columns= ["plot_name","xmin","ymin","xmax","ymax","score","label"])    
        
        return boxes_output
    
    def evaluate_generator(self, annotations, comet_experiment = None, iou_threshold=0.5, score_threshold=0.05, max_detections=200):
        """ Evaluate prediction model using a csv fit_generator
        
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
            score_threshold=args.score_threshold,
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

    def predict_image(self, image_path=None, raw_image=None, return_plot=True, score_threshold=0.05, show=False, color=(0,0,0)):
        """Predict tree crowns based on loaded (or trained) model
        
        Args:
            image_path (str): Path to image on disk
            raw_image (array): Numpy image array in BGR channel order following openCV convention
            color (tuple): Color of bounding boxes in BGR order (0,0,0) black default 
            show (bool): Plot the predicted image with bounding boxes. Ignored if return_plot=False
            return_plot: Whether to return image with annotations overlaid, or just a numpy array of boxes
        
        Returns:
            predictions (array): if return_plot, an image. Otherwise a numpy array of predicted bounding boxes, with scores and labels
        """     
        
        #Check for model save
        if(self.prediction_model is None):
            raise ValueError("Model currently has no prediction weights, either train a new model using deepforest.train, loading existing model, or use prebuilt model (see deepforest.use_release()")
        
        #Check the formatting
        if isinstance(image_path,np.ndarray):
            raise ValueError("image_path should be a string, but is a numpy array. If predicting a loaded image (channel order BGR), use raw_image argument.")
        
        #Check for correct formatting
        #Warning if image is very large and using the release model
        if raw_image is None:
            raw_image = cv2.imread(image_path)    
        
        if self.__release_version__ :
            if any([x > 400 for x in raw_image.shape[:2]]):
                warnings.warn("Input image has a size of {}, but the release model was trained on crops of 400px x 400px, results may be poor."
                              "Use predict_tile for dividing large images into overlapping windows.".format(raw_image.shape[:2]))
        
        #Predict
        prediction = predict.predict_image(self.prediction_model, image_path=image_path, raw_image=raw_image, return_plot=return_plot, score_threshold=score_threshold, color=color)            
            
        #cv2 channel order to matplotlib order
        if return_plot & show:
            plt.imshow(prediction[:,:,::-1])
            plt.show()             

        return prediction            

    def predict_tile(self, path_to_raster, patch_size=400, patch_overlap=0.25, iou_threshold=0.3, return_plot=False):
        """
        For images too large to input into the model, predict_tile cuts the image into overlapping windows, predicts trees on each window and reassambles into a single array. 
    
        Args:
            raster_path: Path to image on disk
            iou_threshold: Minimum iou overlap among predictions between windows to be supressed. Defaults to 0.5. Lower values suppress more boxes at edges.
            return_plot: Should the image be returned with the predictions drawn?
    
        Returns:
            boxes (array): if return_plot, an image. Otherwise a numpy array of predicted bounding boxes, scores and labels
        """   
    
        #Load raster as image
        raster = Image.open(path_to_raster)
        numpy_image = np.array(raster)        
        image_name = os.path.basename(path_to_raster)
    
        #Compute sliding window index
        windows = preprocess.compute_windows(numpy_image, patch_size,patch_overlap)
    
        #Save images to tmpdir
        predicted_boxes = []
    
        for index, window in enumerate(windows):
            #Crop window and predict
            crop = numpy_image[windows[index].indices()] 
    
            #Crop is RGB channel order, change to BGR
            crop = crop[...,::-1]
            boxes = self.predict_image(raw_image=crop, return_plot=False, score_threshold=self.config["score_threshold"])            
    
            #transform coordinates to original system
            xmin, ymin, xmax, ymax = windows[index].getRect()
            boxes.xmin = boxes.xmin + xmin
            boxes.xmax = boxes.xmax + xmin
            boxes.ymin = boxes.ymin + ymin
            boxes.ymax = boxes.ymax + ymin
    
            predicted_boxes.append(boxes)
    
        predicted_boxes = pd.concat(predicted_boxes)
    
        #Non-max supression for overlapping boxes among window 
        with tf.Session() as sess:
            print("{} predictions in overlapping windows, applying non-max supression".format(predicted_boxes.shape[0]))
            new_boxes, new_scores, new_labels = predict.non_max_suppression(sess,
                                                                                predicted_boxes[["xmin","ymin","xmax","ymax"]].values,
                                                                                predicted_boxes.score.values, predicted_boxes.label.values,
                                                                                max_output_size=predicted_boxes.shape[0],
                                                                                iou_threshold=iou_threshold)
    
            image_detections = np.concatenate([new_boxes, np.expand_dims(new_scores, axis=1), np.expand_dims(new_labels, axis=1)], axis=1)
            mosaic_df = pd.DataFrame(image_detections,columns=["xmin","ymin","xmax","ymax","score","label"])
            mosaic_df.label = mosaic_df.label.str.decode("utf-8")
            print("{} predictions kept after non-max suppression".format(mosaic_df.shape[0]))
    
        if return_plot:
            #Draw predictions
            for box in mosaic_df[["xmin","ymin","xmax","ymax"]].values:
                draw_box(numpy_image, box, [0,0,255])
    
            return numpy_image
        else:
            return mosaic_df