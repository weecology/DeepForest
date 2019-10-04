"""
.. module:: deepforest
   :platform: Unix, Windows
   :synopsis: A module for individual tree crown detection using deep learning neural networks. see Weinstein et al. Remote Sensing. 2019

.. moduleauthor:: Ben Weinstein <ben.weinstein@weecology.org>

"""
from deepforest import utilities
from deepforest import predict
import os
from matplotlib import pyplot as plt
from keras_retinanet.models import convert_model

class deepforest:
    ''' Class for training and predicting tree crowns in RGB images
    
    Args:
        weights (str): Path to model weights on disk. Default is None
    
    Attributes:
        model: A keras training model from keras-retinanet
    '''
    
    def __init__(self, weights=None):
        self.weights = weights
        
        #Read config file
        self.config = utilities.read_config()
        
        #Load model weights if needed
        if self.weights is not None:
            self.model = utilities.read_model(self.weights, self.config)
        else:
            self.model = None
            
    def train(self, image_dir,label_csv):
        '''
        Train a deep learning tree detection model
        This is the main entry point for training a new model based on either existing weights or scratch
        
        Args:
            image_dir (str): Directory of images
            label_csv (str): Path to csv label file, labels are in the format -> path/to/image.jpg,x1,y1,x2,y2,class_name
        Returns:
            model (object): A trained keras model
        '''
        
        pass
    
    def use_release(self):
        '''
        Use the latest DeepForest model release from github and load model. Optionally download if release doesn't exist
        
        Returns:
            model (object): A trained keras model
        '''        
        #Download latest model from github release
        weight_path = utilities.use_release()  
        
        #load weights
        self.weights = weight_path
        self.model = utilities.read_model(self.weights, self.config)
        
    def predict_image(self, image_path, plot=True):
        '''
        Predict tree crowns based on loaded (or trained) model
        Args:
            image_path (str): Path to image on disk
            plot (bool): Plot the predicted image with bounding boxes
        Returns:
            predictions (array): Numpy array of predicted bounding boxes
        '''     
        #Check for model weights
        
        if(self.weights is None):
            raise ValueError("Model currently has no weights, either train a new model using deepforest.train, loading existing model, or use prebuilt model (see deepforest.use_release()")
        
        #convert model to prediction
        self.prediction_model = convert_model(self.model)
        image = predict.predict_image(self.prediction_model, image_path, return_plot=True)
        
        #cv2 channel order
        plt.imshow(image[:,:,::-1])
        plt.show()
        
        return image