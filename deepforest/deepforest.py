"""
.. module:: deepforest
   :platform: Unix, Windows
   :synopsis: A module for individual tree crown detection using deep learning neural networks. see Weinstein et al. Remote Sensing. 2019

.. moduleauthor:: Ben Weinstein <ben.weinstein@weecology.org>

"""
from deepforest import utilities
import os

class deepforest:
    ''' Class for training and predicting tree crowns in RGB images
    
    Args:
        weights (str): Path to model weights on disk. Default is None
    
    Attributes:
        model: A keras training model from keras-retinanet
    '''
    
    def __init__(self, weights=""):
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
    
    def download_release(self):
        '''
        Download the latest model release from github release and load model
        
        Returns:
            model (object): A trained keras model
        '''        
        #Download latest model from github release
        weight_path = utilities.download_release()  
        
        #load weights
        self.weights = weight_path
        self.model = utilities.read_model(self.weights, self.config)
        
    def predict(self, image):
        '''
        Predict tree crowns based on loaded (or trained) model
        
        Returns:
            predictions (array): Numpy array of predicted bounding boxes
        '''     
        pass