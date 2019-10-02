"""
.. module:: deepforest
   :platform: Unix, Windows
   :synopsis: A module for individual tree crown detection using deep learning neural networks. see Weinstein et al. Remote Sensing. 2019

.. moduleauthor:: Ben Weinstein <ben.weinstein@weecology.org>

"""
from ..deepforest import utilities
import os


class deepforest:
    """Overall class for training and predicting images
    """
    
    def __init__(self, weights="default"):
        """A deepforest object for model training or prediction
        
        Parameters
        ---------
        weights: str
             Path to model weights.
        
        """
        self.weights = weights
        
        #Read config file
        self.config = utilities.read_config()
        
        #Load model weights if needed
        if self.weights is not None:
            self.model = utilities.read_model(self.weights, self.config)
        else:
            self.model = None
            
    def train(self, image_dir,label_csv):
        """Train a deep learning tree detection model
        
        This is the main entry point for training a new model based on either existing weights or scratch
        
        Parameters
        ---------
        image_dir: str
            Directory of images
        label_csv: str
            Path to csv label file, labels are in the format -> path/to/image.jpg,x1,y1,x2,y2,class_name
        Returns
        -------
             str
                 "A train keras model"

        """
        
        pass
    
    def download_release(self):
        """Download the latest model release from github release
        Returns
        ------
            str
                 "A loaded keras model"
        """        
        #Download latest model from github release
        weight_path = utilities.download_release()  
        
        #load weights
        self.weights = weight_path
        self.model = utilities.read_model(self.weights, self.config)
        
    def predict(self, image):
        """Predict tree crowns in an image
        Parameters
        ---------
        
        Returns
        ------
        """
        pass