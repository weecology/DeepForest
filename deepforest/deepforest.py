"""
.. module:: deepforest
   :platform: Unix, Windows
   :synopsis: A module for individual tree crown detection using deep learning neural networks. see Weinstein et al. Remote Sensing. 2019

.. moduleauthor:: Ben Weinstein <ben.weinstein@weecology.org>

"""
from .utilities import read_model, read_config

class deepforest:
    """Overall class for training and predicting images
    """
    
    def __init__(self, weights=None):
        """A deepforest object for model training or prediction"""
        self.weights = weights
        
        #Read config file
        self.config = read_config()
        
        #Load model weights if needed
        if self.weights is not None:
            read_model(model_path, config)
            
    def train(self):
        """A training model"""
        pass
    
    def download_release(self):
        """Download the latest model release from github"""
        pass
        
    def predict(self):
        """A prediction model"""
        pass