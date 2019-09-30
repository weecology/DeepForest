"""
.. module:: deepforest
   :platform: Unix, Windows
   :synopsis: A module for individual tree crown detection using deep learning neural networks. see Weinstein et al. Remote Sensing. 2019

.. moduleauthor:: Ben Weinstein <ben.weinstein@weecology.org>

"""

class deepforest:
    """Overall Class for training and predicting images
    """
    
    def __init__(self, weights=None):
        """A deepforest object for model training or prediction"""
        self.weights = weights
    
    def train(self):
        """A training model"""
        
    def prediction(self):
        """A prediction model"""