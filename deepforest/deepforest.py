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
    
    def __init__(self, weights="default"):
        """A deepforest object for model training or prediction"""
        self.weights = weights
        
        #Read config file
        self.config = read_config()
        
        #Load model weights if needed
        if self.weights is "default":
            #Check if most recent model has been downloaded
            if os.path.exists("path_to_model"):
                pass
            else:
                path_to_model = self.download_release()
                self.weights = path_to_model
        if self.weights is not None:
            self.model = read_model(self.weights, self.config)
        else:
            self.model = None
            
    def train(self):
        """A training model.
        
        Train a deep learning model for tree crown detection
        """
        pass
    
    def download_release(self):
        """Download the latest model release from github release
        Returns:
            str: "path to downloaded weights on disk"
        """
        pass
        
    def predict(self, image):
        """Predict tree crowns in an image
        Args: 
               image (np.array): Image numpy array
        """
        pass