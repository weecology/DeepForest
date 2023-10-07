# Model - common class
from deepforest.models import *
import torch


class Model():
    """A architecture agnostic class that controls the basic train, eval and predict functions.
    A model should optionally allow a backbone for pretraining. To add new architectures, simply create a new module in models/ and write a create_model. 
    Then add the result to the if else statement below.
    Args:
        num_classes (int): number of classes in the model
        nms_thresh (float): non-max suppression threshold for intersection-over-union [0,1]
        score_thresh (float): minimum prediction score to keep during prediction  [0,1]
    Returns:
        model: a pytorch nn module
    """

    def __init__(self, config):

        # Check for required properties and formats
        self.config = config

        # Check input output format:
        self.check_model()

    def create_model(self):
        """This function converts a deepforest config file into a model. An architecture should have a list of nested arguments in config that match this function"""
        raise ValueError(
            "The create_model class method needs to be implemented. Take in args and return a pytorch nn module."
        )

    def check_model(self):
        """
        Ensure that model follows deepforest guidelines, see #####
        If fails, raise ValueError
        """
        # This assumes model creation is not expensive
        test_model = self.create_model()
        test_model.eval()

        # Create a dummy batch of 3 band data.
        x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]

        predictions = test_model(x)
        # Model takes in a batch of images
        assert len(predictions) == 2

        # Returns a list equal to number of images with proper keys per image
        model_keys = list(predictions[1].keys())
        model_keys.sort()
        assert model_keys == ['boxes', 'labels', 'scores']
