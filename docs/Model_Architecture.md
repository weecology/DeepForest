# Model Architecture

DeepForest allows users to specify custom model architectures if they follow certain guidelines. 
To create a compliant format, follow the recipe below.

## Subclass the model.Model() structure

A subclass is a class instance that inherits the methods and function of super classes. In this cases, model.Model() is defined as:

```
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
    
    def create_model():
        """This function converts a deepforest config file into a model. An architecture should have a list of nested arguments in config that match this function"""
        raise ValueError("The create_model class method needs to be implemented. Take in args and return a pytorch nn module.")
    
    def check_model(self):
        """
        Ensure that model follows deepforest guidelines
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
        assert model_keys == ['boxes','labels','scores']
```

## Match torchvision formats

From this definition we can see three format requirements. The model must be able to take in a batch of images in the order [channels, height, width]. The current model weights are trained on 3 band images, but you can update the check_model function if you have other image dimensions.
The second requirement is that the model ouputs a dictionary with keys ["boxes","labels","scores"], the boxes are formatted following torchvision object detection format. From the [docs](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.retinanet_resnet50_fpn.html#torchvision.models.detection.retinanet_resnet50_fpn)

```
During training, the model expects both the input tensors and targets (list of dictionary), containing:

boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.

labels (Int64Tensor[N]): the class label for each ground-truth box

During inference, the model requires only the input tensors, and returns the post-processed predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as follows, where N is the number of detections:

boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.

labels (Int64Tensor[N]): the predicted labels for each detection

scores (Tensor[N]): the scores of each detection
```
