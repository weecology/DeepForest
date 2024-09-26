# Extending DeepForest with Custom Models and Dataloaders

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

.. note::
   During training, the model expects both the input tensors and targets (list of dictionary), containing:
   boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
   labels (Int64Tensor[N]): the class label for each ground-truth box
   During inference, the model requires only the input tensors, and returns the post-processed predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as follows, where N is the number of detections:
   boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
   labels (Int64Tensor[N]): the predicted labels for each detection
   scores (Tensor[N]): the scores of each detection
```

# Custom Dataloaders

For model training, evaluation, and prediction, we usually let DeepForest create dataloaders with augmentations and formatting starting from a CSV of annotations for training and evaluation or image paths for prediction. That works well, but what happens if your data is already in the form of a PyTorch dataloader? There are a number of emerging benchmarks (e.g., [WILDS](https://github.com/p-lambda/wilds)) that skip the finicky steps of data preprocessing and just yield data directly. We can pass dataloaders directly to DeepForest functions. Because this is a custom route, we leave the responsibility of formatting the data properly to users; see [dataset.TreeDataset](https://deepforest.readthedocs.io/en/latest/source/deepforest.html#deepforest.dataset.TreeDataset) for an example. Any dataloader that meets the needed requirements could be used. It's important to note that every time a dataloader is updated, the `create_trainer()` method needs to be called to update the rest of the object.
For train/test
```
m = main.deepforest()
existing_loader = m.load_dataset(csv_file=m.config["train"]["csv_file"],
                                root_dir=m.config["train"]["root_dir"],
                                batch_size=m.config["batch_size"])

# Can be passed directly to main.deepforest(existing_train_dataloader) or reassign to existing deepforest object
m.existing_train_dataloader_loader
m.create_trainer()
m.trainer.fit()
```

For prediction directly on a dataloader, we need a dataloader that yields images, see [TileDataset](https://deepforest.readthedocs.io/en/latest/source/deepforest.html#deepforest.dataset.TileDataset) for an example. Any dataloader could be supplied to m.trainer.predict as long as it meets this specification.  

```
ds = dataset.TileDataset(tile=np.random.random((400,400,3)).astype("float32"), patch_overlap=0.1, patch_size=100)
existing_loader = m.predict_dataloader(ds)

batches = m.trainer.predict(m, existing_loader)
len(batches[0]) == m.config["batch_size"] + 1
```