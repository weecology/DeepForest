# Model - common class
from deepforest.models import *
import torch
from pytorch_lightning import LightningModule, Trainer
import os
import torchmetrics
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import numpy as np
import rasterio
from torch.utils.data import Dataset
import torch.nn.functional as F
import cv2


class Model():
    """A architecture agnostic class that controls the basic train, eval and
    predict functions. A model should optionally allow a backbone for
    pretraining. To add new architectures, simply create a new module in
    models/ and write a create_model. Then add the result to the if else
    statement below.

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
        """This function converts a deepforest config file into a model.

        An architecture should have a list of nested arguments in config
        that match this function
        """

        raise ValueError(
            "The create_model class method needs to be implemented. Take in args and return a pytorch nn module."
        )

    def check_model(self):
        """Ensure that model follows deepforest guidelines, see ##### If fails,
        raise ValueError."""
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


def simple_resnet_50(num_classes=2):
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = m.fc.in_features
    m.fc = torch.nn.Linear(num_ftrs, num_classes)

    return m


class CropModel(LightningModule):
    """A PyTorch Lightning module for classifying image crops from object
    detection models.

    This class provides a flexible architecture for training classification models on cropped
    regions identified by object detection models. It supports using either a default ResNet-50
    model or a custom provided model.

    Args:
        num_classes (int, optional): Number of classes for classification. If None, it will be inferred from the checkpoint during loading.
        batch_size (int, optional): Batch size for training. Defaults to 4.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 0.
        lr (float, optional): Learning rate for optimization. Defaults to 0.0001.
        model (nn.Module, optional): Custom PyTorch model to use. If None, uses ResNet-50. Defaults to None.
        label_dict (dict, optional): Mapping of class labels to numeric indices. Defaults to None.

    Attributes:
        model (nn.Module): The classification model (ResNet-50 or custom)
        accuracy (torchmetrics.Accuracy): Per-class accuracy metric
        total_accuracy (torchmetrics.Accuracy): Overall accuracy metric
        precision_metric (torchmetrics.Precision): Precision metric
        metrics (torchmetrics.MetricCollection): Collection of all metrics
        batch_size (int): Batch size for training
        num_workers (int): Number of data loading workers
        lr (float): Learning rate
        label_dict (dict): Label to index mapping {"Bird": 0, "Mammal": 1}
    """

    def __init__(self,
                 num_classes=None,
                 batch_size=4,
                 num_workers=0,
                 lr=0.0001,
                 model=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.numeric_to_label_dict = None
        self.save_hyperparameters()

        if num_classes is not None:
            if model is None:
                self.model = simple_resnet_50(num_classes=num_classes)
            else:
                self.model = model

            self.accuracy = torchmetrics.Accuracy(average='none',
                                                  num_classes=num_classes,
                                                  task="multiclass")
            self.total_accuracy = torchmetrics.Accuracy(num_classes=num_classes,
                                                        task="multiclass")
            self.precision_metric = torchmetrics.Precision(num_classes=num_classes,
                                                           task="multiclass")
            self.metrics = torchmetrics.MetricCollection({
                "Class Accuracy": self.accuracy,
                "Accuracy": self.total_accuracy,
                "Precision": self.precision_metric
            })
        else:
            self.model = model

        # Training Hyperparameters
        self.batch_size = batch_size
        self.lr = lr

    def on_save_checkpoint(self, checkpoint):
        checkpoint['label_dict'] = self.label_dict

    def create_trainer(self, **kwargs):
        """Create a pytorch lightning trainer object."""
        self.trainer = Trainer(**kwargs)

    def on_load_checkpoint(self, checkpoint):
        # Now that self.num_classes always exists, this check won't error
        if self.num_classes is None:
            self.num_classes = checkpoint['hyper_parameters']['num_classes']
        if self.model is None:
            self.model = simple_resnet_50(num_classes=self.num_classes)
        self.accuracy = torchmetrics.Accuracy(average='none',
                                              num_classes=self.num_classes,
                                              task="multiclass")
        self.total_accuracy = torchmetrics.Accuracy(num_classes=self.num_classes,
                                                    task="multiclass")
        self.precision_metric = torchmetrics.Precision(num_classes=self.num_classes,
                                                       task="multiclass")
        self.metrics = torchmetrics.MetricCollection({
            "Class Accuracy": self.accuracy,
            "Accuracy": self.total_accuracy,
            "Precision": self.precision_metric
        })
        self.label_dict = checkpoint['label_dict']

    def load_from_disk(self, train_dir, val_dir):
        self.train_ds = ImageFolder(root=train_dir,
                                    transform=self.get_transform(augment=True))
        self.val_ds = ImageFolder(root=val_dir,
                                  transform=self.get_transform(augment=False))
        self.label_dict = self.train_ds.class_to_idx
        # Create a reverse mapping from numeric indices to class labels
        self.numeric_to_label_dict = {v: k for k, v in self.label_dict.items()}

    def get_transform(self, augment):
        """Returns the data transformation pipeline for the model.

        Args:
            augment (bool): Flag indicating whether to apply data augmentation.

        Returns:
            torchvision.transforms.Compose: The composed data transformation pipeline.
        """
        data_transforms = []
        data_transforms.append(transforms.ToTensor())
        data_transforms.append(self.normalize())
        data_transforms.append(transforms.Resize([224, 224]))
        if augment:
            data_transforms.append(transforms.RandomHorizontalFlip(0.5))
        return transforms.Compose(data_transforms)

    def expand_bbox_to_square(self, bbox, image_width, image_height):
        """Expand a bounding box to a square by extending the shorter side.

        Parameters:
        -----------
        bbox : list or tuple
            Bounding box in format [xmin, ymin, xmax, ymax]
        image_width : int
            Width of the original image
        image_height : int
            Height of the original image

        Returns:
        --------
        list
            Square bounding box in format [xmin, ymin, xmax, ymax]
        """
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin

        center_x = xmin + width / 2
        center_y = ymin + height / 2

        side_length = max(width, height)

        new_xmin = center_x - side_length / 2
        new_ymin = center_y - side_length / 2

        new_xmin = max(0, min(new_xmin, image_width - side_length))
        new_ymin = max(0, min(new_ymin, image_height - side_length))

        if side_length > image_width:
            side_length = image_width
            new_xmin = 0

        if side_length > image_height:
            side_length = image_height
            new_ymin = 0

        new_xmax = new_xmin + side_length
        new_ymax = new_ymin + side_length

        return [new_xmin, new_ymin, new_xmax, new_ymax]

    def write_crops(self, root_dir, images, boxes, labels, savedir):
        """Write crops to disk.

        Args:
            root_dir (str): The root directory where the images are located.
            images (list): A list of image filenames.
            boxes (list): A list of bounding box coordinates in the format [xmin, ymin, xmax, ymax].
            labels (list): A list of labels corresponding to each bounding box.
            savedir (str): The directory where the cropped images will be saved.

        Returns:
            None
        """

        # Create a directory for each label
        for label in labels:
            os.makedirs(os.path.join(savedir, label), exist_ok=True)

        # Use rasterio to read the image
        for index, box in enumerate(boxes):
            label = labels[index]
            image = images[index]
            with rasterio.open(os.path.join(root_dir, image)) as src:
                # Get image dimensions
                image_width = src.width
                image_height = src.height

                # Expand the bounding box to a square
                square_box = self.expand_bbox_to_square(box, image_width, image_height)
                xmin, ymin, xmax, ymax = square_box

                # Crop the image using the square box coordinates
                img = src.read(window=((int(ymin), int(ymax)), (int(xmin), int(xmax))))
                # Save the cropped image as a PNG file using opencv
                image_basename = os.path.splitext(os.path.basename(image))[0]
                img_path = os.path.join(savedir, label, f"{image_basename}_{index}.png")
                img = np.rollaxis(img, 0, 3)
                cv2.imwrite(img_path, img)

    def normalize(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        if self.model is None:
            raise AttributeError(
                "CropModel is not initialized. Provide 'num_classes' or load from a checkpoint."
            )
        output = self.model(x)

        return output

    def train_dataloader(self):
        """Train data loader."""
        train_loader = torch.utils.data.DataLoader(self.train_ds,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=self.num_workers)

        return train_loader

    def predict_dataloader(self, ds):
        """Prediction data loader."""
        loader = torch.utils.data.DataLoader(ds,
                                             batch_size=self.batch_size,
                                             shuffle=False,
                                             num_workers=self.num_workers)

        return loader

    def val_dataloader(self):
        """Validation data loader."""
        val_loader = torch.utils.data.DataLoader(self.val_ds,
                                                 batch_size=self.batch_size,
                                                 num_workers=self.num_workers)

        return val_loader

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = F.cross_entropy(outputs, y)
        self.log("train_loss", loss)

        return loss

    def predict_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        yhat = F.softmax(outputs, 1)

        return yhat

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = F.cross_entropy(outputs, y)
        self.log("val_loss", loss)
        metric_dict = self.metrics(outputs, y)
        for key, value in metric_dict.items():
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                for i, v in enumerate(value):
                    self.log(f"{key}_{i}", v, on_step=False, on_epoch=True)
            else:
                self.log(key, value, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.5,
                                                               patience=10,
                                                               threshold=0.0001,
                                                               threshold_mode='rel',
                                                               cooldown=0,
                                                               min_lr=0,
                                                               eps=1e-08)

        # Monitor rate is val data is used
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, "monitor": 'val_loss'}

    def dataset_confusion(self, loader):
        """Create a confusion matrix from a data loader."""
        true_class = []
        predicted_class = []
        self.eval()
        for batch in loader:
            x, y = batch
            true_class.append(F.one_hot(y, num_classes=self.num_classes).detach().numpy())
            prediction = self(x)
            predicted_class.append(prediction.detach().numpy())

        true_class = np.concatenate(true_class)
        predicted_class = np.concatenate(predicted_class)

        return true_class, predicted_class
