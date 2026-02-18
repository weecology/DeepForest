# Model - common class
import json
import os

import cv2
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import torchmetrics
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import LightningModule, Trainer
from safetensors.torch import load_file
from torchvision import models, transforms

from deepforest import utilities
from deepforest.datasets.training import create_aligned_image_folders


class BaseModel:
    """Base class for DeepForest models.

    Provides common train, eval, and predict functionality.
    To add new architectures, create a module in models/ and implement create_model().

    Args:
        config: DeepForest configuration object
    """

    def __init__(self, config) -> None:
        # Check for required properties and formats
        self.config = config

    def create_model(self) -> torch.nn.Module:
        """Create model from configuration.

        Must be implemented by subclasses to return a PyTorch nn.Module.
        """

        raise ValueError(
            "The create_model class method needs to be implemented. "
            "Take in args and return a pytorch nn module."
        )

    def check_model(self) -> None:
        """Validate model follows DeepForest guidelines.

        Tests model with dummy data to ensure proper input/output
        format. Raises ValueError if validation fails.
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
        assert model_keys == ["boxes", "labels", "scores"]


_CROP_BACKBONES: dict[str, tuple] = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
}


def create_crop_backbone(
    architecture: str = "resnet50",
    num_classes: int = 2,
    pretrained: bool = True,
) -> torch.nn.Module:
    """Create a classification backbone for :class:`CropModel`.

    Args:
        architecture: One of the keys in ``_CROP_BACKBONES``
            (currently ``"resnet18"`` or ``"resnet50"``).
        num_classes: Number of output classes for the final layer.
        pretrained: Whether to load ImageNet-pretrained weights.

    Returns:
        torch.nn.Module: Model with the final FC layer adjusted to
        ``num_classes``.

    Raises:
        ValueError: If ``architecture`` is not recognized.
    """
    if architecture not in _CROP_BACKBONES:
        raise ValueError(
            f"Unknown CropModel architecture '{architecture}'. "
            f"Choose from {sorted(_CROP_BACKBONES)}."
        )
    factory, default_weights = _CROP_BACKBONES[architecture]
    m = factory(weights=default_weights if pretrained else None)
    num_ftrs = m.fc.in_features
    m.fc = torch.nn.Linear(num_ftrs, num_classes)
    return m


def simple_resnet_50(num_classes: int = 2) -> torch.nn.Module:
    """Create a simple ResNet-50 model for classification.

    .. deprecated::
        Use :func:`create_crop_backbone` instead.

    Args:
        num_classes: Number of output classes for the final layer

    Returns:
        torch.nn.Module: ResNet-50 model with modified final layer
    """
    return create_crop_backbone("resnet50", num_classes=num_classes)


class CropModel(LightningModule, PyTorchModelHubMixin):
    """A PyTorch Lightning module for classifying image crops from object
    detection models.

    This class provides a flexible architecture for training classification models on cropped
    regions identified by object detection models. It supports using either a default ResNet-50
    model or a custom provided model.

    Args:
        model (nn.Module, optional): Custom PyTorch model to use. If None, uses ResNet-50. Defaults to None.
        config (DictConfig, optional): Full configuration object. If None, loads default config. Defaults to None.
        config_args (dict, optional): Dictionary to override cropmodel config settings (e.g., {"resize": [300, 300], "balance_classes": True}). Defaults to None.

    Attributes:
        model (nn.Module): The classification model (ResNet-50 or custom)
        accuracy (torchmetrics.Accuracy): Per-class accuracy metric
        total_accuracy (torchmetrics.Accuracy): Overall accuracy metric
        precision_metric (torchmetrics.Precision): Precision metric
        metrics (torchmetrics.MetricCollection): Collection of all metrics
        label_dict (dict): Label to index mapping {"Bird": 0, "Mammal": 1}
    """

    def __init__(
        self,
        model=None,
        config=None,
        config_args: dict | None = None,
    ):
        super().__init__()

        self.model = model
        # Set the argument as the self.config, this way when reloading the checkpoint, self.config exists and is not overwritten.
        self.config = config
        if self.config is None:
            if config_args is None:
                # If not provided, load default config via OmegaConf.
                self.config = utilities.load_config()
            else:
                self.config = utilities.load_config(overrides={"cropmodel": config_args})

        if self.config["cropmodel"]["balance_classes"]:
            self._sampler_type = "weighted_random"
        else:
            self._sampler_type = "random"

        self.save_hyperparameters()

    def on_save_checkpoint(self, checkpoint):
        # In case the label dict has been updated on self.load_from_disk, save the hyperparameters
        checkpoint["label_dict"] = self.label_dict

    def on_load_checkpoint(self, checkpoint):
        # Recreate the model architecture BEFORE state_dict is loaded so keys match
        self.label_dict = checkpoint["label_dict"]
        self.numeric_to_label_dict = {v: k for k, v in self.label_dict.items()}
        num_classes = len(self.label_dict)
        self.create_model(num_classes)

    def create_model(self, num_classes: int, architecture: str | None = None):
        """Create classification backbone and metrics for ``num_classes``.

        Args:
            num_classes: Number of output classes.
            architecture: Backbone name (e.g. ``"resnet18"``, ``"resnet50"``).
                Falls back to ``self.config["cropmodel"]["architecture"]``
                then ``"resnet50"``.
        """
        if architecture is None:
            architecture = self.config["cropmodel"].get("architecture", "resnet50")

        self.accuracy = torchmetrics.Accuracy(
            average="none", num_classes=num_classes, task="multiclass"
        )
        self.total_accuracy = torchmetrics.Accuracy(
            num_classes=num_classes, task="multiclass"
        )
        self.precision_metric = torchmetrics.Precision(
            num_classes=num_classes, task="multiclass"
        )
        self.macro_precision = torchmetrics.Precision(
            num_classes=num_classes, task="multiclass", average="macro"
        )
        self.metrics = torchmetrics.MetricCollection(
            {
                "Class Accuracy": self.accuracy,
                "Accuracy": self.total_accuracy,
                "Precision": self.precision_metric,
                "Macro Precision": self.macro_precision,
            }
        )

        self.model = create_crop_backbone(
            architecture=architecture,
            num_classes=num_classes,
        )

    def create_trainer(self, **kwargs):
        """Create a pytorch lightning trainer object."""
        self.trainer = Trainer(**kwargs)

    def load_from_disk(self, train_dir, val_dir):
        """Load the training and validation datasets from disk.

        Args:
            train_dir (str): The directory containing the training dataset.
            val_dir (str): The directory containing the validation dataset.

        Returns:
            None
        """
        self.train_ds, self.val_ds = create_aligned_image_folders(
            train_dir,
            val_dir,
            transform_train=self.get_transform(augmentations=["HorizontalFlip"]),
            transform_val=self.get_transform(augmentations=None),
        )
        self.label_dict = self.train_ds.class_to_idx

        # Create a reverse mapping from numeric indices to class labels
        self.numeric_to_label_dict = {v: k for k, v in self.label_dict.items()}

        self.num_classes = len(self.label_dict)

        if self.model is None:
            self.create_model(self.num_classes)

    def get_transform(self, augmentations):
        """Returns the data transformation pipeline for the model.

        Args:
            augmentations (str, list, dict, optional): Augmentation configuration.
                If None, no augmentations are applied. If "HorizontalFlip" or
                ["HorizontalFlip"], applies random horizontal flip.

        Returns:
            torchvision.transforms.Compose: The composed data transformation pipeline.
        """
        data_transforms = []
        data_transforms.append(transforms.ToTensor())
        data_transforms.append(self.normalize())

        # Get resize dimensions from config, default to [224, 224] if not specified
        resize_dims = self.config["cropmodel"].get("resize", [224, 224])
        data_transforms.append(transforms.Resize(resize_dims))

        # Apply augmentations if specified
        if augmentations is not None:
            if isinstance(augmentations, str) and augmentations == "HorizontalFlip":
                data_transforms.append(transforms.RandomHorizontalFlip(0.5))
            elif isinstance(augmentations, list) and "HorizontalFlip" in augmentations:
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
        sampler = None
        shuffle = True

        # Optional class balancing using WeightedRandomSampler
        if self.config["cropmodel"]["balance_classes"]:
            # Compute class counts and inverse-frequency weights per sample
            counts = {}
            for t in self.train_ds.targets:
                counts[t] = counts.get(t, 0) + 1

            weights = [1.0 / counts[t] for t in self.train_ds.targets]
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights, num_samples=len(weights), replacement=True
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.config["cropmodel"]["batch_size"],
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config["cropmodel"]["num_workers"],
        )

        return train_loader

    def predict_dataloader(self, ds):
        """Prediction data loader."""
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.config["cropmodel"]["batch_size"],
            shuffle=False,
            num_workers=self.config["cropmodel"]["num_workers"],
        )

        return loader

    def val_dataloader(self):
        """Validation data loader."""
        val_loader = torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.config["cropmodel"]["batch_size"],
            num_workers=self.config["cropmodel"]["num_workers"],
        )

        return val_loader

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = F.cross_entropy(outputs, y)
        self.log("train_loss", loss)

        return loss

    def predict_step(self, batch, batch_idx):
        # Check if batch is a tuple for validation_dataloader
        if isinstance(batch, list):
            x, y = batch
        else:
            x = batch
        outputs = self.forward(x)
        yhat = F.softmax(outputs, 1)

        return yhat

    def postprocess_predictions(self, predictions):
        """Postprocess predictions to get class labels and scores."""
        stacked_outputs = np.vstack(np.concatenate(predictions))
        label = np.argmax(stacked_outputs, axis=1)  # Get class with highest probability
        score = np.max(stacked_outputs, axis=1)  # Get confidence score

        return label, score

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = F.cross_entropy(outputs, y)
        self.log("val_loss", loss)

        predictions = torch.argmax(outputs, dim=1)
        self.metrics.update(predictions, y)

        return loss

    def on_validation_epoch_end(self):
        metric_dict = self.metrics.compute()
        # Only log per-class metrics when there are multiple classes
        if len(self.numeric_to_label_dict) > 1:
            for index, value in enumerate(metric_dict["Class Accuracy"]):
                key = self.numeric_to_label_dict[index]
                metric_name = f"Class Accuracy_{key}"
                self.log(metric_name, value, on_step=False, on_epoch=True)

        self.log(
            "Micro-Average Accuracy",
            metric_dict["Accuracy"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "Micro-Average Precision",
            metric_dict["Precision"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "Macro-Average Precision",
            metric_dict["Macro Precision"],
            on_step=False,
            on_epoch=True,
        )
        self.metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["cropmodel"]["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
        )

        # Monitor rate is val data is used
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def val_dataset_confusion(self, return_images=False):
        """Create a labels and predictions from the validation dataset to be
        created into a confusion matrix."""
        dl = self.predict_dataloader(self.val_ds)
        # ensure fast_dev_run is False
        self.trainer.fast_dev_run = False
        predictions = self.trainer.predict(self, dl)
        predicted_label, _ = self.postprocess_predictions(predictions)
        true_label = [self.val_ds[i][1] for i in range(len(self.val_ds))]
        if return_images:
            images = [
                Image.open(self.val_ds.imgs[i][0]) for i in range(len(self.val_ds.imgs))
            ]
            return images, true_label, predicted_label
        else:
            return true_label, predicted_label

    @classmethod
    def load_model(
        cls,
        repo_id,
        revision=None,
    ):
        """Load a model from the Hugging Face Hub.

        Args:
            repo_id: Hugging Face repo id, e.g. "username/my-cropmodel".
            revision: Optional git revision/branch/tag. Defaults to repo default.

        Returns:
            CropModel: The loaded and eval-mode model instance.
        """

        model = cls.from_pretrained(
            repo_id,
            revision=revision,
        )
        model.eval()

        return model

    def push_to_hub_in_memory(self, repo_id, **kwargs):
        """Push the model to the Hugging Face Hub.

        Args:
            repo_id: Hugging Face repo id, e.g. "username/my-cropmodel".
            **kwargs: Additional arguments to pass to the push_to_hub method.
        """
        config = OmegaConf.to_container(self.config, resolve=True, enum_to_str=True)
        config["cropmodel"]["label_dict"] = self.label_dict
        config["cropmodel"].setdefault("architecture", "resnet50")
        super().push_to_hub(repo_id, **kwargs, config=config)

    def push_to_hub(self, repo_id, commit_message="Add model", **kwargs):
        return self.push_to_hub_in_memory(
            repo_id, commit_message=commit_message, **kwargs
        )

    @classmethod
    def from_pretrained(cls, repo_id: str, revision: str | None = None):
        """Load a model from the Hugging Face Hub.

        Reads config.json to determine architecture and labels, creates
        the model, then loads weights.

        Args:
            repo_id: Hugging Face repo id, e.g. "username/my-cropmodel".
            revision: Optional git revision, branch, or tag. Defaults to
                the repo default branch.
        """
        # Download config to determine architecture and labels before loading weights
        cfg_path = hf_hub_download(repo_id, "config.json", revision=revision)
        with open(cfg_path) as f:
            cfg = json.load(f)

        label_dict = {k: int(v) for k, v in cfg["cropmodel"]["label_dict"].items()}
        num_classes = len(label_dict)
        architecture = cfg["cropmodel"].get("architecture", "resnet50")

        # Create instance with correct architecture before loading weights
        instance = cls.__new__(cls)
        LightningModule.__init__(instance)

        instance.config = cfg
        if cfg["cropmodel"].get("balance_classes", False):
            instance._sampler_type = "weighted_random"
        else:
            instance._sampler_type = "random"

        instance.create_model(num_classes, architecture=architecture)

        # Load weights
        model_path = hf_hub_download(repo_id, "model.safetensors", revision=revision)
        state_dict = load_file(model_path)
        instance.load_state_dict(state_dict, strict=True)

        instance.label_dict = label_dict
        instance.numeric_to_label_dict = {v: k for k, v in label_dict.items()}
        instance.num_classes = num_classes
        instance.eval()

        return instance
