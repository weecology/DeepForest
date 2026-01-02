# entry point for deepforest model
import importlib
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.exceptions import MisconfigurationException
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pytorch_lightning.callbacks import LearningRateMonitor
from torch import optim
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.detection import IntersectionOverUnion, MeanAveragePrecision

from deepforest import evaluate as evaluate_iou
from deepforest import predict, utilities
from deepforest.datasets import prediction, training


class deepforest(pl.LightningModule):
    """DeepForest model for tree crown detection in RGB images.

    Args:
        num_classes: Number of classes in the model
        model: DeepForest model object
        existing_train_dataloader: PyTorch dataloader for training data
        existing_val_dataloader: PyTorch dataloader for validation data
        config: DeepForest configuration object or name
        config_args: Dictionary of config overrides
    """

    def __init__(
        self,
        model=None,
        transforms=None,
        existing_train_dataloader=None,
        existing_val_dataloader=None,
        config: str | dict | DictConfig | None = None,
        config_args: dict | None = None,
    ):
        super().__init__()

        if config is None:
            config = utilities.load_config(overrides=config_args)
        # Default/string config name
        elif isinstance(config, str):
            config = utilities.load_config(config_name=config, overrides=config_args)
        # Checkpoint load
        elif isinstance(config, dict):
            config = utilities.load_config(overrides=config)
        # Hub overrides
        elif "config_args" in config:
            config = utilities.load_config(overrides=config["config_args"])
        elif config_args is not None:
            warnings.warn(
                f"Ignoring options as configuration object was provided: {config_args}",
                stacklevel=2,
            )

        self.config = config

        # release version id to flag if release is being used
        self.__release_version__ = None

        self.existing_train_dataloader = existing_train_dataloader
        self.existing_val_dataloader = existing_val_dataloader

        # Metrics
        self.iou_metric = IntersectionOverUnion(
            class_metrics=True, iou_threshold=self.config.validation.iou_threshold
        )
        self.mAP_metric = MeanAveragePrecision(backend="faster_coco_eval")

        # Empty frame accuracy
        self.empty_frame_accuracy = BinaryAccuracy()

        # Create a default trainer.
        self.create_trainer()

        self.model = model

        if self.model is None:
            self.create_model()

        # Add user supplied transforms
        if transforms is None:
            self.transforms = None
        else:
            self.transforms = transforms

        self.save_hyperparameters(
            {"config": OmegaConf.to_container(self.config, resolve=True)}
        )

    def load_model(self, model_name=None, revision=None):
        """Loads a model that has already been pretrained for a specific task,
        like tree crown detection.

        Models (technically model weights) are distributed via Hugging Face
        and designated the Hugging Face repository ID (model_name), which
        is in the form: 'organization/repository'. For a list of models distributed
        by the DeepForest team (and the associated model names) see the
        documentation:
        https://deepforest.readthedocs.io/en/latest/installation_and_setup/prebuilt.html

        Args:
            model_name (str): A repository ID for huggingface in the form of organization/repository
            revision (str): The model version ('main', 'v1.0.0', etc.).

        Returns:
            None
        """

        if model_name is None:
            model_name = self.config.model.name

        if revision is None:
            revision = self.config.model.revision

        model_class = importlib.import_module(
            f"deepforest.models.{self.config.architecture}"
        )
        self.model = model_class.Model(config=self.config).create_model(
            pretrained=model_name, revision=revision
        )

        # Handle label override
        cfg_labels = self.config.label_dict
        model_labels = self.model.label_dict

        # If user specified labels, and they differ from the model:
        if cfg_labels != model_labels:
            warnings.warn(
                "Your supplied label dict differs from the model. "
                "This is expected if you plan to fine-tune this model on your own data.",
                stacklevel=2,
            )
            label_dict = cfg_labels
        else:
            label_dict = model_labels

        self.set_labels(label_dict)

        return

    def set_labels(self, label_dict):
        """Set new label mapping, updating both the label dictionary (str ->
        int) and its inverse (int -> str).

        Args:
            label_dict (dict): Dictionary mapping class names to numeric IDs.
        """
        if label_dict is None:
            raise ValueError(
                "Label dictionary not found. Check it was set in your config file or config_args."
            )

        # Label encoder and decoder
        if not len(label_dict) == self.config.num_classes:
            raise ValueError(
                f"label_dict {label_dict} does not match requested number of "
                f"classes {self.config.num_classes}, please supply a label_dict argument "
                '{"label1":0, "label2":1, "label3":2 ... etc} '
                "for each label in the "
                "dataset"
            )

        # Check for duplicate values in label_dict:
        if len(set(label_dict.values())) != len(label_dict):
            raise ValueError("Found duplicate label IDs in label_dict.")

        self.label_dict = label_dict
        self.numeric_to_label_dict = {v: k for k, v in label_dict.items()}

    def create_model(self, initialize_model=False):
        """Initialize a deepforest architecture. This can be done in two ways.
        Passed as the model argument to deepforest __init__(), or as a named
        architecture in config.architecture, which corresponds to a file in
        models/, as is a subclass of model.Model(). The config args in the
        .yaml are specified.

        Returns:
            None
        """
        if self.config.model.name is None or initialize_model:
            model_class = importlib.import_module(
                f"deepforest.models.{self.config.architecture}"
            )
            self.model = model_class.Model(config=self.config).create_model()
            self.set_labels(self.config.label_dict)
        else:
            self.load_model()

    def create_trainer(self, logger=None, callbacks=None, **kwargs):
        """Create a pytorch lightning training by reading config files.

        Args:
            logger: Optional logger
            callbacks: Optional list of callbacks
            **kwargs: Additional trainer arguments
        """
        if callbacks is None:
            callbacks = []
        # If val data is passed, monitor learning rate and setup classification metrics
        if self.config.validation.csv_file is not None:
            if logger is not None:
                lr_monitor = LearningRateMonitor(logging_interval="epoch")
                callbacks.append(lr_monitor)
            limit_val_batches = 1.0
            num_sanity_val_steps = 2
        else:
            # Disable validation, don't use trainer defaults
            limit_val_batches = 0
            num_sanity_val_steps = 0

        # Check for model checkpoint object
        checkpoint_types = [type(x).__qualname__ for x in callbacks]
        if "ModelCheckpoint" in checkpoint_types:
            enable_checkpointing = True
        else:
            enable_checkpointing = False

        trainer_args = {
            "logger": logger,
            "max_epochs": self.config.train.epochs,
            "enable_checkpointing": enable_checkpointing,
            "devices": self.config.devices,
            "accelerator": self.config.accelerator,
            "fast_dev_run": self.config.train.fast_dev_run,
            "callbacks": callbacks,
            "limit_val_batches": limit_val_batches,
            "num_sanity_val_steps": num_sanity_val_steps,
        }
        # Update with kwargs to allow them to override config
        trainer_args.update(kwargs)

        self.trainer = pl.Trainer(**trainer_args)

    def on_fit_start(self):
        if self.config.train.csv_file is None:
            raise AttributeError(
                "Cannot train with a train annotations file, "
                "please set 'config['train']['csv_file'] before "
                "calling deepforest.create_trainer()'"
            )

    def on_save_checkpoint(self, checkpoint):
        # Update hparams in case they've changed since init
        checkpoint["hyper_parameters"]["config"] = OmegaConf.to_container(
            self.config, resolve=True
        )
        checkpoint["label_dict"] = self.label_dict
        checkpoint["numeric_to_label_dict"] = self.numeric_to_label_dict

        for key in checkpoint:
            if isinstance(checkpoint[key], DictConfig):
                checkpoint[key] = OmegaConf.to_container(checkpoint[key], resolve=True)

    def on_load_checkpoint(self, checkpoint):
        try:
            self.label_dict = checkpoint["label_dict"]
            self.numeric_to_label_dict = checkpoint["numeric_to_label_dict"]
        except KeyError:
            print(
                "No label_dict found in checkpoint, using default label_dict, "
                "please use deepforest.set_labels() to set the label_dict after loading the checkpoint."
            )
        # Pre 2.0 compatibility, the score_threshold used to be stored under retinanet.score_thresh
        try:
            self.config.score_thresh = self.config.retinanet.score_thresh
        except AttributeError:
            pass

        if not hasattr(self.config.validation, "lr_plateau_target"):
            default_config = utilities.load_config()
            self.config.validation.lr_plateau_target = (
                default_config.validation.lr_plateau_target
            )

        if not hasattr(self.config.train, "augmentations"):
            default_config = utilities.load_config()
            self.config.train.augmentations = default_config.train.augmentations

        if not hasattr(self.config.validation, "augmentations"):
            default_config = utilities.load_config()
            self.config.validation.augmentations = default_config.validation.augmentations

    def save_model(self, path):
        """Save the trainer checkpoint in user defined path, in order to access
        in future.

        Args:
            Path: the path located the model checkpoint
        """
        self.trainer.save_checkpoint(path)

    def load_dataset(
        self,
        csv_file,
        root_dir=None,
        shuffle=True,
        transforms=None,
        augmentations=None,
        preload_images=False,
        batch_size=1,
    ):
        """Create a dataset for inference or training. Csv file format is .csv
        file with the columns "image_path", "xmin","ymin","xmax","ymax" for the
        image name and bounding box position. Image_path is the relative
        filename, not absolute path, which is in the root_dir directory. One
        bounding box per line.

        Args:
            csv_file: path to csv file
            root_dir: directory of images. If none, uses "image_dir" in config
            transforms: Albumentations transforms
            batch_size: batch size
            preload_images: if True, preload the images into memory
            augmentations: augmentation configuration (str, list, or dict)
        Returns:
            ds: a pytorch dataset
        """

        ds = training.BoxDataset(
            csv_file=csv_file,
            root_dir=root_dir,
            transforms=transforms,
            label_dict=self.label_dict,
            augmentations=augmentations,
            preload_images=preload_images,
        )
        if len(ds) == 0:
            raise ValueError(
                f"Dataset from {csv_file} is empty. Check CSV for valid entries and columns."
            )

        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=ds.collate_fn,
            num_workers=self.config.workers,
        )

        return data_loader

    def train_dataloader(self):
        """Train loader using the configurations.

        Returns:
            loader
        """
        if self.existing_train_dataloader:
            return self.existing_train_dataloader

        loader = self.load_dataset(
            csv_file=self.config.train.csv_file,
            root_dir=self.config.train.root_dir,
            augmentations=self.config.train.augmentations,
            preload_images=self.config.train.preload_images,
            shuffle=True,
            transforms=self.transforms,
            batch_size=self.config.batch_size,
        )

        return loader

    def val_dataloader(self):
        """Create a val data loader only if specified in config.

        Returns:
            a dataloader or a empty iterable.
        """
        # The preferred route for skipping validation is now (pl-2.0) an empty list,
        # see https://github.com/Lightning-AI/lightning/issues/17154
        loader = []

        if self.existing_val_dataloader:
            return self.existing_val_dataloader

        if self.config.validation.csv_file is not None:
            loader = self.load_dataset(
                csv_file=self.config.validation.csv_file,
                root_dir=self.config.validation.root_dir,
                augmentations=self.config.validation.augmentations,
                shuffle=False,
                preload_images=self.config.validation.preload_images,
                batch_size=self.config.batch_size,
            )

        return loader

    def predict_dataloader(self, ds, batch_size=None):
        """Create a PyTorch dataloader for prediction.

        Args:
            ds (torchvision.datasets.Dataset): A torchvision dataset to be wrapped into a dataloader using config args.

        Returns:
            torch.utils.data.DataLoader: A dataloader object that can be used for prediction.
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        else:
            batch_size = batch_size
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.workers,
            collate_fn=ds.collate_fn,
        )
        return loader

    def predict_image(self, image: np.ndarray | None = None, path: str | None = None):
        """Predict a single image with a deepforest model.

        Args:
            image: a float32 numpy array of a RGB with channels last format
            path: optional path to read image from disk instead of passing image arg

        Returns:
            result: A pandas dataframe of predictions (Default)
        """
        # Ensure we are in eval mode
        self.model.eval()

        if path:
            image = np.array(Image.open(path).convert("RGB")).astype("float32")

        # sanity checks on input images
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Input image is of type {type(image)}, expected numpy, if reading "
                "from PIL, wrap in "
                "np.array(image).astype(float32)"
            )

        if image.dtype != "float32":
            warnings.warn(
                f"Image type is {image.dtype}, transforming to float32. "
                f"This assumes that the range of pixel values is 0-255, as "
                f"opposed to 0-1.To suppress this warning, transform image "
                f"(image.astype('float32')",
                stacklevel=2,
            )
            image = image.astype("float32")

        result = predict._predict_image_(
            model=self.model, image=image, path=path, nms_thresh=self.config.nms_thresh
        )

        # If there were no predictions, return None
        if result is None:
            return None
        else:
            result["label"] = result.label.apply(lambda x: self.numeric_to_label_dict[x])

        if path is None:
            result = utilities.read_file(result)
            warnings.warn(
                "An image was passed directly to predict_image, the result.root_dir attribute "
                "will be None in the output dataframe, to use visualize.plot_results, "
                "please assign results.root_dir = <directory name>",
                stacklevel=2,
            )
        else:
            root_dir = os.path.dirname(path)
            result = utilities.read_file(result, root_dir=root_dir)

        return result

    def predict_file(
        self, csv_file, root_dir, crop_model=None, size=None, batch_size=None
    ):
        """Create a dataset and predict entire annotation file CSV file format
        is .csv file with the columns "image_path", "xmin","ymin","xmax","ymax"
        for the image name and bounding box position. Image_path is the
        relative filename, not absolute path, which is in the root_dir
        directory. One bounding box per line.

        Args:
            csv_file: path to csv file
            root_dir: directory of images. If none, uses "image_dir" in config
            crop_model: a deepforest.model.CropModel object to predict on crops
            size: the size of the image to resize to. Optional, if not provided, the image is not resized.
        Returns:
            df: pandas dataframe with bounding boxes, label and scores for each image in the csv file
        """

        ds = prediction.FromCSVFile(csv_file=csv_file, root_dir=root_dir, size=size)
        dataloader = self.predict_dataloader(ds, batch_size=batch_size)
        results = predict._dataloader_wrapper_(
            model=self,
            crop_model=crop_model,
            trainer=self.trainer,
            dataloader=dataloader,
            root_dir=root_dir,
        )

        results.root_dir = root_dir

        return results

    def predict_tile(
        self,
        path=None,
        image=None,
        patch_size=400,
        patch_overlap=0.05,
        iou_threshold=0.15,
        dataloader_strategy="single",
        crop_model=None,
    ):
        """For images too large to input into the model, predict_tile cuts the
        image into overlapping windows, predicts trees on each window and
        reassambles into a single array.

        Args:
            path: Path or list of paths to images on disk. If a single string is provided, it will be converted to a list.
            image (array): Numpy image array in BGR channel order following openCV convention. Not possible in combination with dataloader_strategy='batch'.
            patch_size: patch size for each window
            patch_overlap: patch overlap among windows
            iou_threshold: Minimum iou overlap among predictions between windows to be suppressed
            dataloader_strategy: "single", "batch", or "window".
                - "Single" loads the entire image into memory and passes individual windows to GPU and cannot be parallelized.
                - "batch" loads the entire image into GPU memory and creates views of an image as batch, requires in the entire tile to fit into GPU memory. CPU parallelization is possible for loading images.
                - "window" loads only the desired window of the image from the raster dataset. Most memory efficient option, but cannot parallelize across windows.
            crop_model: a deepforest.model.CropModel object to predict on crops

        Returns:
            pd.DataFrame or tuple: Predictions dataframe or (predictions, crops) tuple
        """
        self.model.eval()
        self.model.nms_thresh = self.config.nms_thresh

        # Check if path or image is provided
        if dataloader_strategy == "single":
            if path is None and image is None:
                raise ValueError(
                    "Either path or image must be provided for single tile prediction"
                )

        if dataloader_strategy == "batch":
            if path is None:
                raise ValueError(
                    "path argument must be provided when using dataloader_strategy='batch'"
                )

        # Convert single path to list for consistent handling
        if isinstance(path, str):
            paths = [path]
        elif path is None:
            paths = [None]
        else:
            paths = path

        image_results = []
        if dataloader_strategy in ["single", "window"]:
            for image_path in paths:
                if dataloader_strategy == "single":
                    ds = prediction.SingleImage(
                        path=image_path,
                        image=image,
                        patch_overlap=patch_overlap,
                        patch_size=patch_size,
                    )
                else:
                    # Check for workers config when using out of memory dataset
                    if self.config.workers > 0:
                        raise ValueError(
                            "workers must be 0 when using out-of-memory dataset "
                            "(dataloader_strategy='window'). Set config['workers']=0 and recreate "
                            "trainer self.create_trainer()."
                        )
                    ds = prediction.TiledRaster(
                        path=image_path,
                        patch_overlap=patch_overlap,
                        patch_size=patch_size,
                    )

                batched_results = self.trainer.predict(self, self.predict_dataloader(ds))

                # Flatten list from batched prediction
                prediction_list = []
                for batch in batched_results:
                    for images in batch:
                        prediction_list.append(images)
                image_results.append(ds.postprocess(prediction_list))

            results = pd.concat(image_results)

        elif dataloader_strategy == "batch":
            ds = prediction.MultiImage(
                paths=paths, patch_overlap=patch_overlap, patch_size=patch_size
            )

            batched_results = self.trainer.predict(self, self.predict_dataloader(ds))

            # Flatten list from batched prediction
            prediction_list = []
            for batch in batched_results:
                for images in batch:
                    prediction_list.append(images)
            image_results.append(ds.postprocess(prediction_list))
            results = pd.concat(image_results)

        else:
            raise ValueError(f"Invalid dataloader_strategy: {dataloader_strategy}")

        if results.empty:
            warnings.warn("No predictions made, returning None", stacklevel=2)
            return None

        # Perform mosaic for each image_path, or all if image_path is None
        mosaic_results = []
        if results["image_path"].isnull().all():
            mosaic_results.append(predict.mosiac(results, iou_threshold=iou_threshold))
        else:
            for image_path in results["image_path"].unique():
                image_results = results[results["image_path"] == image_path]
                image_mosaic = predict.mosiac(image_results, iou_threshold=iou_threshold)
                image_mosaic["image_path"] = image_path
                mosaic_results.append(image_mosaic)

        mosaic_results = pd.concat(mosaic_results)
        mosaic_results["label"] = mosaic_results.label.apply(
            lambda x: self.numeric_to_label_dict[x]
        )

        if paths[0] is not None:
            root_dir = os.path.dirname(paths[0])
        else:
            print(
                "No image path provided, root_dir will be None, since either "
                "images were directly provided or there were multiple image paths"
            )
            root_dir = None

        if crop_model is not None:
            cropmodel_results = []
            for path in paths:
                image_result = mosaic_results[
                    mosaic_results.image_path == os.path.basename(path)
                ]
                if image_result.empty:
                    continue
                image_result.root_dir = os.path.dirname(path)
                cropmodel_result = predict._crop_models_wrapper_(
                    crop_model, self.trainer, image_result
                )
                cropmodel_results.append(cropmodel_result)
            cropmodel_results = pd.concat(cropmodel_results)
        else:
            cropmodel_results = mosaic_results

        formatted_results = utilities.read_file(cropmodel_results, root_dir=root_dir)

        return formatted_results

    def training_step(self, batch, batch_idx):
        """Train on a loaded dataset."""
        # Confirm model is in train mode
        self.model.train()

        # allow for empty data if data augmentation is generated
        images, targets, image_names = batch
        loss_dict = self.model.forward(images, targets)

        # sum of regression and classification loss
        losses = sum(loss_dict.values())

        # Log loss
        for key, value in loss_dict.items():
            self.log(
                f"train_{key}", value.detach(), on_epoch=True, batch_size=len(images)
            )

        # Log sum of losses
        self.log("train_loss", losses.detach(), on_epoch=True, batch_size=len(images))

        return losses

    def validation_step(self, batch, batch_idx):
        """Evaluate a batch."""
        images, targets, image_names = batch

        # Set model to train mode to return loss, but disable optimization.
        # Torchvision does not return loss in eval mode.
        self.model.train()
        with torch.no_grad():
            loss_dict = self.model.forward(images, targets)

        # sum of regression and classification loss
        losses = sum(loss_dict.values())

        # Log losses
        try:
            for key, value in loss_dict.items():
                self.log(
                    f"val_{key}", value.detach(), on_epoch=True, batch_size=len(images)
                )

            self.log("val_loss", losses.detach(), on_epoch=True, batch_size=len(images))
        except MisconfigurationException:
            pass

        # In eval model, return predictions to calculate prediction metrics
        preds = self.model.eval()
        with torch.no_grad():
            preds = self.model.forward(images, targets)

        if len(targets) > 0:
            # Remove empty targets and corresponding predictions
            filtered_preds = []
            filtered_targets = []
            for i, target in enumerate(targets):
                if target["boxes"].shape[0] > 0:
                    filtered_preds.append(preds[i])
                    filtered_targets.append(target)

            self.iou_metric.update(filtered_preds, filtered_targets)
            self.mAP_metric.update(filtered_preds, filtered_targets)

        # Log the predictions if you want to use them for evaluation logs
        for i, result in enumerate(preds):
            formatted_result = utilities.format_geometry(result)
            if formatted_result is not None:
                formatted_result["image_path"] = image_names[i]
                self.predictions.append(formatted_result)

        return losses

    def on_validation_epoch_start(self):
        self.predictions = []

    def calculate_empty_frame_accuracy(self, ground_df, predictions_df):
        """Calculate accuracy for empty frames (frames with no objects).

        Args:
            ground_df (pd.DataFrame): Ground truth dataframe containing image paths and bounding boxes.
                Must have columns 'image_path', 'xmin', 'ymin', 'xmax', 'ymax'.
            predictions_df (pd.DataFrame): Model predictions dataframe containing image paths and predicted boxes.
                Must have column 'image_path'.

        Returns:
            float or None: Accuracy score for empty frame detection. A score of 1.0 means the model correctly
                identified all empty frames (no false positives), while 0.0 means it predicted objects
                in all empty frames (all false positives). Returns None if there are no empty frames.
        """
        # Find images that are marked as empty in ground truth (all coordinates are 0)
        empty_images = ground_df.loc[
            (ground_df.xmin == 0)
            & (ground_df.ymin == 0)
            & (ground_df.xmax == 0)
            & (ground_df.ymax == 0),
            "image_path",
        ].unique()

        if len(empty_images) == 0:
            return None

        if predictions_df.empty:
            # Empty predictions with empty ground truth = 100% accuracy
            empty_accuracy = 1
        else:
            # Get non-empty predictions for empty images
            non_empty_predictions = predictions_df.loc[predictions_df.xmin.notnull()]
            predictions_for_empty_images = non_empty_predictions.loc[
                non_empty_predictions.image_path.isin(empty_images)
            ]

            # Create prediction tensor - 1 if model predicted objects, 0 if predicted empty
            predictions = torch.zeros(len(empty_images))
            for index, image in enumerate(empty_images):
                if (
                    len(
                        predictions_for_empty_images.loc[
                            predictions_for_empty_images.image_path == image
                        ]
                    )
                    > 0
                ):
                    predictions[index] = 1

            # Ground truth tensor - all zeros since these are empty frames
            gt = torch.zeros(len(empty_images))
            predictions = torch.tensor(predictions)

            # Calculate accuracy using metric
            self.empty_frame_accuracy.update(predictions, gt)
            empty_accuracy = self.empty_frame_accuracy.compute()

        # Log empty frame accuracy
        try:
            self.log("empty_frame_accuracy", empty_accuracy)
        except MisconfigurationException:
            pass

        return empty_accuracy

    def log_epoch_metrics(self):
        if len(self.iou_metric.groundtruth_labels) > 0:
            output = self.iou_metric.compute()
            # Lightning bug: claims this is a warning but it's not. See issue #16218 in Lightning-AI/pytorch-lightning
            try:
                self.log_dict(output)
            except Exception:
                pass

            self.iou_metric.reset()
            output = self.mAP_metric.compute()

            # Remove classes from output dict
            output = {key: value for key, value in output.items() if not key == "classes"}
            try:
                self.log_dict(output)
            except MisconfigurationException:
                pass
            self.mAP_metric.reset()

        # Log empty frame accuracy if it has been updated
        if self.empty_frame_accuracy._update_called:
            empty_accuracy = self.empty_frame_accuracy.compute()

            # Log empty frame accuracy
            try:
                self.log("empty_frame_accuracy", empty_accuracy)
            except MisconfigurationException:
                pass

    def on_validation_epoch_end(self):
        """Compute metrics and predictions at the end of the validation
        epoch."""
        if self.trainer.sanity_checking:  # optional skip
            return

        # Log epoch metrics
        self.log_epoch_metrics()

        if (self.current_epoch + 1) % self.config.validation.val_accuracy_interval == 0:
            if len(self.predictions) > 0:
                predictions = pd.concat(self.predictions)
            else:
                predictions = pd.DataFrame()

            results = self.evaluate(
                self.config.validation.csv_file,
                root_dir=self.config.validation.root_dir,
                size=self.config.validation.size,
                predictions=predictions,
            )

            self.__evaluation_logs__(results)

            return results

    def predict_step(self, batch, batch_idx):
        """Predict a batch of images with the deepforest model. If batch is a
        list, concatenate the images, predict and then split the results,
        useful for main.predict_tile.

        Args:
            batch (torch.Tensor or np.ndarray): A batch of images with shape (B, C, H, W).
            batch_idx (int): The index of the batch.

        Returns:
        """
        split_results = False
        # If batch is a list, concatenate the images, predict and then split the results
        if isinstance(batch, list):
            original_list_length = len(batch)
            combined_batch = torch.cat(batch, dim=0)
            split_results = True
        else:
            combined_batch = batch

        batch_results = self.model(combined_batch)

        # If batch is a list, split the results
        if split_results:
            results = []
            batch_size = len(batch_results) // original_list_length
            for i in range(original_list_length):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                results.append(batch_results[start_idx:end_idx])
            return results
        else:
            return batch_results

    def predict_batch(self, images, preprocess_fn=None):
        """Predict a batch of images with the deepforest model.

        Args:
            images (torch.Tensor or np.ndarray): A batch of images with shape (B, C, H, W).
            preprocess_fn (callable, optional): A function to preprocess images before prediction.
                If None, assumes images are preprocessed.

        Returns:
            List[pd.DataFrame]: A list of dataframes with predictions for each image.
        """
        self.model.eval()

        # convert to tensor if input is array
        if isinstance(images, np.ndarray):
            images = torch.tensor(images, device=self.device)

        # apply preprocessing if available
        if preprocess_fn:
            images = preprocess_fn(images)

        # using Pytorch Ligthning's predict_step
        with torch.no_grad():
            predictions = self.predict_step(images, 0)

        # convert predictions to dataframes
        results = []
        for pred in predictions:
            if len(pred["boxes"]) == 0:
                continue
            geom_type = utilities.determine_geometry_type(pred)
            result = utilities.format_geometry(pred, geom_type=geom_type)
            results.append(utilities.read_file(result))

        return results

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.model.parameters(), lr=self.config.train.lr, momentum=0.9
        )

        scheduler_config = self.config.train.scheduler
        scheduler_type = scheduler_config.type
        params = scheduler_config.params

        # Assume the lambda is a function of epoch
        def lr_lambda(epoch):
            return eval(params.lr_lambda)

        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=params.T_max, eta_min=params.eta_min
            )

        elif scheduler_type == "lambdaLR":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        elif scheduler_type == "multiplicativeLR":
            scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
                optimizer, lr_lambda=lr_lambda
            )

        elif scheduler_type == "stepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=params.step_size, gamma=params.gamma
            )

        elif scheduler_type == "multistepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=params.milestones, gamma=params.gamma
            )

        elif scheduler_type == "exponentialLR":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=params.gamma
            )

        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=params["mode"],
                factor=params["factor"],
                patience=params["patience"],
                threshold=params["threshold"],
                threshold_mode=params["threshold_mode"],
                cooldown=params["cooldown"],
                min_lr=params["min_lr"],
                eps=params["eps"],
            )

        # Monitor learning rate if val data is used
        if self.config.validation.csv_file is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": self.config.validation.lr_plateau_target,
            }
        else:
            return optimizer

    def evaluate(
        self,
        csv_file,
        iou_threshold=None,
        root_dir=None,
        size=None,
        batch_size=None,
        predictions=None,
    ):
        """Compute intersection-over-union and precision/recall for a given
        iou_threshold.

        Args:
            csv_file: location of a csv file with columns "name","xmin","ymin","xmax","ymax","label"
            iou_threshold: float [0,1] intersection-over-union threshold for true positive
            batch_size: int, the batch size to use for prediction. If None, uses the batch size of the model.
            size: int, the size to resize the images to. If None, no resizing is done.
            predictions: list of predictions to use for evaluation. If None, predictions are generated from the model.

        Returns:
            dict: Results dictionary containing precision, recall and other metrics
        """
        self.model.eval()
        ground_df = utilities.read_file(csv_file)
        ground_df["label"] = ground_df.label.apply(lambda x: self.label_dict[x])

        if root_dir is None:
            root_dir = os.path.dirname(csv_file)

        if predictions is None:
            # Get the predict dataloader and use predict_batch
            predictions = self.predict_file(
                csv_file, root_dir, size=size, batch_size=batch_size
            )

        if iou_threshold is None:
            iou_threshold = self.config.validation.iou_threshold

        results = evaluate_iou.__evaluate_wrapper__(
            predictions=predictions,
            ground_df=ground_df,
            iou_threshold=iou_threshold,
            numeric_to_label_dict=self.numeric_to_label_dict,
        )

        # empty frame accuracy
        empty_accuracy = self.calculate_empty_frame_accuracy(ground_df, predictions)
        results["empty_frame_accuracy"] = empty_accuracy

        self.__evaluation_logs__(results)

        return results

    def __evaluation_logs__(self, results):
        """Log metrics from evaluation results."""
        # Log metrics
        for key, value in results.items():
            if type(value) in [pd.DataFrame, gpd.GeoDataFrame]:
                pass
            elif value is None:
                pass
            else:
                try:
                    self.log(key, value)
                except MisconfigurationException:
                    pass

        # Log each key value pair of the results dict
        if results["class_recall"] is not None:
            for key, value in results.items():
                if key in ["class_recall"]:
                    for _, row in value.iterrows():
                        try:
                            self.log(
                                "{}_Recall".format(
                                    self.numeric_to_label_dict[row["label"]]
                                ),
                                row["recall"],
                            )
                            self.log(
                                "{}_Precision".format(
                                    self.numeric_to_label_dict[row["label"]]
                                ),
                                row["precision"],
                            )
                        except MisconfigurationException:
                            pass
                elif key in ["predictions", "results", "ground_df"]:
                    # Don't log dataframes of predictions or IoU results per epoch
                    pass
                elif value is None:
                    pass
                else:
                    try:
                        self.log(key, value)
                    except MisconfigurationException:
                        pass
