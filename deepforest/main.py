# entry point for deepforest model
import importlib
import os
import typing
import warnings

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rasterio as rio
import torch
from PIL import Image
from pytorch_lightning.callbacks import LearningRateMonitor
from torch import optim
from torchmetrics.detection import IntersectionOverUnion, MeanAveragePrecision

from deepforest import dataset, visualize, get_data, utilities, predict
from deepforest import evaluate as evaluate_iou


class deepforest(pl.LightningModule):
    """Class for training and predicting tree crowns in RGB images"""

    def __init__(self,
                 num_classes: int = 1,
                 label_dict: dict = {"Tree": 0},
                 transforms=None,
                 config_file: str = 'deepforest_config.yml',
                 config_args=None,
                 model=None):
        """Args:
            num_classes (int): number of classes in the model
            config_file (str): path to deepforest config file
            model (model.Model()): a deepforest model object, see model.Model().
            config_args (dict): a dictionary of key->value to update
            config file at run time. e.g. {"batch_size":10}
            This is useful for iterating over arguments during model testing.
        Returns:
            self: a deepforest pytorch lightning module
        """
        super().__init__()

        # Read config file. Defaults to deepforest_config.yml in working directory.
        # Falls back to default installed version
        if os.path.exists(config_file):
            config_path = config_file
        else:
            try:
                config_path = get_data("deepforest_config.yml")
            except Exception as e:
                raise ValueError("No config file provided and deepforest_config.yml "
                                 "not found either in local directory or in installed "
                                 "package location. {}".format(e))

        print("Reading config file: {}".format(config_path))
        self.config = utilities.read_config(config_path)
        self.config["num_classes"] = num_classes
        # If num classes is specified, overwrite config
        if not num_classes == 1:
            warnings.warn(
                "Directly specifying the num_classes arg in deepforest.main will be deprecated in 2.0 in favor of config_args. Use main.deepforest(config_args={'num_classes':value})"
            )

        # Update config with user supplied arguments
        if config_args:
            for key, value in config_args.items():
                if key not in self.config.keys():
                    raise ValueError(
                        "Config argument {} not found in config file".format(key))
                if type(value) == dict:
                    for subkey, subvalue in value.items():
                        print("setting config {} to {}".format(subkey, subvalue))
                        self.config[key][subkey] = subvalue
                else:
                    print("setting config {} to {}".format(key, value))
                    self.config[key] = value

        self.model = model

        # release version id to flag if release is being used
        self.__release_version__ = None

        self.create_model()

        # Metrics
        self.iou_metric = IntersectionOverUnion(
            class_metrics=True, iou_threshold=self.config["validation"]["iou_threshold"])
        self.mAP_metric = MeanAveragePrecision()

        # Create a default trainer.
        self.create_trainer()

        # Label encoder and decoder
        if not len(label_dict) == self.config["num_classes"]:
            raise ValueError('label_dict {} does not match requested number of '
                             'classes {}, please supply a label_dict argument '
                             '{{"label1":0, "label2":1, "label3":2 ... etc}} '
                             'for each label in the '
                             'dataset'.format(label_dict, self.config["num_classes"]))

        self.label_dict = label_dict
        self.numeric_to_label_dict = {v: k for k, v in label_dict.items()}

        # Add user supplied transforms
        if transforms is None:
            self.transforms = dataset.get_transform
        else:
            self.transforms = transforms

        self.save_hyperparameters()

    def use_release(self, check_release=True):
        """Use the latest DeepForest model release from github and load model.
        Optionally download if release doesn't exist.
        Args:
            check_release (logical): whether to check github for a model recent release. In cases where you are hitting the github API rate limit, set to False and any local model will be downloaded. If no model has been downloaded an error will raise.
        Returns:
            model (object): A trained PyTorch model
        """
        # Download latest model from github release
        release_tag, self.release_state_dict = utilities.use_release(
            check_release=check_release)
        if self.config["architecture"] != "retinanet":
            warnings.warn(
                "The config file specifies architecture {}, but the release model is torchvision retinanet. Reloading main.deepforest with a retinanet model"
                .format(self.config["architecture"]))
            self.config["architecture"] = "retinanet"
            self.create_model()
        self.model.load_state_dict(torch.load(self.release_state_dict))

        # load saved model and tag release
        self.__release_version__ = release_tag
        print("Loading pre-built model: {}".format(release_tag))

    def use_bird_release(self, check_release=True):
        """Use the latest DeepForest bird model release from github and load model.
        Optionally download if release doesn't exist.
        Args:
            check_release (logical): whether to check github for a model recent release. In cases where you are hitting the github API rate limit, set to False and any local model will be downloaded. If no model has been downloaded an error will raise.
        Returns:
            model (object): A trained pytorch model
        """
        # Download latest model from github release
        release_tag, self.release_state_dict = utilities.use_bird_release(
            check_release=check_release)
        self.model.load_state_dict(torch.load(self.release_state_dict))

        # load saved model and tag release
        self.__release_version__ = release_tag
        print("Loading pre-built model: {}".format(release_tag))

        print("Setting default score threshold to 0.3")
        self.config["score_thresh"] = 0.3

        # Set label dictionary to Bird
        self.label_dict = {"Bird": 0}
        self.numeric_to_label_dict = {v: k for k, v in self.label_dict.items()}

    def create_model(self):
        """Define a deepforest architecture. This can be done in two ways.
        Passed as the model argument to deepforest __init__(),
        or as a named architecture in config["architecture"],
        which corresponds to a file in models/, as is a subclass of model.Model().
        The config args in the .yaml are specified

        >>> # retinanet:
        >>> #   ms_thresh: 0.1
        >>> #   score_thresh: 0.2
        >>> # RCNN:
        >>> #   nms_thresh: 0.1
        >>> # etc.
        """
        if self.model is None:
            model_name = importlib.import_module("deepforest.models.{}".format(
                self.config["architecture"]))
            self.model = model_name.Model(config=self.config).create_model()

    def create_trainer(self, logger=None, callbacks=[], **kwargs):
        """Create a pytorch lightning training by reading config files
        Args:
            callbacks (list): a list of pytorch-lightning callback classes
        """
        # If val data is passed, monitor learning rate and setup classification metrics
        if not self.config["validation"]["csv_file"] is None:
            if logger is not None:
                lr_monitor = LearningRateMonitor(logging_interval='epoch')
                callbacks.append(lr_monitor)
            limit_val_batches = 1.0
            num_sanity_val_steps = 2
        else:
            # Disable validation, don't use trainer defaults
            print("No validation file provided. Turning off validation loop")
            limit_val_batches = 0
            num_sanity_val_steps = 0
        # Check for model checkpoint object
        checkpoint_types = [type(x).__qualname__ for x in callbacks]
        if 'ModelCheckpoint' in checkpoint_types:
            enable_checkpointing = True
        else:
            enable_checkpointing = False

        self.trainer = pl.Trainer(logger=logger,
                                  max_epochs=self.config["train"]["epochs"],
                                  enable_checkpointing=enable_checkpointing,
                                  devices=self.config["devices"],
                                  accelerator=self.config["accelerator"],
                                  fast_dev_run=self.config["train"]["fast_dev_run"],
                                  callbacks=callbacks,
                                  limit_val_batches=limit_val_batches,
                                  num_sanity_val_steps=num_sanity_val_steps,
                                  **kwargs)

    def on_fit_start(self):
        if self.config["train"]["csv_file"] is None:
            raise AttributeError(
                "Cannot train with a train annotations file, please set 'config['train']['csv_file'] before calling deepforest.create_trainer()'"
            )

    def save_model(self, path):
        """
        Save the trainer checkpoint in user defined path, in order to access in future
        Args:
            Path: the path located the model checkpoint

        """
        self.trainer.save_checkpoint(path)

    def load_dataset(self,
                     csv_file,
                     root_dir=None,
                     augment=False,
                     shuffle=True,
                     batch_size=1,
                     train=False):
        """Create a tree dataset for inference
        Csv file format is .csv file with the columns "image_path", "xmin","ymin","xmax","ymax" for the image name and bounding box position.
        Image_path is the relative filename, not absolute path, which is in the root_dir directory. One bounding box per line.

        Args:
            csv_file: path to csv file
            root_dir: directory of images. If none, uses "image_dir" in config
            augment: Whether to create a training dataset, this activates data augmentations
            
        Returns:
            ds: a pytorch dataset
        """
        ds = dataset.TreeDataset(csv_file=csv_file,
                                 root_dir=root_dir,
                                 transforms=self.transforms(augment=augment),
                                 label_dict=self.label_dict,
                                 preload_images=self.config["train"]["preload_images"])

        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=utilities.collate_fn,
            num_workers=self.config["workers"],
        )

        return data_loader

    def train_dataloader(self):
        """
        Train loader using the configurations
        Returns: loader

        """
        loader = self.load_dataset(csv_file=self.config["train"]["csv_file"],
                                   root_dir=self.config["train"]["root_dir"],
                                   augment=True,
                                   shuffle=True,
                                   batch_size=self.config["batch_size"])

        return loader

    def val_dataloader(self):
        """
        Create a val data loader only if specified in config
        Returns: a dataloader or a empty iterable.

        """
        if self.config["validation"]["csv_file"] is not None:
            loader = self.load_dataset(csv_file=self.config["validation"]["csv_file"],
                                       root_dir=self.config["validation"]["root_dir"],
                                       augment=False,
                                       shuffle=False,
                                       batch_size=self.config["batch_size"])
        else:
            # The preferred route for skipping validation is now (pl-2.0) an empty list, see https://github.com/Lightning-AI/lightning/issues/17154
            loader = []

        return loader

    def predict_dataloader(self, ds):
        """
        Create a pytorch dataloader for prediction
        Returns:
        """
        data_loader = torch.utils.data.DataLoader(ds,
                                                  batch_size=self.config["batch_size"],
                                                  shuffle=False,
                                                  num_workers=self.config["workers"])

        return data_loader

    def predict_image(self,
                      image: typing.Optional[np.ndarray] = None,
                      path: typing.Optional[str] = None,
                      return_plot: bool = False,
                      color: typing.Optional[tuple] = (0, 165, 255),
                      thickness: int = 1):
        """Predict a single image with a deepforest model
        Args:
            image: a float32 numpy array of a RGB with channels last format
            path: optional path to read image from disk instead of passing image arg
            return_plot: Return image with plotted detections
            color: color of the bounding box as a tuple of BGR color, e.g. orange annotations is (0, 165, 255)
            thickness: thickness of the rectangle border line in px
        Returns:
            result: A pandas dataframe of predictions (Default)
            img: The input with predictions overlaid (Optional)
        """
        # Ensure we are in eval mode
        self.model.eval()

        if path:
            image = np.array(Image.open(path).convert("RGB")).astype("float32")

        # sanity checks on input images
        if not type(image) == np.ndarray:
            raise TypeError("Input image is of type {}, expected numpy, if reading "
                            "from PIL, wrap in "
                            "np.array(image).astype(float32)".format(type(image)))

        if image.dtype != "float32":
            warnings.warn(f"Image type is {image.dtype}, transforming to float32. "
                          f"This assumes that the range of pixel values is 0-255, as "
                          f"opposed to 0-1.To suppress this warning, transform image "
                          f"(image.astype('float32')")
            image = image.astype("float32")

        image = torch.tensor(image, device=self.device).permute(2, 0, 1)
        image = image / 255

        result = predict._predict_image_(model=self.model,
                                         image=image,
                                         path=path,
                                         nms_thresh=self.config["nms_thresh"],
                                         return_plot=return_plot,
                                         thickness=thickness,
                                         color=color)

        if return_plot:
            return result
        else:
            #If there were no predictions, return None
            if result is None:
                return None
            else:
                result["label"] = result.label.apply(
                    lambda x: self.numeric_to_label_dict[x])

        return result

    def predict_file(self, csv_file, root_dir, savedir=None, color=None, thickness=1):
        """Create a dataset and predict entire annotation file
        Csv file format is .csv file with the columns "image_path", "xmin","ymin","xmax","ymax" for the image name and bounding box position.
        Image_path is the relative filename, not absolute path, which is in the root_dir directory. One bounding box per line.

        Args:
            csv_file: path to csv file
            root_dir: directory of images. If none, uses "image_dir" in config
            savedir: Optional. Directory to save image plots.
            color: color of the bounding box as a tuple of BGR color, e.g. orange annotations is (0, 165, 255)
            thickness: thickness of the rectangle border line in px
        Returns:
            df: pandas dataframe with bounding boxes, label and scores for each image in the csv file
        """
        df = pd.read_csv(csv_file)
        ds = dataset.TreeDataset(csv_file=csv_file,
                                 root_dir=root_dir,
                                 transforms=None,
                                 train=False)
        dataloader = self.predict_dataloader(ds)

        results = predict._dataloader_wrapper_(model=self,
                                               trainer=self.trainer,
                                               annotations=df,
                                               dataloader=dataloader,
                                               root_dir=root_dir,
                                               nms_thresh=self.config["nms_thresh"],
                                               savedir=savedir,
                                               color=color,
                                               thickness=thickness)

        return results

    def predict_tile(self,
                     raster_path=None,
                     image=None,
                     patch_size=400,
                     patch_overlap=0.05,
                     iou_threshold=0.15,
                     return_plot=False,
                     mosaic=True,
                     sigma=0.5,
                     thresh=0.001,
                     color=None,
                     thickness=1):
        """For images too large to input into the model, predict_tile cuts the
        image into overlapping windows, predicts trees on each window and
        reassambles into a single array.

        Args:
            raster_path: Path to image on disk
            image (array): Numpy image array in BGR channel order
                following openCV convention
            patch_size: patch size for each window.
            patch_overlap: patch overlap among windows.
            iou_threshold: Minimum iou overlap among predictions between
                windows to be suppressed.
                Lower values suppress more boxes at edges.
            return_plot: Should the image be returned with the predictions drawn?
            mosaic: Return a single prediction dataframe (True) or a tuple of image crops and predictions (False)
            sigma: variance of Gaussian function used in Gaussian Soft NMS
            thresh: the score thresh used to filter bboxes after soft-nms performed
            color: color of the bounding box as a tuple of BGR color, e.g. orange annotations is (0, 165, 255)
            thickness: thickness of the rectangle border line in px

        Returns:
            boxes (array): if return_plot, an image.
            Otherwise a numpy array of predicted bounding boxes, scores and labels
        """
        self.model.eval()
        self.model.nms_thresh = self.config["nms_thresh"]

        if (raster_path is None) and (image is None):
            raise ValueError(
                "Both tile and tile_path are None. Either supply a path to a tile on disk, or read one into memory!"
            )

        if raster_path is None:
            self.image = image
        else:
            self.image = rio.open(raster_path).read()
            self.image = np.moveaxis(self.image, 0, 2)

        ds = dataset.TileDataset(tile=self.image,
                                 patch_overlap=patch_overlap,
                                 patch_size=patch_size)
        batched_results = self.trainer.predict(self, self.predict_dataloader(ds))

        # Flatten list from batched prediction
        results = []
        for batch in batched_results:
            for boxes in batch:
                results.append(boxes)

        if mosaic:
            results = predict.mosiac(results,
                                     ds.windows,
                                     sigma=sigma,
                                     thresh=thresh,
                                     iou_threshold=iou_threshold)
            results["label"] = results.label.apply(
                lambda x: self.numeric_to_label_dict[x])
            if raster_path:
                results["image_path"] = os.path.basename(raster_path)
            if return_plot:
                # Draw predictions on BGR
                if raster_path:
                    tile = rio.open(raster_path).read()
                else:
                    tile = self.image
                drawn_plot = tile[:, :, ::-1]
                drawn_plot = visualize.plot_predictions(tile,
                                                        results,
                                                        color=color,
                                                        thickness=thickness)
                return drawn_plot
        else:
            for df in results:
                df["label"] = df.label.apply(lambda x: self.numeric_to_label_dict[x])

            # TODO this is the 2nd time the crops are generated? Could be more efficient.
            self.crops = []
            for window in ds.windows:
                crop = self.image[window.indices()]
                self.crops.append(crop)

            return list(zip(results, self.crops))

        return results

    def training_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        # Confirm model is in train mode
        self.model.train()

        # allow for empty data if data augmentation is generated
        path, images, targets = batch

        loss_dict = self.model.forward(images, targets)

        # sum of regression and classification loss
        losses = sum([loss for loss in loss_dict.values()])

        return losses

    def validation_step(self, batch, batch_idx):
        """Evaluate a batch
        """
        try:
            path, images, targets = batch
        except:
            print("Empty batch encountered, skipping")
            return None

        # Get loss from "train" mode, but don't allow optimization
        self.model.train()
        with torch.no_grad():
            loss_dict = self.model.forward(images, targets)

        # sum of regression and classification loss
        losses = sum([loss for loss in loss_dict.values()])

        self.model.eval()
        preds = self.model.forward(images)

        # Calculate intersection-over-union
        self.iou_metric.update(preds, targets)
        self.mAP_metric.update(preds, targets)

        # Log loss
        for key, value in loss_dict.items():
            self.log("val_{}".format(key), value, on_epoch=True)

        for index, result in enumerate(preds):
            boxes = visualize.format_boxes(result)
            boxes["image_path"] = path[index]
            self.predictions.append(boxes)

        return losses

    def on_validation_epoch_start(self):
        self.predictions = []

    def on_validation_epoch_end(self):
        output = self.iou_metric.compute()
        self.log_dict(output)
        self.iou_metric.reset()

        output = self.mAP_metric.compute()

        # Remove classes from output dict
        output = {key: value for key, value in output.items() if not key == "classes"}
        self.log_dict(output)
        self.mAP_metric.reset()

        # Evaluate on validation data predictions
        self.predictions_df = pd.concat(self.predictions)
        ground_df = pd.read_csv(self.config["validation"]["csv_file"])
        ground_df["label"] = ground_df.label.apply(lambda x: self.label_dict[x])

        #Evaluate every n epochs
        if self.current_epoch % self.config["validation"]["val_accuracy_interval"] == 0:
            results = evaluate_iou.__evaluate_wrapper__(
                predictions=self.predictions_df,
                ground_df=ground_df,
                root_dir=self.config["validation"]["root_dir"],
                iou_threshold=self.config["validation"]["iou_threshold"],
                savedir=None,
                numeric_to_label_dict=self.numeric_to_label_dict)

            self.log("box_recall", results["box_recall"])
            self.log("box_precision", results["box_precision"])
            if isinstance(results, pd.DataFrame):
                for index, row in results["class_recall"].iterrows():
                    self.log("{}_Recall".format(self.numeric_to_label_dict[row["label"]]),
                             row["recall"])
                    self.log(
                        "{}_Precision".format(self.numeric_to_label_dict[row["label"]]),
                        row["precision"])

    def predict_step(self, batch, batch_idx):
        batch_results = self.model(batch)

        results = []
        for result in batch_results:
            boxes = visualize.format_boxes(result)
            results.append(boxes)

        return results

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(),
                              lr=self.config["train"]["lr"],
                              momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.1,
                                                               patience=10,
                                                               verbose=True,
                                                               threshold=0.0001,
                                                               threshold_mode='rel',
                                                               cooldown=0,
                                                               min_lr=0,
                                                               eps=1e-08)

        # Monitor rate is val data is used
        if self.config["validation"]["csv_file"] is not None:
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                "monitor": 'val_classification'
            }
        else:
            return optimizer

    def evaluate(self, csv_file, root_dir, iou_threshold=None, savedir=None):
        """Compute intersection-over-union and precision/recall for a given iou_threshold
        Args:
            csv_file: location of a csv file with columns "name","xmin","ymin","xmax","ymax","label", each box in a row
            root_dir: location of files in the dataframe 'name' column.
            iou_threshold: float [0,1] intersection-over-union union between annotation and prediction to be scored true positive
            savedir: optional path dir to save evaluation images
        Returns:
            results: dict of ("results", "precision", "recall") for a given threshold
        """
        ground_df = pd.read_csv(csv_file)
        ground_df["label"] = ground_df.label.apply(lambda x: self.label_dict[x])
        predictions = self.predict_file(csv_file=csv_file,
                                        root_dir=root_dir,
                                        savedir=savedir)

        results = evaluate_iou.__evaluate_wrapper__(
            predictions=predictions,
            ground_df=ground_df,
            root_dir=root_dir,
            iou_threshold=iou_threshold,
            numeric_to_label_dict=self.numeric_to_label_dict,
            savedir=savedir)

        return results
