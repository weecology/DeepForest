"""Deepforest main module.

This module holds the deepforest class for model building and training
"""
import csv
import os
import warnings

from PIL import Image

with warnings.catch_warnings():
    # Suppress some of the verbose tensorboard warnings,
    # compromise to avoid numpy version errors
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf

import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt

from deepforest import get_data
from deepforest import utilities
from deepforest import predict
from deepforest import preprocess
from deepforest.retinanet_train import main as retinanet_train
from deepforest.retinanet_train import parse_args

from deepforest.keras_retinanet import models
from deepforest.keras_retinanet.models import convert_model
from deepforest.keras_retinanet.bin.train import create_models
from deepforest.keras_retinanet.preprocessing.csv_generator import CSVGenerator, _read_classes
from deepforest.keras_retinanet.utils.eval import evaluate
from deepforest.keras_retinanet.utils.visualization import draw_box


class deepforest:
    """Class for training and predicting tree crowns in RGB images.

    Args:
        weights (str): Path to model saved on disk from keras.model.save_weights().
            A new model is created and weights are copied. Default is None.
        saved_model: Path to a saved model from disk using keras.model.save().
            No new model is created.

    Attributes:
        model: A keras training model from keras-retinanet
    """

    def __init__(self, weights=None, saved_model=None):
        self.weights = weights
        self.saved_model = saved_model

        # Read config file - if a config file exists in local dir use it,
        # if not use installed.
        if os.path.exists("deepforest_config.yml"):
            config_path = "deepforest_config.yml"
        else:
            try:
                config_path = get_data("deepforest_config.yml")
            except Exception as e:
                raise ValueError(
                    "No deepforest_config.yml found either in local "
                    "directory or in installed package location. {}".format(e))

        print("Reading config file: {}".format(config_path))
        self.config = utilities.read_config(config_path)

        # Create a label dict, defaults to "Tree"
        self.read_classes()

        # release version id to flag if release is being used
        self.__release_version__ = None

        # Load saved model if needed
        if self.saved_model:
            print("Loading saved model")
            # Capture user warning, not relevant here
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.model = models.load_model(saved_model)
                self.prediction_model = convert_model(self.model)

        elif self.weights:
            print("Creating model from weights")
            backbone = models.backbone(self.config["backbone"])
            self.model, self.training_model, self.prediction_model = create_models(
                backbone.retinanet, num_classes=1, weights=self.weights)
        else:
            print("A blank deepforest object created. "
                  "To perform prediction, either train or load an existing model.")
            self.model = None

    def read_classes(self):
        """Read class file in case of multi-class training.

        If no file has been created, DeepForest assume there is 1 class,
        Tree
        """
        # parse the provided class file
        self.labels = {}
        try:
            with open(self.classes_file, 'r') as file:
                self.classes = _read_classes(csv.reader(file, delimiter=','))
            for key, value in self.classes.items():
                self.labels[value] = key
        except:
            self.labels[0] = "Tree"

    def train(self,
              annotations,
              input_type="fit_generator",
              list_of_tfrecords=None,
              comet_experiment=None,
              images_per_epoch=None):
        """Train a deep learning tree detection model using keras-retinanet.
        This is the main entry point for training a new model based on either
        existing weights or scratch.

        Args:
            annotations (str): Path to csv label file,
                labels are in the format -> path/to/image.png,x1,y1,x2,y2,class_name
            input_type: "fit_generator" or "tfrecord"
            list_of_tfrecords: Ignored if input_type != "tfrecord",
                list of tf records to process
            comet_experiment: A comet ml object to log images. Optional.
            images_per_epoch: number of images to override default config
                of images in annotations file / batch size. Useful for debug

        Returns:
            model (object): A trained keras model
            prediction model: with bbox nms
                trained model: without nms
        """
        # Test if there is a new classes file in case # of classes has changed.
        self.classes_file = utilities.create_classes(annotations)
        self.read_classes()
        arg_list = utilities.format_args(annotations, self.classes_file, self.config,
                                         images_per_epoch)

        print("Training retinanet with the following args {}".format(arg_list))

        # Train model
        self.model, self.prediction_model, self.training_model = retinanet_train(
            forest_object=self,
            args=arg_list,
            input_type=input_type,
            list_of_tfrecords=list_of_tfrecords,
            comet_experiment=comet_experiment)

    def use_release(self, gpus=1):
        """Use the latest DeepForest model release from github and load model.
        Optionally download if release doesn't exist.

        Returns:
            model (object): A trained keras model
            gpus: number of gpus to parallelize, default to 1
        """
        # Download latest model from github release
        release_tag, self.weights = utilities.use_release()

        # load saved model and tag release
        self.__release_version__ = release_tag
        print("Loading pre-built model: {}".format(release_tag))

        if gpus == 1:
            with warnings.catch_warnings():
                # Suppress compilte warning, not relevant here
                warnings.filterwarnings("ignore", category=UserWarning)
                self.model = utilities.read_model(self.weights, self.config)

            # Convert model
            self.prediction_model = convert_model(self.model)
        elif gpus > 1:
            backbone = models.backbone(self.config["backbone"])
            n_classes = len(self.labels.keys())
            self.model, self.training_model, self.prediction_model = create_models(
                backbone.retinanet,
                num_classes=n_classes,
                weights=self.weights,
                multi_gpu=gpus)

        # add to config
        self.config["weights"] = self.weights

    def predict_generator(self,
                          annotations,
                          comet_experiment=None,
                          iou_threshold=0.5,
                          max_detections=200,
                          return_plot=False,
                          color=None):
        """Predict bounding boxes for a model using a csv fit_generator

        Args:
            annotations (str): Path to csv label file, labels are in the
                format -> path/to/image.png,x1,y1,x2,y2,class_name
            comet_experiment(object): A comet experiment class objects to track
            color: rgb color for the box annotations if return_plot is True e.g. (255,140,0) is orange.
            return_plot: Whether to return prediction boxes (False) or Images (True). If True, files will be written to current working directory if model.config["save_path"] is not defined.

        Return:
            boxes_output: If return_plot=False, a pandas dataframe of bounding boxes
                for each image in the annotations file
                None: If return_plot is True, images are written to save_dir as a side effect.
        """
        # Format args for CSV generator
        classes_file = utilities.create_classes(annotations)
        arg_list = utilities.format_args(annotations, classes_file, self.config)
        args = parse_args(arg_list)

        # create generator
        generator = CSVGenerator(
            args.annotations,
            args.classes,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config,
            shuffle_groups=False,
        )

        if self.prediction_model:
            boxes_output = []

            # For each image, gather predictions
            for i in range(generator.size()):
                # pass image as path
                plot_name = generator.image_names[i]
                image_path = os.path.join(generator.base_dir, plot_name)
                result = self.predict_image(image_path,
                                            return_plot=return_plot,
                                            score_threshold=args.score_threshold,
                                            color=color)

                if return_plot:
                    if not self.config["save_path"]:
                        print("model.config['save_path'] is None,"
                              "saving images to current working directory")
                        save_path = "."
                    else:
                        save_path = self.config["save_path"]
                    # Save image
                    fname = os.path.join(save_path, plot_name)
                    cv2.imwrite(fname, result)
                    continue
                else:
                    # Turn boxes to pandas frame and save output
                    box_df = pd.DataFrame(result)
                    # use only plot name, not extension
                    box_df["plot_name"] = os.path.splitext(plot_name)[0]
                    boxes_output.append(box_df)
        else:
            raise ValueError(
                "No prediction model loaded. Either load a retinanet from file, "
                "download the latest release or train a new model")

        if return_plot:
            return None
        else:
            # if boxes, name columns and return box data
            boxes_output = pd.concat(boxes_output)
            boxes_output.columns = [
                "xmin", "ymin", "xmax", "ymax", "score", "label", "plot_name"
            ]
            boxes_output = boxes_output.reindex(
                columns=["plot_name", "xmin", "ymin", "xmax", "ymax", "score", "label"])
            return boxes_output

    def evaluate_generator(self,
                           annotations,
                           comet_experiment=None,
                           iou_threshold=0.5,
                           max_detections=200):
        """Evaluate prediction model using a csv fit_generator.

        Args:
            annotations (str): Path to csv label file, labels are in the
                format -> path/to/image.png,x1,y1,x2,y2,class_name
            comet_experiment(object): A comet experiment class objects to track
            iou_threshold(float): IoU Threshold to count for a positive detection
                (defaults to 0.5)
            max_detections (int): Maximum number of bounding box predictions

        Return:
            mAP: Mean average precision of the evaluated data
        """
        # Format args for CSV generator
        classes_file = utilities.create_classes(annotations)
        arg_list = utilities.format_args(annotations, classes_file, self.config)
        args = parse_args(arg_list)

        # create generator
        validation_generator = CSVGenerator(
            args.annotations,
            args.classes,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config,
            shuffle_groups=False,
        )

        average_precisions = evaluate(validation_generator,
                                      self.prediction_model,
                                      iou_threshold=iou_threshold,
                                      score_threshold=args.score_threshold,
                                      max_detections=max_detections,
                                      save_path=args.save_path,
                                      comet_experiment=comet_experiment)

        # print evaluation
        total_instances = []
        precisions = []
        for label, (average_precision, num_annotations) in average_precisions.items():
            print('{:.0f} instances of class'.format(num_annotations),
                  validation_generator.label_to_name(label),
                  'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)

        if sum(total_instances) == 0:
            print('No test instances found.')
            return

        print('mAP using the weighted average of precisions among classes: {:.4f}'.format(
            sum([a * b for a, b in zip(total_instances, precisions)]) /
            sum(total_instances)))

        mAP = sum(precisions) / sum(x > 0 for x in total_instances)
        print('mAP: {:.4f}'.format(mAP))
        return mAP

    def predict_image(self,
                      image_path=None,
                      numpy_image=None,
                      return_plot=True,
                      score_threshold=0.05,
                      show=False,
                      color=None):
        """Predict tree crowns based on loaded (or trained) model.

        Args:
            image_path (str): Path to image on disk
            numpy_image (array): Numpy image array in BGR channel order
                following openCV convention
            return_plot: Whether to return image with annotations overlaid,
                or just a numpy array of boxes
            score_threshold: score threshold default 0.05,
            show (bool): Plot the predicted image with bounding boxes.
                Ignored if return_plot=False
            color (tuple): Color of bounding boxes in BGR order (0,0,0)
                black default

        Returns:
            predictions (array): if return_plot, an image. Otherwise a numpy array
                of predicted bounding boxes, with scores and labels
        """

        # Check for model save
        if (self.prediction_model is None):
            raise ValueError("Model currently has no prediction weights, "
                             "either train a new model using deepforest.train, "
                             "loading existing model, or use prebuilt model "
                             "(see deepforest.use_release()")

        # Check the formatting
        if isinstance(image_path, np.ndarray):
            raise ValueError("image_path should be a string, but is a numpy array. "
                             "If predicting a loaded image (channel order BGR), "
                             "use numpy_image argument.")

        #Check for correct formatting
        if numpy_image is None:
            if image_path is not None:
                numpy_image = cv2.imread(image_path)
            else:
                raise ValueError("No input specified. deepforest.predict_image() requires either a numpy_image array or a path to a file to read.")

        #Predict
        prediction = predict.predict_image(self.prediction_model,
                                           image_path=image_path,
                                           raw_image=numpy_image,
                                           return_plot=return_plot,
                                           score_threshold=score_threshold,
                                           color=color,
                                           classes=self.labels)

        # cv2 channel order to matplotlib order
        if return_plot & show:
            plt.imshow(prediction[:, :, ::-1])
            plt.show()

        return prediction

    def predict_tile(self,
                     raster_path=None,
                     numpy_image=None,
                     patch_size=400,
                     patch_overlap=0.05,
                     iou_threshold=0.15,
                     return_plot=False):
        """For images too large to input into the model, predict_tile cuts the
        image into overlapping windows, predicts trees on each window and
        reassambles into a single array.

        Args:
            raster_path: Path to image on disk
            numpy_image (array): Numpy image array in BGR channel order
                following openCV convention
            patch_size: patch size default400,
            patch_overlap: patch overlap default 0.15,
            iou_threshold: Minimum iou overlap among predictions between
                windows to be suppressed. Defaults to 0.5.
                Lower values suppress more boxes at edges.
            return_plot: Should the image be returned with the predictions drawn?

        Returns:
            boxes (array): if return_plot, an image.
                Otherwise a numpy array of predicted bounding boxes, scores and labels
        """

        if numpy_image is not None:
            pass
        else:
            # Load raster as image
            raster = Image.open(raster_path)
            numpy_image = np.array(raster)

        # Compute sliding window index
        windows = preprocess.compute_windows(numpy_image, patch_size, patch_overlap)

        # Save images to tmpdir
        predicted_boxes = []

        for index, window in enumerate(windows):
            # Crop window and predict
            crop = numpy_image[windows[index].indices()]

            # Crop is RGB channel order, change to BGR
            crop = crop[..., ::-1]
            boxes = self.predict_image(numpy_image=crop,
                                       return_plot=False,
                                       score_threshold=self.config["score_threshold"])

            # transform coordinates to original system
            xmin, ymin, xmax, ymax = windows[index].getRect()
            boxes.xmin = boxes.xmin + xmin
            boxes.xmax = boxes.xmax + xmin
            boxes.ymin = boxes.ymin + ymin
            boxes.ymax = boxes.ymax + ymin

            predicted_boxes.append(boxes)

        predicted_boxes = pd.concat(predicted_boxes)

        # Non-max supression for overlapping boxes among window
        if patch_overlap == 0:
            mosaic_df = predicted_boxes
        else:
            with tf.Session() as sess:
                print(
                    "{} predictions in overlapping windows, applying non-max supression".
                    format(predicted_boxes.shape[0]))
                new_boxes, new_scores, new_labels = predict.non_max_suppression(
                    sess,
                    predicted_boxes[["xmin", "ymin", "xmax", "ymax"]].values,
                    predicted_boxes.score.values,
                    predicted_boxes.label.values,
                    max_output_size=predicted_boxes.shape[0],
                    iou_threshold=iou_threshold)

                # Recreate box dataframe
                image_detections = np.concatenate([
                    new_boxes,
                    np.expand_dims(new_scores, axis=1),
                    np.expand_dims(new_labels, axis=1)
                ],
                                                  axis=1)

                mosaic_df = pd.DataFrame(
                    image_detections,
                    columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])
                mosaic_df.label = mosaic_df.label.str.decode("utf-8")

                print("{} predictions kept after non-max suppression".format(
                    mosaic_df.shape[0]))

        if return_plot:
            # Draw predictions
            for box in mosaic_df[["xmin", "ymin", "xmax", "ymax"]].values:
                draw_box(numpy_image, box, [0, 0, 255])

            # Mantain consistancy with predict_image
            return numpy_image
        else:
            return mosaic_df

    def plot_curves(self):
        """Plot training curves."""
        if self.history:
            # Plot training & validation regression loss values
            fig, axes, = plt.subplots(nrows=1, ncols=3)
            axes = axes.flatten()

            # Regression Loss
            axes[0].plot(self.history.history['regression_loss'])
            axes[0].set_title('Bounding Box Loss')
            axes[0].set_ylabel('Loss')
            axes[0].set_xlabel('Epoch')

            # Classification Loss
            axes[1].plot(self.history.history['classification_loss'])
            axes[1].set_title('Classification Loss')
            axes[1].set_ylabel('Loss')
            axes[1].set_xlabel('Epoch')

            # Plot validation mAP
            if "mAP" in self.history.history.keys():
                axes[2].plot(self.history.history['mAP'])
                axes[2].set_title('Validation: Mean Average Precision')
                axes[2].set_ylabel('mAP')
                axes[2].set_xlabel('Epoch')
            plt.show()
        else:
            print("No training history found.")
            return None
