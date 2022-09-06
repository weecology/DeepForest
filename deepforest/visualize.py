# Visualize module for plotting and handling predictions
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import pandas.api.types as ptypes
import cv2
import random
import warnings

def view_dataset(ds, savedir=None, color=None, thickness=1):
    """Plot annotations on images for debugging purposes
    Args:
        ds: a deepforest pytorch dataset, see deepforest.dataset or deepforest.load_dataset() to start from a csv file
        savedir: optional path to save figures. If none (default) images will be interactively plotted
        color: color of the bounding box as a tuple of BGR color, e.g. orange annotations is (0, 165, 255)
        thickness: thickness of the rectangle border line in px
    """
    for i in iter(ds):
        image_path, image, targets = i
        df = format_boxes(targets[0], scores=False)
        image = np.moveaxis(image[0].numpy(),0,2)
        image = plot_predictions(image, df, color=color, thickness=thickness)
    
    if savedir:
        cv2.imwrite("{}/{}".format(savedir, image_path[0]), image)
    else:
        cv2.imshow(image)
        cv2.waitKey(0)
            
def format_boxes(prediction, scores=True):
    """Format a retinanet prediction into a pandas dataframe for a single image
       Args:
           prediction: a dictionary with keys 'boxes' and 'labels' coming from a retinanet
           scores: Whether boxes come with scores, during prediction, or without scores, as in during training.
        Returns:
           df: a pandas dataframe
    """

    df = pd.DataFrame(prediction["boxes"].cpu().detach().numpy(),
                      columns=["xmin", "ymin", "xmax", "ymax"])
    df["label"] = prediction["labels"].cpu().detach().numpy()

    if scores:
        df["score"] = prediction["scores"].cpu().detach().numpy()

    return df


def plot_prediction_and_targets(image, predictions, targets, image_name, savedir):
    """Plot an image, its predictions, and its ground truth targets for debugging
    Args:
        image: torch tensor, RGB color order
        targets: torch tensor
    Returns:
        figure_path: path on disk with saved figure
    """
    image = np.array(image)[:,:,::-1].copy()
    prediction_df = format_boxes(predictions)
    image = plot_predictions(image, prediction_df)
    target_df = format_boxes(targets, scores=False)
    image = plot_predictions(image, target_df)
    figure_path = "{}/{}.png".format(savedir, image_name)
    cv2.imwrite(figure_path, image)
    
    return figure_path

def plot_prediction_dataframe(df, root_dir, ground_truth=None, savedir=None):
    """For each row in dataframe, call plot predictions. For multi-class labels, boxes will be colored by labels. Ground truth boxes will all be same color, regardless of class.
    Args:
        df: a pandas dataframe with image_path, xmin, xmax, ymin, ymax and label columns. The image_path column should be the relative path from root_dir, not the full path.
        root_dir: relative dir to look for image names from df.image_path
        ground_truth: an optional pandas dataframe in same format as df holding ground_truth boxes
        savedir: save the plot to an optional directory path.
    Returns:
        written_figures: list of filenames written
        """
    written_figures = []
    for name, group in df.groupby("image_path"):
        image = np.array(Image.open("{}/{}".format(root_dir, name)))[:,:,::-1].copy()
        image = plot_predictions(image, group)
        
        if ground_truth is not None:
            annotations = ground_truth[ground_truth.image_path == name]
            image = plot_predictions(image, annotations)
            
        if savedir:
            figure_name = "{}/{}.png".format(savedir, os.path.splitext(name)[0])
            written_figures.append(figure_name)
            cv2.imwrite(figure_name, image)
    
    return written_figures

def plot_predictions(image, df, color=None, thickness=1):
    """Plot a set of boxes on an image
    By default this function does not show, but only plots an axis
    Label column must be numeric!
    Image must be BGR color order!
    Args:
        image: a numpy array in *BGR* color order! Channel order is channels first 
        df: a pandas dataframe with xmin, xmax, ymin, ymax and label column
        color: color of the bounding box as a tuple of BGR color, e.g. orange annotations is (0, 165, 255)
        thickness: thickness of the rectangle border line in px
    Returns:
        image: a numpy array with drawn annotations
    """    
    if image.shape[0] == 3:
        warnings.warn("Input images must be channels last format [h, w, 3] not channels first [3, h, w], using np.rollaxis(image, 0, 3) to invert!")
        image = np.rollaxis(image, 0, 3)
    if image.dtype == "float32":
        image = image.astype("uint8")
    image = image.copy()
    if not color:
        if not ptypes.is_numeric_dtype(df.label):
            warnings.warn("No color was provided and the label column is not numeric. Using a single default color.")
            color=(0,165,255)

    for index, row in df.iterrows():
        if not color:
            color = label_to_color(row["label"])
        cv2.rectangle(image, (int(row["xmin"]), int(row["ymin"])), (int(row["xmax"]), int(row["ymax"])), color=color, thickness=thickness, lineType=cv2.LINE_AA)
    
    return image


def label_to_color(label):
    color_dict = {}
    
    random.seed(1)
    colors = [list((matplotlib.colors.hsv_to_rgb([x, 1.0, 1.0]) * 255).astype(int)) for x in np.arange(0, 1, 1/80)]
    colors = [tuple([int(y) for y in x]) for x in colors]
    random.shuffle(colors)
    
    for index, color in enumerate(colors):
        color_dict[index] = color

    # hand pick the first few colors
    color_dict[0] = (255,255,0)
    color_dict[1] = (71, 99,255)
    color_dict[2] = (255,0,0)
    color_dict[3] = (50,205,50)
    color_dict[4] = (214,112,214)
    color_dict[5] = (60, 20, 220)
    color_dict[6] = (63, 133, 205)
    color_dict[7] = (255, 144, 30)
    color_dict[8] = (0, 215 ,255)

    return color_dict[label]
