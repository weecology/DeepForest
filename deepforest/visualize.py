# Visualize module for plotting and handling predictions
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import pandas.api.types as ptypes

def view_dataset(ds, savedir=None):
    """Plot annotations on images for debugging purposes
    Args:
        ds: a deepforest-pytorch dataset, see deepforest.dataset or deepforest.load_dataset() to start from a csv file
        savedir: optional path to save figures. If none (default) images will be interactively plotted
    """
    for i in iter(ds):
        image_path, image, targets = i
        df = format_boxes(targets[0], scores=False)
        image = np.moveaxis(image[0].numpy(),0,2)
        plot, ax = plot_predictions(image, df)
    
    if savedir:
        plot.savefig("{}/{}".format(savedir, image_path[0]), dpi=300)
    else:
        plt.show()
            
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
    """Plot an image, its predictions, and its ground truth targets for debugging"""
    prediction_df = format_boxes(predictions)
    plot, ax = plot_predictions(image, prediction_df)
    target_df = format_boxes(targets, scores=False)
    plot = add_annotations(plot, ax, target_df)
    plot.savefig("{}/{}.png".format(savedir, image_name), dpi=300)
    return "{}/{}.png".format(savedir, image_name)


def plot_prediction_dataframe(df, root_dir, ground_truth=None, savedir=None, show=False):
    """For each row in dataframe, call plot predictions. For multi-class labels, boxes will be colored by labels. Ground truth boxes will all be same color, regardless of class.
    Args:
        df: a pandas dataframe with image_path, xmin, xmax, ymin, ymax and label columns. The image_path column should be the relative path from root_dir, not the full path.
        root_dir: relative dir to look for image names from df.image_path
        ground_truth: an optional pandas dataframe in same format as df holding ground_truth boxes
        savedir: save the plot to an optional directory path.
        show (logical): Render the plot in the matplotlib GUI
    Returns:
        None: side-effect plots are saved or generated and viewed
        """
    for name, group in df.groupby("image_path"):
        image = np.array(Image.open("{}/{}".format(root_dir, name)))
        plot, ax = plot_predictions(image, group, show=show)
        
        if ground_truth is not None:
            annotations = ground_truth[ground_truth.image_path == name]
            plot = add_annotations(plot, ax, annotations)
            
        if savedir:
            plot.savefig("{}/{}.png".format(savedir, os.path.splitext(name)[0]))
    

def plot_predictions(image, df, show=False):
    """channel order is channels first for pytorch
    By default this function does not show, but only plots an axis
    Label column must be numeric!
    """
    if not show:
        original_backend = matplotlib.get_backend()
        matplotlib.use("Agg")

    if not ptypes.is_numeric_dtype(df.label):
        raise ValueError("Label column is not numeric, please convert to numeric to correctly color image {}".format(df.label.head()))

    #What size does the figure need to be in inches to fit the image?
    dpi=300
    height, width, nbands = image.shape
    figsize = width / float(dpi), height / float(dpi)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    for index, row in df.iterrows():
        xmin = row["xmin"]
        ymin = row["ymin"]
        width = row["xmax"] - xmin
        height = row["ymax"] - ymin
        color = label_to_color(row["label"])
        rect = create_box(xmin=xmin, ymin=ymin, height=height, width=width, color=color)
        ax.add_patch(rect)
    # no axis show up
    plt.axis('off')
    
    #reload matplotlib to get use back their favorite backend.
    if not show:
        matplotlib.use(original_backend)
    
    return fig, ax


def create_box(xmin, ymin, height, width, color="cyan", linewidth=0.75):
    rect = patches.Rectangle((xmin, ymin),
                             height,
                             width,
                             linewidth=linewidth,
                             edgecolor=color,
                             fill=False)
    return rect


def add_annotations(plot, ax, annotations):
    """Add annotations to an already created visuale.plot_predictions
    Args:
        plot: matplotlib figure object
        ax: maplotlib axes object
        annotations: pandas dataframe of bounding box annotations
    Returns:
        plot: matplotlib figure object
    """
    for index, row in annotations.iterrows():
        xmin = row["xmin"]
        ymin = row["ymin"]
        width = row["xmax"] - xmin
        height = row["ymax"] - ymin
        rect = create_box(xmin=xmin,
                          ymin=ymin,
                          height=height,
                          width=width,
                          color="orange")
        ax.add_patch(rect)

    return plot


def label_to_color(label):
    color_dict = {}
    colors = [
        list((matplotlib.colors.hsv_to_rgb([x, 1.0, 1.0])).astype(int))
        for x in np.arange(0, 1, 1.0 / 80)
    ]
    for index, color in enumerate(colors):
        color_dict[index] = color

    # hand pick the first few colors
    color_dict[0] = "cyan"
    color_dict[1] = "tomato"
    color_dict[2] = "blue"
    color_dict[3] = "limegreen"
    color_dict[4] = "orchid"
    color_dict[5] = "crimson"
    color_dict[6] = "peru"
    color_dict[7] = "dodgerblue"
    color_dict[8] = "gold"
    color_dict[9] = "blueviolet"

    return color_dict[label]
