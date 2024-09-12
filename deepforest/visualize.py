# Visualize module for plotting and handling predictions
import os
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import pandas.api.types as ptypes
import cv2
import random
import warnings
import supervision as sv
import shapely
from deepforest.utilities import determine_geometry_type


def view_dataset(ds, savedir=None, color=None, thickness=1):
    """Plot annotations on images for debugging purposes.

    Args:
        ds: a deepforest pytorch dataset, see deepforest.dataset or deepforest.load_dataset() to start from a csv file
        savedir: optional path to save figures. If none (default) images will be interactively plotted
        color: color of the bounding box as a tuple of BGR color, e.g. orange annotations is (0, 165, 255)
        thickness: thickness of the rectangle border line in px
    """
    for i in iter(ds):
        image_path, image, targets = i
        df = format_boxes(targets[0], scores=False)
        image = np.moveaxis(image[0].numpy(), 0, 2)
        image = plot_predictions(image, df, color=color, thickness=thickness)

    if savedir:
        cv2.imwrite("{}/{}".format(savedir, image_path[0]), image)
    else:
        plt.imshow(image)


def format_geometry(predictions, scores=True):
    """Format a retinanet prediction into a pandas dataframe for a batch of images
    Args:
        predictions: a list of dictionaries with keys 'boxes' and 'labels' coming from a retinanet
        scores: Whether boxes come with scores, during prediction, or without scores, as in during training.
    Returns:
        df: a pandas dataframe
    """
    # Detect geometry type
    geom_type = determine_geometry_type(predictions)

    if geom_type == "box":
        df = format_boxes(predictions, scores=scores)
        df['geometry'] = df.apply(
            lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
    elif geom_type == "polygon":
        raise ValueError("Polygon predictions are not yet supported for formatting")
    elif geom_type == "point":
        raise ValueError("Point predictions are not yet supported for formatting")

    return df


def format_boxes(prediction, scores=True):
    """Format a retinanet prediction into a pandas dataframe for a single
    image.

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
    """Plot an image, its predictions, and its ground truth targets for
    debugging.

    Args:
        image: torch tensor, RGB color order
        targets: torch tensor
    Returns:
        figure_path: path on disk with saved figure
    """
    image = np.array(image)[:, :, ::-1].copy()
    prediction_df = format_boxes(predictions)
    image = plot_predictions(image, prediction_df)
    target_df = format_boxes(targets, scores=False)
    image = plot_predictions(image, target_df)
    figure_path = "{}/{}.png".format(savedir, image_name)
    cv2.imwrite(figure_path, image)

    return figure_path


def plot_prediction_dataframe(df,
                              root_dir,
                              savedir,
                              color=None,
                              thickness=1,
                              ground_truth=None):
    """For each row in dataframe, call plot predictions and save plot files to
    disk. For multi-class labels, boxes will be colored by labels. Ground truth
    boxes will all be same color, regardless of class.

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
        image = np.array(Image.open("{}/{}".format(root_dir, name)))[:, :, ::-1].copy()
        image = plot_predictions(image, group)

        if ground_truth is not None:
            annotations = ground_truth[ground_truth.image_path == name]
            image = plot_predictions(image, annotations, color=color, thickness=thickness)

        figure_name = "{}/{}.png".format(savedir, os.path.splitext(name)[0])
        written_figures.append(figure_name)
        cv2.imwrite(figure_name, image)

    return written_figures


def plot_points(image, points, color=None, radius=5, thickness=1):
    """Plot points on an image
    Args:
        image: a numpy array in *BGR* color order! Channel order is channels first 
        points: a numpy array of shape (N, 2) representing the coordinates of the points
        color: color of the points as a tuple of BGR color, e.g. orange points is (0, 165, 255)
        radius: radius of the points in px
        thickness: thickness of the point border line in px
    Returns:
        image: a numpy array with drawn points
    """
    if image.shape[0] == 3:
        warnings.warn("Input images must be channels last format [h, w, 3] not channels "
                      "first [3, h, w], using np.rollaxis(image, 0, 3) to invert!")
        image = np.rollaxis(image, 0, 3)
    if image.dtype == "float32":
        image = image.astype("uint8")
    image = image.copy()
    if not color:
        color = (0, 165, 255)  # Default color is orange

    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])),
                   color=color,
                   radius=radius,
                   thickness=thickness)

    return image


def plot_predictions(image, df, color=None, thickness=1):
    """Plot a set of boxes on an image By default this function does not show,
    but only plots an axis Label column must be numeric! Image must be BGR
    color order!

    Args:
        image: a numpy array in *BGR* color order! Channel order is channels first
        df: a pandas dataframe with xmin, xmax, ymin, ymax and label column
        color: color of the bounding box as a tuple of BGR color, e.g. orange annotations is (0, 165, 255)
        thickness: thickness of the rectangle border line in px
    Returns:
        image: a numpy array with drawn annotations
    """
    if image.shape[0] == 3:
        warnings.warn("Input images must be channels last format [h, w, 3] not channels "
                      "first [3, h, w], using np.rollaxis(image, 0, 3) to invert!")
        image = np.rollaxis(image, 0, 3)
    if image.dtype == "float32":
        image = image.astype("uint8")
    image = image.copy()
    if not color:
        if not ptypes.is_numeric_dtype(df.label):
            warnings.warn("No color was provided and the label column is not numeric. "
                          "Using a single default color.")
            color = (0, 165, 255)

    for index, row in df.iterrows():
        if not color:
            color = label_to_color(row["label"])

        # Plot rectangle, points or polygon
        if "geometry" in df.columns:
            # Get geometry type
            geometry_type = row["geometry"].geom_type
            if geometry_type == "Polygon":
                # convert to int32 and numpy array
                int_coords = lambda x: np.array(x).round().astype(np.int32)
                polygon = [int_coords(row["geometry"].exterior.coords)]
                cv2.polylines(image, polygon, True, color, thickness=thickness)
            elif geometry_type == "Point":
                int_coords = lambda x: np.array(x).round().astype(np.int32)
                cv2.circle(image,
                           (int_coords(row["geometry"].x), int_coords(row["geometry"].y)),
                           color=color,
                           radius=5,
                           thickness=thickness)
            else:
                raise ValueError("Only polygons and points are supported")
        elif "xmin" in df.columns:
            cv2.rectangle(image, (int(row["xmin"]), int(row["ymin"])),
                          (int(row["xmax"]), int(row["ymax"])),
                          color=color,
                          thickness=thickness,
                          lineType=cv2.LINE_AA)
        elif "x" in df.columns:
            cv2.circle(image, (row["x"], row["y"]),
                       color=color,
                       radius=5,
                       thickness=thickness)
        elif "polygon" in df.columns:
            polygon = np.array(row["polygon"])
            cv2.polylines(image, [polygon], True, color, thickness=thickness)

    return image


def label_to_color(label):
    color_dict = {}

    random.seed(1)
    colors = [
        list((matplotlib.colors.hsv_to_rgb([x, 1.0, 1.0]) * 255).astype(int))
        for x in np.arange(0, 1, 1 / 80)
    ]
    colors = [tuple([int(y) for y in x]) for x in colors]
    random.shuffle(colors)

    for index, color in enumerate(colors):
        color_dict[index] = color

    # hand pick the first few colors
    color_dict[0] = (255, 255, 0)
    color_dict[1] = (71, 99, 255)
    color_dict[2] = (255, 0, 0)
    color_dict[3] = (50, 205, 50)
    color_dict[4] = (214, 112, 214)
    color_dict[5] = (60, 20, 220)
    color_dict[6] = (63, 133, 205)
    color_dict[7] = (255, 144, 30)
    color_dict[8] = (0, 215, 255)

    return color_dict[label]


def convert_to_sv_format(df, width=None, height=None):
    """Convert DeepForest prediction results to a supervision Detections
    object.

    Args:
        df (pd.DataFrame): The results from `predict_image` or `predict_tile`.
                           Expected columns: ['xmin', 'ymin', 'xmax', 'ymax', 'label', 'score', 'image_path'].
        width (int): The width of the image in pixels. Required if the geometry type is 'polygon'.
        height (int): The height of the image in pixels. Required if the geometry type is 'polygon'.

    Returns:
        sv.Detections: A supervision Detections object containing bounding boxes, class IDs,
                       confidence scores, and class names object mapping classes ids to corresponding
                       class names inside of data dictionary.


    Example:
        detections = convert_to_sv_format(result)
    """
    geom_type = determine_geometry_type(df)

    if geom_type == "box":
        # Extract bounding boxes as a 2D numpy array with shape (_, 4)
        boxes = df.geometry.apply(
            lambda x: (x.bounds[0], x.bounds[1], x.bounds[2], x.bounds[3])).values
        boxes = np.stack(boxes)

        label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}

        # Extract labels as a numpy array
        labels = df['label'].map(label_mapping).values.astype(int)

        # Extract scores as a numpy array
        try:
            scores = np.array(df['score'].tolist())
        except KeyError:
            scores = np.ones(len(labels))

        # Create a reverse mapping from integer to string labels
        class_name = {v: k for k, v in label_mapping.items()}

        detections = sv.Detections(
            xyxy=boxes,
            class_id=labels,
            confidence=scores,
            data={"class_name": [class_name[class_id] for class_id in labels]})

    elif geom_type == "polygon":
        # Extract bounding boxes as a 2D numpy array with shape (_, 4)
        boxes = df.geometry.apply(
            lambda x: (x.bounds[0], x.bounds[1], x.bounds[2], x.bounds[3])).values
        boxes = np.stack(boxes)

        label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}

        # Extract labels as a numpy array
        labels = df['label'].map(label_mapping).values.astype(int)

        # Extract scores as a numpy array
        scores = np.array(df['score'].tolist())
        # Create a reverse mapping from integer to string labels
        class_name = {v: k for k, v in label_mapping.items()}

        # Create masks
        if height is None or width is None:
            raise ValueError(
                "height and width of the mask must be provided for polygon predictions")

        polygons = df.geometry.apply(lambda x: np.array(x.exterior.coords)).values
        # as integers
        polygons = [np.array(p).round().astype(np.int32) for p in polygons]
        masks = [sv.polygon_to_mask(p, (width, height)) for p in polygons]
        masks = np.stack(masks)

        detections = sv.Detections(
            xyxy=boxes,
            mask=masks,
            class_id=labels,
            confidence=scores,
            data={"class_name": [class_name[class_id] for class_id in labels]})

    elif geom_type == "point":
        points = df.geometry.apply(lambda x: (x.x, x.y)).values
        points = np.stack(points)
        points = np.expand_dims(points, axis=1)

        label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}

        # Extract labels as a numpy array
        labels = df['label'].map(label_mapping).values.astype(int)

        # Extract scores as a numpy array
        scores = np.array(df['score'].tolist())
        scores = np.expand_dims(np.stack(scores), 1)

        # Create a reverse mapping from integer to string labels
        class_name = {v: k for k, v in label_mapping.items()}

        detections = sv.KeyPoints(
            xy=points,
            class_id=labels,
            confidence=scores,
            data={"class_name": [class_name[class_id] for class_id in labels]})

    return detections


def plot_results(results,
                 ground_truth=None,
                 savedir=None,
                 height=None,
                 width=None,
                 results_color=None,
                 ground_truth_color=None,
                 thickness=2,
                 radius=3):
    """Plot the prediction results.

    Args:
        df: a pandas dataframe with prediction results
        ground_truth: an optional pandas dataframe with ground truth annotations
        savedir: optional path to save the figure. If None (default), the figure will be interactively plotted.
        height: height of the image in pixels. Required if the geometry type is 'polygon'.
        width: width of the image in pixels. Required if the geometry type is 'polygon'.
        results_color (list): color of the results annotations as a tuple of RGB color, e.g. orange annotations is [245, 135, 66]
    Returns:
        None
    """
    # Convert colors, check for multi-class labels
    if results_color is None:
        sv_color = sv.Color(245, 135, 66)
    elif type(results_color) is list:
        sv_color = sv.Color(results_color[0], results_color[1], results_color[2])
    else:
        sv_color = results_color

    num_labels = len(results.label.unique())
    if num_labels > 1 and results_color is not None:
        warnings.warn(
            "Multiple labels detected in the results. Each label will be plotted with a different color using a color ramp, results color argument is ignored."
        )
    if num_labels > 1:
        sv_color = sv.ColorPalette.from_matplotlib('viridis', 5)

    # Read images
    root_dir = results.root_dir
    image_path = os.path.join(root_dir, results.image_path.unique()[0])
    image = np.array(Image.open(image_path))

    # Plot the results following https://supervision.roboflow.com/annotators/
    fig, ax = plt.subplots()
    annotated_scene = _plot_image_with_results(df=results,
                                               image=image,
                                               sv_color=sv_color,
                                               height=height,
                                               width=width,
                                               thickness=thickness,
                                               radius=radius)

    if ground_truth is not None:
        if ground_truth_color is None:
            sv_color = sv.Color(0, 165, 255)
        elif type(ground_truth_color) is list:
            sv_color = sv.Color(ground_truth_color[0], ground_truth_color[1],
                                ground_truth_color[2])
        else:
            sv_color = ground_truth_color
        # Plot the ground truth annotations
        annotated_scene = _plot_image_with_results(df=ground_truth,
                                                   image=annotated_scene,
                                                   sv_color=sv_color,
                                                   height=height,
                                                   width=width,
                                                   thickness=thickness,
                                                   radius=radius)

    if savedir:
        basename = os.path.splitext(os.path.basename(results.image_path.unique()[0]))[0]
        image_name = "{}.png".format(basename)
        plt.savefig(os.path.join(savedir, image_name), bbox_inches='tight', pad_inches=0)
    else:
        # Display the image using Matplotlib
        plt.imshow(annotated_scene)
        plt.axis('off')  # Hide axes for a cleaner look
        plt.show()


def _plot_image_with_results(df,
                             image,
                             sv_color,
                             thickness=1,
                             radius=3,
                             height=None,
                             width=None):
    """Annotates an image with the given results.

    Args:
        df (pandas.DataFrame): The DataFrame containing the results.
        image (numpy.ndarray): The image to annotate.
        sv_color (str): The color of the annotations.
        thickness (int): The thickness of the annotations.

    Returns:
        numpy.ndarray: The annotated image.
    """
    # Determine the geometry type
    geom_type = determine_geometry_type(df)
    detections = convert_to_sv_format(df, height=height, width=width)

    if geom_type == "box":
        bounding_box_annotator = sv.BoxAnnotator(color=sv_color, thickness=thickness)
        annotated_frame = bounding_box_annotator.annotate(
            scene=image.copy(),
            detections=detections,
        )
    elif geom_type == "polygon":

        polygon_annotator = sv.PolygonAnnotator(color=sv_color, thickness=thickness)
        annotated_frame = polygon_annotator.annotate(
            scene=image.copy(),
            detections=detections,
        )
    elif geom_type == "point":
        point_annotator = sv.VertexAnnotator(color=sv_color, radius=radius)
        annotated_frame = point_annotator.annotate(scene=image.copy(),
                                                   key_points=detections)
    return annotated_frame
