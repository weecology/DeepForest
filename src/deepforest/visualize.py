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
from typing import Optional, Union
from deepforest.utilities import determine_geometry_type


def _load_image(image: Optional[Union[np.typing.NDArray, str, Image.Image]] = None,
                df: Optional[pd.DataFrame] = None,
                root_dir: Optional[str] = None) -> np.typing.NDArray:
    """Utility function to load an image from either a path or a
    prediction/annotation dataframe.

    Returns an image in RGB format with HWC channel ordering.

    Args:
        image (optional): Numpy array or string
        df (optiona): Pandas dataframe
        root_dir (optional): Root directory of the image, will override dataframe root_dir attribute

    Returns:
        image: Numpy array
    """

    if image is None and df is None:
        raise ValueError(
            "Either an image or a valid dataframe must be provided for plotting.")

    if df is not None:
        # Resolve image root
        if hasattr(df, 'root_dir') and root_dir is None:
            root_dir = df.root_dir
            # expected str, bytes or os.PathLike object, not Series
            root_dir = df.root_dir.iloc[0] if isinstance(root_dir,
                                                         pd.Series) else root_dir
        elif root_dir is None:
            raise ValueError(
                "Neither root_dir nor a dataframe with the root_dir attribute was provided."
            )

        image_path = os.path.join(root_dir, df.image_path.unique()[0])
        image = np.array(Image.open(image_path))
    elif isinstance(image, str):
        if root_dir is not None:
            image_path = os.path.join(root_dir, image)
        else:
            image_path = image

        image = np.array(Image.open(image_path))
    elif isinstance(image, Image.Image):
        image = np.array(image)
    elif not isinstance(image, np.ndarray):
        raise ValueError("Image should be a numpy array, path or PIL Image.")

    # Fix channel ordering
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[2] != 3:
        warnings.warn("Input images must be channels last format [h, w, 3] not channels "
                      "first [3, h, w], using np.transpose(1,2,0) to invert!")
        image = image.transpose(1, 2, 0)

    # Drop alpha channel if present and warn
    if image.ndim == 3 and image.shape[2] == 4:
        warnings.warn(
            f"Image has {image.ndim} bands (may have an alpha channel). Only keeping first 3."
        )
        image = image[:, :, :3]

    if image.dtype != np.uint8:

        warnings.warn(
            f"Image is {image.dtype}. Will be cast to 8-bit unsigned and clipped to [0,255]"
        )

        # Images in [0,1] are allowed, but should be rescaled
        if image.max() <= 1 and image.min() >= 0:
            warnings.warn(
                f"Image is in [0,1], multiplying by 255. If this is not expected")
            image *= 255

        image = np.clip(image, 0, 255).astype('uint8')

    return image


def plot_points(image: np.typing.NDArray,
                points: np.typing.NDArray,
                color: Optional[tuple] = None,
                radius: int = 5,
                thickness: int = 1) -> np.typing.NDArray:
    """Draw points on an image, returns a copy of the array
    Args:
        image: a numpy array in RGB order, HWC format
        points: a numpy array of shape (N, 2) representing the coordinates of the points
        color: color of the points as a tuple of BGR color, e.g. orange points is (0, 165, 255)
        radius: radius of the points in px
        thickness: thickness of the point border line in px
    Returns:
        image: a numpy array with drawn points
    """
    warnings.warn(
        "plot_points will be deprecated in 2.0, please use draw_points instead.",
        DeprecationWarning)
    draw_points(image, points, color, radius, thickness)


def draw_points(image: np.typing.NDArray,
                points: np.typing.NDArray,
                color: Optional[tuple] = None,
                radius: int = 5,
                thickness: int = 1) -> np.typing.NDArray:
    """Draw points on an image, returns a copy of the array.

    Args:
        image: a numpy array in RGB order, HWC format
        points: a numpy array of shape (N, 2) representing the coordinates of the points
        color: color of the points as a tuple of BGR color, e.g. orange points is (0, 165, 255)
        radius: radius of the points in px
        thickness: thickness of the point border line in px
    Returns:
        image: a numpy array with drawn points
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()

    if not color:
        color = (0, 165, 255)  # Default color is orange

    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])),
                   color=color,
                   radius=radius,
                   thickness=thickness)

    return image


def plot_predictions(image: np.typing.NDArray,
                     df: pd.DataFrame,
                     color: tuple = None,
                     thickness: int = 1) -> np.typing.NDArray:
    """Draw geometries on an image, which can be polygons, boxes or points.

    Returns a copy of the array.

    Args:
        image: a numpy array in RGB order, HWC format
        df: a pandas dataframe with xmin, xmax, ymin, ymax and label column
        color: color of the bounding box as a tuple of BGR color, e.g. orange annotations is (0, 165, 255)
        thickness: thickness of the rectangle border line in px
    Returns:
        image: a numpy array with drawn annotations
    """
    warnings.warn(
        "plot_predictions will be deprecated in 2.0, please use draw_predictions instead. Or plot_results if you need a figure.",
        DeprecationWarning)
    draw_predictions(image, df, color, thickness)


def draw_predictions(image: np.typing.NDArray,
                     df: pd.DataFrame,
                     color: Optional[tuple] = None,
                     thickness: int = 1) -> np.typing.NDArray:
    """Draw geometries on an image, which can be polygons, boxes or points.

    Returns a copy of the array.

    Args:
        image: a numpy array in RGB order, HWC format
        df: a pandas dataframe with xmin, xmax, ymin, ymax and label column
        color: color of the bounding box as a tuple of BGR color, e.g. orange annotations is (0, 165, 255)
        thickness: thickness of the rectangle border line in px
    Returns:
        image: a numpy array with drawn annotations
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()

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


def label_to_color(label: int) -> tuple:
    """Return an RGB color tuple for a given (integer) label."""
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


def convert_to_sv_format(
        df: pd.DataFrame,
        width: Optional[int] = None,
        height: Optional[int] = None) -> Union[sv.Detections, sv.KeyPoints]:
    """Convert DeepForest prediction results to a supervision Detections
    object.

    Args:
        df (pd.DataFrame): The results from `predict_image` or `predict_tile`.
                           Expected columns includes: ['geometry', 'label', 'score', 'image_path'] for bounding boxes
        width (int): The width of the image in pixels. Only required if the geometry type is 'polygon'.
        height (int): The height of the image in pixels. Only required if the geometry type is 'polygon'.

    Returns:
        Depending on the geometry type, the function returns either a Detections or a KeyPoints object from the supervision library.
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
        try:
            scores = np.array(df['score'].tolist())
        except KeyError:
            scores = np.ones(len(labels))

        # Create a reverse mapping from integer to string labels
        class_name = {v: k for k, v in label_mapping.items()}

        # Auto-detect width/height if missing
        if width is None or height is None:
            if 'image_path' not in df.columns:
                raise ValueError("'image_path' column required for polygons.")

            # Use the first image_path entry
            image_path = df['image_path'].iloc[0]
            try:
                with Image.open(image_path) as img:
                    width, height = img.size  # Get dimensions
            except Exception as e:
                raise ValueError(
                    f"Could not read image dimensions from {image_path}: {e}")

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
        try:
            scores = np.array(df['score'].tolist())
        except KeyError:
            scores = np.ones(len(labels))

        scores = np.expand_dims(np.stack(scores), 1)

        # Create a reverse mapping from integer to string labels
        class_name = {v: k for k, v in label_mapping.items()}

        detections = sv.KeyPoints(
            xy=points,
            class_id=labels,
            confidence=scores,
            data={"class_name": [class_name[class_id] for class_id in labels]})

    return detections


def __check_color__(color: Union[list, tuple, sv.ColorPalette],
                    num_labels: int) -> Union[sv.Color, sv.ColorPalette]:
    if isinstance(color, list) and len(color) == 3:
        if num_labels > 1:
            warnings.warn(
                """Multiple labels detected in the results and results_color argument provides a single color.
                Each label will be plotted with a different color using a built-in color ramp.
                If you want to customize colors with multiple labels pass a supervision.ColorPalette object to results_color with the appropriate number of labels"""
            )
            return sv.ColorPalette.from_matplotlib('viridis', num_labels)
        else:
            return sv.Color(color[0], color[1], color[2])
    elif isinstance(color, sv.draw.color.ColorPalette):
        if num_labels > len(color.colors):
            warnings.warn(
                """The number of colors provided in results_color does not match the number number of labels.
                Replacing the provided color palette with a built-in built-in color palette.
                To use a custom color palette make sure the number of values matches the number of labels"""
            )
            return sv.ColorPalette.from_matplotlib('viridis', num_labels)
        else:
            return color
    elif isinstance(color, list):
        raise ValueError(
            "results_color must be either a 3 item list containing RGB values or an sv.ColorPalette instance"
        )
    else:
        raise TypeError(
            "results_color must be either a list of RGB values or an sv.ColorPalette instance"
        )


def plot_annotations(
        annotations: pd.DataFrame,
        savedir: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        color: Union[list, sv.ColorPalette] = [245, 135, 66],
        thickness: int = 2,
        basename: Optional[str] = None,
        root_dir: Optional[str] = None,
        radius: int = 3,
        image: Optional[Union[np.typing.NDArray, str, Image.Image]] = None) -> None:
    """Plot prediction results or ground truth annotations for a single image.

    This function can be used to create a figure which can be saved or shown. If you wish
    to do further plotting, you can return the axis object by passing axes=True.

    Args:
        annotations: a pandas dataframe with prediction results
        savedir: optional path to save the figure. If None (default), the figure will be interactively plotted.
        height: height of the image in pixels. Required if the geometry type is 'polygon'.
        width: width of the image in pixels. Required if the geometry type is 'polygon'.
        results_color (list or sv.ColorPalette): color of the results annotations as a tuple of RGB color (if a single color), e.g. orange annotations is [245, 135, 66], or an supervision.ColorPalette if multiple labels and specifying colors for each label
        thickness: thickness of the rectangle border line in px
        basename: optional basename for the saved figure. If None (default), the basename will be extracted from the image path.
        root_dir: optional path to the root directory of the image. If None (default), the root directory will be extracted from the annotations dataframe.root_dir attribute.
        radius: radius of the points in px
        image: image (numpy array, string, PIL image)
    Returns:
        None
    """
    # Convert colors, check for multi-class labels
    num_labels = len(annotations.label.unique())
    annotation_color = __check_color__(color, num_labels)

    image = _load_image(image, annotations, root_dir)

    # Plot the results following https://supervision.roboflow.com/annotators/
    plt.subplots()
    annotated_scene = _plot_image_with_geometry(df=annotations,
                                                image=image,
                                                sv_color=annotation_color,
                                                height=height,
                                                width=width,
                                                thickness=thickness,
                                                radius=radius)

    if savedir:
        if basename is None:
            basename = os.path.splitext(
                os.path.basename(annotations.image_path.unique()[0]))[0]
        image_name = "{}.png".format(basename)
        image_path = os.path.join(savedir, image_name)
        cv2.imwrite(image_path, annotated_scene)
    else:
        # Display the image using Matplotlib
        plt.imshow(annotated_scene)
        plt.axis('off')  # Hide axes for a cleaner look
        plt.show()


def plot_results(results: pd.DataFrame,
                 ground_truth: Optional[pd.DataFrame] = None,
                 savedir: Optional[str] = None,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 results_color: Union[list, sv.ColorPalette] = [245, 135, 66],
                 ground_truth_color: Union[list, sv.ColorPalette] = [0, 165, 255],
                 thickness: int = 2,
                 basename: Optional[str] = None,
                 radius: int = 3,
                 image: Optional[Union[np.typing.NDArray, str, Image.Image]] = None,
                 axes: bool = False):
    """Plot prediction results and optionally ground truth annotations.

    This function can be used to create a figure which can be saved or shown. If you wish
    to do further plotting, you can return the axis object by passing axes=True.

    Args:
        results: a pandas dataframe with prediction results
        ground_truth: an optional pandas dataframe with ground truth annotations
        savedir: optional path to save the figure. If None (default), the figure will be interactively plotted.
        height: height of the image in pixels. Required if the geometry type is 'polygon'.
        width: width of the image in pixels. Required if the geometry type is 'polygon'.
        results_color (list or sv.ColorPalette): color of the results annotations as a tuple of RGB color (if a single color), e.g. orange annotations is [245, 135, 66], or an supervision.ColorPalette if multiple labels and specifying colors for each label
        ground_truth_color (list): color of the ground truth annotations as a tuple of RGB color, e.g. blue annotations is [0, 165, 255]
        thickness: thickness of the rectangle border line in px
        basename: optional basename for the saved figure. If None (default), the basename will be extracted from the image path.
        radius: radius of the points in px
        image: an optional numpy array or path to an image to annotate. If None (default), the image will be loaded from the results dataframe.
        axes: returns matplotlib axes object if True
    Returns:
        Matplotlib axes object if axes=True, otherwise None
    """
    # Convert colors, check for multi-class labels
    num_labels = len(results.label.unique())
    results_color_sv = __check_color__(results_color, num_labels)
    ground_truth_color_sv = __check_color__(ground_truth_color, num_labels)

    image = _load_image(image, results)

    # Plot the results following https://supervision.roboflow.com/annotators/
    _, ax = plt.subplots()
    annotated_scene = _plot_image_with_geometry(df=results,
                                                image=image,
                                                sv_color=results_color_sv,
                                                height=height,
                                                width=width,
                                                thickness=thickness,
                                                radius=radius)

    if ground_truth is not None:
        # Plot the ground truth annotations
        annotated_scene = _plot_image_with_geometry(df=ground_truth,
                                                    image=annotated_scene,
                                                    sv_color=ground_truth_color_sv,
                                                    height=height,
                                                    width=width,
                                                    thickness=thickness,
                                                    radius=radius)

    if savedir:
        if basename is None:
            basename = os.path.splitext(os.path.basename(
                results.image_path.unique()[0]))[0]
        image_name = "{}.png".format(basename)
        image_path = os.path.join(savedir, image_name)
        cv2.imwrite(image_path, annotated_scene)
    else:
        # Display the image using Matplotlib
        plt.imshow(annotated_scene)
        if axes:
            return ax
        plt.axis('off')  # Hide axes for a cleaner look
        plt.show()


def _plot_image_with_geometry(df,
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
