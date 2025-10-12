# Visualize module for plotting and handling predictions
import os
import random
import warnings

import matplotlib
import numpy as np
import pandas as pd
import supervision as sv
from matplotlib import pyplot as plt
from PIL import Image

from deepforest.utilities import determine_geometry_type


def _load_image(
    image: np.typing.NDArray | str | Image.Image | None = None,
    df: pd.DataFrame | None = None,
    root_dir: str | None = None,
) -> np.typing.NDArray:
    """Utility function to load an image from either a path or a
    prediction/annotation dataframe. If both are passed, image takes
    precedence.

    Returns an image in RGB format with HWC channel ordering.

    Args:
        image (optional): Numpy array or string
        df (optiona): Pandas dataframe
        root_dir (optional): Root directory of the image, will override dataframe root_dir attribute

    Returns:
        image: Numpy array
    """

    if image is not None:
        if isinstance(image, str):
            if root_dir is not None:
                image_path = os.path.join(root_dir, image)
            else:
                image_path = image

            image = np.array(Image.open(image_path))
        elif isinstance(image, Image.Image):
            image = np.array(image)
        elif not isinstance(image, np.ndarray):
            raise ValueError("Image should be a numpy array, path or PIL Image.")

    elif df is not None:
        # Resolve image root
        if hasattr(df, "root_dir") and root_dir is None:
            root_dir = df.root_dir
            # expected str, bytes or os.PathLike object, not Series
            root_dir = (
                df.root_dir.iloc[0] if isinstance(root_dir, pd.Series) else root_dir
            )
        elif root_dir is None:
            raise ValueError(
                "Neither root_dir nor a dataframe with the root_dir attribute was provided."
                "This can happen if your dataframe contains predictions that aren't associated with a file."
                "If you called plot_results, try providing an image (array or path) alongside the dataframe."
            )

        image_path = os.path.join(root_dir, df.image_path.unique()[0])
        image = np.array(Image.open(image_path))
    else:
        raise ValueError(
            "Either an image or a valid dataframe must be provided for plotting."
        )

    # Fix channel ordering
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[2] != 3:
        warnings.warn(
            "Input images must be channels last format [h, w, 3] not channels "
            "first [3, h, w], using np.transpose(1,2,0) to invert!",
            stacklevel=2,
        )
        image = image.transpose(1, 2, 0)

    # Drop alpha channel if present and warn
    if image.ndim == 3 and image.shape[2] == 4:
        warnings.warn(
            f"Image has {image.ndim} bands (may have an alpha channel). Only keeping first 3.",
            stacklevel=2,
        )
        image = image[:, :, :3]

    if image.dtype != np.uint8:
        warnings.warn(
            f"Image is {image.dtype}. Will be cast to 8-bit unsigned and clipped to [0,255]",
            stacklevel=2,
        )

        # Images in [0,1] are allowed, but should be rescaled
        if image.max() <= 1 and image.min() >= 0:
            warnings.warn(
                "Image is in [0,1], multiplying by 255. If this is not expected",
                stacklevel=2,
            )
            image *= 255

        image = np.clip(image, 0, 255).astype("uint8")

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
    df: pd.DataFrame, width: int | None = None, height: int | None = None
) -> sv.Detections | sv.KeyPoints:
    """Convert DeepForest prediction results into a supervision object.

    Args:
        df (pd.DataFrame): Results from `predict_image` or `predict_tile`. Must
            include columns: ['geometry', 'label', 'score', 'image_path'].
        width (int, optional): Image width in pixels. Required for polygon geometry.
        height (int, optional): Image height in pixels. Required for polygon geometry.

    Returns:
        sv.Detections | sv.KeyPoints: Object type depends on geometry.
    """
    geom_type = determine_geometry_type(df)

    if geom_type == "box":
        # Extract bounding boxes as numpy array with shape (_, 4)
        boxes = df.geometry.apply(
            lambda x: (x.bounds[0], x.bounds[1], x.bounds[2], x.bounds[3])
        ).values
        boxes = np.stack(boxes)

        label_mapping = {label: idx for idx, label in enumerate(df["label"].unique())}

        # Extract labels as a numpy array
        labels = df["label"].map(label_mapping).values.astype(int)

        # Extract scores as a numpy array
        try:
            scores = np.array(df["score"].tolist())
        except KeyError:
            scores = np.ones(len(labels))

        # Create a reverse mapping from integer to string labels
        class_name = {v: k for k, v in label_mapping.items()}

        detections = sv.Detections(
            xyxy=boxes,
            class_id=labels,
            confidence=scores,
            data={"class_name": [class_name[class_id] for class_id in labels]},
        )

    elif geom_type == "polygon":
        # Extract bounding boxes as numpy array with shape (_, 4)
        boxes = df.geometry.apply(
            lambda x: (x.bounds[0], x.bounds[1], x.bounds[2], x.bounds[3])
        ).values
        boxes = np.stack(boxes)

        label_mapping = {label: idx for idx, label in enumerate(df["label"].unique())}

        # Extract labels as a numpy array
        labels = df["label"].map(label_mapping).values.astype(int)

        # Extract scores as a numpy array
        try:
            scores = np.array(df["score"].tolist())
        except KeyError:
            scores = np.ones(len(labels))

        # Create a reverse mapping from integer to string labels
        class_name = {v: k for k, v in label_mapping.items()}

        # Auto-detect width/height if missing
        if width is None or height is None:
            if "image_path" not in df.columns:
                raise ValueError("'image_path' column required for polygons.")

            # Use the first image_path entry
            image_path = df["image_path"].iloc[0]
            try:
                with Image.open(image_path) as img:
                    width, height = img.size  # Get dimensions
            except Exception as e:
                raise ValueError(
                    f"Could not read image dimensions from {image_path}: {e}"
                ) from e

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
            data={"class_name": [class_name[class_id] for class_id in labels]},
        )

    elif geom_type == "point":
        points = df.geometry.apply(lambda x: (x.x, x.y)).values
        points = np.stack(points)
        points = np.expand_dims(points, axis=1)

        label_mapping = {label: idx for idx, label in enumerate(df["label"].unique())}

        # Extract labels as a numpy array
        labels = df["label"].map(label_mapping).values.astype(int)

        # Extract scores as a numpy array
        try:
            scores = np.array(df["score"].tolist())
        except KeyError:
            scores = np.ones(len(labels))

        scores = np.expand_dims(np.stack(scores), 1)

        # Create a reverse mapping from integer to string labels
        class_name = {v: k for k, v in label_mapping.items()}

        detections = sv.KeyPoints(
            xy=points,
            class_id=labels,
            confidence=scores,
            data={"class_name": [class_name[class_id] for class_id in labels]},
        )

    return detections


def __check_color__(
    color: list | tuple | sv.ColorPalette | None, num_labels: int
) -> sv.Color | sv.ColorPalette:
    if isinstance(color, sv.draw.color.ColorPalette):
        if num_labels > len(color.colors):
            warnings.warn(
                "results_color count does not match label count. Replacing with a built-in "
                "palette. To use a custom one, ensure values match the number of labels.",
                stacklevel=2,
            )
            return sv.ColorPalette.from_matplotlib("viridis", num_labels)
        else:
            return color
    elif isinstance(color, list):
        if len(color) == 3:
            if num_labels > 1:
                # For multiple labels with a single color, use a color palette
                # without showing a warning since this is expected behavior
                return sv.ColorPalette.from_matplotlib("viridis", num_labels)
            else:
                return sv.Color(color[0], color[1], color[2])
        else:
            raise ValueError("results_color list must contain exactly 3 RGB values")
    elif color is None:
        # Fallback case - should not happen as calling functions provide defaults
        if num_labels > 1:
            return sv.ColorPalette.from_matplotlib("viridis", num_labels)
        else:
            return sv.Color(245, 135, 66)  # Default orange color
    else:
        raise TypeError(
            "results_color must be either a list of RGB values "
            "or an sv.ColorPalette instance"
        )


def plot_annotations(
    annotations: pd.DataFrame,
    savedir: str | None = None,
    height: int | None = None,
    width: int | None = None,
    color: list | sv.ColorPalette | None = None,
    thickness: int = 2,
    basename: str | None = None,
    root_dir: str | None = None,
    radius: int = 3,
    image: np.typing.NDArray | str | Image.Image | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot prediction results or ground truth annotations for a single image.

    Args:
        annotations: DataFrame with annotations
        savedir: Directory to save plot
        height: Image height in pixels
        width: Image width in pixels
        color: Color for annotations
        thickness: Line thickness
        basename: Base name for saved file
        root_dir: Root directory for images
        radius: Point radius
        image: Image array or path
        show: Whether to display the plot (default: True). Set to False for testing.

    Returns:
        matplotlib.figure.Figure: The figure object for further customization
    """
    # Initialize default colors if None
    if color is None:
        color = [245, 135, 66]
    # Convert colors, check for multi-class labels
    num_labels = len(annotations.label.unique())
    annotation_color = __check_color__(color, num_labels)

    image = _load_image(image, annotations, root_dir)

    # Plot results using supervision annotators
    fig, ax = plt.subplots()
    annotated_scene = _plot_image_with_geometry(
        df=annotations,
        image=image,
        sv_color=annotation_color,
        height=height,
        width=width,
        thickness=thickness,
        radius=radius,
    )

    if savedir:
        if basename is None:
            basename = os.path.splitext(
                os.path.basename(annotations.image_path.unique()[0])
            )[0]
        os.makedirs(savedir, exist_ok=True)
        image_name = f"{basename}.png"
        image_path = os.path.join(savedir, image_name)
        Image.fromarray(annotated_scene).save(image_path)
    else:
        # Display the image using Matplotlib
        ax.imshow(annotated_scene)
        ax.axis("off")  # Hide axes for a cleaner look
        if show:
            plt.show()

    return fig


def plot_results(
    results: pd.DataFrame,
    ground_truth: pd.DataFrame | None = None,
    savedir: str | None = None,
    height: int | None = None,
    width: int | None = None,
    results_color: list | sv.ColorPalette | None = None,
    ground_truth_color: list | sv.ColorPalette | None = None,
    thickness: int = 2,
    basename: str | None = None,
    radius: int = 3,
    image: np.typing.NDArray | str | Image.Image | None = None,
    axes: bool = False,
    show: bool = True,
):
    """Plot predicted annotations with optional ground truth.

    Creates a figure that can be displayed or saved. Pass axes=True to return the
    Matplotlib Axes for additional plotting.

    Args:
        results: Pandas DataFrame of prediction results.
        ground_truth: Optional DataFrame of ground-truth annotations.
        savedir: Optional path to save the figure; if None, plots interactively.
        height: Image height in pixels. Required when using polygon geometry.
        width: Image width in pixels. Required when using polygon geometry.
        results_color: Single RGB list (e.g., [245, 135, 66]) or an sv.ColorPalette for per-label colors.
        ground_truth_color: Single RGB list (e.g., [0, 165, 255]) or an sv.ColorPalette.
        thickness: Line thickness in pixels.
        basename: Optional base name for the saved figure; if None, derived from the image path.
        radius: Point marker radius in pixels.
        image: Optional NumPy array, image path, or PIL Image to annotate; if None, loaded from the results DataFrame.
        axes: If True, return the Matplotlib Axes object.
        show: Whether to display the plot (default: True). Set to False for testing.

    Returns:
        matplotlib.figure.Figure | matplotlib.axes.Axes: The Figure (default) or Axes (when axes=True).
    """
    # Initialize default colors if None
    if results_color is None:
        results_color = [245, 135, 66]
    if ground_truth_color is None:
        ground_truth_color = [0, 165, 255]
    # Convert colors, check for multi-class labels
    num_labels = len(results.label.unique())
    results_color_sv = __check_color__(results_color, num_labels)
    ground_truth_color_sv = __check_color__(ground_truth_color, num_labels)

    image = _load_image(image, results)

    fig, ax = plt.subplots()
    annotated_scene = _plot_image_with_geometry(
        df=results,
        image=image,
        sv_color=results_color_sv,
        height=height,
        width=width,
        thickness=thickness,
        radius=radius,
    )

    if ground_truth is not None:
        # Plot the ground truth annotations
        annotated_scene = _plot_image_with_geometry(
            df=ground_truth,
            image=annotated_scene,
            sv_color=ground_truth_color_sv,
            height=height,
            width=width,
            thickness=thickness,
            radius=radius,
        )

    if savedir:
        if basename is None:
            basename = os.path.splitext(os.path.basename(results.image_path.unique()[0]))[
                0
            ]

        os.makedirs(savedir, exist_ok=True)
        image_name = f"{basename}.png"
        image_path = os.path.join(savedir, image_name)
        Image.fromarray(annotated_scene).save(image_path)
    else:
        # Display the image using Matplotlib
        ax.imshow(annotated_scene)
        ax.axis("off")  # Hide axes for a cleaner look
        if show:
            plt.show()

    if axes:
        return ax
    return fig


def _plot_image_with_geometry(
    df, image, sv_color, thickness=1, radius=3, height=None, width=None
):
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
        annotated_frame = point_annotator.annotate(
            scene=image.copy(), key_points=detections
        )
    return annotated_frame
