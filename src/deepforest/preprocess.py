# Deepforest Preprocessing model
"""The preprocessing module is used to reshape data into format suitable for
training or prediction.

For example cutting large tiles into smaller images.
"""

import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import slidingwindow
import torch
from PIL import Image
from shapely import geometry

from deepforest.utilities import determine_geometry_type, read_file


def preprocess_image(image):
    """Preprocess a single RGB numpy array as a prediction from channels last,
    to channels first."""
    image = torch.tensor(image).permute(2, 0, 1)
    image = image / 255

    return image


def image_name_from_path(image_path):
    """Convert path to image name for use in indexing."""
    image_name = os.path.basename(image_path)
    image_name = os.path.splitext(image_name)[0]

    return image_name


def compute_windows(numpy_image, patch_size, patch_overlap):
    """Create a sliding window object from a raster tile.

    Args:
        numpy_image (array): Raster object as numpy array to cut into crops, channels first

    Returns:
        windows (list): a sliding windows object
    """

    if patch_overlap > 1:
        raise ValueError(f"Patch overlap {patch_overlap} must be between 0 - 1")

    # Check that image is channels first
    if numpy_image.shape[0] != 3:
        raise ValueError(f"Image is not channels first, shape is {numpy_image.shape}")

    # Generate overlapping sliding windows
    windows = slidingwindow.generate(
        numpy_image, slidingwindow.DimOrder.ChannelHeightWidth, patch_size, patch_overlap
    )

    return windows


def select_annotations(annotations, window):
    """Select annotations that overlap with selected image crop.

    Args:
        annotations: a geopandas dataframe of annotations with a geometry column
        windows: A sliding window object (see compute_windows)
    Returns:
        selected_annotations: a pandas dataframe of annotations
    """
    # Get window coordinates
    window_xmin, window_ymin, w, h = window.getRect()

    # Create a shapely box from the window
    window_box = geometry.box(window_xmin, window_ymin, window_xmin + w, window_ymin + h)
    selected_annotations = annotations[annotations.intersects(window_box)]

    # cut off any annotations over the border
    original_area = selected_annotations.geometry.area
    clipped_annotations = gpd.clip(selected_annotations, window_box)

    if clipped_annotations.empty:
        return clipped_annotations

    # For points, keep all annotations but translate to window origin
    if selected_annotations.iloc[0].geometry.geom_type == "Point":
        if clipped_annotations.empty:
            return clipped_annotations
        # Translate point coordinates to be relative to the top-left of the window
        clipped_annotations.geometry = clipped_annotations.geometry.translate(
            xoff=-window_xmin, yoff=-window_ymin
        )
        # Update convenience x/y columns if they exist or create them
        clipped_annotations["x"] = clipped_annotations.geometry.x
        clipped_annotations["y"] = clipped_annotations.geometry.y
        return clipped_annotations

    else:
        # Keep clipped boxes if they're more than 50% of original size
        clipped_area = clipped_annotations.geometry.area
        clipped_annotations = clipped_annotations[(clipped_area / original_area) > 0.5]

    clipped_annotations.geometry = clipped_annotations.geometry.translate(
        xoff=-window_xmin, yoff=-window_ymin
    )

    # Update xmin, ymin, xmax, ymax from clipped geometry
    if (
        not clipped_annotations.empty
        and determine_geometry_type(clipped_annotations) == "box"
    ):
        if clipped_annotations.shape[0] > 0:
            clipped_annotations["xmin"] = clipped_annotations.geometry.bounds.minx
            clipped_annotations["ymin"] = clipped_annotations.geometry.bounds.miny
            clipped_annotations["xmax"] = clipped_annotations.geometry.bounds.maxx
            clipped_annotations["ymax"] = clipped_annotations.geometry.bounds.maxy

    return clipped_annotations


def save_crop(base_dir, image_name, index, crop):
    """Save window crop as an image file to be read by PIL.

    Args:
        base_dir (str): The base directory to save the image file.
        image_name (str): The name of the original image.
        index (int): The index of the window crop.
        crop (numpy.ndarray): The window crop as a NumPy array.

    Returns:
        str: The filename of the saved image.
    """
    # Create directory if needed
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Convert NumPy array to PIL image, PIL expects channels last
    crop = np.moveaxis(crop, 0, 2)
    im = Image.fromarray(crop)

    # Extract the basename of the image
    image_basename = os.path.splitext(image_name)[0]

    # Generate the filename for the saved image
    filename = f"{base_dir}/{image_basename}_{index}.png"

    # Save the image
    im.save(filename)

    return filename


def split_raster(
    annotations_file=None,
    path_to_raster=None,
    numpy_image=None,
    root_dir=None,
    patch_size=400,
    patch_overlap=0.05,
    allow_empty=False,
    image_name=None,
    save_dir=".",
):
    """Split a large raster into smaller patches for processing.

    Args:
        annotations_file: Path to annotations CSV or DataFrame with columns:
            image_path, xmin, ymin, xmax, ymax, label
        path_to_raster: Path to raster file on disk
        numpy_image: Numpy array in (channels, height, width) order
        root_dir: Root directory for annotations file
        patch_size: Size of square patches
        patch_overlap: Overlap between patches (0-1)
        allow_empty: Include patches with no annotations
        image_name: Name for the raster image
        save_dir: Directory to save patches

    Returns:
        DataFrame with annotations for training, or list of patch filenames
    """

    # Load raster as image
    if numpy_image is None and path_to_raster is None:
        raise OSError(
            "Supply a raster either as a path_to_raster or if ready "
            "from existing in-memory numpy object, as numpy_image="
        )

    if path_to_raster:
        numpy_image = Image.open(path_to_raster)
        numpy_image = np.array(numpy_image)
    else:
        if image_name is None:
            raise OSError(
                "If passing a numpy_image, please also specify an image_name"
                " to match the column in the annotation.csv file"
            )

    # Convert from channels-last (H x W x C) to channels-first (C x H x W)
    if len(numpy_image.shape) == 3 and numpy_image.shape[2] in [3, 4]:
        print(
            f"Image shape is {numpy_image.shape[2]}, assuming this is channels last, "
            "converting to channels first"
        )
        numpy_image = numpy_image.transpose(2, 0, 1)

    # Check that it's 3 bands (after transpose, shape is (C, H, W), so bands is shape[0])
    bands = numpy_image.shape[0]
    if not bands == 3:
        warnings.warn(
            f"Input image had non-3 band shape of {numpy_image.shape}, selecting first 3 bands",
            UserWarning,
            stacklevel=2,
        )
        try:
            numpy_image = numpy_image[:3, :, :].astype("uint8")
        except Exception:
            raise OSError(
                f"Input file {path_to_raster} has {bands} bands. "
                "DeepForest only accepts 3 band RGB rasters in the order "
                "(channels, height, width). "
                "Selecting the first three bands failed, "
                "please reshape manually. If the image was cropped and "
                "saved, please ensure that no alpha channel "
                "was used."
            ) from None

    # Check that patch size is greater than image size
    height, width = numpy_image.shape[1], numpy_image.shape[2]
    if any(np.array([height, width]) < patch_size):
        raise ValueError(
            f"Patch size of {patch_size} is larger than the image dimensions {[height, width]}"
        )

    # Compute sliding window index
    windows = compute_windows(numpy_image, patch_size, patch_overlap)

    # Get image name for indexing
    if image_name is None:
        image_name = os.path.basename(path_to_raster)

    # Early return for non-annotation case
    if annotations_file is None:
        return process_without_annotations(numpy_image, windows, image_name, save_dir)

    # Process annotations
    annotations = load_annotations(annotations_file, root_dir)
    validate_annotations(annotations, numpy_image, path_to_raster)

    image_annotations = annotations[annotations.image_path == image_name]

    if not allow_empty and image_annotations.empty:
        raise ValueError(
            f"No image names match between the file:{annotations_file} and the image_path: {image_name}. "
            "Reminder that image paths should be the relative "
            "path (e.g. 'image_name.tif'), not the full path "
            "(e.g. path/to/dir/image_name.tif)"
        )

    return process_with_annotations(
        numpy_image=numpy_image,
        windows=windows,
        image_annotations=image_annotations,
        image_name=image_name,
        save_dir=save_dir,
        allow_empty=allow_empty,
    )


def load_annotations(annotations_file, root_dir):
    """Load and validate annotations file.

    Args:
        annotations_file: Path to file, DataFrame, or GeoDataFrame
        root_dir: Root directory for relative paths

    Returns:
        GeoDataFrame with annotations
    """
    if isinstance(annotations_file, str):
        return read_file(annotations_file, root_dir=root_dir)
    elif isinstance(annotations_file, gpd.GeoDataFrame):
        return annotations_file
    elif isinstance(annotations_file, pd.DataFrame):
        if root_dir is None:
            raise ValueError(
                "If passing a pandas DataFrame with relative pathnames in "
                "image_path, please also specify a root_dir"
            )
        return read_file(annotations_file, root_dir=root_dir)
    else:
        raise TypeError(
            f"Annotations file must either be a path, Pandas Dataframe, or Geopandas GeoDataFrame, found {type(annotations_file)}"
        )


def validate_annotations(annotations, numpy_image, path_to_raster):
    """Validate annotation coordinate systems and bounds."""
    if hasattr(annotations, "crs"):
        if annotations.crs is not None and annotations.crs.is_geographic:
            raise ValueError(
                "Annotations appear to be in geographic coordinates (latitude/longitude). "
                "Please convert your annotations to the same projected coordinate system as your raster."
            )

    if hasattr(annotations, "total_bounds"):
        raster_height, raster_width = numpy_image.shape[1], numpy_image.shape[2]
        ann_bounds = annotations.total_bounds

        if (
            ann_bounds[0] < -raster_width * 0.1  # xmin
            or ann_bounds[2] > raster_width * 1.1  # xmax
            or ann_bounds[1] < -raster_height * 0.1  # ymin
            or ann_bounds[3] > raster_height * 1.1
        ):  # ymax
            raise ValueError(
                f"Annotation bounds {ann_bounds} appear to be outside reasonable range for "
                f"raster dimensions ({raster_width}, {raster_height}). "
                "This might indicate your annotations are in a different coordinate system."
            )


def process_without_annotations(numpy_image, windows, image_name, save_dir):
    """Process raster without annotations."""
    crop_filenames = []
    for index, window in enumerate(windows):
        crop = numpy_image[window.indices()]
        if crop.size == 0:
            continue
        crop_filename = save_crop(save_dir, image_name, index, crop)
        crop_filenames.append(crop_filename)
    return crop_filenames


def process_with_annotations(
    numpy_image, windows, image_annotations, image_name, save_dir, allow_empty
):
    """Process raster with annotations."""
    annotations_files = []
    crop_filenames = []
    image_basename = os.path.splitext(image_name)[0]

    for index, window in enumerate(windows):
        crop = numpy_image[window.indices()]
        if crop.size == 0:
            continue

        crop_annotations = select_annotations(image_annotations, window=window)
        crop_annotations["image_path"] = f"{image_basename}_{index}.png"

        if crop_annotations.empty:
            if allow_empty:
                crop_annotations = create_empty_annotation(
                    image_annotations, image_basename, index
                )
            else:
                continue

        annotations_files.append(crop_annotations)
        crop_filename = save_crop(save_dir, image_name, index, crop)
        crop_filenames.append(crop_filename)

    if len(annotations_files) == 0:
        raise ValueError(
            f"Input file has no overlapping annotations and allow_empty is {allow_empty}"
        )

    annotations_files = pd.concat(annotations_files)
    file_path = os.path.join(save_dir, f"{image_basename}.csv")
    annotations_files.to_csv(file_path, index=False, header=True)
    return annotations_files


def create_empty_annotation(image_annotations, image_basename, index):
    """Create empty annotation record when allow_empty=True."""
    geom_type = determine_geometry_type(image_annotations)
    crop_annotations = pd.DataFrame(columns=image_annotations.columns)
    crop_annotations.loc[0, "label"] = image_annotations.label.unique()[0]
    crop_annotations.loc[0, "image_path"] = f"{image_basename}_{index}.png"

    if geom_type == "box":
        crop_annotations.loc[0, "xmin"] = 0
        crop_annotations.loc[0, "ymin"] = 0
        crop_annotations.loc[0, "xmax"] = 0
        crop_annotations.loc[0, "ymax"] = 0
    elif geom_type == "point":
        crop_annotations.loc[0, "geometry"] = geometry.Point(0, 0)
        crop_annotations.loc[0, "x"] = 0
        crop_annotations.loc[0, "y"] = 0
    elif geom_type == "polygon":
        crop_annotations.loc[0, "geometry"] = geometry.Polygon([(0, 0), (0, 0), (0, 0)])
        crop_annotations.loc[0, "polygon"] = 0

    return crop_annotations
