# Deepforest Preprocessing model
"""The preprocessing module is used to reshape data into format suitable for
training or prediction.

For example cutting large tiles into smaller images.
"""
import os
import numpy as np
import pandas as pd
import slidingwindow
from PIL import Image
import torch
import warnings
import rasterio
import geopandas as gpd
from deepforest.utilities import read_file, determine_geometry_type
from shapely import geometry


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
        numpy_image (array): Raster object as numpy array to cut into crops

    Returns:
        windows (list): a sliding windows object
    """

    if patch_overlap > 1:
        raise ValueError("Patch overlap {} must be between 0 - 1".format(patch_overlap))

    # Generate overlapping sliding windows
    windows = slidingwindow.generate(numpy_image,
                                     slidingwindow.DimOrder.HeightWidthChannel,
                                     patch_size, patch_overlap)

    return (windows)


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

    # For points, keep all annotations.
    if selected_annotations.iloc[0].geometry.geom_type == "Point":
        return selected_annotations

    else:
        # Only keep clipped boxes if they are more than 50% of the original size.
        clipped_area = clipped_annotations.geometry.area
        clipped_annotations = clipped_annotations[(clipped_area / original_area) > 0.5]

    clipped_annotations.geometry = clipped_annotations.geometry.translate(
        xoff=-window_xmin, yoff=-window_ymin)

    # Update xmin, ymin, xmax, ymax based on the clipped annotations' geometry
    if not clipped_annotations.empty and determine_geometry_type(
            clipped_annotations) == "box":
        if clipped_annotations.shape[0] > 0:
            clipped_annotations['xmin'] = clipped_annotations.geometry.bounds.minx
            clipped_annotations['ymin'] = clipped_annotations.geometry.bounds.miny
            clipped_annotations['xmax'] = clipped_annotations.geometry.bounds.maxx
            clipped_annotations['ymax'] = clipped_annotations.geometry.bounds.maxy

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

    # Convert NumPy array to PIL image
    im = Image.fromarray(crop)

    # Extract the basename of the image
    image_basename = os.path.splitext(image_name)[0]

    # Generate the filename for the saved image
    filename = "{}/{}_{}.png".format(base_dir, image_basename, index)

    # Save the image
    im.save(filename)

    return filename


def split_raster(annotations_file=None,
                 path_to_raster=None,
                 numpy_image=None,
                 root_dir=None,
                 base_dir=None,
                 patch_size=400,
                 patch_overlap=0.05,
                 allow_empty=False,
                 image_name=None,
                 save_dir="."):
    """Divide a large tile into smaller arrays. Each crop will be saved to
    file.

    Args:
        numpy_image: a numpy object to be used as a raster, usually opened from rasterio.open.read(), in order (height, width, channels)
        root_dir: (str): Root directory of annotations file, if not supplied, will be inferred from annotations_file
        path_to_raster: (str): Path to a tile that can be read by rasterio on disk
        annotations_file (str or pd.DataFrame): A pandas dataframe or path to annotations csv file to transform to cropped images. In the format -> image_path, xmin, ymin, xmax, ymax, label. If None, allow_empty is ignored and the function will only return the cropped images.
        save_dir (str): Directory to save images
        base_dir (str): Directory to save images
        patch_size (int): Maximum dimensions of square window
        patch_overlap (float): Percent of overlap among windows 0->1
        allow_empty: If True, include images with no annotations
            to be included in the dataset. If annotations_file is None, this is ignored.
        image_name (str): If numpy_image arg is used, what name to give the raster?
    Note:
        When allow_empty is True, the function will return 0's for coordinates, following torchvision style, the label will be ignored, so for continuity, the first label in the annotations_file will be used.
    Returns:
        If annotations_file is provided, a pandas dataframe with annotations file for training. A copy of this file is written to save_dir as a side effect.
        If not, a list of filenames of the cropped images.
    """
    # Set deprecation warning for base_dir and set to save_dir
    if base_dir:
        warnings.warn(
            "base_dir argument will be deprecated in 2.0. The naming is confusing, the rest of the API uses 'save_dir' to refer to location of images. Please use 'save_dir' argument.",
            DeprecationWarning)
        save_dir = base_dir

    # Load raster as image
    if numpy_image is None and path_to_raster is None:
        raise IOError("Supply a raster either as a path_to_raster or if ready "
                      "from existing in-memory numpy object, as numpy_image=")

    if path_to_raster:
        numpy_image = rasterio.open(path_to_raster).read()
        numpy_image = np.moveaxis(numpy_image, 0, 2)
    else:
        if image_name is None:
            raise IOError("If passing a numpy_image, please also specify an image_name"
                          " to match the column in the annotation.csv file")

    # Confirm that raster is H x W x C, if not, convert, assuming image is wider/taller than channels
    if numpy_image.shape[0] < numpy_image.shape[-1]:
        warnings.warn(
            "Input rasterio had shape {}, assuming channels first. Converting to channels last"
            .format(numpy_image.shape), UserWarning)
        numpy_image = np.moveaxis(numpy_image, 0, 2)

    # Check that it's 3 bands
    bands = numpy_image.shape[2]
    if not bands == 3:
        warnings.warn(
            "Input rasterio had non-3 band shape of {}, ignoring "
            "alpha channel".format(numpy_image.shape), UserWarning)
        try:
            numpy_image = numpy_image[:, :, :3].astype("uint8")
        except:
            raise IOError("Input file {} has {} bands. "
                          "DeepForest only accepts 3 band RGB rasters in the order "
                          "(height, width, channels). "
                          "Selecting the first three bands failed, "
                          "please reshape manually. If the image was cropped and "
                          "saved as a .jpg, please ensure that no alpha channel "
                          "was used.".format(path_to_raster, bands))

    # Check that patch size is greater than image size
    height, width = numpy_image.shape[0], numpy_image.shape[1]
    if any(np.array([height, width]) < patch_size):
        raise ValueError("Patch size of {} is larger than the image dimensions {}".format(
            patch_size, [height, width]))

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
            "No image names match between the file:{} and the image_path: {}. "
            "Reminder that image paths should be the relative "
            "path (e.g. 'image_name.tif'), not the full path "
            "(e.g. path/to/dir/image_name.tif)".format(annotations_file, image_name))

    return process_with_annotations(numpy_image=numpy_image,
                                    windows=windows,
                                    image_annotations=image_annotations,
                                    image_name=image_name,
                                    save_dir=save_dir,
                                    allow_empty=allow_empty)


def load_annotations(annotations_file, root_dir):
    """Load and validate annotations file."""
    if type(annotations_file) == str:
        return read_file(annotations_file, root_dir=root_dir)
    elif type(annotations_file) == pd.DataFrame:
        if root_dir is None:
            raise ValueError(
                "If passing a pandas DataFrame with relative pathnames in image_path, please also specify a root_dir"
            )
        return read_file(annotations_file, root_dir=root_dir)
    elif type(annotations_file) == gpd.GeoDataFrame:
        return annotations_file
    else:
        raise TypeError(
            "Annotations file must either be a path, Pandas Dataframe, or Geopandas GeoDataFrame, found {}"
            .format(type(annotations_file)))


def validate_annotations(annotations, numpy_image, path_to_raster):
    """Validate annotation coordinate systems and bounds."""
    if hasattr(annotations, 'crs'):
        if annotations.crs is not None and annotations.crs.is_geographic:
            raise ValueError(
                "Annotations appear to be in geographic coordinates (latitude/longitude). "
                "Please convert your annotations to the same projected coordinate system as your raster."
            )

    if hasattr(annotations, 'total_bounds'):
        raster_height, raster_width = numpy_image.shape[0], numpy_image.shape[1]
        ann_bounds = annotations.total_bounds

        if (ann_bounds[0] < -raster_width * 0.1 or  # xmin
                ann_bounds[2] > raster_width * 1.1 or  # xmax
                ann_bounds[1] < -raster_height * 0.1 or  # ymin
                ann_bounds[3] > raster_height * 1.1):  # ymax
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


def process_with_annotations(numpy_image, windows, image_annotations, image_name,
                             save_dir, allow_empty):
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
                crop_annotations = create_empty_annotation(image_annotations,
                                                           image_basename, index)
            else:
                continue

        annotations_files.append(crop_annotations)
        crop_filename = save_crop(save_dir, image_name, index, crop)
        crop_filenames.append(crop_filename)

    if len(annotations_files) == 0:
        raise ValueError(
            "Input file has no overlapping annotations and allow_empty is {}".format(
                allow_empty))

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
