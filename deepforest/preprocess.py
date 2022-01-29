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

def preprocess_image(image, device):
    """Preprocess a single RGB numpy array as a prediction from channels last, to channels first"""
    image = torch.tensor(image, device=device).permute(2, 0, 1).unsqueeze(0)
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


def select_annotations(annotations, windows, index, allow_empty=False):
    """Select annotations that overlap with selected image crop.

    Args:
        image_name (str): Name of the image in the annotations file to lookup.
        annotations_file: path to annotations file in
            the format -> image_path, xmin, ymin, xmax, ymax, label
        windows: A sliding window object (see compute_windows)
        index: The index in the windows object to use a crop bounds
        allow_empty (bool): If True, allow window crops
            that have no annotations to be included

    Returns:
        selected_annotations: a pandas dataframe of annotations
    """

    # Window coordinates - with respect to tile
    window_xmin, window_ymin, w, h = windows[index].getRect()
    window_xmax = window_xmin + w
    window_ymax = window_ymin + h

    # buffer coordinates a bit to grab boxes that might start just against
    # the image edge. Don't allow boxes that start and end after the offset
    offset = 40
    selected_annotations = annotations[(annotations.xmin > (window_xmin - offset)) &
                                       (annotations.xmin < (window_xmax)) &
                                       (annotations.xmax >
                                        (window_xmin)) & (annotations.ymin >
                                                          (window_ymin - offset)) &
                                       (annotations.xmax <
                                        (window_xmax + offset)) & (annotations.ymin <
                                                                   (window_ymax)) &
                                       (annotations.ymax >
                                        (window_ymin)) & (annotations.ymax <
                                                          (window_ymax + offset))]

    # change the image name
    image_basename = os.path.splitext("{}".format(annotations.image_path.unique()[0]))[0]
    selected_annotations.image_path = "{}_{}.png".format(image_basename, index)

    # If no matching annotations, return a line with the image name, but no
    # records
    if selected_annotations.empty:
        if allow_empty:
            selected_annotations = pd.DataFrame(
                ["{}_{}.png".format(image_basename, index)], columns=["image_path"])
            selected_annotations["xmin"] = 0
            selected_annotations["ymin"] = 0
            selected_annotations["xmax"] = 0
            selected_annotations["ymax"] = 0
            #Dummy label
            selected_annotations["label"] = annotations.label.unique()[0]
        else:
            return None
    else:
        # update coordinates with respect to origin
        selected_annotations.xmax = (selected_annotations.xmin - window_xmin) + (
            selected_annotations.xmax - selected_annotations.xmin)
        selected_annotations.xmin = (selected_annotations.xmin - window_xmin)
        selected_annotations.ymax = (selected_annotations.ymin - window_ymin) + (
            selected_annotations.ymax - selected_annotations.ymin)
        selected_annotations.ymin = (selected_annotations.ymin - window_ymin)

        # cut off any annotations over the border.
        selected_annotations.loc[selected_annotations.xmin < 0, "xmin"] = 0
        selected_annotations.loc[selected_annotations.xmax > w, "xmax"] = w
        selected_annotations.loc[selected_annotations.ymin < 0, "ymin"] = 0
        selected_annotations.loc[selected_annotations.ymax > h, "ymax"] = h

    return selected_annotations


def save_crop(base_dir, image_name, index, crop):
    """Save window crop as image file to be read by PIL.

    Filename should match the image_name + window index
    """
    # create dir if needed
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    im = Image.fromarray(crop)
    image_basename = os.path.splitext(image_name)[0]
    filename = "{}/{}_{}.png".format(base_dir, image_basename, index)
    im.save(filename)

    return filename


def split_raster(annotations_file,
                 path_to_raster=None,
                 numpy_image=None,
                 base_dir=".",
                 patch_size=400,
                 patch_overlap=0.05,
                 allow_empty=False,
                 image_name = None):
    """Divide a large tile into smaller arrays. Each crop will be saved to
    file.

    Args:
        numpy_image: a numpy object to be used as a raster, usually opened from rasterio.open.read()
        path_to_raster: (str): Path to a tile that can be read by rasterio on disk
        annotations_file (str): Path to annotations file (with column names)
            data in the format -> image_path, xmin, ymin, xmax, ymax, label
        base_dir (str): Where to save the annotations and image
            crops relative to current working dir
        patch_size (int): Maximum dimensions of square window
        patch_overlap (float): Percent of overlap among windows 0->1
        allow_empty: If True, include images with no annotations
            to be included in the dataset
        image_name (str): If numpy_image arg is used, what name to give the raster?

    Returns:
        A pandas dataframe with annotations file for training.
    """
    
    # Load raster as image
    # Load raster as image
    if (numpy_image is None) & (path_to_raster is None):
        raise IOError("supply a raster either as a path_to_raster or if ready from existing in memory numpy object, as numpy_image=")
    
    if path_to_raster:
        numpy_image = rasterio.open(path_to_raster).read()
        numpy_image = np.moveaxis(numpy_image,0,2)
    else:
        if image_name is None:
            raise(IOError("If passing an numpy_image, please also specify a image_name to match the column in the annotation.csv file"))

    # Check that its 3 band
    bands = numpy_image.shape[2]
    if not bands == 3:
        warnings.warn("Input rasterio had non-3 band shape of {}, ignoring alpha channel".format(numpy_image.shape))
        try:
            numpy_image = numpy_image[:,:,:3].astype("uint8") 
        except:
            raise IOError("Input file {} has {} bands. DeepForest only accepts 3 band RGB "
                          "rasters in the order (height, width, channels). Selecting the first three bands failed, please reshape manually."
                          "If the image was cropped and saved as a .jpg, "
                          "please ensure that no alpha channel was used.".format(
                              path_to_raster, bands))

    # Check that patch size is greater than image size
    height = numpy_image.shape[0]
    width = numpy_image.shape[1]
    if any(np.array([height, width]) < patch_size):
        raise ValueError("Patch size of {} is larger than the image dimensions {}".format(
            patch_size, [height, width]))

    # Compute sliding window index
    windows = compute_windows(numpy_image, patch_size, patch_overlap)

    # Get image name for indexing
    if image_name is None:
        image_name = os.path.basename(path_to_raster)    

    # Load annotations file and coerce dtype
    annotations = pd.read_csv(annotations_file)

    # open annotations file
    image_annotations = annotations[annotations.image_path == image_name]

    # Sanity checks
    if image_annotations.empty:
        raise ValueError(
            "No image names match between the file:{} and the image_path: {}. "
            "Reminder that image paths should be the relative "
            "path (e.g. 'image_name.tif'), not the full path "
            "(e.g. path/to/dir/image_name.tif)".format(annotations_file, image_name))

    if not all([
            x in annotations.columns
            for x in ["image_path", "xmin", "ymin", "xmax", "ymax", "label"]
    ]):
        raise ValueError("Annotations file has {} columns, should have "
                         "format image_path, xmin, ymin, xmax, ymax, label".format(
                             annotations.shape[1]))

    annotations_files = []
    for index, window in enumerate(windows):

        # Crop image
        crop = numpy_image[windows[index].indices()]
        
        #skip if empty crop
        if crop.size == 0:
            continue

        # Find annotations, image_name is the basename of the path
        crop_annotations = select_annotations(image_annotations, windows, index,
                                              allow_empty)

        # If empty images not allowed, select annotations returns None
        if crop_annotations is not None:
            # save annotations
            annotations_files.append(crop_annotations)

            # save image crop
            save_crop(base_dir, image_name, index, crop)
    if len(annotations_files) == 0:
        raise ValueError(
            "Input file has no overlapping annotations and allow_empty is {}".format(
                allow_empty))

    annotations_files = pd.concat(annotations_files)

    # Checkpoint csv files, useful for parallelization
    # Use filename of the raster path to save the annotations
    image_basename = os.path.splitext(image_name)[0]
    file_path = image_basename + ".csv"
    file_path = os.path.join(base_dir, file_path)
    annotations_files.to_csv(file_path, index=False, header=True)

    return annotations_files
