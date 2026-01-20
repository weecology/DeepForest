import os

import numpy as np
import pandas as pd
import rasterio as rio
import slidingwindow
import torch
from PIL import Image
from rasterio.windows import Window
from torch.nn import functional as F
from torch.utils.data import Dataset

from deepforest import preprocess
from deepforest.utilities import apply_nodata_mask, format_geometry, read_file


# Base prediction class
class PredictionDataset(Dataset):
    """Base class for prediction datasets. Defines the common interface and
    accepts either a single image or path, or lists of images/paths, along with
    patch_size and patch_overlap. Images can optionally be resized to a given
    size.

    Args:
        image (PIL.Image.Image): A single image.
        path (str): Path to a single image.
        images (List[PIL.Image.Image]): A list of images.
        patch_size (int): Size of the patches to extract.
        patch_overlap (float): Overlap between patches.
        size (int): Target size to resize images to. Optional; if not provided, no resizing is performed.
    """

    def __init__(
        self,
        image=None,
        path=None,
        images=None,
        patch_size=400,
        patch_overlap=0,
    ):
        self.image = image
        self.images = images
        self.path = path
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.items = self.prepare_items()

    def load_and_preprocess_image(
        self, image_path: str = None, image: np.ndarray | Image.Image = None
    ):
        if image is None:
            if image_path is None:
                raise ValueError("Either image_path or image must be provided")
            image = np.array(Image.open(image_path).convert("RGB"))
        else:
            image = np.array(image)
        # If dtype is not float32, convert to float32
        if image.dtype != "float32":
            image = image.astype("float32")

        # If image is not normalized, normalize to [0, 1]
        if image.max() > 1 or image.min() < 0:
            image = image / 255.0

        # If image is not in CHW format, convert to CHW
        if image.shape[0] != 3:
            if image.shape[-1] != 3:
                raise ValueError(
                    f"Expected 3 channel image, got image shape {image.shape}"
                )
            else:
                image = np.rollaxis(image, 2, 0)

        image = torch.from_numpy(image)

        return image

    def prepare_items(self):
        """Prepare the items for the dataset.

        This is used for special cases before the main.model.forward()
        is called.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        """Get the item at the given index."""
        return self.get_crop(idx)

    def collate_fn(self, batch):
        """Collate the batch into a list."""

        return batch

    def get_crop_bounds(self, idx):
        """Get the crop bounds at the given index, needed to mosaic
        predictions."""
        raise NotImplementedError("Subclasses must implement this method")

    def get_crop(self, idx):
        """Get the crop of the image at the given index."""
        raise NotImplementedError("Subclasses must implement this method")

    def get_image_basename(self, idx):
        """Get the basename of the image at the given index."""
        raise NotImplementedError("Subclasses must implement this method")

    def determine_geometry_type(self, batched_result):
        """Determine the geometry type of the batched result."""
        # Assumes that all geometries are the same in a batch
        if "boxes" in batched_result.keys():
            geom_type = "box"
        elif "points" in batched_result.keys():
            geom_type = "point"
        elif "polygons" in batched_result.keys():
            geom_type = "polygon"
        else:
            raise ValueError(
                f"Unknown geometry type, prediction keys are {batched_result.keys()}"
            )

        return geom_type

    def format_batch(self, batch, idx, sub_idx=None):
        """Format a single prediction dict into a dataframe with metadata.

        Args:
            batch: A single prediction dict (keys: boxes, labels, scores, etc.)
            idx: The dataset index (image index for windowed datasets)
            sub_idx: The sub-index (window index). If None, uses idx.
        """
        if sub_idx is None:
            sub_idx = idx
        geom_type = self.determine_geometry_type(batch)
        result = format_geometry(batch, geom_type=geom_type)
        if result is None:
            return None

        crop_bounds = self.get_crop_bounds(sub_idx)
        if crop_bounds is not None:
            result["window_xmin"] = crop_bounds[0]
            result["window_ymin"] = crop_bounds[1]
        result["image_path"] = self.get_image_basename(idx)

        return result

    def postprocess(self, batch, prediction_index):
        """Postprocess a single prediction result into a dataframe.

        Args:
            batch: A single prediction dict (keys: boxes, labels, scores, etc.)
            prediction_index: The index of this item in the dataset
        """
        result = self.format_batch(batch, prediction_index)
        if result is None:
            return pd.DataFrame()
        return result.reset_index(drop=True)


class SingleImage(PredictionDataset):
    """Take in a single image path, preprocess and batch together."""

    def __init__(self, path=None, image=None, patch_size=400, patch_overlap=0):
        super().__init__(
            path=path, image=image, patch_size=patch_size, patch_overlap=patch_overlap
        )

    def prepare_items(self):
        self.image = self.load_and_preprocess_image(self.path, image=self.image)
        self.windows = preprocess.compute_windows(
            self.image, self.patch_size, self.patch_overlap
        )

    def __len__(self):
        return len(self.windows)

    def window_list(self):
        return [x.getRect() for x in self.windows]

    def get_crop(self, idx):
        crop = self.image[self.windows[idx].indices()]

        return crop

    def get_image_basename(self, idx):
        if self.path is not None:
            return os.path.basename(self.path)
        else:
            return None

    def get_crop_bounds(self, idx):
        return self.windows[idx].getRect()


class FromCSVFile(PredictionDataset):
    """Take in a csv file with image paths and preprocess and batch
    together."""

    def __init__(self, csv_file: str, root_dir: str):
        self.csv_file = csv_file
        self.root_dir = root_dir
        super().__init__()

    def prepare_items(self):
        self.annotations = read_file(self.csv_file)
        if self.root_dir is None:
            self.root_dir = self.annotations.root_dir
        self.image_names = self.annotations.image_path.unique()
        self.image_paths = [os.path.join(self.root_dir, x) for x in self.image_names]

    def __len__(self):
        return len(self.image_paths)

    def get_crop(self, idx):
        image = self.load_and_preprocess_image(image_path=self.image_paths[idx])
        return image

    def get_image_basename(self, idx):
        return os.path.basename(self.image_paths[idx])

    def get_crop_bounds(self, idx):
        return None

    def format_batch(self, batch, idx, sub_idx=None):
        """Format a single prediction dict into a dataframe with metadata.
        Override of base class to skip window coordinates (not applicable for
        full images).

        Args:
            batch: A single prediction dict (keys: boxes, labels, scores, etc.)
            idx: The dataset index (image index)
            sub_idx: Unused (kept for compatibility with base class signature)
        """
        geom_type = self.determine_geometry_type(batch)
        result = format_geometry(batch, geom_type=geom_type)
        if result is None:
            return None

        result["image_path"] = self.get_image_basename(idx)

        return result


class MultiImage(PredictionDataset):
    """Take in a list of image paths, preprocess and batch together.

    Note: This dataset will load the first image to determine the image dimensions. Images are expected to be the same size. For variable sized images, write a csv file and use the FromCSVFile dataset.
    """

    def __init__(self, paths: list[str], patch_size: int, patch_overlap: float):
        """
        Args:
            paths (List[str]): List of image paths.
            patch_size (int): Size of the patches to extract.
            patch_overlap (float): Overlap between patches.
        """
        self.paths = paths
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.sublist_lengths = []

        image = self.load_and_preprocess_image(image_path=self.paths[0])
        self.image_height = image.shape[1]
        self.image_width = image.shape[2]

    def create_overlapping_views(self, input_tensor, size, overlap):
        """Creates overlapping views of a 4D tensor.

        Args:
            input_tensor (torch.Tensor): A 4D tensor of shape [N, C, H, W].
            size (int): The size of the sliding window (square).
            overlap (int): The overlap between adjacent windows.

        Returns:
            torch.Tensor: A tensor containing all the overlapping views.
                          The output shape is [N * num_windows, C, size, size].
        """
        # Get the input tensor shape
        N, C, H, W = input_tensor.shape

        # Calculate step size based on overlap
        step = size - overlap

        # Calculate number of patches needed in each dimension
        n_patches_h = (H - overlap) // step + 1
        n_patches_w = (W - overlap) // step + 1

        # Calculate total padded dimensions needed
        H_padded = n_patches_h * step + overlap
        W_padded = n_patches_w * step + overlap

        # Calculate padding needed
        padding_h = max(0, H_padded - H)
        padding_w = max(0, W_padded - W)

        # Pad the input tensor
        padded_tensor = F.pad(input_tensor, (0, padding_w, 0, padding_h))

        # Use unfold to create views of the tensor
        # This creates views rather than copies
        unfolded_h = padded_tensor.unfold(2, size, step)  # unfold height dimension
        unfolded = unfolded_h.unfold(3, size, step)  # unfold width dimension

        # Reshape to [N * num_windows, C, size, size]
        # This is still a view operation
        output = unfolded.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, size, size)

        return output

    def _create_patches(self, image):
        image_tensor = image.unsqueeze(0)  # Convert to (N, C, H, W)
        patch_overlap_size = int(self.patch_size * self.patch_overlap)
        patches = self.create_overlapping_views(
            image_tensor, self.patch_size, patch_overlap_size
        )

        return patches

    def window_list(self):
        """Get the original positions of patches in the image.

        Returns:
            list: List of tuples containing (x, y, w, h) coordinates of each patch
        """
        H = self.image_height
        W = self.image_width

        patch_overlap_size = int(self.patch_size * self.patch_overlap)
        step = self.patch_size - patch_overlap_size

        # Calculate number of patches needed in each dimension
        n_patches_h = (H - patch_overlap_size) // step + 1
        n_patches_w = (W - patch_overlap_size) // step + 1

        # Generate window coordinates for unfolded tensor views
        windows = []
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                y = i * step
                x = j * step
                # Only add window if it contains any real data
                if x < W and y < H:
                    windows.append((x, y, self.patch_size, self.patch_size))

        return windows

    def collate_fn(self, batch):
        """Collate the batch into a single list of crops.

        Keep track of the lengths of each sublist.
        """
        # Create a list of lengths of each sublist
        sub_list_length = [
            [idx, sub_idx]
            for idx, sublist in enumerate(batch)
            for sub_idx in range(len(sublist))
        ]
        self.sublist_lengths.append(sub_list_length)

        # Flatten list of lists of crops
        flattened_batch = [crop for sublist in batch for crop in sublist]
        sublist_lengths = [
            [idx, sub_idx]
            for idx, sublist in enumerate(batch)
            for sub_idx in range(len(sublist))
        ]

        return {"images": flattened_batch, "sublist_lengths": sublist_lengths}

    def __len__(self):
        return len(self.paths)

    def get_crop(self, idx):
        image = self.load_and_preprocess_image(image_path=self.paths[idx])
        crops = self._create_patches(image)

        # Return as a list of crops, each with shape (3, 300, 300)
        return [crops[i] for i in range(crops.shape[0])]

    def get_image_basename(self, idx):
        return os.path.basename(self.paths[idx])

    def get_crop_bounds(self, idx):
        return self.window_list()[idx]

    def postprocess(self, batch, prediction_index, original_batch_structure):
        """Postprocess flattened batch of predictions from multiple images.

        Args:
            batch: List of prediction dicts (all windows from all images in batch)
            prediction_index: Index of this batch from trainer.predict
        """
        if prediction_index >= len(original_batch_structure):
            raise ValueError(
                f"prediction_index {prediction_index} exceeds sublist_lengths length {len(original_batch_structure)}. "
                "This may indicate a mismatch between collate_fn calls and postprocess calls."
            )

        batch_sublist_lengths = original_batch_structure[prediction_index]
        formatted_results = []

        # batch_sublist_lengths[i] = [image_idx, window_idx] corresponds to batch[i]
        for batch_position, (image_idx, window_idx) in enumerate(batch_sublist_lengths):
            prediction = batch[batch_position]

            # Format with correct image index and window index
            result = self.format_batch(prediction, image_idx, window_idx)
            if result is not None:
                formatted_results.append(result)

        if len(formatted_results) > 0:
            return pd.concat(formatted_results).reset_index(drop=True)
        else:
            return pd.DataFrame()


class TiledRaster(PredictionDataset):
    """Dataset for predicting on raster windows.

    This dataset is useful for predicting on a large raster that is too large to fit into memory.

    Args:
        path (str): Path to raster file
        patch_size (int): Size of windows to predict on
        patch_overlap (float): Overlap between windows as fraction (0-1)
    Returns:
        A dataset of raster windows
    """

    def __init__(self, path, patch_size, patch_overlap):
        if path is None:
            raise ValueError("path is required for a memory raster dataset")
        super().__init__(path=path, patch_size=patch_size, patch_overlap=patch_overlap)

    def prepare_items(self):
        # Get raster shape without keeping file open
        with rio.open(self.path) as src:
            width = src.shape[0]
            height = src.shape[1]

            # Check is tiled
            if not src.is_tiled:
                raise ValueError(
                    "Out-of-memory dataset is selected, but raster is not tiled, "
                    "leading to entire raster being read into memory and defeating "
                    "the purpose of an out-of-memory dataset. "
                    "\nPlease run: "
                    "\ngdal_translate -of GTiff -co TILED=YES <input> <output> "
                    "to create a tiled raster"
                )

        # Generate sliding windows
        self.windows = slidingwindow.generateForSize(
            height,
            width,
            dimOrder=slidingwindow.DimOrder.ChannelHeightWidth,
            maxWindowSize=self.patch_size,
            overlapPercent=self.patch_overlap,
        )

    def __len__(self):
        return len(self.windows)

    def window_list(self):
        return [x.getRect() for x in self.windows]

    def get_crop(self, idx):
        window = self.windows[idx]
        with rio.open(self.path) as src:
            window_data = apply_nodata_mask(
                src, Window(window.x, window.y, window.w, window.h)
            )

        # Rasterio already returns (C, H, W), just normalize and convert
        window_data = window_data.astype("float32") / 255.0
        window_data = torch.from_numpy(window_data).float()
        if window_data.shape[0] != 3:
            raise ValueError(
                f"Expected 3 channel image, got {window_data.shape[0]} channels"
            )

        return window_data

    def get_image_basename(self, idx):
        return os.path.basename(self.path)

    def get_crop_bounds(self, idx):
        return self.window_list()[idx]
