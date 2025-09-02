# Standard library imports
import os
from typing import List

# Third party imports
import numpy as np
import rasterio as rio
import slidingwindow
import torch
from PIL import Image
from rasterio.windows import Window
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import default_collate
import pandas as pd

from deepforest.utilities import format_geometry

# Local imports
from deepforest import preprocess


# Base prediction class
class PredictionDataset(Dataset):
    """This is the base class for all prediction datasets. It defines the
    interface for all prediction datasets. It flexibly accepts a single image
    or a list of images, a single path or a list of paths, and a patch_size and
    patch_overlap.

    Args:
        image (PIL.Image.Image): A single image.
        path (str): A single image path.
        images (List[PIL.Image.Image]): A list of images.
        paths (List[str]): A list of image paths.
        patch_size (int): The size of the patches to extract.
        patch_overlap (float): The overlap between patches.
        size (int): The size of the image to resize to. Optional, if not provided, the image is not resized.
    """

    def __init__(self,
                 image=None,
                 path=None,
                 images=None,
                 paths=None,
                 patch_size=400,
                 patch_overlap=0,
                 size=None):
        self.image = image
        self.images = images
        self.path = path
        self.paths = paths
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.size = size
        self.items = self.prepare_items()

    def _load_and_preprocess_image(self, image_path, image=None, size=None):
        """Load and preprocess an image.

        Datasets should load using PIL and transpose the image to (C, H,
        W) before main.model.forward() is called.
        """
        if image is None:
            image = Image.open(image_path)
        else:
            image = image
        image = np.array(image)
        if not image.shape[2] == 3:
            raise ValueError(
                "Only three band raster are accepted. Input tile has shape {}. Check for transparent alpha channel and remove if present"
                .format(image.shape))

        image = np.transpose(image, (2, 0, 1))
        image = self.preprocess_crop(image, size)

        return image

    def preprocess_crop(self, image, size=None):
        """Preprocess a crop to a float32 tensor between 0 and 1."""
        image = np.array(image)
        image = image / 255.0
        image = image.astype(np.float32)

        if size is not None:
            image = self.resize_image(image, size)

        return image

    def resize_image(self, image, size):
        """Resize an image to a new size."""
        image = np.resize(image, (image.shape[0], size, size))
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
        """Collate the batch into a single tensor."""
        # Check if all images in batch have same dimensions
        try:
            return default_collate(batch)
        except RuntimeError:
            raise RuntimeError(
                "Images in batch have different dimensions. Set validation.size in config.yaml to resize all images to a common size."
            )

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
            raise ValueError("Unknown geometry type, prediction keys are {}".format(
                batched_result.keys()))

        return geom_type

    def format_batch(self, batch, idx, sub_idx=None):
        """Format the batch into a single dataframe.

        Args:
            batch (list): The batch to format.
            idx (int): The index of the batch.
            sub_idx (int): The index of the subbatch. If None, the index is the subbatch index.
        """
        if sub_idx is None:
            sub_idx = idx
        geom_type = self.determine_geometry_type(batch)
        result = format_geometry(batch, geom_type=geom_type)
        if result is None:
            return None
        result["window_xmin"] = self.get_crop_bounds(sub_idx)[0]
        result["window_ymin"] = self.get_crop_bounds(sub_idx)[1]
        result["image_path"] = self.get_image_basename(idx)

        return result

    def postprocess(self, batched_result):
        """Postprocess the batched result into a single dataframe.

        In the case of sub-batches, the index is the sub-batch index.
        """
        formatted_result = []
        for idx, batch in enumerate(batched_result):
            if isinstance(batch, list):
                for sub_idx, sub_batch in enumerate(batch):
                    result = self.format_batch(sub_batch, idx, sub_idx)
                    if result is not None:
                        formatted_result.append(result)
            else:
                result = self.format_batch(batch, idx)
                if result is not None:
                    formatted_result.append(result)

        if len(formatted_result) > 0:
            formatted_result = pd.concat(formatted_result)
        else:
            formatted_result = pd.DataFrame()

        # reset index
        formatted_result = formatted_result.reset_index(drop=True)

        return formatted_result


class SingleImage(PredictionDataset):
    """Take in a single image path, preprocess and batch together."""

    def __init__(self, path=None, image=None, patch_size=400, patch_overlap=0):
        super().__init__(path=path,
                         image=image,
                         patch_size=patch_size,
                         patch_overlap=patch_overlap)

    def prepare_items(self):
        self.image = self._load_and_preprocess_image(self.path, self.image)
        self.windows = preprocess.compute_windows(self.image, self.patch_size,
                                                  self.patch_overlap)

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

    def __init__(self, csv_file: str, root_dir: str, size: int = None):
        self.csv_file = csv_file
        self.root_dir = root_dir
        super().__init__(size=size)
        self.prepare_items()

    def prepare_items(self):
        self.annotations = pd.read_csv(self.csv_file)
        self.image_names = self.annotations.image_path.unique()
        self.image_paths = [os.path.join(self.root_dir, x) for x in self.image_names]

    def __len__(self):
        return len(self.image_paths)

    def get_crop(self, idx):
        image = self._load_and_preprocess_image(self.image_paths[idx], size=self.size)
        return image

    def get_image_basename(self, idx):
        return os.path.basename(self.image_paths[idx])

    def get_crop_bounds(self, idx):
        return None

    def format_batch(self, batch, idx, sub_idx=None):
        """Format the batch into a single dataframe.

        Args:
            batch (list): The batch to format.
            idx (int): The index of the batch.
            sub_idx (int): The index of the subbatch. If None, the index is the subbatch index.
        """
        if sub_idx is None:
            sub_idx = idx
        geom_type = self.determine_geometry_type(batch)
        result = format_geometry(batch, geom_type=geom_type)
        if result is None:
            return None
        result["image_path"] = self.get_image_basename(idx)

        return result


class MultiImage(PredictionDataset):
    """Take in a list of image paths, preprocess and batch together."""

    def __init__(self, paths: List[str], patch_size: int, patch_overlap: float):
        """
        Args:
            paths (List[str]): List of image paths.
            patch_size (int): Size of the patches to extract.
            patch_overlap (float): Overlap between patches.
        """
        # Runtime type checking
        if not isinstance(paths, list):
            raise TypeError(f"paths must be a list, got {type(paths)}")

        self.paths = paths
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap

        image = self._load_and_preprocess_image(self.paths[0])
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
        image_tensor = torch.tensor(image).unsqueeze(0)  # Convert to (N, C, H, W)
        patch_overlap_size = int(self.patch_size * self.patch_overlap)
        patches = self.create_overlapping_views(image_tensor, self.patch_size,
                                                patch_overlap_size)

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

        # Calculate number of patches needed in each dimension to match unfold + padding
        n_patches_h = (H - patch_overlap_size) // step + 1
        n_patches_w = (W - patch_overlap_size) // step + 1

        # Generate window coordinates matching the unfolded tensor views (including padded areas)
        # Clamp coordinates so that each window is anchored within the image bounds
        max_y = max(0, H - self.patch_size)
        max_x = max(0, W - self.patch_size)
        windows = []
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                y = min(i * step, max_y)
                x = min(j * step, max_x)
                windows.append((x, y, self.patch_size, self.patch_size))

        return windows

    def collate_fn(self, batch):
        # Comes pre-batched
        return batch

    def __len__(self):
        return len(self.paths)

    def get_crop(self, idx):
        image = self._load_and_preprocess_image(self.paths[idx])
        return self._create_patches(image)

    def get_image_basename(self, idx):
        return os.path.basename(self.paths[idx])

    def get_crop_bounds(self, idx):
        return self.window_list()[idx]


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
        self.path = path
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.prepare_items()

        if path is None:
            raise ValueError("path is required for a memory raster dataset")

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
                    "to create a tiled raster")

        # Generate sliding windows
        self.windows = slidingwindow.generateForSize(
            height,
            width,
            dimOrder=slidingwindow.DimOrder.ChannelHeightWidth,
            maxWindowSize=self.patch_size,
            overlapPercent=self.patch_overlap)

    def __len__(self):
        return len(self.windows)

    def window_list(self):
        return [x.getRect() for x in self.windows]

    def get_crop(self, idx):
        window = self.windows[idx]
        with rio.open(self.path) as src:
            window_data = src.read(window=Window(window.x, window.y, window.w, window.h))

        # Convert to torch tensor and rearrange dimensions
        window_data = torch.from_numpy(window_data).float()  # Convert to torch tensor
        window_data = window_data / 255.0  # Normalize

        return window_data

    def get_image_basename(self, idx):
        return os.path.basename(self.path)

    def get_crop_bounds(self, idx):
        return self.window_list()[idx]
