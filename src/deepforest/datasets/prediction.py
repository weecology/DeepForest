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
    def __init__(self, image=None, path=None, patch_size=None, patch_overlap=None):
        self.image = image
        self.path = path
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.items = self.prepare_items()

    def _load_and_preprocess_image(self, image_path):
        image = Image.open(image_path)
        image = self.preprocess_crop(image)
 
        if not image.shape[2] == 3:
            raise ValueError(
                "Only three band raster are accepted. Channels should be the final dimension. Input tile has shape {}. Check for transparent alpha channel and remove if present"
                .format(image.shape))
        
        return image
    
    def preprocess_crop(self, image):
        image = np.array(image)
        image = image / 255.0
        image = image.astype(np.float32)
        return image
    
    def prepare_items(self):
        raise NotImplementedError("Subclasses must implement this method")

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        """
        Get the item at the given index
        """
        return self.get_crop(idx)
    
    def collate_fn(self, batch):
        """
        Collate the batch into a single tensor
        """
        return default_collate(batch)
    
    def get_crop_bounds(self, idx):
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_crop(self, idx):
        """
        Get the crop of the image at the given index
        """
        raise NotImplementedError("Subclasses must implement this method")  
    
    def get_image_basename(self, idx):
        """
        Get the basename of the image at the given index
        """
        raise NotImplementedError("Subclasses must implement this method")

    def postprocess(self, batched_result, idx): 
        """
        Postprocess the batched result into a single dataframe
        """
        # Assumes that all geometries are the same in a batch
        if "boxes" in batched_result[0].keys():
            geom_type = "box"
        elif "points" in batched_result[0].keys():
            geom_type = "point"
        elif "polygons" in batched_result[0].keys():
            geom_type = "polygon"
        else:
            raise ValueError("Unknown geometry type, prediction keys are {}".format(batched_result[0].keys()))

        formatted_result = []
        for batch in batched_result:
            result = format_geometry(batch, geom_type=geom_type)
            if result is None:
                continue
            result["window_xmin"] = self.get_crop_bounds(idx)[0] 
            result["window_ymin"] = self.get_crop_bounds(idx)[1]
            result["image_path"] = self.get_image_basename(idx)
            formatted_result.append(result)

        # Concatenate all results into a single dataframe
        if len(formatted_result) > 0:
            formatted_result = pd.concat(formatted_result)
        else:
            formatted_result = pd.DataFrame()
        return formatted_result

class SingleImage(PredictionDataset):
    def __init__(self, path=None, image=None, patch_size=None, patch_overlap=None):
        super().__init__(path=path, image=image, patch_size=patch_size, patch_overlap=patch_overlap)

    def prepare_items(self):
        if self.path is not None:
            self.image = self._load_and_preprocess_image(self.path)
        else:
            self.image = self.image
        self.windows = preprocess.compute_windows(self.image, self.patch_size, self.patch_overlap)

    def __len__(self):
        return len(self.windows)
    
    def window_list(self):
        return [x.getRect() for x in self.windows]
    
    def get_crop(self, idx):
        crop = self.image[self.windows[idx].indices()]
        crop = preprocess.preprocess_image(crop)

        return crop
    
    def get_image_basename(self, idx):
        if self.path is not None:
            return os.path.basename(self.path)
        else:
            return None
    
    def get_crop_bounds(self, idx):
        return self.windows[idx].getRect()
    
class FromCSVFile(PredictionDataset):
    def __init__(self, csv_file: str, root_dir: str):
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.prepare_items()

    def prepare_items(self):
        self.annotations = pd.read_csv(self.csv_file)
        self.image_names = self.annotations.image_path.unique()
        self.image_paths = [os.path.join(self.root_dir, x) for x in self.image_names]
    
    def __len__(self):
        return len(self.image_paths)
    
    def get_crop(self, idx):
        image = self._load_and_preprocess_image(self.image_paths[idx])
        return image
    
    def get_image_basename(self, idx):
        return os.path.basename(self.image_paths[idx])
    
    def get_crop_bounds(self, idx):
        return None

    
class MultiImage(PredictionDataset):
    def __init__(self, image_paths: List[str], patch_size: int, patch_overlap: float):
        """
        Args:
            image_paths (List[str]): List of image paths.
            patch_size (int): Size of the patches to extract.
            patch_overlap (float): Overlap between patches.
        """
        # Runtime type checking
        if not isinstance(image_paths, list):
            raise TypeError(f"image_paths must be a list, got {type(image_paths)}")
            
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap

    def create_overlapping_views(self, input_tensor, size, overlap):
        """
        Creates overlapping views of a 4D tensor.

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

        # Calculate the number of padding pixels needed, accounting for overlap
        padding_h = size - ((H - overlap) % (size - overlap)) if H % (size - overlap) != 0 else 0
        padding_w = size - ((W - overlap) % (size - overlap)) if W % (size - overlap) != 0 else 0

        # Pad the input tensor
        padded_tensor = F.pad(input_tensor, (padding_w, padding_w, padding_h, padding_h), "constant", 0)

        # Calculate the step size for the sliding window
        step = size - overlap

        # Use torch.unfold to create the overlapping views
        unfolded_h = padded_tensor.unfold(2, size, step)
        unfolded = unfolded_h.unfold(3, size, step)

        # The unfolded tensor has shape:
        # [N, C, H', W', size, size]
        # where H' and W' are the number of sliding windows in the height and width dimensions

        # Reshape to [N * H' * W', C, size, size]
        output = unfolded.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, size, size)
                
        return output

    def _create_patches(self, image):
        image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0) # Convert to (N, C, H, W)
        patch_overlap_size = int(self.patch_size * self.patch_overlap)
        patches = self.create_overlapping_views(image_tensor, self.patch_size, patch_overlap_size)

        return patches
    
    def window_list(self):
        """Get the original positions of patches in the image.
        
        Returns:
            list: List of tuples containing (x, y, w, h) coordinates of each patch
        """
        image_tensor = torch.tensor(self.image).permute(2, 0, 1).unsqueeze(0)  # Convert to (N, C, H, W)
        N, C, H, W = image_tensor.shape
        patch_overlap_size = int(self.patch_size * self.patch_overlap)
        step = self.patch_size - patch_overlap_size

        # Calculate padding needed
        padding_h = self.patch_size - ((H - patch_overlap_size) % (self.patch_size - patch_overlap_size)) if H % (self.patch_size - patch_overlap_size) != 0 else 0
        padding_w = self.patch_size - ((W - patch_overlap_size) % (self.patch_size - patch_overlap_size)) if W % (self.patch_size - patch_overlap_size) != 0 else 0

        # Adjust H and W to include padding
        H_padded = H + padding_h
        W_padded = W + padding_w

        windows = []
        for y in range(0, H_padded - self.patch_size + 1, step):
            for x in range(0, W_padded - self.patch_size + 1, step):
                windows.append((x, y, self.patch_size, self.patch_size))

        return windows

    def collate_fn(self, batch):
        # Separate first and second positions from each batch item
        crops = [item[0] for item in batch]
        crops = torch.cat(crops, dim=0)
        
        return crops

    def __len__(self):
        return len(self.image_paths)

    def get_crop(self, idx):
        self.image = self._load_and_preprocess_image(self.image_paths[idx])
        return self._create_patches(self.image)
    
    def get_image_basename(self, idx):
        return os.path.basename(self.image_paths[idx])
    
    def get_crop_bounds(self, idx):
        return self.window_list()

class TiledRaster(PredictionDataset):
    """Dataset for predicting on raster windows

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

    def collate_fn(self, batch):
        # Separate first and second positions from each batch item
        crops = [item[0] for item in batch]
        
        # Concatenate crops and return with windows
        return torch.stack(crops, dim=0)
    
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
        return os.path.basename(self.raster_path)
    
    def get_crop_bounds(self, idx):
        return self.window_list()[idx]