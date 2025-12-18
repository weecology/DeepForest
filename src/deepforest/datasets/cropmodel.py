"""Crop model dataset for DeepForest.

Provides dataset classes and transforms for training crop detection
models on bounding box annotations.
"""

import os

import numpy as np
import rasterio as rio
from rasterio.windows import Window
from torch.utils.data import Dataset
from torchvision import transforms

from deepforest.utilities import read_raster_window


def bounding_box_transform(augmentations=None, resize=None):
    """Create transform pipeline for bounding box data.

    Args:
        augmentations: Augmentation configuration (str, list, or dict)
        resize: Optional list of [height, width] for resizing. Defaults to [224, 224]

    Returns:
        Composed transform pipeline
    """
    if resize is None:
        resize = [224, 224]

    data_transforms = []
    data_transforms.append(transforms.ToTensor())
    data_transforms.append(resnet_normalize)
    data_transforms.append(transforms.Resize(resize))
    if augmentations:
        data_transforms.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(data_transforms)


resnet_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


class BoundingBoxDataset(Dataset):
    """Dataset for bounding box predictions on single images.

    Args:
        df: DataFrame with image_path and bounding box columns
        root_dir: Directory containing images
        transform: Optional transform function
        augmentations: Augmentation configuration
        resize: Optional list of [height, width] for resizing. Defaults to [224, 224]
        expand: expand: number of context pixels to add to input bounding boxes

    Note: Use the expand option to sample a larger area around the source bounding box. This may improve classification accuracy by providing the model with increased context. The sampling window is increased by `expand` pixels on all sides and then clamped to the image bounds.

    Returns:
        Tensor of shape (3, height, width)
    """

    def __init__(
        self,
        df,
        root_dir,
        transform=None,
        augmentations=None,
        resize=None,
        expand: int = 0,
    ):
        self.df = df

        if transform is None:
            self.transform = bounding_box_transform(
                augmentations=augmentations, resize=resize
            )
        else:
            self.transform = transform

        # Check that expand is non-negative
        if expand < 0:
            raise ValueError("expand must be >= 0")
        self.expand = int(expand)

        unique_image = self.df["image_path"].unique()
        assert len(unique_image) == 1, (
            "There should be only one unique image for this class object"
        )

        # Open the image using rasterio
        self.src = rio.open(os.path.join(root_dir, unique_image[0]))
        self._image_width = self.src.width
        self._image_height = self.src.height

    def __len__(self):
        """Return number of samples."""
        return len(self.df)

    def __getitem__(self, idx):
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Transformed image tensor
        """
        row = self.df.iloc[idx]
        xmin = float(row["xmin"])
        xmax = float(row["xmax"])
        ymin = float(row["ymin"])
        ymax = float(row["ymax"])

        # Expand the box equally on all sides by self.expand pixels (context window)
        if self.expand > 0:
            xmin = max(0, xmin - self.expand)
            ymin = max(0, ymin - self.expand)
            xmax = min(self._image_width, xmax + self.expand)
            ymax = min(self._image_height, ymax + self.expand)

        # Read the RGB data
        col_off = int(xmin)
        row_off = int(ymin)
        width = int(max(1, xmax - xmin))
        height = int(max(1, ymax - ymin))
        box = self.src.read(window=Window(col_off, row_off, width, height))
        box = read_raster_window(box, nodata_value=self.src.nodata)
        box = np.rollaxis(box, 0, 3)

        if self.transform:
            image = self.transform(box)
        else:
            image = box

        return image
