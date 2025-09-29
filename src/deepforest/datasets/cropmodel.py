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


def bounding_box_transform(augment=False):
    """Create transform pipeline for bounding box data.

    Args:
        augment: Whether to apply data augmentation

    Returns:
        Composed transform pipeline
    """
    data_transforms = []
    data_transforms.append(transforms.ToTensor())
    data_transforms.append(resnet_normalize)
    data_transforms.append(transforms.Resize([224, 224]))
    if augment:
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

    Returns:
        Tensor of shape (3, height, width)
    """

    def __init__(self, df, root_dir, transform=None, augment=False):
        self.df = df

        if transform is None:
            self.transform = bounding_box_transform(augment=augment)
        else:
            self.transform = transform

        unique_image = self.df["image_path"].unique()
        assert len(unique_image) == 1, (
            "There should be only one unique image for this class object"
        )

        # Open the image using rasterio
        self.src = rio.open(os.path.join(root_dir, unique_image[0]))

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
        xmin = row["xmin"]
        xmax = row["xmax"]
        ymin = row["ymin"]
        ymax = row["ymax"]

        # Read the RGB data
        box = self.src.read(window=Window(xmin, ymin, xmax - xmin, ymax - ymin))
        box = np.rollaxis(box, 0, 3)

        if self.transform:
            image = self.transform(box)
        else:
            image = box

        return image
