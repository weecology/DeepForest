"""Dataset model for object detection tasks."""

# Standard library imports
import os
from typing import Dict, List, Optional, Union
import warnings

# Third party imports
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import shapely
from deepforest.augmentations import get_transform


class BoxDataset(Dataset):

    def __init__(self,
                 csv_file,
                 root_dir,
                 *,
                 transforms=None,
                 augment=None,
                 augmentations=None,
                 label_dict={"Tree": 0},
                 preload_images=False):
        """
        Args:
            csv_file (string): Path to a single csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            label_dict: a dictionary where keys are labels from the csv column and values are numeric labels "Tree" -> 0
            augment: if True, apply augmentations to the images
            augmentations: augmentation configuration (str, list, or dict)
            preload_images: if True, preload the images into memory
        Returns:
            List of images and targets. Targets are dictionaries with keys "boxes" and "labels". Boxes are numpy arrays with shape (N, 4) and labels are numpy arrays with shape (N,).
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        if transforms is None:

            if augment is not None:
                warnings.warn(
                    "The `augment` parameter is deprecated. Please use `augmentations` instead and provide an empty list or None to disable augmentations."
                )
                if not augment:
                    augmentations = None

            self.transform = get_transform(augmentations=augmentations)
        else:
            self.transform = transforms
        self.image_names = self.annotations.image_path.unique()
        self.label_dict = label_dict
        self.preload_images = preload_images

        self._validate_labels()

        # Pin data to memory if desired
        if self.preload_images:
            print("Pinning dataset to GPU memory")
            self.image_dict = {}
            for idx, x in enumerate(self.image_names):
                self.image_dict[idx] = self.load_image(idx)

    def _validate_labels(self):
        """Validate that all labels in annotations exist in label_dict.

        Raises:
            ValueError: If any label in annotations is missing from label_dict
        """
        csv_labels = self.annotations['label'].unique()
        missing_labels = [label for label in csv_labels if label not in self.label_dict]

        if missing_labels:
            raise ValueError(
                f"Labels {missing_labels} are missing from label_dict. "
                f"Please ensure all labels in the annotations exist as keys in label_dict."
            )

    def __len__(self):
        return len(self.image_names)

    def collate_fn(self, batch):
        """Collate function for DataLoader."""
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        image_names = [item[2] for item in batch]

        return images, targets, image_names

    def load_image(self, idx):
        img_name = os.path.join(self.root_dir, self.image_names[idx])
        image = np.array(Image.open(img_name).convert("RGB")) / 255
        image = image.astype("float32")
        return image

    def __getitem__(self, idx):

        # Read image if not in memory
        if self.preload_images:
            image = self.image_dict[idx]
        else:
            image = self.load_image(idx)

        # select annotations
        image_annotations = self.annotations[self.annotations.image_path ==
                                             self.image_names[idx]]
        targets = {}

        if "geometry" in image_annotations.columns:
            targets["boxes"] = np.array([
                shapely.wkt.loads(x).bounds for x in image_annotations.geometry
            ]).astype("float32")
        else:
            targets["boxes"] = image_annotations[["xmin", "ymin", "xmax",
                                                  "ymax"]].values.astype("float32")

        # Labels need to be encoded
        targets["labels"] = image_annotations.label.apply(
            lambda x: self.label_dict[x]).values.astype(np.int64)

        # If image has no annotations, don't augment
        if np.sum(targets["boxes"]) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            # channels last
            image = np.rollaxis(image, 2, 0)
            image = torch.from_numpy(image).float()
            targets = {"boxes": boxes, "labels": labels}

            return image, targets, self.image_names[idx]

        # Apply augmentations
        augmented = self.transform(image=image,
                                   bboxes=targets["boxes"],
                                   category_ids=targets["labels"].astype(np.int64))
        image = augmented["image"]

        # Convert boxes to tensor
        boxes = np.array(augmented["bboxes"])
        boxes = torch.from_numpy(boxes).float()
        labels = np.array(augmented["category_ids"])
        labels = torch.from_numpy(labels.astype(np.int64))
        targets = {"boxes": boxes, "labels": labels}

        return image, targets, self.image_names[idx]
