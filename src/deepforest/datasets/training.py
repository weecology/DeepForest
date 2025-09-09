"""Dataset model for object detection tasks."""

import os
import warnings

import numpy as np
import pandas as pd
import shapely
import torch
from PIL import Image
from torch.utils.data import Dataset

from deepforest.augmentations import get_transform


class BoxDataset(Dataset):
    """Dataset for object detection with bounding boxes.

    Args:
        csv_file: Path to CSV file with annotations
        root_dir: Directory containing images
        transforms: Function applied to each sample
        augment: Deprecated - use augmentations instead
        augmentations: Augmentation configuration
        label_dict: Mapping from string labels to class IDs
        preload_images: Preload all images into memory

    Returns:
        List of (image, target) pairs where target contains:
        - "boxes": numpy array of shape (N, 4)
        - "labels": numpy array of shape (N,)
    """

    def __init__(
        self,
        csv_file,
        root_dir,
        *,
        transforms=None,
        augment=None,
        augmentations=None,
        label_dict=None,
        preload_images=False,
    ):
        """
        Args:
            csv_file (str): Path to the CSV file containing annotations.
            root_dir (str): Directory containing all referenced images.
            transform (callable, optional): Function applied to each sample (e.g., image and target). Defaults to None.
            label_dict (dict[str, int]): Mapping from string labels in the CSV to integer class IDs (e.g., {"Tree": 0}).
            augment (bool): Whether to apply data augmentations. Defaults to False.
            augmentations (str | list | dict, optional): Augmentation configuration used when `augment` is True.
            preload_images (bool): If True, preload all images into memory. Defaults to False.

        Returns:
            list: A list of (image, target) pairs, where each target is a dict with:
                - "boxes": numpy.ndarray of shape (N, 4)
                - "labels": numpy.ndarray of shape (N,)
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir

        # Initialize label_dict with default if None
        if label_dict is None:
            label_dict = {"Tree": 0}
        self.label_dict = label_dict

        if transforms is None:
            if augment is not None:
                warnings.warn(
                    "The `augment` parameter is deprecated. Please use `augmentations`"
                    "Use empty list or None to disable augmentations.",
                    stacklevel=2,
                )
                if not augment:
                    augmentations = None

            self.transform = get_transform(augmentations=augmentations)
        else:
            self.transform = transforms
        self.image_names = self.annotations.image_path.unique()
        self.preload_images = preload_images

        self._validate_labels()

        # Pin data to memory if desired
        if self.preload_images:
            print("Pinning dataset to GPU memory")
            self.image_dict = {}
            for idx, _x in enumerate(self.image_names):
                self.image_dict[idx] = self.load_image(idx)

    def _validate_labels(self):
        """Validate that all labels in annotations exist in label_dict.

        Raises:
            ValueError: If any label in annotations is missing from label_dict
        """
        csv_labels = self.annotations["label"].unique()
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
        image_annotations = self.annotations[
            self.annotations.image_path == self.image_names[idx]
        ]
        targets = {}

        if "geometry" in image_annotations.columns:
            targets["boxes"] = np.array(
                [shapely.wkt.loads(x).bounds for x in image_annotations.geometry]
            ).astype("float32")
        else:
            targets["boxes"] = image_annotations[
                ["xmin", "ymin", "xmax", "ymax"]
            ].values.astype("float32")

        # Labels need to be encoded
        targets["labels"] = image_annotations.label.apply(
            lambda x: self.label_dict[x]
        ).values.astype(np.int64)

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
        augmented = self.transform(
            image=image,
            bboxes=targets["boxes"],
            category_ids=targets["labels"].astype(np.int64),
        )
        image = augmented["image"]

        # Convert boxes to tensor
        boxes = np.array(augmented["bboxes"])
        boxes = torch.from_numpy(boxes).float()
        labels = np.array(augmented["category_ids"])
        labels = torch.from_numpy(labels.astype(np.int64))
        targets = {"boxes": boxes, "labels": labels}

        return image, targets, self.image_names[idx]
