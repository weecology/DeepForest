"""Dataset model for object detection tasks."""

# Standard library imports
import os
from typing import Dict, List, Optional, Union

# Third party imports
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import shapely

def get_transform(augment: bool) -> A.Compose:
    """Create Albumentations transformation for bounding boxes."""
    bbox_params = A.BboxParams(format='pascal_voc', label_fields=["category_ids"])
    
    if augment:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            ToTensorV2()
        ], bbox_params=bbox_params)
    else:
        return A.Compose([ToTensorV2()], bbox_params=bbox_params)


class BoxDataset(Dataset):
    def __init__(self,
                 csv_file,
                 root_dir,
                 transforms=None,
                 label_dict={"Tree": 0},
                 train=True,
                 preload_images=False):
        """
        
        Args:
            csv_file (string): Path to a single csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            label_dict: a dictionary where keys are labels from the csv column and values are numeric labels "Tree" -> 0
        
        Returns:
            If train, path, image, targets else image
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        if transforms is None:
            self.transform = get_transform(augment=train)
        else:
            self.transform = transforms
        self.image_names = self.annotations.image_path.unique()
        self.label_dict = label_dict
        self.train = train
        self.image_converter = A.Compose([ToTensorV2()])
        self.preload_images = preload_images

        # Pin data to memory if desired
        if self.preload_images:
            print("Pinning dataset to GPU memory")
            self.image_dict = {}
            for idx, x in enumerate(self.image_names):
                img_name = os.path.join(self.root_dir, x)
                image = np.array(Image.open(img_name).convert("RGB")) / 255
                self.image_dict[idx] = image.astype("float32")

    def __len__(self):
        return len(self.image_names)

    def collate_fn(self, batch):
        """
        Collate function for DataLoader.
        """
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return images, targets
    
    def __getitem__(self, idx):

        # Read image if not in memory
        if self.preload_images:
            image = self.image_dict[idx]
        else:
            img_name = os.path.join(self.root_dir, self.image_names[idx])
            image = np.array(Image.open(img_name).convert("RGB")) / 255
            image = image.astype("float32")

        if self.train:
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

                return self.image_names[idx], image, targets

            augmented = self.transform(image=image,
                                       bboxes=targets["boxes"],
                                       category_ids=targets["labels"].astype(np.int64))
            image = augmented["image"]

            boxes = np.array(augmented["bboxes"])
            boxes = torch.from_numpy(boxes).float()
            labels = np.array(augmented["category_ids"])
            labels = torch.from_numpy(labels.astype(np.int64))
            targets = {"boxes": boxes, "labels": labels}

            return self.image_names[idx], image, targets

        else:
            # Mimic the train augmentation
            converted = self.image_converter(image=image)
            return converted["image"]

