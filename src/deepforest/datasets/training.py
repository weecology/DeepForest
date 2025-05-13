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
    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        transforms: Optional[A.Compose] = None,
        label_dict: Dict[str, int] = {"Tree": 0},
        train: bool = True,
        preload_images: bool = False
    ):
        """
        Initialize the TrainDataset for Box Geometry.

        Args:
            csv_file (str): Path to a single csv file with annotations.
            root_dir (str): Directory with all the images.
            transforms (Optional[A.Compose]): Optional transform to be applied on a sample.
            label_dict (Dict[str, int]): A dictionary mapping labels to numeric values.
            train (bool): Whether to trigger default train augmentations.
            preload_images (bool): Whether to preload images into memory.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms if transforms is not None else get_transform(augment=train)
        self.image_names = self.annotations.image_path.unique()
        self.label_dict = label_dict
        self.train = train
        self.image_converter = A.Compose([ToTensorV2()])
        self.preload_images = preload_images

        if self.preload_images:
            self._preload_images()

    def _preload_images(self):
        """Preload images into memory if desired."""
        print("Pinning dataset to GPU memory")
        self.image_dict = {}
        for idx, img_path in enumerate(self.image_names):
            img_name = os.path.join(self.root_dir, img_path)
            image = np.array(Image.open(img_name).convert("RGB")) / 255
            self.image_dict[idx] = image.astype("float32")

    def __len__(self) -> int:
        return len(self.image_names)

    def collate_fn(self, batch: List[tuple]) -> tuple:
        """Collate function for DataLoader."""
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return images, targets

    def __getitem__(self, idx: int) -> Union[tuple, torch.Tensor]:
        """Get item from dataset."""
        if self.preload_images:
            image = self.image_dict[idx]
        else:
            img_name = os.path.join(self.root_dir, self.image_names[idx])
            image = np.array(Image.open(img_name).convert("RGB")) / 255
            image = image.astype("float32")

        if self.train:
            return self._get_train_item(idx, image)
        else:
            return self._get_test_item(image)

    def _get_train_item(self, idx: int, image: np.ndarray) -> tuple:
        """Process and return a training item."""
        image_annotations = self.annotations[self.annotations.image_path == self.image_names[idx]]
        targets = self._prepare_targets(image_annotations)

        if np.sum(targets["boxes"]) == 0:
            return self._handle_empty_annotations(idx, image)

        augmented = self.transform(
            image=image,
            bboxes=targets["boxes"],
            category_ids=targets["labels"].astype(np.int64)
        )

        return self._process_augmented_data(idx, augmented)

    def _get_test_item(self, image: np.ndarray) -> torch.Tensor:
        """Process and return a test item."""
        converted = self.image_converter(image=image)
        return converted["image"]

    def _prepare_targets(self, image_annotations: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Prepare target boxes and labels."""
        if "geometry" in image_annotations.columns:
            boxes = np.array([
                shapely.wkt.loads(x).bounds for x in image_annotations.geometry
            ]).astype("float32")
        else:
            boxes = image_annotations[["xmin", "ymin", "xmax", "ymax"]].values.astype("float32")

        labels = image_annotations.label.apply(lambda x: self.label_dict[x]).values.astype(np.int64)

        return {"boxes": boxes, "labels": labels}

    def _handle_empty_annotations(self, idx: int, image: np.ndarray) -> tuple:
        """Handle cases where there are no annotations."""
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros(0, dtype=torch.int64)
        image = torch.from_numpy(np.rollaxis(image, 2, 0)).float()
        targets = {"boxes": boxes, "labels": labels}
        return self.image_names[idx], image, targets

    def _process_augmented_data(self, idx: int, augmented: Dict[str, Union[torch.Tensor, List]]) -> tuple:
        """Process augmented data and prepare final output."""
        image = augmented["image"]
        boxes = torch.from_numpy(np.array(augmented["bboxes"])).float()
        labels = torch.from_numpy(np.array(augmented["category_ids"]).astype(np.int64))
        targets = {"boxes": boxes, "labels": labels}
        return self.image_names[idx], image, targets
