"""
Dataset model

https://pytorch.org/docs/stable/torchvision/models.html#object-detection-instance-segmentation-and-person-keypoint-detection

During training, the model expects both the input tensors, as well as a targets (list of dictionary), containing:

boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W

labels (Int64Tensor[N]): the class label for each ground-truth box

https://colab.research.google.com/github/benihime91/pytorch_retinanet/blob/master/demo.ipynb#scrollTo=0zNGhr6D7xGN

"""
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations import functional as F
from albumentations.pytorch import ToTensorV2
import torch
import typing
from PIL import Image
import rasterio as rio
from deepforest import preprocess


def get_transform(augment):
    """Albumentations transformation of bounding boxs"""
    if augment:
        transform = A.Compose(
            [A.HorizontalFlip(p=0.5), ToTensorV2()],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=["category_ids"]))

    else:
        transform = A.Compose([ToTensorV2()],
                              bbox_params=A.BboxParams(format='pascal_voc',
                                                       label_fields=["category_ids"]))

    return transform


class TreeDataset(Dataset):

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
            targets["boxes"] = image_annotations[["xmin", "ymin", "xmax",
                                                  "ymax"]].values.astype(float)

            # Labels need to be encoded
            targets["labels"] = image_annotations.label.apply(
                lambda x: self.label_dict[x]).values.astype(np.int64)

            # If image has no annotations, don't augment
            if np.sum(targets["boxes"]) == 0:
                boxes = boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.from_numpy(targets["labels"])
                # channels last
                image = np.rollaxis(image, 2, 0)
                image = torch.from_numpy(image)
                targets = {"boxes": boxes, "labels": labels}
                return self.image_names[idx], image, targets

            augmented = self.transform(image=image,
                                       bboxes=targets["boxes"],
                                       category_ids=targets["labels"])
            image = augmented["image"]

            boxes = np.array(augmented["bboxes"])
            boxes = torch.from_numpy(boxes)
            labels = np.array(augmented["category_ids"])
            labels = torch.from_numpy(labels)
            targets = {"boxes": boxes, "labels": labels}

            return self.image_names[idx], image, targets

        else:
            # Mimic the train augmentation
            converted = self.image_converter(image=image)
            return converted["image"]


class TileDataset(Dataset):

    def __init__(self,
                 tile: typing.Optional[np.ndarray],
                 preload_images: bool = False,
                 patch_size: int = 400,
                 patch_overlap: float = 0.05):
        """
        Args:
            tile: an in memory numpy array.
            patch_size (int): The size for the crops used to cut the input raster into smaller pieces. This is given in pixels, not any geographic unit.
            patch_overlap (float): The horizontal and vertical overlap among patches
        Returns:
            ds: a pytorch dataset
        """
        if not tile.shape[2] == 3:
            raise ValueError(
                "Only three band raster are accepted. Channels should be the final dimension. Input tile has shape {}. Check for transparent alpha channel and remove if present"
                .format(tile.shape))

        self.image = tile
        self.preload_images = preload_images
        self.windows = preprocess.compute_windows(self.image, patch_size, patch_overlap)

        if self.preload_images:
            self.crops = []
            for window in self.windows:
                crop = self.image[window.indices()]
                crop = preprocess.preprocess_image(crop)
                self.crops.append(crop)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Read image if not in memory
        if self.preload_images:
            crop = self.crops[idx]
        else:
            crop = self.image[self.windows[idx].indices()]
            crop = preprocess.preprocess_image(crop)

        return crop
