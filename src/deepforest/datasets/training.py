"""Dataset model for object detection tasks."""

import math
import os

import kornia.augmentation as K
import numpy as np
import shapely
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from deepforest import utilities
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
            augmentations (str | list | dict, optional): Augmentation configuration.
            preload_images (bool): If True, preload all images into memory. Defaults to False.

        Returns:
            list: A list of (image, target) pairs, where each target is a dict with:
                - "boxes": numpy.ndarray of shape (N, 4)
                - "labels": numpy.ndarray of shape (N,)
        """
        self.annotations = utilities.read_file(csv_file, root_dir=root_dir)
        self.root_dir = root_dir

        # Initialize label_dict with default if None
        if label_dict is None:
            label_dict = {"Tree": 0}
        self.label_dict = label_dict

        if transforms is None:
            self.transform = get_transform(augmentations=augmentations)
        else:
            if not isinstance(transforms, K.AugmentationSequential):
                raise ValueError(
                    "User-supplied dataset transform must be a kornia AugmentationSequential object."
                )
            self.transform = transforms

        self.image_names = self.annotations.image_path.unique()
        self.preload_images = preload_images

        self._validate_labels()
        self._validate_coordinates()

        # Pin data to memory if desired
        if self.preload_images:
            print("Pinning dataset to GPU memory")
            self.image_dict = {}
            for idx, _ in enumerate(self.image_names):
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

    def _validate_coordinates(self):
        """Validate that all bounding box coordinates occur within the image.

        Raises:
            ValueError: If any bounding box coordinate occurs outside the image
        """
        errors = []
        for _idx, row in self.annotations.iterrows():
            img_path = os.path.join(self.root_dir, row["image_path"])
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                errors.append(f"Failed to open image {img_path}: {e}")
                continue

            # Extract bounding box
            geom = row["geometry"]
            xmin, ymin, xmax, ymax = geom.bounds

            # All coordinates equal to zero is how we code empty frames.
            # Therefore these are valid coordinates even though they would
            # fail other checks.
            if xmin == 0 and ymin == 0 and xmax == 0 and ymax == 0:
                continue

            # Check if box is valid
            oob_issues = []
            if not geom.equals(shapely.envelope(geom)):
                oob_issues.append(f"geom ({geom}) is not a valid bounding box")
            if xmin < 0:
                oob_issues.append(f"xmin ({xmin}) < 0")
            if xmax > width:
                oob_issues.append(f"xmax ({xmax}) > image width ({width})")
            if ymin < 0:
                oob_issues.append(f"ymin ({ymin}) < 0")
            if ymax > height:
                oob_issues.append(f"ymax ({ymax}) > image height ({height})")
            if math.isclose(geom.area, 1):
                oob_issues.append("area of bounding box is a single pixel")

            if oob_issues:
                errors.append(
                    f"Box, ({xmin}, {ymin}, {xmax}, {ymax}) exceeds image dimensions, ({width}, {height}). Issues: {', '.join(oob_issues)}."
                )

        if errors:
            raise ValueError("\n".join(errors))

    def filter_boxes(self, boxes, labels, width, height, min_size=1):
        """Clamp boxes to image bounds and filter by minimum dimension.

        Args:
            boxes (torch.Tensor): Bounding boxes of shape (N, 4) in xyxy format.
            labels (torch.Tensor): Labels of shape (N,).
            width (int): Image width in pixels.
            height (int): Image height in pixels.
            min_size (int): Minimum box width/height in pixels. Defaults to 1.

        Returns:
            tuple: A tuple of (filtered_boxes, filtered_labels)
        """

        # Clamp boxes to image bounds
        boxes[:, 0] = torch.clamp(boxes[:, 0], min=0, max=width)  # x1
        boxes[:, 1] = torch.clamp(boxes[:, 1], min=0, max=height)  # y1
        boxes[:, 2] = torch.clamp(boxes[:, 2], min=0, max=width)  # x2
        boxes[:, 3] = torch.clamp(boxes[:, 3], min=0, max=height)  # y2

        # Filter boxes with minimum size
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        valid_mask = (width >= min_size) & (height >= min_size)

        return boxes[valid_mask], labels[valid_mask]

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

    def annotations_for_path(self, image_path, return_tensor=False):
        """Construct target dictionary for a given image path, optionally
        convert to tensor.

        Args:
            image_path (str): Path to image, expected to be in dataframe
            return_tensor (bool): If true, convert fields from numpy to tensor

        Returns:
            target dictionary with boxes and labels entries
        """
        image_annotations = self.annotations[self.annotations.image_path == image_path]
        targets = {}

        if "geometry" in image_annotations.columns:
            # Handle both shapely geometry objects and WKT strings
            targets["boxes"] = np.array(
                [
                    x.bounds if hasattr(x, "bounds") else shapely.wkt.loads(x).bounds
                    for x in image_annotations.geometry
                ]
            ).astype("float32")
        else:
            targets["boxes"] = image_annotations[
                ["xmin", "ymin", "xmax", "ymax"]
            ].values.astype("float32")

        # Labels need to be encoded
        targets["labels"] = image_annotations.label.apply(
            lambda x: self.label_dict[x]
        ).values.astype(np.int64)

        if return_tensor:
            for k, v in targets.items():
                targets[k] = torch.from_numpy(v)

        return targets

    def __getitem__(self, idx):
        # Read image if not in memory
        if self.preload_images:
            image = self.image_dict[idx]
        else:
            image = self.load_image(idx)

        targets = self.annotations_for_path(self.image_names[idx])

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
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        boxes_tensor = torch.from_numpy(targets["boxes"]).unsqueeze(0).float()
        augmented_image, augmented_boxes = self.transform(image_tensor, boxes_tensor)

        # Convert boxes to tensor
        image = augmented_image.squeeze(0)
        boxes = augmented_boxes.squeeze(0)
        labels = torch.from_numpy(targets["labels"].astype(np.int64))

        # Filter invalid boxes after augmentation
        # Since the augmentation operation may change image size, we take the smallest
        # of the source and transformed dimensions assuming that padding is always the
        # last operation in the pipeline.
        boxes, labels = self.filter_boxes(
            boxes,
            labels,
            width=min(image_tensor.shape[3], image.shape[2]),
            height=min(image_tensor.shape[2], image.shape[1]),
        )

        # Edge case if all labels were augmented away, keep the image
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)

        targets = {"boxes": boxes, "labels": labels}

        return image, targets, self.image_names[idx]


# ---------- ImageFolder alignment utilities ----------


class FixedClassImageFolder(ImageFolder):
    """ImageFolder that enforces a provided class_to_idx mapping.

    Samples and targets are remapped based on the folder names so that
    the numeric labels strictly follow the supplied mapping. Classes
    absent in the dataset are simply not represented in samples.
    """

    def __init__(self, root, class_to_idx, transform=None):
        # Build with super to scan files
        super().__init__(root=root, transform=transform)

        # Force mapping and classes order
        self.class_to_idx = dict(class_to_idx)
        classes = [None] * len(self.class_to_idx)
        for name, idx in self.class_to_idx.items():
            if idx < len(classes):
                classes[idx] = name
        self.classes = classes

        # Remap samples/targets by reading class name from path
        remapped_samples = []
        remapped_targets = []
        for path, _ in self.samples:
            cls_name = os.path.basename(os.path.dirname(path))
            if cls_name in self.class_to_idx:
                target = self.class_to_idx[cls_name]
                remapped_samples.append((path, target))
                remapped_targets.append(target)

        self.samples = remapped_samples
        # Torchvision aliases 'imgs' to 'samples' historically
        self.imgs = list(remapped_samples)
        self.targets = remapped_targets


def create_aligned_image_folders(
    train_root, val_root, transform_train=None, transform_val=None
):
    """Create train/val ImageFolders that share an aligned class_to_idx.

    - Computes the union of class folder names across train and val roots
    - Uses a single mapping (sorted by class name) for both datasets
    - Returns two ImageFolder instances with remapped samples/targets
    """

    def _classes_in(root):
        try:
            return sorted(
                [e.name for e in os.scandir(root) if e.is_dir()], key=lambda x: x
            )
        except FileNotFoundError:
            return []

    train_classes = set(_classes_in(train_root))
    val_classes = set(_classes_in(val_root))
    union_classes = sorted(train_classes.union(val_classes))
    class_to_idx = {name: idx for idx, name in enumerate(union_classes)}

    train_ds = FixedClassImageFolder(
        train_root, class_to_idx=class_to_idx, transform=transform_train
    )
    val_ds = FixedClassImageFolder(
        val_root, class_to_idx=class_to_idx, transform=transform_val
    )

    return train_ds, val_ds
