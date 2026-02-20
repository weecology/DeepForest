"""Dataset model for object detection tasks."""

import math
import os
from abc import abstractmethod
from typing import Any

import kornia.augmentation as K
import numpy as np
import shapely
import torch
import torchvision
from kornia.constants import DataKey
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from deepforest import utilities
from deepforest.augmentations import get_transform


class TrainingDataset(Dataset):
    _data_keys = [DataKey.IMAGE, DataKey.BBOX_XYXY]

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
        """
        self.annotations = utilities.read_file(csv_file, root_dir=root_dir)
        self.root_dir = root_dir

        # Initialize label_dict with default if None
        if label_dict is None:
            label_dict = {"Tree": 0}
        self.label_dict = label_dict

        if transforms is None:
            self.transform = get_transform(
                augmentations=augmentations, data_keys=self._data_keys
            )
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

    def _validate_labels(self) -> None:
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

    @abstractmethod
    def _validate_coordinates(self) -> None:
        """Validate geometries in the annotation data. Must be overidden by
        child classes to implement task-specific checks (e.g., boxes vs
        points).

        Should raise ValueError with details if any invalid geometries
        are found.
        """

    def __len__(self) -> int:
        """Dataset length is the number of unique images."""
        return len(self.image_names)

    def collate_fn(self, batch) -> tuple:
        """Collate function for DataLoader."""
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        image_names = [item[2] for item in batch]

        return images, targets, image_names

    def load_image(self, idx) -> np.typing.NDArray[np.float32]:
        """Load image from disk and convert to float32 numpy array in [0,
        1]."""
        img_name = os.path.join(self.root_dir, self.image_names[idx])
        image = np.array(Image.open(img_name).convert("RGB")) / 255
        image = image.astype("float32")
        return image

    @abstractmethod
    def annotations_for_path(self, image_path, return_tensor=False) -> Any:
        """Construct target dictionary for a given image path, optionally
        convert to tensor."""

    @abstractmethod
    def __getitem__(self, index) -> tuple:
        """Return a single item from the dataset."""
        pass


class BoxDataset(TrainingDataset):
    """Dataset for object detection with bounding boxes."""

    def _validate_coordinates(self) -> None:
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

    def filter_boxes(self, boxes, labels, image_shape, min_size=1) -> tuple:
        """Clamp boxes to image bounds and filter by minimum dimension.

        Args:
            boxes (torch.Tensor): Bounding boxes of shape (N, 4) in xyxy format.
            labels (torch.Tensor): Labels of shape (N,).
            image_shape (tuple): Image shape as (C, H, W).
            min_size (int): Minimum box width/height in pixels. Defaults to 1.

        Returns:
            tuple: A tuple of (filtered_boxes, filtered_labels)
        """
        _, H, W = image_shape

        # Clamp boxes to image bounds
        boxes[:, 0] = torch.clamp(boxes[:, 0], min=0, max=W)  # x1
        boxes[:, 1] = torch.clamp(boxes[:, 1], min=0, max=H)  # y1
        boxes[:, 2] = torch.clamp(boxes[:, 2], min=0, max=W)  # x2
        boxes[:, 3] = torch.clamp(boxes[:, 3], min=0, max=H)  # y2

        # Filter boxes with minimum size
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        valid_mask = (width >= min_size) & (height >= min_size)

        return boxes[valid_mask], labels[valid_mask]

    def annotations_for_path(self, image_path, return_tensor=False) -> dict:
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

    def __getitem__(self, idx) -> tuple:
        # Read image if not in memory
        if self.preload_images:
            image = self.image_dict[idx]
        else:
            image = self.load_image(idx)

        targets = self.annotations_for_path(self.image_names[idx])

        # If image has no annotations, add a dummy
        if np.sum(targets["boxes"]) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros(0, dtype=np.int64)
            targets = {"boxes": boxes, "labels": labels}

        # Apply augmentations
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        boxes_tensor = torch.from_numpy(targets["boxes"]).unsqueeze(0).float()
        augmented_image, augmented_boxes = self.transform(image_tensor, boxes_tensor)

        # Convert boxes to tensor
        image = augmented_image.squeeze(0)
        boxes = augmented_boxes.squeeze(0)
        labels = torch.from_numpy(targets["labels"].astype(np.int64))

        # Filter invalid boxes after augmentation
        boxes, labels = self.filter_boxes(boxes, labels, image.shape)

        # Edge case if all labels were augmented away, keep the image
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)

        targets = {"boxes": boxes, "labels": labels}

        return image, targets, self.image_names[idx]


class KeypointDataset(TrainingDataset):
    """Dataset for keypoint detection tasks."""

    _data_keys = [DataKey.IMAGE, DataKey.KEYPOINTS]

    def __init__(
        self,
        csv_file,
        root_dir,
        *,
        transforms=None,
        augmentations=None,
        label_dict=None,
        preload_images=False,
        gaussian_radius=2,
        output="centroid",
    ):
        """This dataset class returns keypoint annotations in one of two common
        formats.

        If the output parameter is set to "centroid", each target is a dictionary with "points"
        and "labels" entries. The "points" entry is a tensor of shape (N, 2) containing the xy
        coordinates of each point, and the "labels" entry is a tensor of shape (N,)
        containing the class label for each point.

        If the output parameter is set to "density", each target is a dictionary with only a "labels" entry.
        In this case, "labels" is a tensor of shape (num_classes, H, W) representing a class-wise density map.
        The size of the Gaussian at each point is determined by the gaussian_radius parameter.

        Args:
            csv_file (str): Path to the CSV file containing annotations.
            root_dir (str): Directory containing all referenced images.
            transform (callable, optional): Function applied to each sample (e.g., image and target). Defaults to None.
            label_dict (dict[str, int]): Mapping from string labels in the CSV to integer class IDs (e.g., {"Tree": 0}).
            augmentations (str | list | dict, optional): Augmentation configuration.
            preload_images (bool): If True, preload all images into memory. Defaults to False.
            gaussian_radius (int): Radius of Gaussian kernel for density map generation. Defaults to 2.
            output (str): Output format, either "centroid" for point coordinates or "density" for Gaussian density maps. Defaults to "centroid".
        """
        super().__init__(
            csv_file=csv_file,
            root_dir=root_dir,
            transforms=transforms,
            augmentations=augmentations,
            label_dict=label_dict,
            preload_images=preload_images,
        )

        self.gaussian_radius = gaussian_radius

        if output not in ["centroid", "density"]:
            raise ValueError(
                f"Invalid output type: {output}. Supported options are 'centroid' and 'density'."
            )
        self.output = output

    def _validate_coordinates(self) -> None:
        """Validate that all points occur within the image.

        Raises:
            ValueError: If any point occurs outside the image
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

            # Extract point coordinates (use centroid so boxes/polygons also work)
            centroid = row["geometry"].centroid
            x, y = centroid.x, centroid.y

            # All coordinates equal to zero is how we code empty frames.
            if x == 0 and y == 0:
                continue

            # Check if point is valid
            oob_issues = []
            if x < 0:
                oob_issues.append(f"x ({x}) < 0")
            if x > width:
                oob_issues.append(f"x ({x}) > image width ({width})")
            if y < 0:
                oob_issues.append(f"y ({y}) < 0")
            if y > height:
                oob_issues.append(f"y ({y}) > image height ({height})")

            if oob_issues:
                errors.append(
                    f"Point, ({x}, {y}) exceeds image dimensions, ({width}, {height}). Issues: {', '.join(oob_issues)}."
                )

        if errors:
            raise ValueError("\n".join(errors))

    def filter_points(self, points, labels, image_shape) -> tuple:
        """Filter points to be within the image.

        Args:
            points (torch.Tensor): Points of shape (N, 2) in xy format.
            labels (torch.Tensor): Labels of shape (N,).
            image_shape (tuple): Image shape as (C, H, W).

        Returns:
            tuple: A tuple of (filtered_points, filtered_labels)
        """
        _, H, W = image_shape

        # Filter out of bounds
        valid_mask = (
            (points[:, 0] >= 0)
            & (points[:, 0] <= W)
            & (points[:, 1] >= 0)
            & (points[:, 1] <= H)
        )

        return points[valid_mask], labels[valid_mask]

    def annotations_for_path(self, image_path, return_tensor=False) -> dict:
        """Construct target dictionary for a given image path, optionally
        convert to tensor.

        Args:
            image_path (str): Path to image, expected to be in dataframe
            return_tensor (bool): If true, convert fields from numpy to tensor

        Returns:
            target dictionary with points and labels entries
        """
        image_annotations = self.annotations[self.annotations.image_path == image_path]
        targets = {}

        if "geometry" in image_annotations.columns:
            # Handle both shapely geometry objects and WKT strings
            targets["points"] = np.array(
                [
                    x.centroid.coords[0]
                    if hasattr(x, "centroid")
                    else shapely.wkt.loads(x).centroid.coords[0]
                    for x in image_annotations.geometry
                ]
            ).astype("float32")
        else:
            targets["points"] = image_annotations[["x", "y"]].values.astype("float32")

        # Labels need to be encoded
        targets["labels"] = image_annotations.label.apply(
            lambda x: self.label_dict[x]
        ).values.astype(np.int64)

        if return_tensor:
            for k, v in targets.items():
                targets[k] = torch.from_numpy(v)

        return targets

    def gaussian_density(self, points, labels, shape) -> torch.Tensor:
        """Convert points to a Gaussian density representation. The radius is
        set by the dataset attribute gaussian_radius. The map is produced by
        placing a 1 at each point location and applying a Gaussian blur. The
        resulting density map is normalized to have a maximum value of 1.

        Args:
            points (torch.Tensor): Points of shape (N, 2) in xy format.
            labels (torch.Tensor): Labels of shape (N,).
            shape (tuple): Image shape as (C, H, W).

        Returns:
            torch.Tensor: Density map of shape (num_classes, H, W)
        """
        if len(shape) == 3:
            _, H, W = shape
        elif len(shape) == 2:
            H, W = shape
        else:
            raise ValueError(
                f"image_shape must be length 2 (H, W) or 3 (C, H, W), got {shape}."
            )

        # torchvision gaussian_blur expects (C, H, W)
        num_classes = len(self.label_dict)
        density = torch.zeros((num_classes, H, W), dtype=torch.float32)

        if len(points) == 0:
            return density

        # Place a delta function for each point.
        for point, label in zip(points, labels, strict=True):
            r_x, r_y = round(point[0].item()), round(point[1].item())
            x, y = int(r_x), int(r_y)
            class_index = int(label.item())
            if 0 <= x < W and 0 <= y < H:
                density[class_index, y, x] = 1.0

        # Apply Gaussian blur, the kernel size is chosen to be
        # large enough to capture the distribution and not clip
        # the tails.
        sigma = self.gaussian_radius
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        density = torchvision.transforms.functional.gaussian_blur(
            density, kernel_size=kernel_size, sigma=sigma
        )

        # Normalize by the maximum value to handle overlapping points
        max_val = torch.max(density)
        if max_val > 0:
            density = density / max_val

        # Clamp to 1.0
        density = torch.clamp(density, max=1.0)

        return density

    def __getitem__(self, idx) -> tuple:
        """Returns a transformed data sample from the dataset.

        Returns:
            image (torch.Tensor): Image tensor of shape (C, H, W).
            targets (dict): Dictionary containing either:
                - 'points': Tensor of shape (N, 2) with point coordinates in xy format.
                - 'labels': Either a tensor of shape (N,) with class labels for 'centroid' output,
                            or a tensor of shape (num_classes, H, W) with density maps for 'density' output.
            image_name (str): The name of the image corresponding to the returned data.
        """
        # Read image if not in memory
        if self.preload_images:
            image = self.image_dict[idx]
        else:
            image = self.load_image(idx)

        targets = self.annotations_for_path(self.image_names[idx])

        # Dummy annotations for empty image
        if np.sum(targets["points"]) == 0:
            targets = {
                "points": np.zeros((0, 2), dtype=np.float32),
                "labels": np.zeros(0, dtype=np.int64),
            }

        # Apply augmentations
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        points_tensor = torch.from_numpy(targets["points"]).unsqueeze(0).float()
        augmented_image, augmented_points = self.transform(image_tensor, points_tensor)

        # Convert to tensor
        image = augmented_image.squeeze(0)
        points = augmented_points.squeeze(0)
        labels = torch.from_numpy(targets["labels"].astype(np.int64))

        # Filter out-of-bounds points after augmentation
        points, labels = self.filter_points(points, labels, image.shape)

        # Edge case if all labels were augmented away, keep the image
        if len(points) == 0:
            points = torch.zeros((0, 2), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)

        if self.output == "density":
            # Mask is NHW for N classes.
            targets = {"labels": self.gaussian_density(points, labels, image.shape[1:])}
        elif self.output == "centroid":
            targets = {"points": points, "labels": labels}
        else:
            raise ValueError(
                f"Invalid output type: {self.output}. Supported options are 'centroid' and 'density'."
            )

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
