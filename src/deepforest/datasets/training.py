"""Dataset model for object detection tasks."""

import math
import os
from abc import abstractmethod
from typing import Any

import cv2
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
        validate_coordinates=True,
    ):
        """
        Args:
            csv_file (str): Path to the CSV file containing annotations.
            root_dir (str): Directory containing all referenced images.
            transform (callable, optional): Function applied to each sample (e.g., image and target). Defaults to None.
            label_dict (dict[str, int]): Mapping from string labels in the CSV to integer class IDs (e.g., {"Tree": 0}).
            augmentations (str | list | dict, optional): Augmentation configuration.
            preload_images (bool): If True, preload all images into memory. Defaults to False.
            validate_coordinates (bool): If True, check that all annotation coordinates fall
                within image bounds before training. Defaults to True.
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
        if validate_coordinates:
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
        """Validate geometries in the annotation data.

        Must be overidden by child classes to implement task-specific
        checks (e.g., boxes vs points).

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
        for image_path, group in self.annotations.groupby("image_path"):
            img_path = os.path.join(self.root_dir, image_path)
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                errors.append(f"Failed to open image {img_path}: {e}")
                continue

            for _idx, row in group.iterrows():
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


class PointDataset(TrainingDataset):
    """Dataset for point detection tasks."""

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
        validate_coordinates=True,
        density_sigma=4.0,
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
        The map is count-normalized so that each channel sums to the number of points of that class.

        Args:
            csv_file (str): Path to the CSV file containing annotations.
            root_dir (str): Directory containing all referenced images.
            transform (callable, optional): Function applied to each sample (e.g., image and target). Defaults to None.
            label_dict (dict[str, int]): Mapping from string labels in the CSV to integer class IDs (e.g., {"Tree": 0}).
            augmentations (str | list | dict, optional): Augmentation configuration.
            preload_images (bool): If True, preload all images into memory. Defaults to False.
            validate_coordinates (bool): If True, check that all annotation coordinates fall
                within image bounds. Defaults to True.
            density_sigma (float): Standard deviation of the Gaussian kernel for density map generation. Defaults to 4.0.
            output (str): Output format, either "centroid" for point coordinates or "density" for Gaussian density maps. Defaults to "centroid".
        """
        super().__init__(
            csv_file=csv_file,
            root_dir=root_dir,
            transforms=transforms,
            augmentations=augmentations,
            label_dict=label_dict,
            preload_images=preload_images,
            validate_coordinates=validate_coordinates,
        )

        self.density_sigma = density_sigma

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

    def filter_points(self, points, labels, width, height) -> tuple:
        """Filter points to be within the image.

        Args:
            points (torch.Tensor): Points of shape (N, 2) in xy format.
            labels (torch.Tensor): Labels of shape (N,).
            width (int): Image width in pixels.
            height (int): Image height in pixels.

        Returns:
            tuple: A tuple of (filtered_points, filtered_labels)
        """
        valid_mask = (
            (points[:, 0] >= 0)
            & (points[:, 0] <= width)
            & (points[:, 1] >= 0)
            & (points[:, 1] <= height)
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
        """Convert points to a Gaussian density representation.

        Places a delta at each point location, applies a Gaussian blur with
        sigma=density_sigma, then count-normalizes each class channel so that
        channel.sum() == number of points of that class.

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

        # Apply Gaussian blur; kernel size chosen to cover +-3*sigma without clipping tails.
        sigma = self.density_sigma
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        density = torchvision.transforms.functional.gaussian_blur(
            density, kernel_size=kernel_size, sigma=sigma
        )

        # Count-normalize each class channel so channel.sum() == num_points_in_class.
        for cls_idx in range(density.shape[0]):
            s = density[cls_idx].sum()
            if s > 0:
                n_cls = (labels == cls_idx).sum().item()
                density[cls_idx] = density[cls_idx] * (n_cls / s)

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

        # Filter out-of-bounds points after augmentation accounting for
        # potential size changes due to augmentation.
        points, labels = self.filter_points(
            points,
            labels,
            width=min(image_tensor.shape[3], image.shape[2]),
            height=min(image_tensor.shape[2], image.shape[1]),
        )

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


class PolygonDataset(TrainingDataset):
    """Dataset for instance segmentation with polygon masks.

    Parallels :class:`BoxDataset` but additionally rasterizes each polygon
    geometry into a binary instance mask. Targets contain ``boxes``
    (the polygon bounds), ``labels`` and ``masks`` for use with Mask R-CNN.
    Labels are zero-indexed to match the box and point workflows.
    """

    _data_keys = [DataKey.IMAGE, DataKey.BBOX_XYXY, DataKey.MASK]

    def _validate_coordinates(self) -> None:
        """Validate that all polygons are valid and within the image.

        Raises:
            ValueError: If any polygon is invalid or exceeds image bounds.
        """
        errors = []
        for image_path, group in self.annotations.groupby("image_path"):
            img_path = os.path.join(self.root_dir, image_path)
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                errors.append(f"Failed to open image {img_path}: {e}")
                continue

            for _idx, row in group.iterrows():
                geom = row["geometry"]
                if geom is None or geom.is_empty:
                    errors.append(f"Empty polygon geometry for image {img_path}")
                    continue

                if geom.area == 0:
                    errors.append(f"Polygon has zero area for image {img_path}")
                if not geom.is_valid:
                    errors.append(f"Invalid polygon (self-intersecting) for {img_path}")

                xmin, ymin, xmax, ymax = geom.bounds
                oob_issues = []
                if xmin < 0:
                    oob_issues.append(f"xmin ({xmin}) < 0")
                if xmax > width:
                    oob_issues.append(f"xmax ({xmax}) > image width ({width})")
                if ymin < 0:
                    oob_issues.append(f"ymin ({ymin}) < 0")
                if ymax > height:
                    oob_issues.append(f"ymax ({ymax}) > image height ({height})")

                if oob_issues:
                    errors.append(
                        f"Polygon, ({xmin}, {ymin}, {xmax}, {ymax}) exceeds image "
                        f"dimensions, ({width}, {height}) for {img_path}. "
                        f"Issues: {', '.join(oob_issues)}."
                    )

        if errors:
            raise ValueError("\n".join(errors))

    def generate_mask(self, geom, width, height) -> np.typing.NDArray[np.uint8]:
        """Rasterize a Shapely polygon into a binary instance mask.

        Args:
            geom: A Shapely Polygon/MultiPolygon (or its WKT string).
            width: Mask width in pixels.
            height: Mask height in pixels.

        Returns:
            A ``(height, width)`` uint8 array with the polygon interior set.
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        if isinstance(geom, str):
            geom = shapely.wkt.loads(geom)

        if geom.geom_type == "Polygon":
            coords = np.array(geom.exterior.coords, dtype=np.int32)
            cv2.fillPoly(mask, [coords], color=1)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                if not poly.is_empty:
                    coords = np.array(poly.exterior.coords, dtype=np.int32)
                    cv2.fillPoly(mask, [coords], color=1)

        return mask

    def annotations_for_path(self, image_path, return_tensor=False) -> dict:
        """Construct target dictionary for a given image path.

        Args:
            image_path (str): Path to image, expected to be in dataframe
            return_tensor (bool): If true, convert fields from numpy to tensor

        Returns:
            target dictionary with boxes, labels and geometry entries
        """
        image_annotations = self.annotations[self.annotations.image_path == image_path]
        targets = {}

        # Polygon bounds give the enclosing box used by Mask R-CNN.
        targets["boxes"] = np.array(
            [
                x.bounds if hasattr(x, "bounds") else shapely.wkt.loads(x).bounds
                for x in image_annotations.geometry
            ]
        ).astype("float32")

        targets["labels"] = image_annotations.label.apply(
            lambda x: self.label_dict[x]
        ).values.astype(np.int64)

        # Keep geometry so __getitem__ can rasterize masks at image size.
        targets["geometry"] = image_annotations.geometry.values

        if return_tensor:
            targets["boxes"] = torch.from_numpy(targets["boxes"])
            targets["labels"] = torch.from_numpy(targets["labels"])

        return targets

    def __getitem__(self, idx) -> tuple:
        # Read image if not in memory
        if self.preload_images:
            image = self.image_dict[idx]
        else:
            image = self.load_image(idx)

        height, width = image.shape[:2]
        targets = self.annotations_for_path(self.image_names[idx])

        boxes = targets["boxes"]
        labels = targets["labels"]
        geometries = targets["geometry"]

        # If image has no annotations, add a dummy empty frame
        if boxes.size == 0 or np.sum(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros(0, dtype=np.int64)
            geometries = []

        # Rasterize each polygon into a binary instance mask
        if len(geometries) > 0:
            masks = np.stack(
                [self.generate_mask(geom, width, height) for geom in geometries]
            ).astype(np.uint8)
        else:
            masks = np.zeros((0, height, width), dtype=np.uint8)

        # Apply augmentations jointly to image, boxes and masks
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        boxes_tensor = torch.from_numpy(boxes).unsqueeze(0).float()
        masks_tensor = torch.from_numpy(masks).unsqueeze(0).float()
        augmented_image, augmented_boxes, augmented_masks = self.transform(
            image_tensor, boxes_tensor, masks_tensor
        )

        image = augmented_image.squeeze(0)
        boxes = augmented_boxes.squeeze(0)
        masks = augmented_masks.squeeze(0)
        labels = torch.from_numpy(labels.astype(np.int64))

        boxes, labels, masks = self.filter_masks(
            boxes,
            labels,
            masks,
            width=min(image_tensor.shape[3], image.shape[2]),
            height=min(image_tensor.shape[2], image.shape[1]),
        )

        # Edge case if all labels were augmented away, keep the image
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            masks = torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8)

        targets = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks.to(torch.uint8),
        }

        return image, targets, self.image_names[idx]

    def filter_masks(self, boxes, labels, masks, width, height, min_size=1) -> tuple:
        """Clamp boxes to image bounds and filter boxes, labels and masks
        together by minimum box dimension.

        Args:
            boxes (torch.Tensor): Bounding boxes of shape (N, 4) in xyxy format.
            labels (torch.Tensor): Labels of shape (N,).
            masks (torch.Tensor): Instance masks of shape (N, H, W).
            width (int): Image width in pixels.
            height (int): Image height in pixels.
            min_size (int): Minimum box width/height in pixels. Defaults to 1.

        Returns:
            tuple: (filtered_boxes, filtered_labels, filtered_masks)
        """
        if boxes.dim() != 2 or boxes.shape[0] == 0:
            return boxes, labels, masks

        boxes[:, 0] = torch.clamp(boxes[:, 0], min=0, max=width)
        boxes[:, 1] = torch.clamp(boxes[:, 1], min=0, max=height)
        boxes[:, 2] = torch.clamp(boxes[:, 2], min=0, max=width)
        boxes[:, 3] = torch.clamp(boxes[:, 3], min=0, max=height)

        box_w = boxes[:, 2] - boxes[:, 0]
        box_h = boxes[:, 3] - boxes[:, 1]
        valid = (box_w >= min_size) & (box_h >= min_size)

        masks = masks > 0.5
        if masks.dim() == 3:
            masks = masks[valid]

        return boxes[valid], labels[valid], masks


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
