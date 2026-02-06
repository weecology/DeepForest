import warnings

import geopandas as gpd
import pandas as pd
import torch
from torch import Tensor
from torchmetrics import Metric

from deepforest import utilities
from deepforest.evaluate import __evaluate_wrapper__


class RecallPrecision(Metric):
    """DeepForest box recall and precision metric.

    This class is a thin wrapper around evaluate_boxes to compute box
    recall and precision during validation. In multi-GPU environments,
    each rank runs evaluation on the full (gathered) dataset.
    """

    boxes: list[Tensor]
    labels: list[Tensor]
    scores: list[Tensor]
    image_indices: list[Tensor]

    def __init__(
        self,
        csv_file: str,
        label_dict: dict,
        task="box",
        iou_threshold: float = 0.4,
        **kwargs,
    ) -> None:
        """This metric performs DeepForest's box recall and precision
        evaluation.

        Args:
            csv_file (str): Path to CSV file with ground truth boxes.
            label_dict (dict): Dictionary mapping string labels to numeric labels.
            iou_threshold (float, optional): IOU threshold for evaluation. Defaults to 0.4.
        """
        super().__init__(**kwargs)

        self.csv_file = csv_file
        self.iou_threshold = iou_threshold
        self.label_dict = label_dict

        if task != "box":
            raise NotImplementedError("Only 'box' task is currently supported.")

        # Create image path index mappings. This is necessary
        # as we can't sync strings across multiple GPUs by default.
        ground_df = utilities.read_file(csv_file)
        unique_paths = sorted(ground_df["image_path"].unique())
        self.path_to_index = {path: idx for idx, path in enumerate(unique_paths)}
        self.index_to_path = {idx: path for path, idx in self.path_to_index.items()}

        self.add_state("boxes", default=[], dist_reduce_fx=None)
        self.add_state("labels", default=[], dist_reduce_fx=None)
        self.add_state("scores", default=[], dist_reduce_fx=None)
        self.add_state("image_indices", default=[], dist_reduce_fx=None)

    def update(self, preds: list[dict[str, Tensor]], image_names: list[str]) -> None:
        """Update the metric with new predictions.

        Args:
            preds (list[dict[str, Tensor]]): List of prediction dictionaries from the model.
            image_names (list): List of image names corresponding to the predictions.
        """
        for pred, image_name in zip(preds, image_names, strict=True):
            # Look up image index; skip if not in ground truth
            if image_name not in self.path_to_index:
                warnings.warn(
                    f"Image '{image_name}' not found in ground truth CSV. Skipping.",
                    stacklevel=2,
                )
                continue

            image_idx = self.path_to_index[image_name]
            num_boxes = len(pred["boxes"])

            # Store predictions and image indices as tensors
            self.boxes.append(pred["boxes"].detach())
            self.labels.append(pred["labels"].detach())
            self.scores.append(pred["scores"].detach())
            # Create a 1D tensor with one index per box
            self.image_indices.append(
                torch.full((num_boxes,), image_idx, dtype=torch.long).to(
                    pred["boxes"].device
                )
            )

    def compute(self) -> dict[str, float]:
        """Computes the recall/precision metrics."""

        ground_df = utilities.read_file(self.csv_file)
        numeric_to_label_dict = {v: k for k, v in self.label_dict.items()}
        ground_df["label"] = ground_df.label.apply(lambda x: self.label_dict[x])

        predictions = pd.DataFrame()
        if self.boxes:
            combined = {
                "boxes": torch.cat(self.boxes),
                "labels": torch.cat(self.labels),
                "scores": torch.cat(self.scores),
            }
            predictions = utilities.format_geometry(combined)
            if predictions is None:
                predictions = pd.DataFrame()  # Reset to empty DataFrame
            else:
                # Expand image names to one entry per box
                predictions["image_path"] = [
                    self.index_to_path[int(idx.item())]
                    for idx in torch.cat(self.image_indices)
                ]

        # Filter ground_df to only include images that were actually predicted
        if not predictions.empty:
            predicted_images = predictions["image_path"].unique()
            ground_df = ground_df[ground_df["image_path"].isin(predicted_images)]

        results = __evaluate_wrapper__(
            predictions=predictions,
            ground_df=ground_df,
            iou_threshold=self.iou_threshold,
            numeric_to_label_dict=numeric_to_label_dict,
        )

        filtered_results = {}

        # Extract per-class recall/precision for multi class prediction only.
        if len(self.label_dict) > 1:
            if "class_recall" in results and results["class_recall"] is not None:
                for _, row in results["class_recall"].iterrows():
                    filtered_results[
                        "{}_Recall".format(numeric_to_label_dict[row["label"]])
                    ] = row["recall"]
                    filtered_results[
                        "{}_Precision".format(numeric_to_label_dict[row["label"]])
                    ] = row["precision"]

        # Filter out values that cannot be logged
        for key, value in results.items():
            if isinstance(value, (pd.DataFrame, gpd.GeoDataFrame)):
                pass
            elif value is None:
                pass
            else:
                filtered_results[key] = value

        return filtered_results

    def reset(self) -> None:
        """Reset metric state."""
        super().reset()
