import geopandas as gpd
import pandas as pd
from torch import Tensor
from torchmetrics import Metric

from deepforest import utilities
from deepforest.evaluate import __evaluate_wrapper__


class BoxRecallPrecision(Metric):
    """DeepForest box recall and precision metric.

    This class is a thin wrapper around evaluate_boxes to compute box
    recall and precision during validation.
    """

    boxes: list[Tensor]
    labels: list[Tensor]
    scores: list[Tensor]
    image_names: list[str]

    def __init__(
        self, csv_file: str, label_dict: dict, iou_threshold: float = 0.4, **kwargs
    ) -> None:
        """
        Args:
            csv_file (str): Path to CSV file with ground truth boxes.
            label_dict (dict): Dictionary mapping string labels to numeric labels.
            iou_threshold (float, optional): IOU threshold for evaluation. Defaults to 0.4.
        """
        super().__init__(**kwargs)

        self.csv_file = csv_file
        self.iou_threshold = iou_threshold
        self.label_dict = label_dict
        self.add_state("boxes", default=[], dist_reduce_fx=None)
        self.add_state("labels", default=[], dist_reduce_fx=None)
        self.add_state("scores", default=[], dist_reduce_fx=None)
        # Don't use add_state for strings - they're not tensors
        self.image_names = []

    def update(self, preds: list[dict[str, Tensor]], image_names: list[str]) -> None:
        """Update the metric with new predictions.

        Args:
            preds (list[dict[str, Tensor]]): List of prediction dictionaries from the model.
            image_names (list): List of image names corresponding to the predictions.
        """
        for pred, image_name in zip(preds, image_names, strict=True):
            # Dicts can't be hashed, so store as lists
            self.boxes.append(pred["boxes"].detach().cpu())
            self.labels.append(pred["labels"].detach().cpu())
            self.scores.append(pred["scores"].detach().cpu())
            self.image_names.append(image_name)

    def compute(self) -> dict[str, float]:
        """Computes the recall/precision metrics."""
        ground_df = utilities.read_file(self.csv_file)
        numeric_to_label_dict = {v: k for k, v in self.label_dict.items()}
        ground_df["label"] = ground_df.label.apply(lambda x: self.label_dict[x])

        # Convert to dataframe for evaluation
        predictions = []
        for idx in range(len(self.boxes)):
            pred = {
                "boxes": self.boxes[idx],
                "labels": self.labels[idx],
                "scores": self.scores[idx],
            }
            df = utilities.format_geometry(pred)
            if df is not None:
                df["image_path"] = self.image_names[idx]
                predictions.append(df)

        if len(predictions) > 0:
            predictions = pd.concat(predictions)
        else:
            predictions = pd.DataFrame()

        results = __evaluate_wrapper__(
            predictions=predictions,
            ground_df=ground_df,
            iou_threshold=self.iou_threshold,
            numeric_to_label_dict=numeric_to_label_dict,
        )

        # Extract per-class recall/precision
        filtered_results = {}
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
        self.image_names = []
