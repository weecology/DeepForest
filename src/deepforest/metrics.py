import pandas as pd
import torch
from torch import Tensor
from torchmetrics import Metric

from deepforest import utilities
from deepforest.evaluate import compute_class_recall, match_predictions


class RecallPrecision(Metric):
    """DeepForest box recall and precision metric.

    This class is a thin wrapper around evaluate_boxes to compute box
    recall and precision during validation.
    """

    image_indices: list

    def __init__(
        self,
        task="box",
        iou_threshold: float = 0.4,
        distance_threshold: float = 10.0,
        label_dict: dict | None = None,
        **kwargs,
    ) -> None:
        """This metric performs DeepForest's recall and precision evaluation.

        Args:
            task (str, optional): The type of task to evaluate. One of
                ``"box"``, ``"polygon"``, or ``"keypoint"``. Defaults to ``"box"``.
            iou_threshold (float, optional): IoU threshold for box/polygon matching.
                Defaults to 0.4.
            distance_threshold (float, optional): Pixel distance threshold for
                keypoint matching. Defaults to 10.0.
            label_dict (dict | None): Mapping of class name to numeric ID. When
                provided and more than one class is present, per-class recall and
                precision are included in the ``compute()`` output.
        """
        super().__init__(**kwargs)

        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.task = task
        self.label_dict = label_dict or {}
        self.numeric_to_label_dict = {v: k for k, v in self.label_dict.items()}

        if task not in ("box", "point", "keypoint"):
            raise ValueError(f"Unsupported task: {task!r}. Use 'box' or 'keypoint'.")

        if self.task == "box":
            self.pred_key = "boxes"
            self.geom_type = "box"
        elif self.task == "point":
            self.pred_key = "points"
            self.geom_type = "point"

        self.add_state("precision", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("recall", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_images", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(
            "num_images_with_predictions", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state("num_empty_frames", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(
            "correct_empty_predictions", default=torch.tensor(0), dist_reduce_fx="sum"
        )

        # Accumulated per-image match results for class_recall (not a metric
        # state since string labels cannot be stored as tensors)
        self.results: list[pd.DataFrame] = []
        self._all_results: pd.DataFrame = pd.DataFrame()

    def update(
        self,
        preds: list[dict[str, Tensor]],
        targets: list[dict[str, Tensor]],
        image_names: list[str] | None = None,
    ) -> None:
        """Update the metric with a batch of predictions and targets.

        Args:
            preds: List of prediction dicts (keys: ``boxes``, ``labels``, ``scores``).
            targets: List of target dicts (keys: ``boxes``, ``labels``).
            image_names: Optional list of image path strings, one per image.
        """
        if image_names is None:
            image_names = ["unknown"] * len(preds)
        for pred, target, image_path in zip(preds, targets, image_names, strict=True):
            self._update_single(pred, target, image_path)

    def _update_single(
        self,
        pred: dict[str, Tensor],
        target: dict[str, Tensor],
        image_path: str = "unknown",
    ) -> None:
        """Update metric state for a single image."""

        self.num_images += 1
        n_pred = len(pred[self.pred_key])
        n_target = len(target[self.pred_key])

        # Early exit for prediction/target base cases.
        is_empty_frame = n_target == 0 or torch.all(target[self.pred_key] == 0)
        if is_empty_frame:
            self.num_empty_frames += 1
            if n_pred == 0:
                self.correct_empty_predictions += 1
            else:
                # Predictions in an empty frame are all FP: precision = 0
                self.num_images_with_predictions += 1
            return

        if n_pred == 0:
            # No predictions but ground truth exists: recall=0, precision undefined
            return

        self.num_images_with_predictions += 1

        # Note: format_geometry handles detach + CPU. IoU == 0 represents not matched.
        ground_df = utilities.format_geometry(
            target, scores=False, geom_type=self.geom_type
        )
        pred_df = utilities.format_geometry(pred, scores=True, geom_type=self.geom_type)
        ground_df["image_path"] = image_path
        pred_df["image_path"] = image_path
        result = match_predictions(
            predictions=pred_df,
            ground_df=ground_df,
            task=self.task,
        )
        result["image_path"] = image_path

        if self.task == "box" or self.task == "polygon":
            result["match"] = result.IoU > self.iou_threshold
        elif self.task == "point":
            result["match"] = result.distance < self.distance_threshold

        # Compute per-image precision + recall
        result["match"] = result["match"].fillna(False)
        true_positive = sum(result["match"])
        recall = true_positive / result.shape[0]
        precision = true_positive / n_pred

        self.recall += recall
        self.precision += precision
        self.results.append(result)

    def _sync_dist(self, dist_sync_fn=None, process_group=None) -> None:
        """Sync metric states and gather non-tensor results list across
        ranks."""
        super()._sync_dist(dist_sync_fn=dist_sync_fn, process_group=process_group)

        # Gather results dataframes
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size(group=process_group)
            torch.distributed.barrier(group=process_group)
            gathered = [None] * world_size
            torch.distributed.all_gather_object(
                gathered, self.results, group=process_group
            )
            self.results = [df for rank_results in gathered for df in rank_results]

    def compute(self) -> dict:
        """Computes the recall/precision metrics.

        DataFrames (match results and class recall) are stored as instance
        attributes ``_all_results`` and ``_class_recall`` for callers that
        need them. Only loggable scalar/tensor values are returned.
        Per-class recall and precision are included when more than one class
        is present in ``label_dict``.
        """

        # Map numeric label IDs to strings
        if self.results:
            self._all_results = pd.concat(self.results, ignore_index=True)
            # Free up memory once converted to single dataframe
            self.results = []
            for col in ["predicted_label", "true_label"]:
                if col in self._all_results.columns:
                    self._all_results[col] = self._all_results[col].map(
                        lambda x: self.numeric_to_label_dict.get(int(x), x)
                        if pd.notna(x)
                        else x
                    )
            # TODO Check why this fails for keypoint
            if self.task == "box" and len(self.label_dict) > 1:
                self._class_recall = compute_class_recall(
                    self._all_results[self._all_results["match"]]
                )
        else:
            self._all_results = pd.DataFrame()
            self._class_recall = None

        # Reduce precision and recall to per-image
        output = {
            f"{self.task}_precision": (
                self.precision.float() / self.num_images_with_predictions.float()
                if self.num_images_with_predictions > 0
                else torch.tensor(float("nan"))
            ),
            f"{self.task}_recall": self.recall.float() / self.num_images.float(),
        }

        # Only log class metrics if multi-class targets
        if len(self.label_dict) > 1 and self._class_recall is not None:
            for _, row in self._class_recall.iterrows():
                output[f"{row['label']}_Recall"] = row["recall"]
                output[f"{row['label']}_Precision"] = row["precision"]

        # Empty frame accuracy, undefined if no empty frames are present
        if self.num_empty_frames > 0:
            output["empty_frame_accuracy"] = (
                self.correct_empty_predictions.float() / self.num_empty_frames.float()
            )

        return output

    def get_results(self) -> pd.DataFrame:
        """Return the full per-box match results from the last ``compute()``
        call.

        Returns:
            DataFrame with columns ``truth_id``, ``prediction_id``,
            ``predicted_label``, ``true_label``, ``match``, ``IoU``,
            ``score``, ``geometry``, and ``image_path``. Empty if
            ``compute()`` has not been called.
        """
        return self._all_results

    def reset(self) -> None:
        """Reset metric state."""
        super().reset()
        self.results = []
