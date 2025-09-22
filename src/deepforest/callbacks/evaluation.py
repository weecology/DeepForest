"""Evaluation callback for prediction saving and evaluation during training."""

import gzip
import json
import os
import warnings

import pandas as pd
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.core import LightningModule


class EvaluationCallback(Callback):
    """Accumulate validation predictions and save to disk during training.

    This callback accumulates predictions during validation and writes them
    incrementally to a CSV file. At the end of validation, it saves metadata
    and optionally runs evaluation. The saved predictions are in the format
    expected by DeepForest's evaluation functions.

    The saved files follow this naming convention:
    - Predictions: {save_dir}/predictions_epoch_{epoch}.csv (or .csv.gz if compressed)
    - Metadata: {save_dir}/predictions_epoch_{epoch}_metadata.json

    Args:
        save_dir (str): Directory to save prediction files. Will be created if it doesn't exist.
        every_n_epochs (int, optional): Run interval in epochs. Set to -1 to disable callback.
            Defaults to 5.
        iou_threshold (float, optional): IoU threshold for evaluation when run_evaluation=True.
            Defaults to 0.4.
        run_evaluation (bool, optional): Whether to run evaluate_boxes at epoch end and log metrics.
            Defaults to False.
        compress (bool, optional): Whether to compress CSV files using gzip. When True, saves as
            .csv.gz files for better storage efficiency. When False (default), saves as plain .csv
            Defaults to False.

    Attributes:
        save_dir (str): Directory where files are saved
        every_n_epochs (int): Epoch interval for running callback
        iou_threshold (float): IoU threshold used for evaluation
        run_evaluation (bool): Whether evaluation is run at epoch end
        predictions_written (int): Number of predictions written in current epoch

    Note:
        This callback should be used with `val_accuracy_interval = -1` in the model config
        to disable the built-in evaluation and avoid duplicate processing.
    """

    def __init__(
        self,
        save_dir: str,
        every_n_epochs: int = 5,
        iou_threshold: float = 0.4,
        run_evaluation: bool = False,
        compress: bool = False,
    ) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.every_n_epochs = every_n_epochs
        self.iou_threshold = iou_threshold
        self.run_evaluation = run_evaluation
        self.compress = compress
        self.predictions_written = 0

    def _should_skip(self, trainer: Trainer) -> bool:
        """Check if callback should be skipped for the current trainer
        state."""
        return (
            trainer.sanity_checking
            or trainer.fast_dev_run
            or self.every_n_epochs == -1
            or (trainer.current_epoch + 1) % self.every_n_epochs != 0
        )

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Initialize CSV file for writing predictions at validation start.

        Creates the save directory if it doesn't exist and opens a CSV
        file for incremental writing of predictions during validation.
        """
        if self._should_skip(trainer):
            return

        os.makedirs(self.save_dir, exist_ok=True)

        # Set file extension based on compression setting
        csv_filename = f"predictions_epoch_{trainer.current_epoch + 1}.csv"
        if self.compress:
            csv_filename += ".gz"

        csv_path = os.path.join(self.save_dir, csv_filename)

        # Open CSV file for writing (compressed or uncompressed)
        if self.compress:
            self.csv_file = gzip.open(csv_path, "wt", encoding="utf-8")
        else:
            self.csv_file = open(csv_path, "w")

        self.csv_path = csv_path
        self.predictions_written = 0

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Write predictions from current validation batch to CSV file."""
        if self._should_skip(trainer) or self.csv_file is None:
            return

        # Get predictions from this batch
        batch_preds = pl_module.last_preds

        for pred in batch_preds:
            if pred is not None and not pred.empty:
                pred.to_csv(
                    self.csv_file, index=False, header=(self.predictions_written == 0)
                )
                self.predictions_written += len(pred)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Close CSV file, save metadata, and optionally run evaluation."""
        if self._should_skip(trainer):
            return

        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None

            # Save metadata JSON
            metadata = {
                "epoch": trainer.current_epoch + 1,
                "current_step": trainer.global_step,
                "predictions_count": self.predictions_written,
                "target_csv_file": getattr(pl_module.config.validation, "csv_file", None),
                "target_root_dir": getattr(pl_module.config.validation, "root_dir", None),
            }

            metadata_path = os.path.join(
                self.save_dir,
                f"predictions_epoch_{trainer.current_epoch + 1}_metadata.json",
            )
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Optionally run evaluation
            if self.run_evaluation and self.predictions_written > 0:
                try:
                    # Load predictions
                    predictions = pd.read_csv(self.csv_path)

                    # Load ground truth
                    if metadata["target_csv_file"] and os.path.exists(
                        metadata["target_csv_file"]
                    ):
                        from deepforest.utilities import read_file

                        ground_truth = read_file(metadata["target_csv_file"])
                        pl_module.evaluate(
                            predictions=predictions, ground_truth=ground_truth
                        )

                except Exception as e:
                    warnings.warn(f"Evaluation failed: {e}", stacklevel=2)
