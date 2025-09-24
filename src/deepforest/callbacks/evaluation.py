"""Evaluation callback for prediction saving and evaluation during training."""

import gzip
import json
import os
import tempfile
import warnings
from glob import glob

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
        save_dir: str = None,
        every_n_epochs: int = 5,
        iou_threshold: float = 0.4,
        run_evaluation: bool = False,
        compress: bool = False,
    ) -> None:
        super().__init__()

        self.temp_dir_obj = None
        if not save_dir:
            self.temp_dir_obj = tempfile.TemporaryDirectory()
            save_dir = self.temp_dir_obj.name

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
        if self._should_skip(trainer):
            return

        # Create once to avoid races
        if trainer.is_global_zero:
            os.makedirs(self.save_dir, exist_ok=True)
        trainer.strategy.barrier()

        # Per-rank shard filename
        rank = trainer.global_rank
        csv_filename = f"predictions_epoch_{trainer.current_epoch + 1}_rank{rank}.csv"
        if self.compress:
            csv_filename += ".gz"
        csv_path = os.path.join(self.save_dir, csv_filename)

        self.csv_path = csv_path  # path to this rank's shard
        self.predictions_written = 0

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
        """Clean up at end of epoch.

        Handles DDP sync and merging of output shards.
        """
        if self._should_skip(trainer):
            return

        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None

        # All ranks finished writing
        trainer.strategy.barrier()

        # Merge on global rank 0
        if trainer.is_global_zero:
            epoch = trainer.current_epoch + 1
            pattern = os.path.join(self.save_dir, f"predictions_epoch_{epoch}_rank*.csv")
            if self.compress:
                pattern += ".gz"

            shard_paths = sorted(glob(pattern))
            merged_filename = f"predictions_epoch_{epoch}.csv"
            if self.compress:
                merged_filename += ".gz"
            merged_path = os.path.join(self.save_dir, merged_filename)

            # Concatenate shards
            dfs = []
            for p in shard_paths:
                if p.endswith(".gz"):
                    dfs.append(pd.read_csv(p, compression="gzip"))
                else:
                    dfs.append(pd.read_csv(p))
            if dfs:
                merged = pd.concat(dfs, ignore_index=True)
                if self.compress:
                    merged.to_csv(merged_path, index=False, compression="gzip")
                else:
                    merged.to_csv(merged_path, index=False)
                total_written = len(merged)
            else:
                total_written = 0
                merged_path = None

            # Save metadata
            metadata = {
                "epoch": epoch,
                "current_step": trainer.global_step,
                "predictions_count": total_written,
                "target_csv_file": getattr(pl_module.config.validation, "csv_file", None),
                "target_root_dir": getattr(pl_module.config.validation, "root_dir", None),
                "shards": shard_paths,
                "merged_predictions": merged_path,
                "world_size": trainer.world_size,
            }
            with open(
                os.path.join(self.save_dir, f"predictions_epoch_{epoch}_metadata.json"),
                "w",
            ) as f:
                json.dump(metadata, f, indent=2)

            # Optional: cleanup shards after merge
            for p in shard_paths:
                os.remove(p)

            # Run evaluation only if we have predictions and user requested it
            if self.run_evaluation and total_written > 0 and merged_path is not None:
                try:
                    pl_module.evaluate(
                        predictions=merged_path, csv_file=metadata["target_csv_file"]
                    )
                except Exception as e:
                    warnings.warn(f"Evaluation failed: {e}", stacklevel=2)
            elif self.run_evaluation:
                warnings.warn(
                    "No predictions written to disk, skipping evaluate.", stacklevel=2
                )

        # Ensure rank 0 finished before next stage
        trainer.strategy.barrier()

        if self.temp_dir_obj is not None and trainer.is_global_zero:
            self.temp_dir_obj.cleanup()
