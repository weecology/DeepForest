"""DeepForest callback for logging images during training.

Callbacks must implement on_epoch_begin, on_epoch_end, on_fit_end,
on_fit_begin methods and inject model and epoch kwargs.
"""

import json
import os
import random
import warnings
from pathlib import Path
from typing import TextIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import supervision as sv
import torch
from PIL import Image
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.core import LightningModule

from deepforest import evaluate, utilities, visualize
from deepforest.datasets.training import BoxDataset


class ImagesCallback(Callback):
    """Log evaluation images during training.

    Args:
        save_dir: Directory to save predicted images
        n: Number of images to process
        every_n_epochs: Run interval in epochs
        select_random: Whether to select random images
        color: Bounding box color as BGR tuple
        thickness: Border line thickness in pixels
    """

    def __init__(
        self,
        save_dir,
        sample_batches=2,
        images_per_batch=1,
        dataset_samples=5,
        every_n_epochs=5,
        select_random=False,
        color=None,
        thickness=2,
    ):
        self.savedir = save_dir
        self.sample_batches = sample_batches
        self.dataset_samples = dataset_samples
        self.color = color
        self.thickness = thickness
        self.select_random = select_random
        self.every_n_epochs = every_n_epochs
        self.num_val_batches = 0
        self.images_per_batch = images_per_batch

    def on_train_start(self, trainer, pl_module):
        """Log sample images from training and validation datasets at training
        start."""

        if trainer.fast_dev_run:
            return

        self.trainer = trainer
        self.pl_module = pl_module

        # Training samples
        pl_module.print("Logging training dataset samples.")
        train_ds = trainer.train_dataloader.dataset
        self._log_dataset_sample(train_ds, split="train")

        # Validation samples
        if trainer.val_dataloaders:
            pl_module.print("Logging validation dataset samples.")
            val_ds = trainer.val_dataloaders.dataset
            self._log_dataset_sample(val_ds, split="validation")
            self.num_val_batches = len(trainer.val_dataloaders)

    def on_validation_start(self, trainer, pl_module):
        """Pick batch indices for plotting, or skip."""
        self.batch_indices = set()

        if trainer.sanity_checking or trainer.fast_dev_run:
            return

        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            indices = list(range(self.num_val_batches))

            if self.select_random:
                random.shuffle(indices)

            self.batch_indices = set(indices[: self.sample_batches])

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Determine whether to sample predictions from this batch."""
        # NB: Dataloader idx is the i'th dataloader
        if batch_idx in self.batch_indices:
            # Last predictions (validation_step)
            _, batch_targets, image_names = batch
            batch_preds = pl_module.last_preds

            # Sample at most self.images_per_batch
            for idx in list(range(min(len(batch_preds), self.images_per_batch))):
                targets = utilities.format_geometry(batch_targets[idx], scores=False)
                preds = batch_preds[idx]
                image_name = image_names[idx]

                if preds.image_path.unique()[0] != image_name:
                    warnings.warn(
                        "Image names and predictions are out of sync, skipping sample.",
                        stacklevel=2,
                    )
                else:
                    self._log_prediction_sample(
                        trainer, pl_module, preds, targets, image_name
                    )

    def _log_prediction_sample(self, trainer, pl_module, preds, targets, image_name):
        dataset = trainer.val_dataloaders.dataset
        """Log one sample."""
        # Add root_dir to the dataframe
        if "root_dir" not in preds.columns:
            preds["root_dir"] = dataset.root_dir

        # Ensure color is correctly assigned
        if self.color is None:
            num_classes = len(preds["label"].unique())
            results_color = sv.ColorPalette.from_matplotlib("viridis", num_classes)
        else:
            results_color = self.color

        out_dir = os.path.join(self.savedir, "predictions")
        os.makedirs(out_dir, exist_ok=True)

        basename = Path(image_name).stem + f"_{trainer.global_step}"
        fig = visualize.plot_results(
            basename=basename,
            results=preds,
            ground_truth=targets,
            savedir=out_dir,
            results_color=results_color,
            thickness=self.thickness,
            show=False,
        )
        plt.close(fig)

        # Pred metadata, if supported.
        stats = (
            preds["score"]
            .agg(
                mean_confidence="mean",
                max_confidence="max",
                min_confidence="min",
                std_confidence="std",
            )
            .to_dict()
        )

        metadata = {"pred_count": len(preds), "gt_count": len(targets)}
        metadata.update(stats)

        with open(os.path.join(out_dir, basename + ".json"), "w") as fp:
            json.dump(metadata, fp, indent=1)

        self._log_to_all(
            image=os.path.join(out_dir, basename + ".png"),
            trainer=trainer,
            tag="prediction sample",
            metadata=metadata,
        )

    def _log_dataset_sample(self, dataset: BoxDataset, split: str):
        """Log random samples from a DeepForest BoxDataset."""

        if self.dataset_samples == 0:
            return

        out_dir = os.path.join(self.savedir, split + "_sample")
        os.makedirs(out_dir, exist_ok=True)
        n_samples = min(self.dataset_samples, len(dataset))
        sample_indices = torch.randperm(len(dataset))[:n_samples]

        sample_data = [dataset[idx] for idx in sample_indices]
        sample_images = [data[0] for data in sample_data]
        sample_targets = [data[1] for data in sample_data]
        sample_paths = [data[2] for data in sample_data]

        for image, target, path in zip(
            sample_images, sample_targets, sample_paths, strict=False
        ):
            image_annotations = target.copy()
            image_annotations = utilities.format_geometry(image_annotations, scores=False)
            image_annotations.root_dir = dataset.root_dir
            image_annotations["image_path"] = path

            # Plot transformed image
            basename = Path(path).stem
            image = (255 * image.cpu().numpy().transpose((1, 2, 0))).astype(np.uint8)
            fig = visualize.plot_annotations(
                image=image,
                annotations=image_annotations,
                savedir=out_dir,
                basename=basename,
                thickness=self.thickness,
                show=False,
            )
            plt.close(fig)

            self._log_to_all(
                image=os.path.join(out_dir, basename + ".png"),
                trainer=self.trainer,
                tag=f"{split} dataset sample",
            )

    def _log_to_all(self, image: str, trainer, tag, metadata: dict | None = None):
        """Log to all connected loggers.

        Since Comet will pickup image logs to Tensorboard by default, we
        add a check to log images preferentially to Tensorboard if both
        are enabled.
        """
        try:
            img = np.array(Image.open(image).convert("RGB"))

            loggers = [lg for lg in trainer.loggers if hasattr(lg, "experiment")]

            tb = next((lg for lg in loggers if hasattr(lg.experiment, "add_image")), None)
            if tb is not None:
                tb.experiment.add_image(
                    tag=f"{tag}/{os.path.basename(image)}",
                    img_tensor=img,
                    global_step=trainer.global_step,
                    dataformats="HWC",
                )
                return

            comet = next(
                (lg for lg in loggers if hasattr(lg.experiment, "log_image")),
                None,
            )
            if comet is not None:
                meta = {
                    "image_name": os.path.basename(image),
                    "context": tag,
                    "step": trainer.global_step,
                }

                if metadata:
                    meta.update(metadata)

                comet.experiment.log_image(
                    img,
                    name=tag,
                    step=trainer.global_step,
                    metadata=meta,
                )

        except Exception as e:
            warnings.warn(f"Tried to log {image} exception raised: {e}", stacklevel=2)


class images_callback(ImagesCallback):
    def __init__(self, savedir, **kwargs):
        warnings.warn(
            "Please use ImagesCallback instead.", DeprecationWarning, stacklevel=2
        )
        super().__init__(save_dir=savedir, **kwargs)


class EvaluationCallback(Callback):
    """Accumulate validation predictions and save to disk during training.

    This callback accumulates predictions during validation and writes them
    incrementally to a CSV file. At the end of validation, it saves metadata
    and optionally runs evaluation. The saved predictions are in the format
    expected by DeepForest's evaluation functions.

    The callback provides two modes:
    1. Save mode (default): Saves predictions to CSV with metadata JSON
    2. Evaluation mode: Additionally runs evaluate_boxes() and logs metrics

    The saved files follow this naming convention:
    - Predictions: {save_dir}/predictions_epoch_{epoch}.csv
    - Metadata: {save_dir}/predictions_epoch_{epoch}_metadata.json

    Args:
        save_dir (str): Directory to save prediction files. Will be created if it doesn't exist.
        every_n_epochs (int, optional): Run interval in epochs. Set to -1 to disable callback.
            Defaults to 5.
        iou_threshold (float, optional): IoU threshold for evaluation when run_evaluation=True.
            Defaults to 0.4.
        run_evaluation (bool, optional): Whether to run evaluate_boxes at epoch end and log metrics.
            Defaults to False.

    Attributes:
        save_dir (str): Directory where files are saved
        every_n_epochs (int): Epoch interval for running callback
        iou_threshold (float): IoU threshold used for evaluation
        run_evaluation (bool): Whether evaluation is run at epoch end
        csv_file (Optional[TextIO]): Open file handle for CSV writing
        csv_path (Optional[str]): Path to current CSV file being written
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
    ) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.every_n_epochs = every_n_epochs
        self.iou_threshold = iou_threshold
        self.run_evaluation = run_evaluation
        self.csv_file: TextIO | None = None
        self.csv_path: str | None = None
        self.predictions_written = 0

    def _should_skip(self, trainer: Trainer) -> bool:
        """Check if callback should be skipped for current conditions.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance

        Returns:
            bool: True if callback should skip execution, False otherwise

        Note:
            Callback is skipped during sanity checking, fast dev runs,
            when disabled (every_n_epochs=-1), or when not on the target epoch.
        """
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

        Creates the save directory if it doesn't exist and opens a CSV file
        for incremental writing of predictions during validation.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance
            pl_module (LightningModule): The Lightning module being trained

        Note:
            The CSV header will be written when the first prediction is encountered.
            File handle is stored in self.csv_file for batch-by-batch writing.
        """
        if self._should_skip(trainer):
            return

        os.makedirs(self.save_dir, exist_ok=True)
        csv_path = os.path.join(
            self.save_dir, f"predictions_epoch_{trainer.current_epoch + 1}.csv"
        )

        # Open CSV file for writing
        self.csv_file = open(csv_path, "w")
        self.csv_path = csv_path
        self.predictions_written = 0

        # We'll write the header when we get the first prediction

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Write predictions from current validation batch to CSV file.

        Extracts predictions from pl_module.last_preds and writes them incrementally
        to the open CSV file. Only evaluation-compatible columns are saved to ensure
        compatibility with DeepForest evaluation functions.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance
            pl_module (LightningModule): The Lightning module being trained
            outputs: Output of validation_step (unused)
            batch: Current validation batch (unused)
            batch_idx (int): Index of current batch (unused)
            dataloader_idx (int, optional): Index of dataloader. Defaults to 0.

        Note:
            Predictions are filtered to include only columns compatible with evaluation:
            ['xmin', 'ymin', 'xmax', 'ymax', 'label', 'score', 'image_path']
        """
        if self._should_skip(trainer) or self.csv_file is None:
            return

        # Get predictions from this batch
        batch_preds = pl_module.last_preds

        for pred in batch_preds:
            if pred is not None and not pred.empty:
                # Remove geometry column for CSV compatibility - keep only evaluation-compatible columns
                eval_columns = [
                    "xmin",
                    "ymin",
                    "xmax",
                    "ymax",
                    "label",
                    "score",
                    "image_path",
                ]
                pred_for_csv = pred[eval_columns].copy()

                # Write header on first prediction
                if self.predictions_written == 0:
                    pred_for_csv.to_csv(self.csv_file, index=False)
                else:
                    pred_for_csv.to_csv(self.csv_file, index=False, header=False)
                self.predictions_written += len(pred)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Close CSV file, save metadata, and optionally run evaluation.

        Finalizes the CSV file writing process by closing the file handle and
        creating a metadata JSON file with information about the saved predictions.
        If run_evaluation=True, loads the saved predictions and runs evaluation
        against the ground truth data.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance
            pl_module (LightningModule): The Lightning module being trained

        Note:
            Metadata JSON includes epoch number, prediction count, IoU threshold,
            and paths to target validation data. When evaluation is run, metrics
            are logged using the same naming convention as the built-in evaluation.
        """
        if self._should_skip(trainer):
            return

        if self.csv_file is not None:
            # Close CSV file
            self.csv_file.close()
            self.csv_file = None

            # Save metadata JSON
            metadata = {
                "epoch": trainer.current_epoch + 1,
                "predictions_count": self.predictions_written,
                "iou_threshold": self.iou_threshold,
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

                        # Run evaluation
                        results = evaluate.evaluate_boxes(
                            predictions=predictions,
                            ground_df=ground_truth,
                            iou_threshold=self.iou_threshold,
                        )

                        # Log results
                        pl_module.log_dict(
                            {
                                "box_recall": results["box_recall"],
                                "box_precision": results["box_precision"],
                            }
                        )

                        if results["class_recall"] is not None:
                            # Log per-class recall and precision like main.py does
                            for _, row in results["class_recall"].iterrows():
                                pl_module.log(
                                    f"{pl_module.numeric_to_label_dict[row['label']]}_Recall",
                                    row["recall"],
                                )
                                pl_module.log(
                                    f"{pl_module.numeric_to_label_dict[row['label']]}_Precision",
                                    row["precision"],
                                )

                except Exception as e:
                    warnings.warn(f"Evaluation failed: {e}", stacklevel=2)
