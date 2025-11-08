"""DeepForest callback for logging images during training.

Callbacks must implement on_epoch_begin, on_epoch_end, on_fit_end,
on_fit_begin methods and inject model and epoch kwargs.
"""

import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import supervision as sv
import torch
from PIL import Image
from pytorch_lightning import Callback

from deepforest import utilities, visualize
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
        prediction_samples=2,
        dataset_samples=5,
        every_n_epochs=5,
        select_random=False,
        color=None,
        thickness=2,
    ):
        self.savedir = save_dir
        self.prediction_samples = prediction_samples
        self.dataset_samples = dataset_samples
        self.color = color
        self.thickness = thickness
        self.select_random = select_random
        self.every_n_epochs = every_n_epochs

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

    def on_validation_end(self, trainer, pl_module):
        """Run callback at validation end."""
        if trainer.sanity_checking or trainer.fast_dev_run:
            return

        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            pl_module.print("Logging prediction samples")
            self._log_last_predictions(trainer, pl_module)

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

    def _log_last_predictions(self, trainer, pl_module):
        """Log sample of predictions + targets from last validation."""
        if self.prediction_samples == 0:
            return

        if len(pl_module.predictions) > 0:
            df = pd.concat(pl_module.predictions)
        else:
            df = pd.DataFrame()

        out_dir = os.path.join(self.savedir, "predictions")
        os.makedirs(out_dir, exist_ok=True)

        dataset = trainer.val_dataloaders.dataset

        # Add root_dir to the dataframe
        if "root_dir" not in df.columns:
            df["root_dir"] = dataset.root_dir

        # Limit to n images, potentially randomly selected
        if self.select_random:
            selected_images = np.random.choice(
                df.image_path.unique(), self.prediction_samples
            )
        else:
            selected_images = df.image_path.unique()[: self.prediction_samples]

        # Ensure color is correctly assigned
        if self.color is None:
            num_classes = len(df["label"].unique())
            results_color = sv.ColorPalette.from_matplotlib("viridis", num_classes)
        else:
            results_color = self.color

        for image_name in selected_images:
            pred_df = df[df.image_path == image_name]

            targets = utilities.format_geometry(
                dataset.annotations_for_path(image_name, return_tensor=True), scores=False
            )

            # Assume that validation images are un-augmented
            basename = Path(image_name).stem + f"_{trainer.global_step}"
            fig = visualize.plot_results(
                basename=basename,
                results=pred_df,
                ground_truth=targets,
                savedir=out_dir,
                results_color=results_color,
                thickness=self.thickness,
                show=False,
            )
            plt.close(fig)

            # Pred metadata, if supported.
            stats = (
                pred_df["score"]
                .agg(
                    mean_confidence="mean",
                    max_confidence="max",
                    min_confidence="min",
                    std_confidence="std",
                )
                .to_dict()
            )

            metadata = {"pred_count": len(pred_df), "gt_count": len(targets)}
            metadata.update(stats)

            with open(os.path.join(out_dir, basename + ".json"), "w") as fp:
                json.dump(metadata, fp, indent=1)

            self._log_to_all(
                image=os.path.join(out_dir, basename + ".png"),
                trainer=trainer,
                tag="prediction sample",
                metadata=metadata,
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
