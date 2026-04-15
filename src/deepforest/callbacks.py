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
            if hasattr(pl_module, "density_samples") and pl_module.density_samples:
                pl_module.print("Logging density map samples")
                self._log_density_plots(trainer, pl_module)

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

            basename = Path(path).stem
            image = (255 * image.cpu().numpy().transpose((1, 2, 0))).astype(np.uint8)
            out_path = os.path.join(out_dir, basename + ".png")

            if image_annotations is not None:
                image_annotations.root_dir = dataset.root_dir
                image_annotations["image_path"] = path

                # Plot transformed image
                fig = visualize.plot_annotations(
                    image=image,
                    annotations=image_annotations,
                    savedir=out_dir,
                    basename=basename,
                    thickness=self.thickness,
                    show=False,
                )
                plt.close(fig)
            else:
                # Save un-annotated image
                Image.fromarray(image).save(out_path)

            self._log_to_all(
                image=out_path,
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

        if df.empty or "image_path" not in df.columns:
            pl_module.print("No predictions above score_thresh to log.")
            return

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

            density_sum = getattr(pl_module, "density_sum_by_image", {}).get(image_name)
            metadata = {
                "pred_count": len(pred_df),
                "gt_count": len(targets),
                "density_sum": round(density_sum, 2) if density_sum is not None else None,
            }
            metadata.update(stats)

            with open(os.path.join(out_dir, basename + ".json"), "w") as fp:
                json.dump(metadata, fp, indent=1)

            self._log_to_all(
                image=os.path.join(out_dir, basename + ".png"),
                trainer=trainer,
                tag="prediction sample",
                metadata=metadata,
            )

    def _log_density_plots(self, trainer, pl_module):
        """Save and log side-by-side predicted vs GT density map plots."""
        out_dir = os.path.join(self.savedir, "density_plots")
        os.makedirs(out_dir, exist_ok=True)

        for sample in pl_module.density_samples:
            image_name = sample["image_name"]
            basename = Path(image_name).stem + f"_{trainer.global_step}"

            pred = sample["pred_density"].numpy()
            gt = sample["gt_density"].numpy()
            # RGB image from CHW float tensor
            img = (
                (255 * sample["image"].numpy().transpose(1, 2, 0))
                .clip(0, 255)
                .astype(np.uint8)
            )
            gt_points = sample.get("gt_points")

            vmax = max(pred.max(), gt.max(), 1e-6)

            fig = plt.figure(figsize=(16, 5))
            gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3)
            ax0 = fig.add_subplot(gs[0])
            ax1 = fig.add_subplot(gs[1])
            ax2 = fig.add_subplot(gs[2])
            cax = fig.add_subplot(gs[3])

            ax0.imshow(img)
            if gt_points is not None and len(gt_points) > 0:
                ax0.scatter(
                    gt_points[:, 0].numpy(),
                    gt_points[:, 1].numpy(),
                    s=10,
                    c="lime",
                    linewidths=0.5,
                    edgecolors="black",
                )
            ax0.set_title(
                f"RGB  (gt_count={len(gt_points) if gt_points is not None else 0})"
            )
            ax0.axis("off")

            ax1.imshow(gt, cmap="hot", vmin=0, vmax=vmax)
            ax1.set_title(f"GT density  (sum={gt.sum():.1f})")
            ax1.axis("off")

            im2 = ax2.imshow(pred, cmap="hot", vmin=0, vmax=vmax)
            ax2.set_title(f"Pred density  (sum={pred.sum():.1f})")
            ax2.axis("off")

            fig.colorbar(im2, cax=cax)

            fig.suptitle(
                f"{os.path.basename(image_name)}  epoch={trainer.current_epoch}",
                fontsize=10,
            )

            out_path = os.path.join(out_dir, basename + ".png")
            fig.savefig(out_path, dpi=100)
            plt.close(fig)

            self._log_to_all(
                image=out_path,
                trainer=trainer,
                tag="density plot",
                metadata={
                    "gt_sum": float(gt.sum()),
                    "pred_sum": float(pred.sum()),
                },
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
