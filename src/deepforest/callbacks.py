"""DeepForest callback for logging images during training.

Callbacks must implement on_epoch_begin, on_epoch_end, on_fit_end,
on_fit_begin methods and inject model and epoch kwargs.
"""

import glob

import numpy as np
import supervision as sv
from pytorch_lightning import Callback

from deepforest import visualize


class images_callback(Callback):
    """Log evaluation images during training.

    Args:
        savedir: Directory to save predicted images
        n: Number of images to process
        every_n_epochs: Run interval in epochs
        select_random: Whether to select random images
        color: Bounding box color as BGR tuple
        thickness: Border line thickness in pixels
    """

    def __init__(
        self, savedir, n=2, every_n_epochs=5, select_random=False, color=None, thickness=1
    ):
        self.savedir = savedir
        self.n = n
        self.color = color
        self.thickness = thickness
        self.select_random = select_random
        self.every_n_epochs = every_n_epochs

    def log_images(self, pl_module):
        """Log images to the logger."""
        df = pl_module.predictions

        # Limit to n images, potentially randomly selected
        if self.select_random:
            selected_images = np.random.choice(df.image_path.unique(), self.n)
        else:
            selected_images = df.image_path.unique()[: self.n]
        df = df[df.image_path.isin(selected_images)]

        # Add root_dir to the dataframe
        if "root_dir" not in df.columns:
            df["root_dir"] = pl_module.config.validation.root_dir

        # Ensure color is correctly assigned
        if self.color is None:
            num_classes = len(df["label"].unique())
            results_color = sv.ColorPalette.from_matplotlib("viridis", num_classes)
        else:
            results_color = self.color

        # Plot results
        visualize.plot_results(
            results=df,
            savedir=self.savedir,
            results_color=results_color,
            thickness=self.thickness,
        )

        try:
            saved_plots = glob.glob(f"{self.savedir}/*.png")
            for x in saved_plots:
                pl_module.logger.experiment.log_image(x)
        except Exception as e:
            print(
                "Could not find comet logger in lightning module, "
                f"skipping upload, images were saved to {self.savedir}, "
                f"error was raised {e}"
            )

    def on_validation_end(self, trainer, pl_module):
        """Run callback at validation end."""
        if trainer.sanity_checking:
            return

        if trainer.current_epoch % self.every_n_epochs == 0:
            print("Running image callback")
            self.log_images(pl_module)
