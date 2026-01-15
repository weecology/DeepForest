import os

from omegaconf import DictConfig

from deepforest.main import deepforest
from deepforest.visualize import plot_results


def predict(
    config: DictConfig,
    input_path: str | None = None,
    output_path: str | None = None,
    plot: bool | None = False,
    root_dir: str | None = None,
) -> None:
    """Run prediction for the given image or CSV file, optionally saving the
    results to the provided path and optionally visualizing the results.

    Args:
        config (DictConfig): Hydra configuration.
        input_path (Optional[str]): Path to the input image or CSV file. If None, uses config.validation.csv_file.
        output_path (Optional[str]): Path to save the prediction results.
        plot (Optional[bool]): Whether to plot the results.
        root_dir (Optional[str]): Root directory containing images when input_path is a CSV file.

    Returns:
        None
    """
    m = deepforest(config=config)

    if input_path is None:
        if config.validation.csv_file is None:
            raise ValueError(
                "No input file provided and config.validation.csv_file is not set"
            )
        input_path = config.validation.csv_file
        print(f"Using validation CSV from config: {input_path}")

    if input_path.endswith(".csv") and root_dir is None:
        root_dir = config.validation.root_dir
        if root_dir is None:
            root_dir = os.path.dirname(input_path)
            if root_dir:
                print(f"Using CSV directory as root: {root_dir}")
        else:
            print(f"Using root directory from config: {root_dir}")

    if input_path.endswith(".csv"):
        # CSV batch prediction
        res = m.predict_file(
            csv_file=input_path,
            root_dir=root_dir,
            batch_size=config.batch_size,
        )
    else:
        # Single image prediction
        res = m.predict_tile(
            path=input_path,
            patch_size=config.patch_size,
            patch_overlap=config.patch_overlap,
            iou_threshold=config.nms_thresh,
        )

    if output_path is not None:
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        res.to_csv(output_path, index=False)

    if plot:
        plot_results(res)
