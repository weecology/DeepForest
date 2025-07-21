import argparse
import os
from typing import Optional

from hydra import compose, initialize_config_dir, initialize
from omegaconf import DictConfig, OmegaConf

from deepforest.main import deepforest
from deepforest.visualize import plot_results


def train(config: DictConfig) -> None:
    m = deepforest(config=config)
    m.trainer.fit(m)


def predict(config: DictConfig,
            input_path: str,
            output_path: Optional[str] = None,
            plot: Optional[bool] = False) -> None:
    """Run prediction for the given image, optionally saving the results to the
    provided path and optionally visualizing the results.

    Args:
        config (DictConfig): Hydra configuration.
        input_path (str): Path to the input image.
        output_path (Optional[str]): Path to save the prediction results.
        plot (Optional[bool]): Whether to plot the results.

    Returns:
        None
    """
    m = deepforest(config=config)
    res = m.predict_tile(path=input_path,
                         patch_size=config.patch_size,
                         patch_overlap=config.patch_overlap,
                         iou_threshold=config.nms_thresh)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        res.to_csv(output_path, index=False)

    if plot:
        plot_results(res)


def main():
    parser = argparse.ArgumentParser(description='DeepForest CLI')
    subparsers = parser.add_subparsers(dest="command")

    # Train subcommand
    _ = subparsers.add_parser(
        "train",
        help="Train a model",
        epilog=
        'Any remaining arguments <key>=<value> will be passed to Hydra to override the current config.'
    )

    # Predict subcommand
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run prediction on input",
        epilog=
        'Any remaining arguments <key>=<value> will be passed to Hydra to override the current config.'
    )
    predict_parser.add_argument("input", help="Path to input raster")
    predict_parser.add_argument("-o", "--output", help="Path to prediction results")
    predict_parser.add_argument("--plot", action="store_true", help="Plot results")

    # Show config subcommand
    subparsers.add_parser("config", help="Show the current config")

    # Config options for Hydra
    parser.add_argument("--config-dir", help="Show available config overrides and exit")
    parser.add_argument("--config-name",
                        help="Show available config overrides and exit",
                        default="config")

    args, overrides = parser.parse_known_args()

    if args.config_dir is not None:
        initialize_config_dir(version_base=None, config_dir=args.config_dir)
    else:
        initialize(version_base=None, config_path="pkg://deepforest.conf")

    cfg = compose(config_name=args.config_name, overrides=overrides)

    if args.command == "predict":
        predict(cfg, input_path=args.input, output_path=args.output, plot=args.plot)
    elif args.command == "train":
        train(cfg)
    elif args.command == "config":
        print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()
