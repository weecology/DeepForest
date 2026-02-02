import argparse
import json
import os

from hydra import compose, initialize, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from deepforest.conf.schema import Config as StructuredConfig
from deepforest.main import deepforest
from deepforest.visualize import plot_results


def train(config: DictConfig) -> None:
    """Train a DeepForest model with the given configuration.

    Args:
        config: Hydra configuration object containing training parameters
    """
    m = deepforest(config=config)
    m.trainer.fit(m)


def predict(
    config: DictConfig,
    input_path: str,
    output_path: str | None = None,
    plot: bool | None = False,
) -> None:
    """Run prediction for the given image, optionally saving the results to the
    provided path and optionally visualizing the results.

    Args:
        config (DictConfig): DeepForest configuration.
        input_path (str): Path to the input image.
        output_path (Optional[str]): Path to save the prediction results.
        plot (Optional[bool]): Whether to plot the results.

    Returns:
        None
    """
    m = deepforest(config=config)
    res = m.predict_tile(
        path=input_path,
        patch_size=config.patch_size,
        patch_overlap=config.patch_overlap,
        iou_threshold=config.nms_thresh,
    )

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        res.to_csv(output_path, index=False)

    if plot:
        plot_results(res)


def evaluate(config: DictConfig, output_path: str | None = None) -> None:
    """Evaluate predictions against ground truth using the specified
    configuration. Target data is taken from config (e.g. validation.csv_file,
    validation.root_dir.)

    Output documents (predictions.csv and results.json) are saved to output_path if provided.

    Args:
            config (DictConfig): DeepForest configuration.
            output_path (str): Directory to save evaluation results.

    Returns:
        None
    """
    config.validation.val_accuracy_interval = 1
    m = deepforest(config=config)

    if output_path:
        os.makedirs(output_path, exist_ok=True)

    eval_results = m.trainer.validate(m)[0]

    if output_path and eval_results.get("results", None):
        eval_results.to_csv(os.path.join(output_path, "predictions.csv"), index=False)

    with (
        open(os.path.join(output_path, "results.json"), "w")
        if output_path
        else os.sys.stdout as f
    ):
        json.dump(eval_results, f, indent=1)


def main():
    """Main CLI entry point for DeepForest.

    Provides subcommands for training, prediction, and configuration
    management.
    """
    parser = argparse.ArgumentParser(description="DeepForest CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Train subcommand
    _ = subparsers.add_parser(
        "train",
        help="Train a model",
        epilog="Any remaining arguments <key>=<value> will be passed to Hydra to override the current config.",
    )

    # Predict subcommand
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run prediction on input",
        epilog="Any remaining arguments <key>=<value> will be passed to Hydra to override the current config.",
    )
    predict_parser.add_argument("input", help="Path to input raster")
    predict_parser.add_argument("-o", "--output", help="Path to prediction results")
    predict_parser.add_argument("--plot", action="store_true", help="Plot results")

    # Evaluate subcommand
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate predictions against the given validation dataset. Set validation parameters like thresholds via config.",
        epilog="Any remaining arguments <key>=<value> will be passed to Hydra to override the current config.",
    )
    evaluate_parser.add_argument(
        "-o", "--output", help="Directory to save evaluation results"
    )

    # Show config subcommand``
    subparsers.add_parser("config", help="Show the current config")

    # Config options for Hydra
    parser.add_argument("--config-dir", help="Show available config overrides and exit")
    parser.add_argument(
        "--config-name", help="Show available config overrides and exit", default="config"
    )

    args, overrides = parser.parse_known_args()

    if args.config_dir is not None:
        initialize_config_dir(version_base=None, config_dir=args.config_dir)
    else:
        initialize(version_base=None, config_path="pkg://deepforest.conf")

    base = OmegaConf.structured(StructuredConfig)
    cfg = compose(config_name=args.config_name, overrides=overrides)
    cfg = OmegaConf.merge(base, cfg)

    if args.command == "predict":
        predict(cfg, input_path=args.input, output_path=args.output, plot=args.plot)
    elif args.command == "train":
        train(cfg)
    elif args.command == "config":
        print(OmegaConf.to_yaml(cfg, resolve=True))
    elif args.command == "evaluate":
        evaluate(cfg, output_path=args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
