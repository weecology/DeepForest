import argparse
import sys

from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf

from deepforest.conf.schema import Config as StructuredConfig
from deepforest.scripts.evaluate import evaluate
from deepforest.scripts.predict import predict
from deepforest.scripts.train import train


def main():
    parser = argparse.ArgumentParser(description="DeepForest CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Train subcommand
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model.",
        epilog="Any remaining arguments <key>=<value> will be passed to Hydra to override the current config.",
    )
    train_parser.add_argument(
        "--disable-checkpoint",
        help="Disable model checkpointing during training",
        action="store_true",
    )
    train_parser.add_argument(
        "--comet",
        help="Enable logging to Comet ML, requires comet to be logged in.",
        action="store_true",
    )
    train_parser.add_argument(
        "--tensorboard",
        help="Enable local logging to Tensorboard",
        action="store_true",
    )
    train_parser.add_argument(
        "--trace",
        help="Enable PyTorch memory profiling.",
        action="store_true",
    )
    train_parser.add_argument(
        "--strategy",
        help="Training strategy to use (e.g., 'ddp', 'auto')",
        default="auto",
    )
    train_parser.add_argument(
        "--resume",
        help="Path to checkpoint to resume training from",
    )
    train_parser.add_argument(
        "--experiment-name",
        help="Experiment name for loggers. Overrides Comet's auto-generated name if set.",
    )
    train_parser.add_argument(
        "--tag",
        action="append",
        default=[],
        dest="tags",
        help="Tag for the experiment (can be repeated, e.g. --tag baseline --tag v2). Applied to Comet if enabled.",
    )

    # Predict subcommand
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run prediction on input image or CSV file",
        epilog="Any remaining arguments <key>=<value> will be passed to Hydra to override the current config.",
    )
    predict_parser.add_argument(
        "input",
        nargs="?",
        help="Path to input image or CSV file (optional if specified in config)",
    )
    predict_parser.add_argument(
        "-o", "--output", help="Path to save prediction results CSV"
    )
    predict_parser.add_argument(
        "--plot", action="store_true", help="Visualize predictions"
    )
    predict_parser.add_argument(
        "--root-dir",
        help="Root directory containing images when input is a CSV file. Defaults to CSV directory if not specified.",
    )
    predict_parser.add_argument(
        "--mode",
        choices=["single", "tile", "csv"],
        default="single",
        help="Prediction mode: 'single' for single image, 'tile' for tiled image prediction, 'csv' for batch prediction from CSV file. Defaults to 'single'.",
    )

    # Evaluate subcommand
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run evaluation on ground truth annotations. Use --predictions-csv to provide existing predictions, or omit to generate them.",
        epilog="Any remaining arguments <key>=<value> will be passed to Hydra to override the current config.",
    )
    evaluate_parser.add_argument(
        "csv_file",
        nargs="?",
        help="Path to ground truth CSV file (optional if specified in config)",
    )
    evaluate_parser.add_argument(
        "--root-dir",
        help="Root directory containing images. Defaults to CSV directory if not specified.",
    )
    evaluate_parser.add_argument(
        "--predictions",
        help="Path to existing predictions CSV file. If not provided, predictions will be generated.",
    )
    evaluate_parser.add_argument(
        "--save-predictions",
        help="Path to save generated predictions CSV (only used when --predictions is not provided)",
    )
    evaluate_parser.add_argument(
        "-o", "--output", help="Path to save evaluation metrics summary CSV"
    )

    # Show config subcommand
    subparsers.add_parser("config", help="Show the current config")

    # Config options for Hydra
    parser.add_argument("--config-dir", help="Path to custom configuration directory")
    parser.add_argument(
        "--config-name", help="Name of configuration file to use", default="config"
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
        predict(
            cfg,
            input_path=args.input,
            output_path=args.output,
            plot=args.plot,
            root_dir=args.root_dir,
            mode=args.mode,
        )
    elif args.command == "train":
        res = train(
            cfg,
            checkpoint=not args.disable_checkpoint,
            comet=args.comet,
            tensorboard=args.tensorboard,
            trace=args.trace,
            strategy=args.strategy,
            resume=args.resume,
            experiment_name=args.experiment_name,
            tags=args.tags,
        )

        sys.exit(0 if res else 1)

    elif args.command == "evaluate":
        evaluate(
            cfg,
            ground_truth=args.csv_file,
            root_dir=args.root_dir,
            predictions=args.predictions,
            output_path=args.output,
            save_predictions=args.save_predictions,
        )

    elif args.command == "config":
        print(OmegaConf.to_yaml(cfg, resolve=True))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
