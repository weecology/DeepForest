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


def predict(input, output=None, plot=False, **kwargs):
    """Predict on a single image or directory of images."""
    m = deepforest()
    m.load_model()

    # Predict on a single image
    if os.path.isfile(input):
        res = m.predict_tile(paths=input, **kwargs)
    else:
        raise ValueError("Input must be a file")

    if output:
        res.to_csv(output, index=False)

    if plot:
        plot_results(res)

    return res


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
        predict(args.input, args.output, args.plot)
    elif args.command == "train":
        train(cfg)
    elif args.command == "config":
        print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()
