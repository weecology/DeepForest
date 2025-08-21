import argparse
import os

from hydra import compose, initialize, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from deepforest.conf.schema import Config as StructuredConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, CometLogger

from deepforest.main import deepforest
from deepforest.visualize import plot_results


def train(config: DictConfig, checkpoint=True) -> None:
    m = deepforest(config=config)

    callbacks = []
    if checkpoint:
        checkpoint_callback = ModelCheckpoint(
            filename=f"{config.architecture}-{{epoch:02d}}-{{val_map_50:.2f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

    loggers = []

    try:
        comet_logger = CometLogger(api_key=os.environ.get("COMET_API_KEY"),
                                   workspace=os.environ.get("COMET_WORKSPACE"),
                                   project="DeepForest")
        comet_experiment_name = comet_logger.experiment.get_name()
        loggers.append(comet_logger)

        full_log_dir = (f"{config.train.log_root}/{comet_experiment_name}")
        csv_logger = CSVLogger(save_dir=full_log_dir, name="", version="")
    except ModuleNotFoundError:
        csv_logger = CSVLogger(save_dir=config.train.log_root)

    loggers.append(csv_logger)

    m.create_trainer(logger=loggers, callbacks=callbacks)
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
        config (DictConfig): Hydra configuration.
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


def main():
    parser = argparse.ArgumentParser(description="DeepForest CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Train subcommand
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model",
        epilog="Any remaining arguments <key>=<value> will be passed to Hydra to override the current config.",
    )
    train_parser.add_argument("--no-checkpoint",
                              help="Path to log folder",
                              action='store_true')

    # Predict subcommand
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run prediction on input",
        epilog="Any remaining arguments <key>=<value> will be passed to Hydra to override the current config.",
    )
    predict_parser.add_argument("input", help="Path to input raster")
    predict_parser.add_argument("-o", "--output", help="Path to prediction results")
    predict_parser.add_argument("--plot", action="store_true", help="Plot results")

    # Show config subcommand
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
        train(cfg, not args.no_checkpoint)
    elif args.command == "config":
        print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()
