import argparse
import datetime
import glob
import os
import sys
import traceback
import warnings
from pathlib import Path

import torch
from hydra import compose, initialize, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import DeviceStatsMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from deepforest.callbacks import ImagesCallback
from deepforest.conf.schema import Config as StructuredConfig
from deepforest.main import deepforest
from deepforest.visualize import plot_results


def train(
    config: DictConfig,
    checkpoint: bool = True,
    comet: bool = False,
    tensorboard: bool = False,
    trace: bool = False,
) -> None:
    """This training function demonstrates basic setup for the DeepForest
    Trainer.

    Most
    experimental parameters are defined in the config, but we include some additional
    logic for logging here with sensible defaults: CSV/tensorboard
    """

    if trace:
        if not torch.cuda.is_available():
            warnings.warn("Cuda is not available, skipping trace.", stacklevel=2)
        else:
            torch.cuda.memory._record_memory_history()

    m = deepforest(config=config)

    callbacks = []
    loggers = []
    log_root = Path(config.train.log_root)

    # Use defaults from Lightning unless overriden by Comet
    experiment_name = None
    # Store as %YYYY%mm%ddT%HH:%MM:%SS
    version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Comet setup requires an external dependency
    if comet and not m.config.train.fast_dev_run:
        try:
            from pytorch_lightning.loggers import CometLogger

            comet_logger = CometLogger(
                api_key=os.environ.get("COMET_API_KEY"),
                workspace=os.environ.get("COMET_WORKSPACE"),
                project=os.environ.get("COMET_PROJECT", default="DeepForest"),
                offline_directory=config.train.log_root,
            )

            experiment_name = comet_logger.experiment.get_name()
            version = ""
            loggers.append(comet_logger)
        except ImportError:
            warnings.warn(
                "Failed to import Comet, check if comet-ml is installed", stacklevel=2
            )
        except Exception as e:
            warnings.warn(f"Failed to set up comet logger. {e}", stacklevel=2)
    else:
        callbacks.append(DeviceStatsMonitor())

    # By default, create a CSV logger and monitor stats
    csv_logger = CSVLogger(save_dir=log_root, name=experiment_name, version=version)
    loggers.append(csv_logger)

    if tensorboard:
        tensorboard_logger = TensorBoardLogger(
            save_dir=log_root,
            sub_dir="tensorboard",
            name=experiment_name,
            version=version,
        )
        loggers.append(tensorboard_logger)

    callbacks.append(ImagesCallback(save_dir=Path(csv_logger.log_dir) / "images"))

    # Setup checkpoint to store in log directory
    if checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(csv_logger.log_dir) / "checkpoints",
            filename=f"{config.architecture}-{{epoch:02d}}-{{val_map_50:.2f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

    m.create_trainer(logger=loggers, callbacks=callbacks, gradient_clip_val=0.5)

    train_success = False
    try:
        m.trainer.fit(m)
        train_success = True
    except Exception as e:
        warnings.warn(
            f"Training failed with exception {e}. Will attempt to upload any existing checkpoints if enabled.",
            stacklevel=2,
        )
        warnings.warn(traceback.format_exc(), stacklevel=2)

    if trace and torch.cuda.is_available():
        torch.cuda.memory._dump_snapshot(
            filename=Path(csv_logger.log_dir) / "dump_snapshot.pickle"
        )

    if checkpoint:
        for logger in m.trainer.loggers:
            if hasattr(logger.experiment, "log_model"):
                for checkpoint in glob.glob(
                    os.path.join((checkpoint_callback.dirpath), "*.ckpt")
                ):
                    m.print(f"Uploading checkpoint {checkpoint}")
                    logger.experiment.log_model(
                        name=os.path.basename(checkpoint), file_or_folder=str(checkpoint)
                    )

    return train_success


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
        help="Train a model. It is strongly recommended that you enable either Tensorboard or Comet logging so you can track your experiment visually.",
        epilog="Any remaining arguments <key>=<value> will be passed to Hydra to override the current config.",
    )
    train_parser.add_argument(
        "--disable-checkpoint", help="Path to log folder", action="store_true"
    )
    train_parser.add_argument(
        "--comet",
        help="Enable logging to Comet ML, requires comet to be logged in.",
        action="store_true",
    )
    train_parser.add_argument(
        "--tensorboard",
        help="Enable logging to Tensorboard",
        action="store_true",
    )
    train_parser.add_argument(
        "--trace",
        help="Enable PyTorch memory profiling.",
        action="store_true",
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
        res = train(
            cfg,
            checkpoint=not args.disable_checkpoint,
            comet=args.comet,
            tensorboard=args.tensorboard,
            trace=args.trace,
        )

        sys.exit(0 if res else 1)

    elif args.command == "config":
        print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()
