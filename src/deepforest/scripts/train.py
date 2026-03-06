import datetime
import glob
import os
import traceback
import warnings
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import DeviceStatsMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from deepforest.callbacks import ImagesCallback
from deepforest.main import deepforest


def train(
    config: DictConfig,
    checkpoint: bool = True,
    comet: bool = False,
    tensorboard: bool = False,
    trace: bool = False,
    resume: str | None = None,
    strategy: str = "auto",
    experiment_name: str | None = None,
    tags: list[str] | None = None,
) -> bool:
    """Train a DeepForest model with configurable logging and experiment
    tracking.

    This training function sets up PyTorch Lightning trainer with various logging
    options including CSV, TensorBoard, and Comet ML. When Comet logging is enabled,
    the experiment ID is automatically captured and stored in the model's
    hyperparameters for later use.

    Args:
        config (DictConfig): DeepForest configuration containing model and training parameters
        checkpoint (bool, optional): Whether to enable model checkpointing. Defaults to True.
        comet (bool, optional): Whether to enable Comet ML logging. Requires comet-ml
            package and proper environment variables (COMET_API_KEY, COMET_WORKSPACE).
            Defaults to False.
        tensorboard (bool, optional): Whether to enable TensorBoard logging in addition
            to CSV logging. Defaults to False.
        trace (bool, optional): Whether to enable PyTorch memory profiling for debugging.
            Only works when CUDA is available. Defaults to False.
        resume (str | None, optional): Path to checkpoint to resume training from.
            Defaults to None.
        experiment_name (str | None, optional): Custom experiment name for loggers.
            Overrides Comet's auto-generated name if set. Defaults to None.
        tags (list[str] | None, optional): Tags to apply to the Comet experiment.
            Defaults to None.

    Returns:
        bool: True if training completed successfully, False if training failed

    Note:
        When Comet logging is enabled, the experiment ID (key) is automatically added
        to the model's hyperparameters as 'experiment_id' for later re-logging to
        the same experiment.
    """

    if trace:
        if not torch.cuda.is_available():
            warnings.warn("Cuda is not available, skipping trace.", stacklevel=2)
        else:
            torch.cuda.memory._record_memory_history()

    m = deepforest(config=config)

    callbacks = []
    loggers = []
    log_root = Path(config.log_root)

    # Use defaults from Lightning unless overridden by caller or Comet
    experiment_id = None
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
                experiment_name=experiment_name,
                offline_directory=config.log_root,
            )

            if experiment_name is None:
                experiment_name = comet_logger.experiment.get_name()
            # Store experiment ID (key) for later re-logging to comet
            experiment_id = comet_logger.experiment.get_key()
            if experiment_name is None:
                version = ""
            if tags:
                comet_logger.experiment.add_tags(tags)
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
    csv_logger = CSVLogger(save_dir=log_root, name=experiment_name or "", version=version)
    loggers.append(csv_logger)

    if tensorboard:
        tensorboard_logger = TensorBoardLogger(
            save_dir=log_root,
            sub_dir="tensorboard",
            name=experiment_name or "",
            version=version,
        )
        loggers.append(tensorboard_logger)

    callbacks.append(
        ImagesCallback(
            save_dir=Path(csv_logger.log_dir) / "images",
            every_n_epochs=config.validation.val_accuracy_interval,
            select_random=True,
        )
    )

    # Setup checkpoint to store in log directory
    if checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(csv_logger.log_dir) / "checkpoints",
            filename=f"{config.architecture}-{{epoch:02d}}-{{map_50:.2f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        # Using equals causes a lot of strife with Hydra, so use colon instead.
        checkpoint_callback.CHECKPOINT_EQUALS_CHAR = ":"
        callbacks.append(checkpoint_callback)

    m.create_trainer(
        logger=loggers,
        callbacks=callbacks,
        gradient_clip_val=0.5,
        accelerator=config.accelerator,
        strategy=strategy,
    )

    # Add experiment ID to hyperparameters if available
    if experiment_id is not None:
        # Update the saved hyperparameters to include experiment ID
        current_hparams = m.hparams.copy()
        current_hparams["experiment_id"] = experiment_id
        m.save_hyperparameters(current_hparams)

    os.makedirs(csv_logger.log_dir, exist_ok=True)
    OmegaConf.save(config, Path(csv_logger.log_dir) / "config.yaml")

    train_success = False
    try:
        m.trainer.fit(m, ckpt_path=resume)
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
