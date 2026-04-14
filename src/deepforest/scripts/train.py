import datetime
import glob
import os
import traceback
import warnings
from datetime import timedelta
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.strategies import DDPStrategy

from deepforest.callbacks import ImagesCallback
from deepforest.main import deepforest


def _find_last_checkpoint(log_root: Path, experiment_name: str) -> str:
    """Find the most recent last.ckpt under log_root/experiment_name/."""
    pattern = log_root / experiment_name / "**" / "last.ckpt"
    candidates = sorted(glob.glob(str(pattern), recursive=True))
    if not candidates:
        raise FileNotFoundError(f"No last.ckpt found matching {pattern}")
    return candidates[-1]


def _load_comet_key(log_root: Path, experiment_name: str) -> str | None:
    """Return a previously saved Comet experiment key, or None."""
    key_file = log_root / experiment_name / "comet_key.txt"
    if key_file.exists():
        return key_file.read_text().strip()
    return None


def train(
    config: DictConfig,
    checkpoint: bool = True,
    comet: bool = False,
    tensorboard: bool = False,
    trace: bool = False,
    resume: str | bool | None = None,
    strategy: str = "auto",
    experiment_name: str | None = None,
    tags: list[str] | None = None,
    export_hf: bool = False,
    slurm_auto_requeue: bool = False,
    ddp_timeout: int = 1800,
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
        resume (str | bool | None, optional): Path to checkpoint to resume training from,
            True to auto-find last.ckpt under log_root/experiment_name/, or None for
            a fresh run. Defaults to None.
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

    # matmul_precision is now set via config in create_trainer()

    if resume is True:
        if experiment_name is None:
            raise ValueError("--resume without a path requires --experiment-name")
        resume = _find_last_checkpoint(Path(config.log_root), experiment_name)

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

            resume_comet_key = None
            if resume is not None and experiment_name is not None:
                resume_comet_key = _load_comet_key(log_root, experiment_name)
                if resume_comet_key is None:
                    warnings.warn(
                        f"Could not find Comet experiment key for '{experiment_name}'. "
                        "A new experiment will be created.",
                        stacklevel=2,
                    )

            comet_logger = CometLogger(
                api_key=os.environ.get("COMET_API_KEY"),
                workspace=os.environ.get("COMET_WORKSPACE"),
                project=os.environ.get("COMET_PROJECT", default="DeepForest"),
                name=experiment_name,
                offline_directory=config.log_root,
                experiment_key=resume_comet_key,
            )

            # Always fetch experiment name and key from API
            experiment_id = comet_logger.experiment.get_key()
            experiment_name = comet_logger.experiment.get_name()
            # Save key so future resumes don't need an API search
            key_file = log_root / experiment_name / "comet_key.txt"
            key_file.parent.mkdir(parents=True, exist_ok=True)
            key_file.write_text(experiment_id)
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

    callbacks.append(TQDMProgressBar(refresh_rate=1))
    if config.get("ema_decay") is not None:
        from pytorch_lightning.callbacks import EMAWeightAveraging

        callbacks.append(EMAWeightAveraging(decay=config.ema_decay))
    callbacks.append(
        ImagesCallback(
            save_dir=Path(csv_logger.log_dir) / "images",
            every_n_epochs=config.validation.val_accuracy_interval,
            select_random=True,
        )
    )

    # Setup checkpoint to store in log directory
    if checkpoint:
        val_interval = config.validation.val_accuracy_interval
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(csv_logger.log_dir) / "checkpoints",
            filename=f"{config.architecture}-{{epoch:02d}}-{{val_mae:.2f}}",
            monitor="val_mae",
            mode="min",
            save_top_k=1,
            save_last=True,
            every_n_epochs=val_interval,
        )
        # Using equals causes a lot of strife with Hydra, so use colon instead.
        checkpoint_callback.CHECKPOINT_EQUALS_CHAR = ":"
        callbacks.append(checkpoint_callback)

    # Timeout DDP processes that may hang so that upstream
    # managers like SLURM can kill the running job. Without this,
    # if one process crashes (e.g. due to OOM), the whole job will
    # often hang indefinitely.
    if isinstance(strategy, str) and "ddp" in strategy:
        find_unused = strategy == "ddp_find_unused_parameters_true"
        strategy = DDPStrategy(
            find_unused_parameters=find_unused,
            timeout=timedelta(seconds=ddp_timeout),
        )

    m.create_trainer(
        logger=loggers,
        callbacks=callbacks,
        gradient_clip_val=0.5,
        accelerator=config.accelerator,
        strategy=strategy,
        plugins=[SLURMEnvironment(auto_requeue=slurm_auto_requeue)]
        if os.environ.get("SLURM_JOB_ID")
        else None,
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
        print(f"[train] calling trainer.fit, strategy={strategy}", flush=True)
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
                for ckpt_path in glob.glob(
                    os.path.join((checkpoint_callback.dirpath), "*.ckpt")
                ):
                    m.print(f"Uploading checkpoint {ckpt_path}")
                    logger.experiment.log_model(
                        name=os.path.basename(ckpt_path), file_or_folder=str(ckpt_path)
                    )

    is_rank_zero = m.trainer.is_global_zero
    if export_hf and checkpoint and train_success and is_rank_zero:
        hf_export_path = Path(csv_logger.log_dir) / "hf_model"
        best_ckpt = checkpoint_callback.best_model_path
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        state = {
            k.removeprefix("model."): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
        m.model.load_state_dict(state)
        m.model.save_pretrained(str(hf_export_path))
        m.print(f"Exported HF model to {hf_export_path}")

    return train_success
