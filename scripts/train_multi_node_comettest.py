"""Train DeepForest bird detection model.

This script trains a bird detection model using the weecology/deepforest-tree
pretrained model as a starting point. A small fraction of the training images is
held out as the validation set used during training; test.csv is only evaluated
once at the end.

Example usage:
    python train_multi_node_comettest.py --data_dir /path/to/prepared/data --batch_size 12 --workers 5
"""

import argparse
import hashlib
import os
import time

import pandas as pd
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

from deepforest import callbacks, main


def get_slurm_experiment_key():
    """Derive a deterministic Comet experiment key from SLURM_JOB_ID.

    All nodes in the same SLURM job share the same SLURM_JOB_ID, so they
    will all compute the same experiment key. This allows non-rank-0 nodes
    to report system metrics (GPU utilization, etc.) to the same Comet
    experiment that rank 0 creates via the PYL CometLogger.
    """
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id is None:
        return None
    # Experiment keys must be alphanumeric, 32-50 characters
    return hashlib.sha256(f"comet-slurm-{slurm_job_id}".encode()).hexdigest()[:50]


def setup_worker_system_metrics(experiment_key):
    """On non-zero SLURM nodes, create a Comet experiment that only
    collects system metrics (GPU, CPU, etc.) and reports them to the same
    experiment that rank 0 on node 0 manages via the PYL CometLogger.

    Only the local-rank-0 process on each node creates this experiment,
    since GPU/CPU metrics are per-node (not per-process).

    SLURM variable semantics:
        SLURM_NODEID  — node/server index (0, 1, 2, …)
        SLURM_LOCALID — rank within the node  (0 to ntasks-per-node-1)
        SLURM_PROCID  — global rank            (0 to ntasks-1)

    Returns the experiment object so it can be ended at cleanup time,
    or None if not applicable.
    """
    slurm_node_id = os.environ.get("SLURM_NODEID")
    slurm_local_id = os.environ.get("SLURM_LOCALID")
    if slurm_node_id is None or slurm_local_id is None or experiment_key is None:
        return None

    # Node 0 is already covered by PYL's CometLogger on global rank 0
    if int(slurm_node_id) == 0:
        return None

    # Only one system-metrics experiment per node — let local rank 0 own it
    if int(slurm_local_id) != 0:
        return None

    # CometLogger.__init__ already set COMET_DISABLE_AUTO_LOGGING=1 and
    # imported comet_ml, so this import is safe
    import comet_ml

    experiment = comet_ml.start(
        project="bird-detector",
        experiment_key=experiment_key,
        mode="get_or_create",
        experiment_config=comet_ml.ExperimentConfig(
            distributed_node_identifier=f"node-{slurm_node_id}",
            # Disable everything except system metrics — rank 0 on node 0
            # handles parameter/metric/code logging via the CometLogger
            auto_param_logging=False,
            auto_metric_logging=False,
            auto_output_logging="false",
            parse_args=False,
            log_code=False,
            log_git_metadata=False,
            log_git_patch=False,
        ),
    )
    print(f"Node {slurm_node_id}: reporting system metrics to experiment {experiment_key}")
    return experiment


def run():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train DeepForest bird detection model")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing train.csv, test.csv, and images",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
        help="Batch size for training (default: 12)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of workers for data loading (default: 5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Number of training epochs (default: 12)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to save model checkpoints (default: data_dir/checkpoints)",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes for distributed training (default: 1)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.1,
        help="Score threshold for the model and validation metrics (default: 0.1). "
        "Keep this low and choose the operating threshold after training.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.02,
        help="Fraction of training images held out as the validation set (default: 0.02)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the train/validation image split (default: 0)",
    )
    parser.add_argument(
        "--validate_coordinates",
        action="store_true",
        help="Check every annotation falls inside its image before training. Off by default: "
        "it opens every image serially at startup, which takes hours on the full dataset.",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Run a fast development run with a single batch",
    )

    args = parser.parse_args()

    # Set matmul precision to high for faster training on Tensor Core GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        print("Set torch.float32_matmul_precision to 'high' for faster training")

    # Set up paths
    train_csv = os.path.join(args.data_dir, "train.csv")
    test_csv = os.path.join(args.data_dir, "test.csv")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    if args.checkpoint_dir is None:
        checkpoint_dir = os.path.join(args.data_dir, "checkpoints")
    else:
        checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Initializing DeepForest model...")
    # Initialize DeepForest model
    m = main.deepforest()

    # Load the pretrained tree model as a starting point
    # print("Loading pretrained tree model: weecology/deepforest-tree")
    m.load_model("weecology/deepforest-tree")

    # Set label dictionaries for single "Bird" class
    m.label_dict = {"Bird": 0}
    m.numeric_to_label_dict = {0: "Bird"}
    m.config.label_dict = {"Bird": 0}
    m.config.num_classes = 1

    m.config.score_thresh = args.score_thresh
    m.model.score_thresh = args.score_thresh

    # Hold out a fraction of training images for validation (by image, seeded, so
    # every rank derives the same split). test.csv stays untouched until the end.
    full_train = pd.read_csv(train_csv)
    images = pd.Series(sorted(full_train.image_path.unique()))
    val_images = set(images.sample(frac=args.val_fraction, random_state=args.seed))
    is_val = full_train.image_path.isin(val_images)
    split_dir = os.path.join(checkpoint_dir, "split")
    os.makedirs(split_dir, exist_ok=True)
    train_split_csv = os.path.join(split_dir, "train_split.csv")
    val_split_csv = os.path.join(split_dir, "val_split.csv")
    for df, path in [(full_train[~is_val], train_split_csv), (full_train[is_val], val_split_csv)]:
        # Atomic write: all ranks produce identical content, so replacement is safe
        tmp = f"{path}.{os.getpid()}.tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    print(
        f"Train: {(~is_val).sum()} boxes / {len(images) - len(val_images)} images; "
        f"validation: {is_val.sum()} boxes / {len(val_images)} images"
    )

    # Configure training data paths
    m.config["train"]["csv_file"] = train_split_csv
    m.config["train"]["root_dir"] = args.data_dir
    m.config["train"]["fast_dev_run"] = args.fast_dev_run
    m.config["train"]["validate_coordinates"] = args.validate_coordinates
    m.config["train"]["epochs"] = args.epochs
    m.config["train"]["lr"] = args.lr
    m.config["train"]["scheduler"]["params"]["patience"] = 3

    # Configure validation data paths (held-out training images, not test.csv)
    m.config["validation"]["csv_file"] = val_split_csv
    m.config["validation"]["root_dir"] = args.data_dir
    m.config["validation"]["validate_coordinates"] = args.validate_coordinates
    m.config["validation"]["val_accuracy_interval"] = 1
    m.config["validation"]["size"] = 800

    # Configure data loading
    m.config["batch_size"] = args.batch_size
    m.config["workers"] = args.workers

    # Configure augmentations with modern options
    # Using zoom augmentations (RandomResizedCrop), rotations, and other augmentations
    # Use OmegaConf.update to bypass strict type validation
    augmentations_config = OmegaConf.create(
        {
            "train": {
                "augmentations": [
                    {
                        "RandomResizedCrop": {
                            "size": (800, 800),
                            "scale": (0.3, 1.0),
                            "p": 0.5,
                        }
                    },
                    {"Rotate": {"degrees": 15, "p": 0.5}},
                    {"HorizontalFlip": {"p": 0.5}},
                    {"VerticalFlip": {"p": 0.3}},
                    {"PadIfNeeded": {"size": (1000, 1000)}},
                    # {"RandomBrightnessContrast": {"brightness": 0.2, "contrast": 0.2, "p": 0.5}},
                    # {"HueSaturationValue": {"hue": 0.1, "saturation": 0.1, "p": 0.3}},
                    # {"ZoomBlur": {"max_factor": (1.0, 1.03), "step_factor": (0.01, 0.02), "p": 0.3}},
                ]
            }
        }
    )
    OmegaConf.set_struct(m.config, False)
    m.config = OmegaConf.merge(m.config, augmentations_config)
    OmegaConf.set_struct(m.config, True)

    # Configure scheduler (similar to BOEM script)
    m.config["train"]["scheduler"]["params"]["eps"] = 0

    # Derive a shared experiment key from SLURM_JOB_ID so all nodes can
    # report to the same Comet experiment
    experiment_key = get_slurm_experiment_key()

    # Set up Comet logger (optional, will skip if not configured)
    comet_logger = None
    system_metrics_exp = None
    try:
        # On rank 0: creates the experiment with the shared key
        # On non-rank-0: CometLogger skips experiment creation (PYL default)
        comet_logger = CometLogger(
            project="bird-detector",
            experiment_key=experiment_key,
        )

        # On non-rank-0 SLURM nodes: open a lightweight connection to the
        # same experiment purely for system metrics (GPU, CPU, etc.)
        system_metrics_exp = setup_worker_system_metrics(experiment_key)

        comet_logger.experiment.add_tag("bird-detection")

        # Log training and test set sizes

        train_df = full_train[~is_val]
        val_df = full_train[is_val]
        test_df = pd.read_csv(test_csv)
        comet_logger.experiment.log_table("train_split.csv", train_df)
        comet_logger.experiment.log_table("val_split.csv", val_df)
        comet_logger.experiment.log_table("test.csv", test_df)

        # Log training parameters
        devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
        comet_logger.experiment.log_parameter("devices", devices)
        comet_logger.experiment.log_parameter("workers", m.config["workers"])
        comet_logger.experiment.log_parameter("batch_size", m.config["batch_size"])
        comet_logger.experiment.log_parameter("train_size", len(train_df))
        comet_logger.experiment.log_parameter("val_size", len(val_df))
        comet_logger.experiment.log_parameter("test_size", len(test_df))
        comet_logger.experiment.log_parameter("score_thresh", args.score_thresh)
        comet_logger.experiment.log_parameter("epochs", args.epochs)
        comet_logger.experiment.log_parameter("learning_rate", args.lr)

        print(f"Comet logging enabled: {comet_logger.experiment.get_key()}")
    except Exception as e:
        print(f"Warning: Could not initialize Comet logger: {e}")
        print("Continuing without Comet logging...")
        comet_logger = None

    # Set up image callback for validation visualization
    images_dir = os.path.join(checkpoint_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    im_callback = callbacks.ImagesCallback(
        save_dir=images_dir,
        prediction_samples=20,  # Number of validation images to log
        dataset_samples=20,  # Number of dataset samples to log at start
        every_n_epochs=1,  # Log predictions every epoch
    )

    # Keep the best checkpoint by validation mAP50 and the latest one, so a crash
    # during a long run does not lose the model
    run_id = (
        os.environ.get("SLURM_JOB_ID") or time.strftime("%Y%m%d_%H%M%S")
    )
    ckpt_callback = ModelCheckpoint(
        dirpath=os.path.join(checkpoint_dir, run_id),
        filename="bird-{epoch:02d}-{map_50:.3f}",
        monitor="map_50",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    # Create trainer with GPU support
    print("Creating trainer...")

    m.create_trainer(
        logger=comet_logger,
        callbacks=[im_callback, ckpt_callback],
        strategy="ddp",
        num_nodes=args.num_nodes,
        precision="16-mixed",  # Use mixed precision training for faster performance
        fast_dev_run=args.fast_dev_run,
        enable_progress_bar=True,
    )

    # Train the model
    print("\nStarting training...")
    m.trainer.fit(m)

    # Save the final-epoch model. The run id comes from SLURM or the clock so this
    # works with or without Comet, and all ranks agree on it.
    checkpoint_path = os.path.join(checkpoint_dir, f"{run_id}.ckpt")
    print(f"\nSaving checkpoint to: {checkpoint_path}")
    m.trainer.save_checkpoint(checkpoint_path)
    print(f"Best checkpoint: {ckpt_callback.best_model_path}")
    global_steps = m.trainer.global_step

    # Final evaluation on the held-out test split (all ranks participate)
    m.config.validation.csv_file = test_csv
    m.trainer.validate(m)

    # The remaining steps build a single-device trainer, which cannot run inside a
    # multi-process job: do them on one process only, and only for single-process runs.
    if m.trainer.world_size > 1:
        if comet_logger:
            comet_logger.experiment.log_metric("global_steps", global_steps)
        print(
            "\nSkipping zero-shot evaluation in a multi-process run. "
            f"Evaluate {checkpoint_path} separately on one GPU."
        )
        if system_metrics_exp is not None:
            system_metrics_exp.end()
        return
    if not m.trainer.is_global_zero:
        return

    # Evaluate on zero-shot dataset
    print("\n" + "=" * 80)
    print("Evaluating on zero-shot dataset (DeepWater Horizon)")
    print("=" * 80)

    # Update validation config for zero-shot dataset
    m.config.validation.csv_file = "/blue/ewhite/b.weinstein/bird_detector_retrain/zero_shot/avian_images_annotated/test_splits//patch_600/test_split_patch_600.csv"
    m.config.validation.root_dir = "/blue/ewhite/b.weinstein/bird_detector_retrain/zero_shot/avian_images_annotated/test_splits/patch_600"
    m.config.validation.iou_threshold = 0.4

    # Create new trainer for zero-shot evaluation
    m.create_trainer()

    # Evaluate on zero-shot dataset
    zero_shot_results = m.trainer.validate(m)
    zero_shot_metrics = zero_shot_results[0] if zero_shot_results else {}

    print("\nZero-shot evaluation results:")
    print(f"  Box Precision: {zero_shot_metrics.get('box_precision', 'N/A')}")
    print(f"  Box Recall: {zero_shot_metrics.get('box_recall', 'N/A')}")
    print(
        f"  Empty Frame Accuracy: {zero_shot_metrics.get('empty_frame_accuracy', 'N/A')}"
    )

    # Log the zero-shot results to Comet, skipping metrics that were not produced
    if comet_logger:
        for key in ["box_precision", "box_recall", "empty_frame_accuracy"]:
            value = zero_shot_metrics.get(key)
            if isinstance(value, int | float):
                comet_logger.experiment.log_metric(f"zero_shot_{key}", value)
        comet_logger.experiment.log_metric("global_steps", global_steps)

    # End the worker system metrics experiment if we created one
    if system_metrics_exp is not None:
        system_metrics_exp.end()

    print("\nTraining complete!")


if __name__ == "__main__":
    run()
