"""Train DeepForest bird detection model.

This script trains a bird detection model using the weecology/deepforest-bird
pretrained model as a starting point.

Example usage:
    python train_birds.py --data_dir /path/to/prepared/data --batch_size 12 --workers 5
"""

import argparse
import os

import torch
from pytorch_lightning.loggers import CometLogger
from omegaconf import OmegaConf

from deepforest import main
import pandas as pd


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
        "--fast_dev_run",
        action="store_true",
        help="Run a fast development run with a single batch",
    )

    args = parser.parse_args()

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
    print("Loading pretrained tree model: weecology/deepforest-tree")
    m.load_model("weecology/deepforest-tree")

    # Set label dictionaries for single "Bird" class
    m.label_dict = {"Bird": 0}
    m.numeric_to_label_dict = {0: "Bird"}
    m.config.label_dict = {"Bird": 0}
    m.config.num_classes = 1

    # Configure training data paths
    m.config["train"]["csv_file"] = train_csv
    m.config["train"]["root_dir"] = args.data_dir
    m.config["train"]["fast_dev_run"] = args.fast_dev_run
    m.config["train"]["epochs"] = args.epochs
    m.config["train"]["lr"] = args.lr

    # Configure validation data paths
    m.config["validation"]["csv_file"] = test_csv
    m.config["validation"]["root_dir"] = args.data_dir
    m.config["validation"]["val_accuracy_interval"] = 1
    m.config["validation"]["size"] = 800

    # Configure data loading
    m.config["batch_size"] = args.batch_size
    m.config["workers"] = args.workers

    # Configure augmentations with modern options
    # Using zoom augmentations (RandomResizedCrop), rotations, and other augmentations
    # Use OmegaConf.update to bypass strict type validation
    augmentations_config = OmegaConf.create({
        "train": {
            "augmentations": [
                {"RandomResizedCrop": {"size": (800, 800), "scale": (0.5, 1.0), "p": 0.3}},
                {"Rotate": {"degrees": 15, "p": 0.5}},
                {"HorizontalFlip": {"p": 0.5}},
                {"VerticalFlip": {"p": 0.3}},
                {"RandomBrightnessContrast": {"brightness": 0.2, "contrast": 0.2, "p": 0.5}},
                {"HueSaturationValue": {"hue": 0.1, "saturation": 0.1, "p": 0.3}},
                {"ZoomBlur": {"max_factor": (1.0, 1.03), "step_factor": (0.01, 0.02), "p": 0.3}},
            ]
        }
    })
    OmegaConf.set_struct(m.config, False)
    m.config = OmegaConf.merge(m.config, augmentations_config)
    OmegaConf.set_struct(m.config, True)

    # Configure scheduler (similar to BOEM script)
    m.config["train"]["scheduler"]["params"]["eps"] = 0

    # Set up Comet logger (optional, will skip if not configured)
    comet_logger = None
    try:
        comet_logger = CometLogger()
        comet_logger.experiment.add_tag("bird-detection")

        # Log training and test set sizes

        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        comet_logger.experiment.log_table("train.csv", train_df)
        comet_logger.experiment.log_table("test.csv", test_df)

        # Log training parameters
        devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
        comet_logger.experiment.log_parameter("devices", devices)
        comet_logger.experiment.log_parameter("workers", m.config["workers"])
        comet_logger.experiment.log_parameter("batch_size", m.config["batch_size"])
        comet_logger.experiment.log_parameter("train_size", len(train_df))
        comet_logger.experiment.log_parameter("test_size", len(test_df))
        comet_logger.experiment.log_parameter("epochs", args.epochs)
        comet_logger.experiment.log_parameter("learning_rate", args.lr)

        print(f"Comet logging enabled: {comet_logger.experiment.get_key()}")
    except Exception as e:
        print(f"Warning: Could not initialize Comet logger: {e}")
        print("Continuing without Comet logging...")
        comet_logger = None

    # Create trainer with GPU support
    print("Creating trainer...")
    devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

    m.create_trainer(
        logger=comet_logger,
        accelerator="auto",
        strategy="ddp",
        num_nodes=1,
        devices=devices,
        fast_dev_run=args.fast_dev_run,
    )

    # Evaluate before training
    print("\nEvaluating model before training...")
    results_before = m.evaluate(csv_file=test_csv, root_dir=args.data_dir, size=800)
    print(f"Before training - Box Precision: {results_before['box_precision']:.4f}, "
          f"Box Recall: {results_before['box_recall']:.4f}")

    if comet_logger:
        comet_logger.experiment.log_metric("box_precision_before", results_before["box_precision"])
        comet_logger.experiment.log_metric("box_recall_before", results_before["box_recall"])

    # Train the model
    print("\nStarting training...")
    m.trainer.fit(m)

    # Save the model checkpoint
    if comet_logger:
        checkpoint_path = os.path.join(checkpoint_dir, f"{comet_logger.experiment.id}.ckpt")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, "bird_model.ckpt")

    print(f"\nSaving checkpoint to: {checkpoint_path}")
    m.trainer.save_checkpoint(checkpoint_path)

    # Evaluate after training
    print("\nEvaluating model after training...")
    results_after = m.evaluate(csv_file=test_csv, root_dir=args.data_dir, size=800)
    print(f"After training - Box Precision: {results_after['box_precision']:.4f}, "
          f"Box Recall: {results_after['box_recall']:.4f}")

    if comet_logger:
        comet_logger.experiment.log_metric("box_precision_after", results_after["box_precision"])
        comet_logger.experiment.log_metric("box_recall_after", results_after["box_recall"])

        # Log global steps
        global_steps = torch.tensor(m.trainer.global_step, dtype=torch.int32, device=m.device)
        comet_logger.experiment.log_metric("global_steps", global_steps.item())

    print("\nTraining complete!")


if __name__ == "__main__":
    run()

