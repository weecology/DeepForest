"""Train an Alive/Dead CropModel on a split made by make_split.py.

Optionally starts from another checkpoint (pretraining -> fine-tuning).
Keeps the checkpoint with the lowest validation loss, then evaluates it on the
test split with evaluate_cropmodel.py.

Usage:
    python train_cropmodel.py --split-dir <root>/splits/neon_hand_v1 \
        --out <root>/runs/baseline_resnet50 --epochs 40
"""

import argparse
import json
from pathlib import Path

import torch
from evaluate_cropmodel import main as evaluate
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from deepforest.model import CropModel


def build_model(args: argparse.Namespace) -> CropModel:
    """Create a CropModel from config overrides, optionally from a checkpoint."""
    config_args = {
        "architecture": args.architecture,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "balance_classes": args.balance_classes,
    }
    m = CropModel(config_args=config_args)
    m.load_from_disk(
        train_dir=str(args.split_dir / "train"), val_dir=str(args.split_dir / "val")
    )
    if args.init_from:
        src = (
            CropModel.load_from_checkpoint(args.init_from, map_location="cpu")
            if args.init_from.endswith(".ckpt")
            else CropModel.load_model(args.init_from)
        )
        assert src.label_dict == m.label_dict, (src.label_dict, m.label_dict)
        m.model.load_state_dict(src.model.state_dict())
    return m


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2)
    )

    m = build_model(args)
    checkpoint = ModelCheckpoint(
        dirpath=args.out, filename="best", monitor="val_loss", mode="min", save_last=True
    )
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=CSVLogger(args.out, name="logs"),
        callbacks=[checkpoint, LearningRateMonitor()],
        log_every_n_steps=10,
    )
    trainer.fit(m)
    evaluate(args.split_dir, "val", checkpoint.best_model_path, args.out / "eval")
    evaluate(args.split_dir, "test", checkpoint.best_model_path, args.out / "eval")
    for extra in args.extra_eval:
        evaluate(
            extra, "test", checkpoint.best_model_path, args.out / f"eval_{extra.name}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--architecture", default="resnet50")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--balance-classes", action="store_true")
    parser.add_argument(
        "--init-from", default=None, help=".ckpt or HF repo id to start from"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--extra-eval",
        type=Path,
        nargs="*",
        default=[],
        help="other split dirs whose test crops are also evaluated (e.g. benchmarks)",
    )
    main(parser.parse_args())
