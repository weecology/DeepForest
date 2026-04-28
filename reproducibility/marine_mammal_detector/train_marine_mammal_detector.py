"""Train and optionally publish a marine mammal DeepForest detector.

This script is intentionally lightweight for internal reproducibility:
- uses a dedicated config preset
- trains on train/test CSVs
- optionally evaluates on a zero-shot CSV
- optionally exports/pushes Hugging Face compatible weights
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from hydra import compose, initialize
from omegaconf import OmegaConf

from deepforest.conf.schema import Config as StructuredConfig
from deepforest.main import deepforest
from deepforest.scripts.train import train as train_script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train marine mammal detector.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--zero-shot-csv", type=Path)
    parser.add_argument("--experiment-name", default="marine-mammal-detector")
    parser.add_argument("--strategy", default="ddp")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--comet", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--hub-repo",
        type=str,
        help="Optional Hugging Face repo, e.g. weecology/deepforest-marine-mammal.",
    )
    parser.add_argument(
        "--hub-token-env",
        default="HF_TOKEN",
        help="Environment variable used by huggingface_hub for auth.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace):
    with initialize(version_base=None, config_path="pkg://deepforest.conf"):
        base_cfg = compose(config_name="marine_mammal_detector")

    structured = OmegaConf.structured(StructuredConfig)
    cfg = OmegaConf.merge(structured, base_cfg)
    cfg.train.csv_file = str(args.train_csv)
    cfg.validation.csv_file = str(args.test_csv)
    cfg.train.root_dir = str(args.root_dir)
    cfg.validation.root_dir = str(args.root_dir)
    cfg.log_root = str(args.log_root)
    return cfg


def latest_run_dir(log_root: Path, experiment_name: str) -> Path:
    parent = log_root / experiment_name
    if not parent.exists():
        raise FileNotFoundError(f"No run directory found in {parent}")
    run_dirs = sorted([p for p in parent.iterdir() if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No timestamped runs found in {parent}")
    return run_dirs[-1]


def pick_checkpoint(run_dir: Path) -> Path:
    checkpoint_dir = run_dir / "checkpoints"
    last_ckpt = checkpoint_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt
    ckpts = sorted(checkpoint_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    return ckpts[-1]


def run_zero_shot_eval(checkpoint: Path, config, zero_shot_csv: Path, output_csv: Path):
    model = deepforest.load_from_checkpoint(str(checkpoint))
    model.config = config
    model.config.validation.csv_file = str(zero_shot_csv)
    model.config.validation.root_dir = config.validation.root_dir
    model.config.validation.val_accuracy_interval = 1
    model.create_trainer()
    metrics = model.trainer.validate(model)

    rows = metrics if isinstance(metrics, list) else [metrics]
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved zero-shot metrics to {output_csv}")


def push_to_hub(checkpoint: Path, repo_id: str, hub_token_env: str) -> None:
    model = deepforest.load_from_checkpoint(str(checkpoint))
    model.label_dict = {"Object": 0}
    model.numeric_to_label_dict = {0: "Object"}
    token = None
    if hub_token_env:
        from os import environ

        token = environ.get(hub_token_env)
    model.model.push_to_hub(repo_id=repo_id, token=token)
    print(f"Pushed model weights to https://huggingface.co/{repo_id}")


def main() -> None:
    args = parse_args()
    args.log_root.mkdir(parents=True, exist_ok=True)
    config = build_config(args)

    success = train_script(
        config=config,
        checkpoint=True,
        comet=args.comet,
        tensorboard=args.tensorboard,
        resume=str(args.resume) if args.resume else None,
        strategy=args.strategy,
        experiment_name=args.experiment_name,
        tags=["marine-mammal", "reproducible"],
    )
    if not success:
        raise RuntimeError("Training did not complete successfully.")

    run_dir = latest_run_dir(args.log_root, args.experiment_name)
    checkpoint = pick_checkpoint(run_dir)
    print(f"Using checkpoint: {checkpoint}")

    if args.zero_shot_csv:
        run_zero_shot_eval(
            checkpoint=checkpoint,
            config=config,
            zero_shot_csv=args.zero_shot_csv,
            output_csv=run_dir / "zero_shot_metrics.csv",
        )

    if args.hub_repo:
        push_to_hub(
            checkpoint=checkpoint,
            repo_id=args.hub_repo,
            hub_token_env=args.hub_token_env,
        )


if __name__ == "__main__":
    main()
