#!/usr/bin/env python
"""HPC-only multi-GPU training smoke test (DDP).

Run with:
  torchrun --nproc_per_node=2 tests/hpc_multi_gpu_train.py
"""
from __future__ import annotations

import os
import sys

import torch

from deepforest import get_data
from deepforest.main import deepforest


def _require_hpc() -> None:
    if os.environ.get("GITHUB_ACTIONS") or os.environ.get("CI"):
        raise SystemExit("CI environment detected; skip HPC-only test.")
    if not os.environ.get("HIPERGATOR") and not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit(
            "This script is intended for HPC use only. "
            "Set HIPERGATOR=1 or run under SLURM."
        )


def _require_ddp() -> None:
    if "LOCAL_RANK" not in os.environ and "RANK" not in os.environ:
        raise SystemExit(
            "DDP environment not detected. Run with:\n"
            "  torchrun --nproc_per_node=2 tests/hpc_multi_gpu_train.py"
        )


def main() -> int:
    _require_hpc()
    _require_ddp()

    if torch.cuda.device_count() < 2:
        raise SystemExit("Need at least 2 GPUs for this test.")

    m = deepforest()
    m.config.workers = 0
    m.config.batch_size = 1
    m.config.num_classes = 1
    m.config.label_dict = {"Tree": 0}
    train_csv = get_data("example.csv")
    m.config.train.csv_file = train_csv
    m.config.train.root_dir = os.path.dirname(train_csv)
    m.config.validation.csv_file = train_csv
    m.config.validation.root_dir = os.path.dirname(train_csv)
    m.create_model(initialize_model=True)

    # Keep this fast but avoid fast_dev_run's zero-length warning in DDP.
    m.create_trainer(
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        fast_dev_run=False,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        log_every_n_steps=1,
    )
    m.trainer.fit(m)

    # Multi-GPU evaluation pass (uses same example.csv)
    m.trainer.validate(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
