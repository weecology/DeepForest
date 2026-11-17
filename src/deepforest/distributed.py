"""Helpers for distributed-safe logging and object gathering."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def is_distributed() -> bool:
    """Return True when torch.distributed is initialized."""
    return dist.is_available() and dist.is_initialized()


def is_global_zero(trainer: Any | None = None) -> bool:
    """Return True on the global-zero rank."""
    if trainer is not None:
        return bool(getattr(trainer, "is_global_zero", True))

    if not is_distributed():
        return True

    return dist.get_rank() == 0


def should_sync(trainer: Any | None = None) -> bool:
    """Return True when metrics should be synchronized across ranks."""
    if trainer is not None:
        world_size = getattr(trainer, "world_size", 1)
        return world_size is not None and world_size > 1

    return is_distributed()


def get_rank() -> int:
    """Return the current distributed rank, defaulting to zero."""
    if not is_distributed():
        return 0

    return dist.get_rank()


def get_world_size() -> int:
    """Return the distributed world size, defaulting to one."""
    if not is_distributed():
        return 1

    return dist.get_world_size()


class FixedOrderSampler(Sampler[int]):
    """Yield a fixed sequence of dataset indices."""

    def __init__(self, indices: list[int]):
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def gather_object(obj: Any) -> list[Any]:
    """Gather a Python object from every rank."""
    if not is_distributed():
        return [obj]

    gathered = [None] * get_world_size()
    dist.all_gather_object(gathered, obj)
    return gathered


def _move_to_device(value: Any, device: torch.device) -> Any:
    """Move tensors onto one device.

    all_gather_object keeps each rank's GPU.
    """
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def compute_metric_across_ranks(metric: Any) -> dict[str, Any]:
    """Compute a detection metric when ranks saw different numbers of crops.

    TorchMetrics stores one tensor per crop and gathers those tensors
    one at a time. A rank that skipped an empty crop sends fewer tensors
    than a rank that did not, and the gather does not finish. Gather
    each whole list once, move every tensor onto this rank's device,
    then compute from the combined crops.
    """
    if not is_distributed():
        return metric.compute()

    sync_was = metric._to_sync
    metric._to_sync = False
    saw_crops = False
    try:
        for name in list(metric._reductions):
            value = getattr(metric, name)
            if not isinstance(value, list):
                continue
            gathered = gather_object(value)
            flat = _move_to_device(
                [item for part in gathered for item in part],
                metric.device,
            )
            setattr(metric, name, flat)
            saw_crops = saw_crops or bool(flat)
        if not saw_crops:
            return {}
        if metric._update_count == 0:
            metric._update_count = 1
        return metric.compute()
    finally:
        metric._to_sync = sync_was


def gather_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Gather pandas DataFrames from every rank."""
    gathered = gather_object(frame)
    non_empty_frames = [
        item for item in gathered if isinstance(item, pd.DataFrame) and not item.empty
    ]

    if non_empty_frames:
        return pd.concat(non_empty_frames, ignore_index=True)

    if isinstance(frame, pd.DataFrame):
        return frame.iloc[0:0].copy()

    return pd.DataFrame()
