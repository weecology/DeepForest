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


def any_across_ranks(flag: bool, device: torch.device | None = None) -> bool:
    """Return True if *flag* is true on any rank; *flag* when not
    distributed."""
    if not is_distributed():
        return bool(flag)

    value = torch.tensor(
        [1 if flag else 0],
        device=device if device is not None else torch.device("cpu"),
        dtype=torch.int64,
    )
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


def sum_across_ranks(value: int, device: torch.device | None = None) -> int:
    """Sum an integer across ranks; return *value* when not distributed."""
    if not is_distributed():
        return int(value)

    tensor = torch.tensor(
        [int(value)],
        device=device if device is not None else torch.device("cpu"),
        dtype=torch.int64,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def mean_metric_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Element-wise mean of metric dicts; skips non-tensor entries like
    classes."""
    if not dicts:
        return {}
    if len(dicts) == 1:
        return dict(dicts[0])

    merged: dict[str, Any] = {}
    for key in dicts[0]:
        if key == "classes":
            continue
        values = [metric_dict[key] for metric_dict in dicts if key in metric_dict]
        if not values:
            continue
        if isinstance(values[0], torch.Tensor):
            merged[key] = torch.stack([value.float() for value in values]).mean(dim=0)
        else:
            merged[key] = values[0]
    return merged


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
