"""Helpers for distributed-safe logging and object gathering."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch.distributed as dist


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


def gather_object(obj: Any) -> list[Any]:
    """Gather a Python object from every rank."""
    if not is_distributed():
        return [obj]

    gathered = [None] * dist.get_world_size()
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
