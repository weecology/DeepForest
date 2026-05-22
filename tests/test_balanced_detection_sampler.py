"""Tests for balanced hard-negative batch sampling during detection training."""

import math
import os
import shutil

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from deepforest import get_data
from deepforest.datasets.training import BalancedDetectionBatchSampler, BoxDataset


def _make_mixed_csv(tmp_path, n_positive: int, n_negative: int) -> tuple[str, str]:
    """Build a CSV with n_positive annotated images and n_negative empty images."""
    root_dir = str(tmp_path / "images")
    os.makedirs(root_dir, exist_ok=True)
    source_image = get_data("OSBS_029.png")
    rows = []
    for i in range(n_positive):
        image_name = f"positive_{i}.png"
        shutil.copy(source_image, os.path.join(root_dir, image_name))
        rows.append(
            pd.DataFrame(
                {
                    "image_path": [image_name],
                    "xmin": [10],
                    "ymin": [10],
                    "xmax": [50],
                    "ymax": [50],
                    "label": ["Tree"],
                }
            )
        )
    for i in range(n_negative):
        image_name = f"negative_{i}.png"
        shutil.copy(source_image, os.path.join(root_dir, image_name))
        rows.append(
            pd.DataFrame(
                {
                    "image_path": [image_name],
                    "xmin": [0],
                    "ymin": [0],
                    "xmax": [0],
                    "ymax": [0],
                    "label": ["Tree"],
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    csv_path = str(tmp_path / "mixed.csv")
    df.to_csv(csv_path, index=False)
    return csv_path, root_dir


def test_balanced_batch_sampler_composition(tmp_path):
    """Each batch has a fixed positive/negative count for the given fraction."""
    csv_path, root_dir = _make_mixed_csv(tmp_path, n_positive=4, n_negative=20)
    ds = BoxDataset(csv_file=csv_path, root_dir=root_dir, label_dict={"Tree": 0})
    batch_size = 8
    fraction = 0.75
    generator = torch.Generator().manual_seed(0)
    sampler = BalancedDetectionBatchSampler(
        positive_indices=ds.positive_indices,
        negative_indices=ds.negative_indices,
        batch_size=batch_size,
        positive_batch_fraction=fraction,
        generator=generator,
    )
    n_positive = min(batch_size, max(1, round(batch_size * fraction)))
    n_negative = batch_size - n_positive
    positive_set = set(ds.positive_indices)
    negative_set = set(ds.negative_indices)

    for batch in sampler:
        assert len(batch) == batch_size
        assert sum(idx in positive_set for idx in batch) == n_positive
        assert sum(idx in negative_set for idx in batch) == n_negative


def test_balanced_sampler_caps_empty_per_batch(tmp_path):
    """Balanced sampling limits empty images per batch under heavy negative skew."""
    csv_path, root_dir = _make_mixed_csv(tmp_path, n_positive=1, n_negative=50)
    ds = BoxDataset(csv_file=csv_path, root_dir=root_dir, label_dict={"Tree": 0})
    batch_size = 8
    fraction = 0.75
    n_positive = min(batch_size, max(1, round(batch_size * fraction)))
    max_empty_balanced = batch_size - n_positive

    balanced = BalancedDetectionBatchSampler(
        positive_indices=ds.positive_indices,
        negative_indices=ds.negative_indices,
        batch_size=batch_size,
        positive_batch_fraction=fraction,
        generator=torch.Generator().manual_seed(1),
    )
    negative_set = set(ds.negative_indices)
    for batch in balanced:
        empty_count = sum(idx in negative_set for idx in batch)
        assert empty_count <= max_empty_balanced

    max_empty_seen = 0
    for _ in range(20):
        perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(1))
        for start in range(0, len(ds), batch_size):
            batch_idx = perm[start : start + batch_size].tolist()
            empty_count = sum(idx in negative_set for idx in batch_idx)
            max_empty_seen = max(max_empty_seen, empty_count)
    assert max_empty_seen > max_empty_balanced


def test_balanced_sampler_epoch_length(tmp_path):
    """Epoch length is one pass over positive images when balancing is enabled."""
    csv_path, root_dir = _make_mixed_csv(tmp_path, n_positive=5, n_negative=30)
    ds = BoxDataset(csv_file=csv_path, root_dir=root_dir, label_dict={"Tree": 0})
    batch_size = 4
    fraction = 0.75
    sampler = BalancedDetectionBatchSampler(
        positive_indices=ds.positive_indices,
        negative_indices=ds.negative_indices,
        batch_size=batch_size,
        positive_batch_fraction=fraction,
    )
    n_positive = min(batch_size, max(1, round(batch_size * fraction)))
    expected_len = math.ceil(len(ds.positive_indices) / n_positive)
    assert len(sampler) == expected_len

    loader = DataLoader(ds, batch_sampler=sampler)
    assert len(loader) == expected_len


def test_balanced_sampler_requires_both_pools():
    with pytest.raises(ValueError, match="positive_indices must not be empty"):
        BalancedDetectionBatchSampler(
            positive_indices=[],
            negative_indices=[0],
            batch_size=4,
            positive_batch_fraction=0.75,
        )
