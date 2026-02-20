import os

import pytest
import torch

from deepforest import get_data
from deepforest.main import deepforest


@pytest.mark.skipif(
    not os.environ.get("HIPERGATOR"),
    reason="Only run on HIPERGATOR (requires GPU + model downloads).",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available in this test environment.",
)
def test_predict_tile_uses_cuda_when_requested():
    """Ensure predict_tile runs on CUDA when accelerator/devices request GPU.

    This is a regression test to catch silent CPU fallbacks on GPU nodes.
    """
    m = deepforest(config_args={"accelerator": "gpu", "devices": 1, "workers": 0})
    m.load_model(model_name="weecology/deepforest-tree", revision="main")
    m.create_trainer(accelerator="gpu", devices=1)

    results = m.predict_tile(
        path=get_data("OSBS_029.png"),
        patch_size=400,
        patch_overlap=0.0,
        iou_threshold=0.15,
        dataloader_strategy="single",
    )
    assert results is not None and not results.empty

    # Assert trainer is actually using a GPU accelerator (no silent CPU fallback).
    assert m.trainer is not None
    accel_name = type(m.trainer.accelerator).__name__.lower()
    assert "cuda" in accel_name or "gpu" in accel_name
