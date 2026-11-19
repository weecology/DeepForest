import os
from pathlib import Path

import pandas as pd
import pytest

from deepforest.main import deepforest


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is not set")
    return value


@pytest.mark.integration
@pytest.mark.cluster
def test_validation_finishes_when_one_rank_has_only_an_empty_frame():
    if os.environ.get("RUN_CLUSTER_TESTS") != "1":
        pytest.skip("Set RUN_CLUSTER_TESTS=1 to enable cluster integration tests")

    if "SLURM_JOB_ID" not in os.environ:
        pytest.skip("This test must run inside a Slurm allocation")

    csv_file = Path(_required_env("DEEPFOREST_HPC_VALIDATION_CSV"))
    root_dir = Path(_required_env("DEEPFOREST_HPC_ROOT_DIR"))
    annotations = pd.read_csv(csv_file)
    coordinates = ["xmin", "ymin", "xmax", "ymax"]
    empty_by_image = annotations.groupby("image_path")[coordinates].apply(
        lambda frame: bool((frame == 0).all(axis=None))
    )
    assert sorted(empty_by_image.tolist()) == [False, True]

    model = deepforest()
    model.config.accelerator = "gpu"
    model.config.devices = int(os.environ.get("DEEPFOREST_HPC_GPUS_PER_NODE", "2"))
    model.config.num_nodes = int(os.environ.get("SLURM_NNODES", "1"))
    model.config.strategy = "ddp"
    model.config.workers = 0
    model.config.batch_size = 1
    model.config.validation.csv_file = str(csv_file)
    model.config.validation.root_dir = str(root_dir)
    model.config.validation.val_accuracy_interval = 1
    model.create_trainer(num_sanity_val_steps=0)

    results = model.trainer.validate(model)

    if model.trainer.is_global_zero:
        assert results
        assert "iou" in results[0]
