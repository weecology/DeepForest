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


def _is_empty_frame(frame: pd.DataFrame) -> bool:
    coordinates = ["xmin", "ymin", "xmax", "ymax"]
    return bool((frame[coordinates] == 0).all(axis=None))


@pytest.mark.integration
@pytest.mark.cluster
def test_validation_finishes_when_ranks_keep_uneven_nonempty_crops():
    """#1460: after empty-frame filter, ranks can keep 2 vs 1 crops.

    Four images, two GPUs, DistributedSampler shuffle=False:
    rank0 gets images 0 and 2, rank1 gets 1 and 3. With three nonempty
    images then one blank, that is 2 nonempty vs 1 nonempty.
    """
    if os.environ.get("RUN_CLUSTER_TESTS") != "1":
        pytest.skip("Set RUN_CLUSTER_TESTS=1 to enable cluster integration tests")

    if "SLURM_JOB_ID" not in os.environ:
        pytest.skip("This test must run inside a Slurm allocation")

    csv_file = Path(_required_env("DEEPFOREST_HPC_VALIDATION_CSV"))
    root_dir = Path(_required_env("DEEPFOREST_HPC_ROOT_DIR"))
    annotations = pd.read_csv(csv_file)
    image_order = annotations.drop_duplicates("image_path")["image_path"].tolist()
    assert len(image_order) == 4

    empty_by_image = {
        image_path: _is_empty_frame(group)
        for image_path, group in annotations.groupby("image_path", sort=False)
    }
    assert sum(empty_by_image.values()) == 1
    assert sum(not empty for empty in empty_by_image.values()) == 3

    world_size = int(os.environ.get("DEEPFOREST_HPC_GPUS_PER_NODE", "2"))
    assert world_size == 2
    nonempty_per_rank = []
    for rank in range(world_size):
        rank_images = image_order[rank::world_size]
        nonempty_per_rank.append(
            sum(1 for image_path in rank_images if not empty_by_image[image_path])
        )
    assert sorted(nonempty_per_rank) == [1, 2]

    model = deepforest()
    model.config.accelerator = "gpu"
    model.config.devices = world_size
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
