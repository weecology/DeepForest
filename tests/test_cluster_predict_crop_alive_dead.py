import os
from pathlib import Path

import pandas as pd
import pytest

from deepforest.main import deepforest
from deepforest.model import CropModel


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is not set")
    return value


@pytest.mark.integration
@pytest.mark.cluster
def test_predict_file_labels_both_images_under_ddp():
    """Tree detector + dead/alive crop model on two photos under DDP.

    Each image should appear once with non-null crop labels. The old wrapper
    passed the full prediction frame into every image loop and duplicated rows.
    """
    if os.environ.get("RUN_CLUSTER_TESTS") != "1":
        pytest.skip("Set RUN_CLUSTER_TESTS=1 to enable cluster integration tests")

    if "SLURM_JOB_ID" not in os.environ:
        pytest.skip("This test must run inside a Slurm allocation")

    csv_file = Path(_required_env("DEEPFOREST_HPC_PREDICT_CSV"))
    root_dir = Path(_required_env("DEEPFOREST_HPC_ROOT_DIR"))
    annotations = pd.read_csv(csv_file)
    expected_images = sorted(annotations["image_path"].unique().tolist())
    assert expected_images == ["treeA.jpg", "treeB.jpg"]

    devices = int(os.environ.get("DEEPFOREST_HPC_GPUS_PER_NODE", "2"))
    num_nodes = int(os.environ.get("SLURM_NNODES", "1"))

    detector = deepforest()
    detector.config.score_thresh = 0.3
    detector.load_model("weecology/deepforest-tree")
    detector.config.accelerator = "gpu"
    detector.config.devices = devices
    detector.config.num_nodes = num_nodes
    detector.config.strategy = "ddp"
    detector.config.workers = 0
    detector.config.batch_size = 1
    detector.create_trainer()

    crop = CropModel.load_model("weecology/cropmodel-deadtrees")
    results = detector.predict_file(
        csv_file=str(csv_file),
        root_dir=str(root_dir),
        crop_model=crop,
    )

    if not detector.trainer.is_global_zero:
        return

    assert not results.empty
    present = sorted(results["image_path"].unique().tolist())
    assert present == expected_images
    assert "cropmodel_label" in results.columns
    assert results["cropmodel_label"].notna().all()

    # Fixed wrapper: one row set per image, not one full-frame copy per image.
    counts = results.groupby("image_path").size()
    assert counts.sum() == len(results)
    assert (counts > 0).all()
