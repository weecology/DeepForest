import os
import shlex
import subprocess
from pathlib import Path

import pandas as pd
import pytest


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is not set")
    return value


def _build_predict_tile_command(output_path: Path) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    tile_path = _required_env("DEEPFOREST_HPC_TILE_PATH")
    model_name = os.environ.get(
        "DEEPFOREST_HPC_TILE_MODEL_NAME",
        "weecology/everglades-bird-species-detector",
    )
    nnodes = os.environ.get("DEEPFOREST_HPC_NNODES", os.environ.get("SLURM_NNODES", "1"))
    gpus_per_node = os.environ.get("DEEPFOREST_HPC_GPUS_PER_NODE", "1")
    patch_size = os.environ.get("DEEPFOREST_HPC_TILE_PATCH_SIZE", "1500")
    patch_overlap = os.environ.get("DEEPFOREST_HPC_TILE_PATCH_OVERLAP", "0")
    iou_threshold = os.environ.get("DEEPFOREST_HPC_TILE_IOU_THRESHOLD", "0.15")
    dataloader_strategy = os.environ.get("DEEPFOREST_HPC_TILE_DATALOADER_STRATEGY", "window")
    master_port = os.environ.get("DEEPFOREST_HPC_MASTER_PORT", "29500")

    bash_script = f"""
set -euo pipefail
cd {shlex.quote(str(repo_root))}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
uv run torchrun \\
  --nnodes={shlex.quote(str(nnodes))} \\
  --nproc_per_node={shlex.quote(str(gpus_per_node))} \\
  --node_rank=$SLURM_NODEID \\
  --master_addr="$MASTER_ADDR" \\
  --master_port={shlex.quote(str(master_port))} \\
  tests/hipergator_predict_tile_driver.py \\
  --input-path {shlex.quote(tile_path)} \\
  --output-path {shlex.quote(str(output_path))} \\
  --model-name {shlex.quote(str(model_name))} \\
  --patch-size {shlex.quote(str(patch_size))} \\
  --patch-overlap {shlex.quote(str(patch_overlap))} \\
  --iou-threshold {shlex.quote(str(iou_threshold))} \\
  --dataloader-strategy {shlex.quote(str(dataloader_strategy))} \\
  --devices {shlex.quote(str(gpus_per_node))} \\
  --num-nodes {shlex.quote(str(nnodes))}
""".strip()

    return [
        "srun",
        f"--nodes={nnodes}",
        f"--ntasks={nnodes}",
        "--ntasks-per-node=1",
        "bash",
        "-lc",
        bash_script,
    ]


@pytest.mark.integration
@pytest.mark.hipergator
def test_multinode_predict_tile_on_hipergator(tmp_path):
    if os.environ.get("RUN_HIPERGATOR_TESTS") != "1":
        pytest.skip("Set RUN_HIPERGATOR_TESTS=1 to enable Hipergator integration tests")

    if "SLURM_JOB_ID" not in os.environ:
        pytest.skip("This test must run inside a Slurm allocation")

    output_dir = Path(
        os.environ.get("DEEPFOREST_HPC_OUTPUT_DIR", str(tmp_path / "hipergator_outputs"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"predict_tile_job_{os.environ['SLURM_JOB_ID']}.csv"
    if output_path.exists():
        output_path.unlink()

    command = _build_predict_tile_command(output_path)
    subprocess.run(command, check=True)

    assert output_path.exists()

    predictions = pd.read_csv(output_path)
    assert not predictions.empty
    assert {"image_path", "xmin", "ymin", "xmax", "ymax", "score", "label"}.issubset(
        predictions.columns
    )
