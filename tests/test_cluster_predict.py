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


def _build_predict_command(output_path: Path) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    input_csv = _required_env("DEEPFOREST_HPC_PREDICT_CSV")
    root_dir = _required_env("DEEPFOREST_HPC_ROOT_DIR")
    nnodes = os.environ.get("DEEPFOREST_HPC_NNODES", os.environ.get("SLURM_NNODES", "1"))
    gpus_per_node = os.environ.get("DEEPFOREST_HPC_GPUS_PER_NODE", "1")

    return [
        "srun",
        f"--nodes={nnodes}",
        f"--ntasks-per-node={gpus_per_node}",
        "bash",
        "-lc",
        (
            f"set -euo pipefail && cd {shlex.quote(str(repo_root))} && "
            f"uv run deepforest predict {shlex.quote(input_csv)} "
            f"--mode csv --root-dir {shlex.quote(root_dir)} "
            f"-o {shlex.quote(str(output_path))} "
            f"--strategy ddp accelerator=gpu devices={shlex.quote(str(gpus_per_node))} "
            f"num_nodes={shlex.quote(str(nnodes))}"
        ),
    ]


@pytest.mark.integration
@pytest.mark.cluster
def test_multinode_predict_cli_on_cluster(tmp_path):
    if os.environ.get("RUN_CLUSTER_TESTS") != "1":
        pytest.skip("Set RUN_CLUSTER_TESTS=1 to enable cluster integration tests")

    if "SLURM_JOB_ID" not in os.environ:
        pytest.skip("This test must run inside a Slurm allocation")

    output_dir = Path(
        os.environ.get("DEEPFOREST_HPC_OUTPUT_DIR", str(tmp_path / "cluster_outputs"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"predict_job_{os.environ['SLURM_JOB_ID']}.csv"
    if output_path.exists():
        output_path.unlink()

    command = _build_predict_command(output_path)
    subprocess.run(command, check=True)

    assert output_path.exists()

    predictions = pd.read_csv(output_path)
    assert not predictions.empty
    assert {"image_path", "xmin", "ymin", "xmax", "ymax", "score", "label"}.issubset(
        predictions.columns
    )
