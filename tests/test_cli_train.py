import os
from pathlib import Path
import re
import subprocess
import sys
from importlib.resources import files

import pytest
from omegaconf import OmegaConf

from deepforest import get_data
from deepforest.utilities import load_config
from deepforest.scripts.train import train

SCRIPT = files("deepforest.scripts").joinpath("cli.py")


@pytest.fixture()
def train_config(tmp_path):
    return load_config(
        overrides={
            "train": {
                "epochs": 1,
                "csv_file": get_data("OSBS_029.csv"),
                "root_dir": os.path.dirname(get_data("OSBS_029.csv")),
            },
            "validation": {
                "csv_file": get_data("OSBS_029.csv"),
                "root_dir": os.path.dirname(get_data("OSBS_029.csv")),
            },
            "log_root": str(tmp_path),
        }
    )


def test_train_creates_artifacts(train_config, tmp_path):
    assert train(train_config, tensorboard=True, checkpoint=True)
    log_root = Path(train_config["log_root"])

    log_dirs = [d for d in log_root.iterdir() if d.is_dir()]
    assert len(log_dirs) == 1
    log_dir = log_dirs[0]

    assert re.match(r"(version_)?\d{8}_\d{6}$", log_dir.name)

    assert (log_dir / "config.yaml").exists()
    assert (log_dir / "checkpoints").is_dir()
    assert (log_dir / "images").is_dir()

    tb_dirs = list(log_root.glob("*/tensorboard"))
    assert len(tb_dirs) == 1

    tfevents = list(tb_dirs[0].glob("events.out.tfevents.*"))
    assert len(tfevents) >= 1

    checkpoints = list(log_root.glob("*/checkpoints/*.ckpt"))
    assert len(checkpoints) >= 1


def test_train_without_checkpoint(train_config, tmp_path):
    assert train(train_config, checkpoint=False)

    log_root = Path(train_config["log_root"])
    checkpoints = list(log_root.glob("*/checkpoints/*.ckpt"))
    assert len(checkpoints) == 0


def test_train_experiment_name(train_config, tmp_path):
    assert train(train_config, checkpoint=False, experiment_name="my-experiment")
    log_root = Path(train_config["log_root"])

    # The experiment name should be used as the logger's subdirectory name
    assert (log_root / "my-experiment").is_dir()


def test_train_experiment_versions(train_config, tmp_path):
    """Running train twice with the same experiment name should produce two version dirs."""
    assert train(train_config, checkpoint=False, experiment_name="repeated")
    assert train(train_config, checkpoint=False, experiment_name="repeated")

    log_root = Path(train_config["log_root"])
    versions = sorted((log_root / "repeated").iterdir())
    assert len(versions) == 2
    assert versions[0].name != versions[1].name


def test_cli_train_subcommand(tmp_path):
    data_csv = get_data("OSBS_029.csv")
    data_root = os.path.dirname(data_csv)

    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "train",
            "--disable-checkpoint",
            f"log_root={tmp_path}",
            f"train.csv_file={data_csv}",
            f"train.root_dir={data_root}",
            f"validation.csv_file={data_csv}",
            f"validation.root_dir={data_root}",
            "train.fast_dev_run=true",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0


def test_cli_train_experiment_name(tmp_path):
    data_csv = get_data("OSBS_029.csv")
    data_root = os.path.dirname(data_csv)

    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "train",
            "--disable-checkpoint",
            "--experiment-name",
            "cli-test-name",
            f"log_root={tmp_path}",
            f"train.csv_file={data_csv}",
            f"train.root_dir={data_root}",
            f"validation.csv_file={data_csv}",
            f"validation.root_dir={data_root}",
            "train.fast_dev_run=true",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0
    assert (tmp_path / "cli-test-name").is_dir()
