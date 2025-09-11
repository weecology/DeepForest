import os
import subprocess
import sys
from importlib.resources import files

from omegaconf import OmegaConf

from deepforest import get_data

SCRIPT = files("deepforest.scripts").joinpath("cli.py")


def test_train_cli(tmpdir):
    """Check a basic training run, including overrides for unit testing
    see test_main.py fixtures for setup reference."""

    test_labels = get_data("OSBS_029.csv")

    args = [
        sys.executable,
        str(SCRIPT),
        "train",
        "train.fast_dev_run=True",
        f"train.csv_file={test_labels}",
        f"train.root_dir={os.path.dirname(test_labels)}"
    ]

    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"


def test_train_cli_fail(tmpdir):
    """Check that training fails if no dataset paths are provided"""

    args = [
        sys.executable,
        str(SCRIPT),
        "train",
        "train.fast_dev_run=True",
    ]

    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode != 0


def test_train_cli_user_config(tmpdir):
    """Check whether we can provide a custom YAML file for configuration"""

    # Create a modified config
    test_labels = get_data("OSBS_029.csv")
    config = OmegaConf.load(get_data("config.yaml"))
    config.train.csv_file = test_labels
    config.train.root_dir = os.path.dirname(test_labels)
    OmegaConf.save(config, tmpdir.join("user_config.yaml").open('w'))

    # This will fail if the config is not correctly created
    # as the csv/root parameters are not set by default.
    args = [
        sys.executable,
        str(SCRIPT),
        f"--config-dir", tmpdir,
        f"--config-name", "user_config",
        "train",
        "train.fast_dev_run=True"
    ]

    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"


def test_predict_cli(tmp_path):
    """Check we can predict an image and save results"""
    input_path = get_data("OSBS_029.png")
    output_path = tmp_path / "result.csv"
    args = [input_path, "-o", str(output_path)]

    result = subprocess.run(
        [sys.executable, SCRIPT, "predict"] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert output_path.exists(), f"Expected output file not found: {output_path}"


def test_predict_cli_with_opt(tmp_path):
    """Check we can predict an image and save results"""
    input_path = get_data("OSBS_029.png")
    output_path = tmp_path / "result.csv"
    args = [input_path, "-o", str(output_path), "patch_size=250"]

    result = subprocess.run(
        [sys.executable, SCRIPT, "predict"] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert output_path.exists(), f"Expected output file not found: {output_path}"


def test_predict_cli_missing_input(tmp_path):
    # Running the script without any inputs should yield an error
    result = subprocess.run(
        [sys.executable, SCRIPT, "predict"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode != 0


def test_predict_cli_config_help(tmp_path):
    # Script should show config without requiring input
    result = subprocess.run(
        [sys.executable, SCRIPT, "config"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0
    assert len(result.stdout) > 0
