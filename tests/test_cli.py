import os
import subprocess
import sys
from importlib.resources import files

from omegaconf import OmegaConf
import pytest

from deepforest import get_data
from deepforest.utilities import load_config
from deepforest.scripts.train import train
from deepforest.scripts.predict import predict
from deepforest.scripts.evaluate import evaluate


SCRIPT = files("deepforest.scripts").joinpath("cli.py")

@pytest.fixture
def config():
    """Load default config for testing, load default train/val data"""
    return load_config(overrides={"train": {"fast_dev_run": True,
                                            "csv_file": get_data("OSBS_029.csv"),
                                            "root_dir": os.path.dirname(get_data("OSBS_029.csv"))},
                                  "validation": {"csv_file": get_data("OSBS_029.csv"),
                                                 "root_dir": os.path.dirname(get_data("OSBS_029.csv"))}})
def test_train_cli_direct(config):
    """Direct call train for coverage"""
    assert train(config)

def test_evaluate_cli_direct(tmpdir, config):
    """Direct call evaluate for coverage"""
    evaluate(config,
             ground_truth=get_data("OSBS_029.csv"),
             root_dir=os.path.dirname(get_data("OSBS_029.csv")),
             output_path=tmpdir / "eval_results.csv")

    assert os.path.exists(tmpdir / "eval_results.csv"), "Expected output file not found"

def test_predict_cli_direct(tmpdir, config):
    """Direct call predict for coverage"""
    prediction_path = tmpdir / "OSBS_029_predictions.csv"
    predict(config,
                   input_path=get_data("OSBS_029.png"),
                   output_path=prediction_path)

    assert os.path.exists(prediction_path), f"Expected output file not found: {prediction_path}"

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


def test_evaluate_cli(tmp_path):
    """Check basic evaluation with generated predictions"""
    test_labels = get_data("OSBS_029.csv")

    args = [
        sys.executable,
        str(SCRIPT),
        "evaluate",
        test_labels,
        f"--root-dir={os.path.dirname(test_labels)}",
        "train.fast_dev_run=True",
    ]

    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert "Evaluation Results:" in result.stdout


def test_evaluate_cli_with_output(tmp_path):
    """Check evaluation with output CSV file"""
    test_labels = get_data("OSBS_029.csv")
    output_path = tmp_path / "eval_results.csv"

    args = [
        sys.executable,
        str(SCRIPT),
        "evaluate",
        test_labels,
        f"--root-dir={os.path.dirname(test_labels)}",
        "-o", str(output_path),
        "train.fast_dev_run=True",
    ]

    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert output_path.exists(), f"Expected output file not found: {output_path}"


def test_evaluate_cli_missing_input(tmp_path):
    """Check that evaluation fails if no CSV file provided"""
    args = [
        sys.executable,
        str(SCRIPT),
        "evaluate",
    ]

    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode != 0
