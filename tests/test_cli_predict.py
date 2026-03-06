import os
import subprocess
import sys
from importlib.resources import files

import pandas as pd
import pytest

from deepforest import get_data
from deepforest.utilities import load_config
from deepforest.scripts.predict import predict

SCRIPT = files("deepforest.scripts").joinpath("cli.py")


@pytest.fixture()
def predict_config():
    return load_config(
        overrides={
            "validation": {
                "csv_file": get_data("OSBS_029.csv"),
                "root_dir": os.path.dirname(get_data("OSBS_029.csv")),
            },
        }
    )


def test_predict_single_image(predict_config, tmp_path):
    # Test basic prediction
    output_path = tmp_path / "predictions.csv"
    predict(
        predict_config,
        input_path=get_data("OSBS_029.png"),
        output_path=str(output_path),
    )

    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert not df.empty

def test_predict_tile_image(predict_config, tmp_path):
    # Test tiled prediction
    output_path = tmp_path / "predictions.csv"
    predict(
        predict_config,
        input_path=get_data("australia.tif"),
        output_path=str(output_path),
        mode="tile"
    )

    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert not df.empty


def test_predict_missing_input_raises():
    config = load_config()
    with pytest.raises(ValueError, match="No input file provided"):
        predict(config)


def test_predict_csv_uses_root_dir_from_config(tmp_path):
    # Test that when input is a CSV file and no root_dir is provided, the function uses config.validation.root_dir as root_dir
    csv_file = get_data("OSBS_029.csv")
    root_dir = os.path.dirname(csv_file)
    config = load_config(
        overrides={
            "validation": {
                "root_dir": root_dir,
            },
        }
    )
    output_path = tmp_path / "predictions.csv"
    predict(config, input_path=csv_file, output_path=str(output_path), mode="csv")
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert not df.empty


def test_predict_csv_infers_root(tmp_path):
    # Test that when input is a CSV file and no root_dir is provided, the function uses the CSV directory as root_dir
    csv_file = get_data("OSBS_029.csv")
    config = load_config(
        overrides={
            "validation": {
                "csv_file": csv_file,
                "root_dir": None,
            }
        }
    )
    output_path = tmp_path / "predictions.csv"
    predict(config, input_path=csv_file, output_path=str(output_path), mode="csv")
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert not df.empty


def test_predict_with_plot(predict_config, tmp_path, monkeypatch):
    # Check that plotting works (patched to avoid show)
    plot_called = []

    def mock_plot_results(df):
        plot_called.append(df)

    monkeypatch.setattr("deepforest.scripts.predict.plot_results", mock_plot_results)

    output_path = tmp_path / "predictions.csv"
    predict(predict_config, input_path=get_data("OSBS_029.png"), output_path=str(output_path), plot=True)
    assert output_path.exists()
    assert len(plot_called) == 1


def test_cli_predict_subcommand(tmp_path):
    image_path = get_data("OSBS_029.png")
    output_path = tmp_path / "predictions.csv"

    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "predict",
            image_path,
            "-o",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0
    assert output_path.exists()
