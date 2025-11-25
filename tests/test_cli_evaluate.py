import os
import subprocess
import sys
from importlib.resources import files

import pandas as pd
import pytest

from deepforest import get_data
from deepforest.utilities import load_config
from deepforest.scripts.evaluate import evaluate

SCRIPT = files("deepforest.scripts").joinpath("cli.py")


@pytest.fixture()
def evaluate_config():
    return load_config(
        overrides={
            "validation": {
                "csv_file": get_data("OSBS_029.csv"),
                "root_dir": os.path.dirname(get_data("OSBS_029.csv")),
            },
        }
    )


def test_evaluate_with_ground_truth(evaluate_config, tmp_path):
    output_path = tmp_path / "results.csv"
    evaluate(
        evaluate_config,
        ground_truth=get_data("OSBS_029.csv"),
        root_dir=os.path.dirname(get_data("OSBS_029.csv")),
        output_path=str(output_path),
    )
    assert output_path.exists()
    results = pd.read_csv(output_path)
    assert len(results) > 0


def test_evaluate_saves_output(evaluate_config, tmp_path):
    output_path = tmp_path / "eval_results.csv"
    evaluate(
        evaluate_config,
        ground_truth=get_data("OSBS_029.csv"),
        root_dir=os.path.dirname(get_data("OSBS_029.csv")),
        output_path=str(output_path),
    )
    assert output_path.exists()
    results = pd.read_csv(output_path)
    assert "metric" in results.columns
    assert "value" in results.columns
    assert len(results) > 0


def test_evaluate_missing_input_raises():
    config = load_config()
    with pytest.raises(ValueError, match="No CSV file provided"):
        evaluate(config)


def test_evaluate_uses_validation_csv_from_config(evaluate_config, tmp_path):
    output_path = tmp_path / "results.csv"
    evaluate(evaluate_config, ground_truth=None, output_path=str(output_path))
    assert output_path.exists()
    results = pd.read_csv(output_path)
    assert len(results) > 0


def test_evaluate_uses_root_dir_from_config(tmp_path):
    csv_file = get_data("OSBS_029.csv")
    root_dir = os.path.dirname(csv_file)
    config = load_config(
        overrides={
            "validation": {
                "root_dir": root_dir,
            },
        }
    )
    output_path = tmp_path / "results.csv"
    evaluate(config, ground_truth=csv_file, root_dir=None, output_path=str(output_path))
    assert output_path.exists()
    results = pd.read_csv(output_path)
    assert len(results) > 0


def test_evaluate_saves_predictions(tmp_path):
    csv_file = get_data("OSBS_029.csv")
    root_dir = os.path.dirname(csv_file)
    config = load_config()
    predictions_path = tmp_path / "saved_predictions.csv"
    evaluate(
        config,
        ground_truth=csv_file,
        root_dir=root_dir,
        save_predictions=str(predictions_path),
    )
    assert predictions_path.exists()
    predictions = pd.read_csv(predictions_path)
    assert "xmin" in predictions.columns
    assert "ymin" in predictions.columns
    assert len(predictions) > 0


def test_evaluate_with_existing_predictions(tmp_path):
    csv_file = get_data("OSBS_029.csv")
    root_dir = os.path.dirname(csv_file)
    config = load_config()

    predictions_path = tmp_path / "predictions.csv"
    evaluate(
        config,
        ground_truth=csv_file,
        root_dir=root_dir,
        save_predictions=str(predictions_path),
    )
    assert predictions_path.exists()

    output_path = tmp_path / "eval_results.csv"
    evaluate(
        config,
        ground_truth=csv_file,
        root_dir=root_dir,
        predictions=str(predictions_path),
        output_path=str(output_path),
    )
    assert output_path.exists()


def test_cli_evaluate_subcommand(tmp_path):
    csv_file = get_data("OSBS_029.csv")
    root_dir = os.path.dirname(csv_file)
    output_path = tmp_path / "cli_eval.csv"

    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "evaluate",
            csv_file,
            "--root-dir",
            root_dir,
            "-o",
            str(output_path),
            f"validation.csv_file={csv_file}",
            f"validation.root_dir={root_dir}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0
    assert output_path.exists()
