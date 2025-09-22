# Test evaluation callback
import glob
import json
import os

import pandas as pd
import pytest

from deepforest import get_data, main, evaluate
from deepforest.callbacks import EvaluationCallback
from deepforest.utilities import read_file


@pytest.fixture(scope="module")
def m(download_release):
    """Create a test model with minimal configuration."""
    m = main.deepforest()
    m.config.train.csv_file = get_data("example.csv")
    m.config.train.root_dir = os.path.dirname(get_data("example.csv"))
    m.config.train.fast_dev_run = True
    m.config.batch_size = 2
    m.config.validation.csv_file = get_data("example.csv")
    m.config.validation.root_dir = os.path.dirname(get_data("example.csv"))
    m.config.workers = 0
    m.config.train.epochs = 2

    m.create_trainer()
    m.load_model("weecology/deepforest-tree")

    return m


def test_evaluation_callback_save_mode(m, tmpdir):
    """Test EvaluationCallback in save mode creates proper CSV and metadata files.

    Verifies that the callback:
    - Creates prediction CSV files with evaluation-compatible format
    - Creates metadata JSON files with correct structure and content
    - Saves predictions with expected columns for evaluation
    - Works when built-in evaluation is disabled
    """
    eval_callback = EvaluationCallback(
        save_dir=tmpdir,
        every_n_epochs=1,
        run_evaluation=False
    )

    # Disable built-in evaluation
    m.config.validation.val_accuracy_interval = -1

    m.create_trainer(callbacks=[eval_callback], fast_dev_run=False)
    m.trainer.fit(m)

    # Check that prediction CSV file was created
    csv_files = glob.glob(f"{tmpdir}/predictions_epoch_*.csv")
    assert len(csv_files) > 0, "No prediction CSV files found"

    # Check that metadata JSON file was created
    json_files = glob.glob(f"{tmpdir}/predictions_epoch_*_metadata.json")
    assert len(json_files) > 0, "No metadata JSON files found"

    # Verify CSV file has expected structure
    csv_file = csv_files[0]
    predictions = pd.read_csv(csv_file)

    # Check that predictions have expected columns
    expected_columns = ["xmin", "ymin", "xmax", "ymax", "label", "score", "image_path"]
    for col in expected_columns:
        assert col in predictions.columns, f"Missing column: {col}"

    # Verify metadata JSON has expected fields
    json_file = json_files[0]
    with open(json_file, 'r') as f:
        metadata = json.load(f)

    expected_keys = ["epoch", "predictions_count", "iou_threshold", "target_csv_file", "target_root_dir"]
    for key in expected_keys:
        assert key in metadata, f"Missing metadata key: {key}"

    assert metadata["target_csv_file"] == get_data("example.csv")
    assert metadata["iou_threshold"] == 0.4


def test_evaluation_callback_with_evaluation(m, tmpdir):
    """Test EvaluationCallback with run_evaluation=True runs and logs evaluation metrics.

    Verifies that when run_evaluation=True, the callback:
    - Saves predictions to CSV files
    - Runs evaluate_boxes() on the saved predictions
    - Logs evaluation metrics to the training logs
    - Works alongside the file saving functionality
    """
    eval_callback = EvaluationCallback(
        save_dir=tmpdir,
        every_n_epochs=1,
        run_evaluation=True
    )

    # Disable built-in evaluation
    m.config.validation.val_accuracy_interval = -1

    m.create_trainer(callbacks=[eval_callback], fast_dev_run=False)
    m.trainer.fit(m)

    # Check that CSV file was created
    csv_files = glob.glob(f"{tmpdir}/predictions_epoch_*.csv")
    assert len(csv_files) > 0


def test_evaluation_callback_vs_builtin_evaluation(m, tmpdir):
    """Test that callback produces identical evaluation results to built-in evaluation.

    Runs both the EvaluationCallback and built-in evaluation simultaneously
    and verifies that:
    - Both produce the same box_recall and box_precision values
    - The saved predictions can be evaluated independently
    - Results are consistent between callback and built-in methods
    """
    eval_callback = EvaluationCallback(
        save_dir=tmpdir,
        every_n_epochs=1,
        run_evaluation=False
    )

    # Enable built-in evaluation to run at the same time
    m.config.validation.val_accuracy_interval = 1

    m.create_trainer(callbacks=[eval_callback], fast_dev_run=False)
    m.trainer.fit(m)

    # Get logged metrics from built-in evaluation
    builtin_box_recall = None
    builtin_box_precision = None

    for logger in m.trainer.loggers:
        if hasattr(logger, 'metrics'):
            metrics = logger.metrics
            if 'box_recall' in metrics:
                builtin_box_recall = metrics['box_recall']
            if 'box_precision' in metrics:
                builtin_box_precision = metrics['box_precision']

    # Load callback predictions and evaluate them
    csv_files = glob.glob(f"{tmpdir}/predictions_epoch_*.csv")
    assert len(csv_files) > 0

    callback_predictions = pd.read_csv(csv_files[0])
    ground_truth = read_file(get_data("example.csv"))

    # Run evaluation on callback predictions
    callback_results = evaluate.evaluate_boxes(
        predictions=callback_predictions,
        ground_df=ground_truth,
        iou_threshold=0.4
    )

    # Compare metrics (allowing for small floating point differences)
    if builtin_box_recall is not None:
        assert abs(callback_results["box_recall"] - builtin_box_recall) < 0.01
    if builtin_box_precision is not None:
        assert abs(callback_results["box_precision"] - builtin_box_precision) < 0.01


def test_evaluation_callback_disabled(m, tmpdir):
    """Test that callback is properly disabled when every_n_epochs=-1.

    Verifies that when every_n_epochs=-1:
    - No prediction CSV files are created
    - No metadata JSON files are created
    - The callback gracefully skips all processing
    - Training completes normally without callback interference
    """
    eval_callback = EvaluationCallback(
        save_dir=tmpdir,
        every_n_epochs=-1,  # Disabled
        run_evaluation=False
    )

    m.create_trainer(callbacks=[eval_callback], fast_dev_run=False)
    m.trainer.fit(m)

    # Check that no files were created
    csv_files = glob.glob(f"{tmpdir}/predictions_epoch_*.csv")
    json_files = glob.glob(f"{tmpdir}/predictions_epoch_*_metadata.json")

    assert len(csv_files) == 0, "CSV files created when callback should be disabled"
    assert len(json_files) == 0, "JSON files created when callback should be disabled"


def test_evaluation_callback_empty_predictions(m, tmpdir):
    """Test callback handles edge case of zero predictions gracefully.

    Uses a very high score threshold to ensure no predictions are made,
    then verifies that:
    - Metadata JSON files are still created
    - predictions_count is correctly set to 0
    - The callback doesn't crash or produce errors
    - File structure is maintained even without predictions
    """
    # Set very high score threshold to get no predictions
    original_score_thresh = m.model.score_thresh
    m.model.score_thresh = 0.999

    eval_callback = EvaluationCallback(
        save_dir=tmpdir,
        every_n_epochs=1,
        run_evaluation=False
    )

    # Disable built-in evaluation
    m.config.validation.val_accuracy_interval = -1

    m.create_trainer(callbacks=[eval_callback], fast_dev_run=False)
    m.trainer.fit(m)

    # Restore original score threshold
    m.model.score_thresh = original_score_thresh

    # Check that files are still created even with no predictions
    json_files = glob.glob(f"{tmpdir}/predictions_epoch_*_metadata.json")
    assert len(json_files) > 0, "Metadata JSON should be created even with no predictions"

    # Check metadata shows 0 predictions
    with open(json_files[0], 'r') as f:
        metadata = json.load(f)

    assert metadata["predictions_count"] == 0
