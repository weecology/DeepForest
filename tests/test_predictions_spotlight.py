"""Test Spotlight integration with DeepForest predictions.

This module tests the integration between DeepForest model predictions and
Spotlight visualization, ensuring that prediction-specific fields like scores
are properly handled and exported.
"""

import pytest
import pandas as pd
from deepforest import get_data
from deepforest.main import deepforest
from deepforest.visualize import view_with_spotlight


@pytest.fixture
def prediction_results():
    """Generate prediction results for testing."""
    model = deepforest()
    model.load_model("Weecology/deepforest-tree")
    image_path = get_data("OSBS_029.tif")
    return model.predict_image(path=image_path)


def test_predictions_have_score_column(prediction_results):
    """Verify prediction results include score column."""
    assert "score" in prediction_results.columns
    assert "label" in prediction_results.columns
    assert "xmin" in prediction_results.columns
    assert len(prediction_results) > 0

    # Verify scores are in reasonable range
    scores = prediction_results["score"]
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_spotlight_with_predictions_objects_format(prediction_results):
    """Test Spotlight objects format includes prediction scores."""
    result = view_with_spotlight(prediction_results, format="objects")

    assert "version" in result
    assert "bbox_format" in result
    assert "images" in result
    assert len(result["images"]) == 1

    # Check annotations include scores
    annotations = result["images"][0]["annotations"]
    assert len(annotations) > 0

    for ann in annotations:
        assert "bbox" in ann
        assert "label" in ann
        assert "score" in ann
        assert isinstance(ann["score"], float)
        assert 0.0 <= ann["score"] <= 1.0


def test_spotlight_with_predictions_lightly_format(prediction_results):
    """Test Spotlight lightly format includes prediction scores."""
    result = view_with_spotlight(prediction_results, format="lightly")

    assert "samples" in result
    assert "version" in result
    assert "bbox_format" in result
    assert len(result["samples"]) == 1

    # Check annotations include scores
    sample = result["samples"][0]
    assert "annotations" in sample
    annotations = sample["annotations"]
    assert len(annotations) > 0

    for ann in annotations:
        assert "bbox" in ann
        assert "label" in ann
        assert "score" in ann
        assert isinstance(ann["score"], float)


def test_dataframe_accessor_with_predictions(prediction_results):
    """Test DataFrame accessor works with prediction results."""
    result = prediction_results.spotlight(format="objects")

    assert isinstance(result, dict)
    assert "images" in result

    # Verify scores are preserved
    annotations = result["images"][0]["annotations"]
    assert all("score" in ann for ann in annotations)


def test_prediction_workflow_complete():
    """Test the complete prediction → Spotlight workflow."""
    # Load model and make predictions
    model = deepforest()
    model.load_model("Weecology/deepforest-tree")
    image_path = get_data("OSBS_029.tif")
    results = model.predict_image(path=image_path)

    # Test both Spotlight approaches
    spotlight_accessor = results.spotlight()
    spotlight_function = view_with_spotlight(results)

    # Both should work and include scores
    assert isinstance(spotlight_accessor, dict)
    assert isinstance(spotlight_function, dict)

    # Verify both include prediction scores
    for result in [spotlight_accessor, spotlight_function]:
        if "samples" in result:  # lightly format
            annotations = result["samples"][0]["annotations"]
        else:  # objects format
            annotations = result["images"][0]["annotations"]

        assert all("score" in ann for ann in annotations)


def test_score_preservation_accuracy(prediction_results):
    """Test that scores are accurately preserved in conversion."""
    # Use 3 decimal places for comparison to account for float precision
    original_scores = set(round(score, 3) for score in prediction_results["score"])

    result = view_with_spotlight(prediction_results, format="objects")
    converted_scores = set(
        round(ann["score"], 3)
        for ann in result["images"][0]["annotations"]
    )

    # All original scores should be preserved (within reasonable precision)
    assert original_scores == converted_scores


def test_mixed_data_with_and_without_scores():
    """Test handling of mixed data (some with scores, some without)."""
    # Create mixed DataFrame
    df_with_scores = pd.DataFrame({
        "image_path": ["test1.jpg"] * 2,
        "xmin": [10, 20],
        "ymin": [10, 20],
        "xmax": [50, 60],
        "ymax": [50, 60],
        "label": ["Tree", "Tree"],
        "score": [0.8, 0.9]
    })

    df_without_scores = pd.DataFrame({
        "image_path": ["test2.jpg"] * 2,
        "xmin": [15, 25],
        "ymin": [15, 25],
        "xmax": [55, 65],
        "ymax": [55, 65],
        "label": ["Tree", "Tree"]
    })

    # Test each separately
    result_with = view_with_spotlight(df_with_scores, format="objects")
    result_without = view_with_spotlight(df_without_scores, format="objects")

    # With scores should include them
    anns_with = result_with["images"][0]["annotations"]
    assert all("score" in ann for ann in anns_with)

    # Without scores should not include them
    anns_without = result_without["images"][0]["annotations"]
    assert all("score" not in ann for ann in anns_without)
