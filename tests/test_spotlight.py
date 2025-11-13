"""Test Spotlight integration for DeepForest."""

import json
import pandas as pd
import pytest

try:
    from deepforest.visualize import view_with_spotlight
    SPOTLIGHT_AVAILABLE = True
except ImportError:
    SPOTLIGHT_AVAILABLE = False

try:
    from deepforest import get_data
    from deepforest.main import deepforest
    DEEPFOREST_AVAILABLE = True
except ImportError:
    DEEPFOREST_AVAILABLE = False


def test_empty_dataframe():
    """Test handling of empty DataFrame."""
    try:
        from deepforest.visualize import view_with_spotlight
    except ImportError:
        pytest.skip("Spotlight functionality not available")

    df = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame is empty"):
        view_with_spotlight(df)


def test_minimal_valid_dataframe():
    """Test with minimal valid DataFrame."""
    try:
        from deepforest.visualize import view_with_spotlight
    except ImportError:
        pytest.skip("Spotlight functionality not available")

    df = pd.DataFrame({
        "image_path": ["test.jpg"],
        "xmin": [10.0], "ymin": [10.0], "xmax": [50.0], "ymax": [50.0]
    })
    result = view_with_spotlight(df, format="objects")

    assert "version" in result
    assert "bbox_format" in result
    assert "images" in result
    assert len(result["images"]) == 1

    image = result["images"][0]
    assert image["file_name"] == "test.jpg"
    assert len(image["annotations"]) == 1

    ann = image["annotations"][0]
    assert ann["bbox"] == [10.0, 10.0, 50.0, 50.0]


def test_dataframe_accessor_error_handling():
    """Test DataFrame accessor handles errors properly."""
    df = pd.DataFrame()  # Empty DataFrame
    with pytest.raises(ValueError, match="DataFrame is empty"):
        df.spotlight()


@pytest.mark.skipif(not DEEPFOREST_AVAILABLE, reason="DeepForest not available")
def test_predictions_have_score_column():
    """Verify prediction results include score column."""
    model = deepforest()
    model.load_model("Weecology/deepforest-tree")
    image_path = get_data("OSBS_029.tif")
    prediction_results = model.predict_image(path=image_path)

    assert "score" in prediction_results.columns
    assert "label" in prediction_results.columns
    assert "xmin" in prediction_results.columns
    assert len(prediction_results) > 0

    # Verify scores are in reasonable range
    scores = prediction_results["score"]
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


@pytest.mark.skipif(not SPOTLIGHT_AVAILABLE, reason="Spotlight functionality not available")
def test_file_output_creation(tmp_path):
    """Test that file output is created correctly."""
    df = pd.DataFrame({
        "image_path": ["test.jpg"],
        "xmin": [10], "ymin": [10], "xmax": [50], "ymax": [50],
        "label": ["Tree"]
    })

    out_dir = tmp_path / "spotlight_output"
    result = view_with_spotlight(df, format="lightly", out_dir=str(out_dir))

    # Check file was created
    manifest_file = out_dir / "manifest.json"
    assert manifest_file.exists()

    # Check file content matches returned result
    with manifest_file.open() as f:
        file_content = json.load(f)

    assert file_content == result


@pytest.mark.skipif(not SPOTLIGHT_AVAILABLE, reason="Spotlight functionality not available")
def test_missing_bbox_column_error():
    """Test error handling for missing required bbox columns."""
    df = pd.DataFrame({
        "image_path": ["test.jpg"],
        "xmin": [10],
        "ymin": [15],
        "xmax": [110]
        # Missing ymax column
    })

    with pytest.raises(ValueError, match="Missing required bbox column: ymax"):
        view_with_spotlight(df, format="objects")
