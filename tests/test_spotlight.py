"""Test Spotlight integration for DeepForest.

Consolidated tests for Spotlight visualization functionality including
basic behavior, predictions, export, and adapter functionality.
"""

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Import with try/except to handle missing dependencies gracefully
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from deepforest import get_data
    from deepforest.main import deepforest
    DEEPFOREST_AVAILABLE = True
except ImportError:
    DEEPFOREST_AVAILABLE = False

try:
    from deepforest.visualize import export_to_gallery, view_with_spotlight
    SPOTLIGHT_AVAILABLE = True
except ImportError:
    SPOTLIGHT_AVAILABLE = False


# Test fixtures
@pytest.fixture
def sample_df():
    """Sample DataFrame for testing."""
    return pd.DataFrame([
        {"image_path": "images/img_0001.png", "xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1, "label": "tree", "score": 0.9, "width": 1, "height": 1},
        {"image_path": "images/img_0002.png", "xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1, "label": "tree", "score": 0.8, "width": 1, "height": 1},
    ])


@pytest.fixture
def prediction_results():
    """Generate prediction results for testing."""
    if not DEEPFOREST_AVAILABLE:
        pytest.skip("DeepForest not available")

    model = deepforest()
    model.load_model("Weecology/deepforest-tree")
    image_path = get_data("OSBS_029.tif")
    return model.predict_image(path=image_path)


# Basic behavior tests
def test_empty_dataframe():
    """Test handling of empty DataFrame."""
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame is empty"):
        view_with_spotlight(df)


def test_missing_required_columns():
    """Test error handling for missing required columns."""
    # Missing image reference column
    df = pd.DataFrame({
        "xmin": [10], "ymin": [10], "xmax": [50], "ymax": [50], "label": ["Tree"]
    })
    with pytest.raises(ValueError, match="DataFrame must contain an image reference column"):
        view_with_spotlight(df)

    # Missing bbox columns
    df = pd.DataFrame({
        "image_path": ["test.jpg"], "label": ["Tree"]
    })
    with pytest.raises(ValueError, match="Missing required bbox column"):
        view_with_spotlight(df)


def test_invalid_format():
    """Test error handling for invalid format parameter."""
    df = pd.DataFrame({
        "image_path": ["test.jpg"],
        "xmin": [10], "ymin": [10], "xmax": [50], "ymax": [50],
        "label": ["Tree"]
    })
    with pytest.raises(ValueError, match="Unsupported format: invalid"):
        view_with_spotlight(df, format="invalid")


def test_minimal_valid_dataframe():
    """Test with minimal valid DataFrame."""
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
    assert "label" not in ann  # No label provided
    assert "score" not in ann  # No score provided


def test_different_image_reference_columns():
    """Test different column names for image references."""
    test_cases = [
        ("image_path", "test1.jpg"),
        ("file_name", "test2.jpg"),
        ("source_image", "test3.jpg"),
        ("image", "test4.jpg")
    ]

    for col_name, image_name in test_cases:
        df = pd.DataFrame({
            col_name: [image_name],
            "xmin": [10], "ymin": [10], "xmax": [50], "ymax": [50],
            "label": ["Tree"]
        })
        result = view_with_spotlight(df, format="objects")
        assert result["images"][0]["file_name"] == image_name


def test_multiple_images():
    """Test handling multiple images in one DataFrame."""
    df = pd.DataFrame({
        "image_path": ["img1.jpg", "img1.jpg", "img2.jpg"],
        "xmin": [10, 20, 30], "ymin": [10, 20, 30],
        "xmax": [50, 60, 70], "ymax": [50, 60, 70],
        "label": ["Tree", "Tree", "Bird"]
    })
    result = view_with_spotlight(df, format="objects")

    assert len(result["images"]) == 2

    # Check first image has 2 annotations
    img1 = next(img for img in result["images"] if img["file_name"] == "img1.jpg")
    assert len(img1["annotations"]) == 2

    # Check second image has 1 annotation
    img2 = next(img for img in result["images"] if img["file_name"] == "img2.jpg")
    assert len(img2["annotations"]) == 1
    assert img2["annotations"][0]["label"] == "Bird"


def test_nan_values_handling():
    """Test handling of NaN values in optional columns."""
    df = pd.DataFrame({
        "image_path": ["test.jpg", "test.jpg"],
        "xmin": [10, 20], "ymin": [10, 20], "xmax": [50, 60], "ymax": [50, 60],
        "label": ["Tree", None],  # One NaN label
        "score": [0.8, None]      # One NaN score
    })
    result = view_with_spotlight(df, format="objects")
    annotations = result["images"][0]["annotations"]

    # First annotation should have both label and score
    ann1 = annotations[0]
    assert ann1["label"] == "Tree"
    assert ann1["score"] == 0.8

    # Second annotation should not have label or score (NaN values excluded)
    ann2 = annotations[1]
    assert "label" not in ann2
    assert "score" not in ann2


def test_format_consistency():
    """Test that both formats produce consistent data."""
    df = pd.DataFrame({
        "image_path": ["test.jpg"],
        "xmin": [10], "ymin": [10], "xmax": [50], "ymax": [50],
        "label": ["Tree"], "score": [0.85]
    })

    objects_result = view_with_spotlight(df, format="objects")
    lightly_result = view_with_spotlight(df, format="lightly")

    # Both should have same basic structure
    assert objects_result["version"] == lightly_result["version"]
    assert objects_result["bbox_format"] == lightly_result["bbox_format"]

    # Check data consistency
    objects_ann = objects_result["images"][0]["annotations"][0]
    lightly_ann = lightly_result["samples"][0]["annotations"][0]

    assert objects_ann["bbox"] == lightly_ann["bbox"]
    assert objects_ann["label"] == lightly_ann["label"]
    assert objects_ann["score"] == lightly_ann["score"]


# Adapter tests
def test_view_with_spotlight_lightly(sample_df, tmp_path):
    """Test lightly format output."""
    out = view_with_spotlight(sample_df, format="lightly")

    assert isinstance(out, dict)
    assert "samples" in out
    assert len(out["samples"]) == 2

    # Test file output
    out_dir = tmp_path / "pkg"
    out_dir.mkdir()
    res = view_with_spotlight(sample_df, format="lightly", out_dir=str(out_dir))
    manifest_file = out_dir / "manifest.json"
    assert manifest_file.exists()
    with open(manifest_file, "r", encoding="utf8") as fh:
        loaded = json.load(fh)
    assert "samples" in loaded


def test_dataframe_spotlight_accessor(sample_df, tmp_path):
    """Test DataFrame accessor functionality."""
    out_direct = view_with_spotlight(sample_df, format="lightly")
    out_accessor = sample_df.spotlight(format="lightly")
    assert out_direct == out_accessor

    # Test file output via accessor
    out_dir = tmp_path / "pkg_accessor"
    out_dir.mkdir()
    _ = sample_df.spotlight(format="lightly", out_dir=str(out_dir))
    manifest_file = out_dir / "manifest.json"
    assert manifest_file.exists()
    with open(manifest_file, "r", encoding="utf8") as fh:
        loaded = json.load(fh)
    assert loaded == out_direct


def test_dataframe_accessor_error_handling():
    """Test DataFrame accessor handles errors properly."""
    df = pd.DataFrame()  # Empty DataFrame
    with pytest.raises(ValueError, match="DataFrame is empty"):
        df.spotlight()


# Predictions tests
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


# Export tests
def test_prepare_spotlight_package(tmp_path):
    """Test preparing a Spotlight package from gallery directory."""
    # Create a small test image
    img_path = tmp_path / "img1.png"
    Image.new("RGB", (64, 64), color=(100, 120, 140)).save(img_path)

    # Build dataframe-like input
    df = pd.DataFrame([
        {"image_path": img_path.name, "xmin": 1, "ymin": 1, "xmax": 20, "ymax": 20, "label": "Tree", "score": 0.9}
    ])
    df.root_dir = str(tmp_path)

    gallery_dir = tmp_path / "gallery"
    export_to_gallery(df, str(gallery_dir), root_dir=None, max_crops=10)

    from deepforest.visualize.spotlight_export import prepare_spotlight_package

    out = tmp_path / "spot_pkg"
    res = prepare_spotlight_package(gallery_dir, out_dir=out)
    assert Path(res["manifest"]).exists()

    # Check images dir populated
    images_dir = out / "images"
    assert images_dir.exists()
    assert any(images_dir.iterdir())

    # Verify manifest content
    with open(res["manifest"]) as f:
        manifest = json.load(f)
    assert "version" in manifest
    assert "bbox_format" in manifest
    assert "images" in manifest
    assert len(manifest["images"]) > 0


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
