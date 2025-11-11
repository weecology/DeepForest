"""Test basic behavior and edge cases for Spotlight integration.

This module tests fundamental behaviors, error handling, and edge cases
to ensure robust operation of the Spotlight integration.
"""

import pytest
import pandas as pd
from deepforest.visualize import view_with_spotlight


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


def test_width_height_handling():
    """Test handling of optional width/height columns."""
    df = pd.DataFrame({
        "image_path": ["test.jpg"],
        "xmin": [10], "ymin": [10], "xmax": [50], "ymax": [50],
        "width": [1000], "height": [800]
    })

    result = view_with_spotlight(df, format="objects")
    image = result["images"][0]

    assert image["width"] == 1000
    assert image["height"] == 800


def test_mixed_width_height_values():
    """Test handling of mixed valid/invalid width/height values."""
    df = pd.DataFrame({
        "image_path": ["test1.jpg", "test2.jpg"],
        "xmin": [10, 20], "ymin": [10, 20], "xmax": [50, 60], "ymax": [50, 60],
        "width": [1000, None],  # One valid, one NaN
        "height": [None, 800]   # One NaN, one valid
    })

    result = view_with_spotlight(df, format="objects")

    # First image should have width but not height
    img1 = result["images"][0]
    assert img1["width"] == 1000
    assert "height" not in img1

    # Second image should have height but not width
    img2 = result["images"][1]
    assert img2["height"] == 800
    assert "width" not in img2


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


def test_dataframe_accessor_error_handling():
    """Test DataFrame accessor handles errors properly."""
    df = pd.DataFrame()  # Empty DataFrame

    with pytest.raises(ValueError, match="DataFrame is empty"):
        df.spotlight()


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
    import json
    with manifest_file.open() as f:
        file_content = json.load(f)

    assert file_content == result
