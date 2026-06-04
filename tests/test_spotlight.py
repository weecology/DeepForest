"""Test Spotlight integration for DeepForest."""

import json
import pandas as pd
import pytest
from unittest.mock import patch

from deepforest import get_data
from deepforest.main import deepforest
from deepforest.visualize import view_with_spotlight
import deepforest.visualize.spotlight_adapter as spotlight_adapter
from deepforest.visualize.spotlight_adapter import prepare_spotlight_package

REAL_IMAGE_PATH = get_data("OSBS_029.tif")


# Test helper functions
def create_sample_dataframe(num_annotations=1, include_optional=False):
    """Create a sample DataFrame for testing."""
    data = {
        "image_path": [REAL_IMAGE_PATH] * num_annotations,
        "xmin": [10 + i * 50 for i in range(num_annotations)],
        "ymin": [10 + i * 50 for i in range(num_annotations)],
        "xmax": [50 + i * 50 for i in range(num_annotations)],
        "ymax": [50 + i * 50 for i in range(num_annotations)],
        "label": ["Tree"] * num_annotations,
        "score": [0.95 - i * 0.05 for i in range(num_annotations)],
    }

    if include_optional:
        data.update({
            "width": [500] * num_annotations,
            "height": [400] * num_annotations,
        })

    return pd.DataFrame(data)


def create_test_gallery(gallery_dir, format_type="csv", num_annotations=1):
    """Create a test gallery directory with metadata."""
    gallery_dir.mkdir(exist_ok=True)

    df = create_sample_dataframe(num_annotations)

    if format_type == "csv":
        metadata_file = gallery_dir / "detections.csv"
        df.to_csv(metadata_file, index=False)
    elif format_type == "json":
        metadata_file = gallery_dir / "annotations.json"
        with open(metadata_file, "w", encoding="utf8") as f:
            json.dump(df.to_dict("records"), f)

    return metadata_file


def test_empty_dataframe():
    """Test handling of empty DataFrame."""
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame is empty"):
        view_with_spotlight(df)


def test_minimal_valid_dataframe():
    """Test with minimal valid DataFrame."""
    df = pd.DataFrame({
        "image_path": [REAL_IMAGE_PATH],
        "xmin": [10.0], "ymin": [10.0], "xmax": [50.0], "ymax": [50.0]
    })
    result = view_with_spotlight(df, format="objects")

    assert "version" in result
    assert "bbox_format" in result
    assert "images" in result
    assert len(result["images"]) == 1

    image = result["images"][0]
    assert image["file_name"] == REAL_IMAGE_PATH
    assert len(image["annotations"]) == 1

    annotation = image["annotations"][0]
    assert annotation["bbox"] == [10.0, 10.0, 50.0, 50.0]


def test_dataframe_accessor_error_handling():
    """Test DataFrame accessor handles errors properly."""
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame is empty"):
        df.spotlight()


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

    scores = prediction_results["score"]
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_file_output_creation(tmp_path):
    """Test that file output is created correctly."""
    df = pd.DataFrame({
        "image_path": [REAL_IMAGE_PATH],
        "xmin": [10], "ymin": [10], "xmax": [50], "ymax": [50],
        "label": ["Tree"]
    })

    out_dir = tmp_path / "spotlight_output"
    result = view_with_spotlight(df, format="lightly", out_dir=str(out_dir))

    manifest_file = out_dir / "manifest.json"
    assert manifest_file.exists()

    with manifest_file.open(encoding="utf8") as f:
        file_content = json.load(f)

    assert file_content == result


def test_missing_bbox_column_error():
    """Test error handling for missing required bbox columns."""
    df = pd.DataFrame({
        "image_path": [REAL_IMAGE_PATH],
        "xmin": [10],
        "ymin": [15],
        "xmax": [110]
    })

    with pytest.raises(ValueError, match="Missing required bbox column: ymax"):
        view_with_spotlight(df, format="objects")


def test_spotlight_launch_parameter_without_launch():
    """Test that basic conversion works without launching."""
    df = pd.DataFrame({
        "image_path": [REAL_IMAGE_PATH],
        "xmin": [10], "ymin": [10], "xmax": [50], "ymax": [50],
        "label": ["Tree"]
    })

    result = view_with_spotlight(df, format="objects")
    assert "version" in result


def test_spotlight_integration_basic():
    """Test basic Spotlight integration functionality."""
    df = pd.DataFrame({
        "image_path": [REAL_IMAGE_PATH],
        "xmin": [10], "ymin": [10], "xmax": [50], "ymax": [50],
        "label": ["Tree"], "score": [0.95]
    })

    result = view_with_spotlight(df, format="objects")

    assert "version" in result
    assert "images" in result
    assert len(result["images"]) == 1

    image = result["images"][0]
    assert image["file_name"] == REAL_IMAGE_PATH
    assert len(image["annotations"]) == 1

    annotation = image["annotations"][0]
    assert annotation["bbox"] == [10.0, 10.0, 50.0, 50.0]
    assert annotation["label"] == "Tree"
    assert annotation["score"] == 0.95


def test_dataframe_accessor_launch_forwards_to_spotlight_launcher():
    """Test that the accessor forwards launch requests to the launcher helper."""
    df = pd.DataFrame({
        "image_path": [REAL_IMAGE_PATH],
        "xmin": [10], "ymin": [10], "xmax": [50], "ymax": [50],
        "label": ["Tree"], "score": [0.95]
    })

    with patch.object(
        spotlight_adapter, "_launch_spotlight_from_manifest"
    ) as mock_launch:
        result = df.spotlight(format="objects", launch=True, port=9999, host="127.0.0.1")

    mock_launch.assert_called_once()

    call_args = mock_launch.call_args
    assert isinstance(call_args[0][0], dict)
    assert call_args[1]["port"] == 9999
    assert call_args[1]["host"] == "127.0.0.1"
    assert result["images"][0]["annotations"][0]["label"] == "Tree"


def test_launch_spotlight_from_manifest_formats_dataframe():
    """Test that the internal launcher converts Spotlight manifests correctly."""
    pytest.importorskip("renumics.spotlight")

    manifest = {
        "version": "1.0",
        "bbox_format": "pixels",
        "images": [
            {
                "file_name": REAL_IMAGE_PATH,
                "annotations": [
                    {
                        "bbox": [10.0, 10.0, 50.0, 50.0],
                        "label": "Tree",
                        "score": 0.95,
                    }
                ],
            }
        ],
    }

    with patch("renumics.spotlight.show") as mock_show:
        spotlight_adapter._launch_spotlight_from_manifest(
            manifest, port=9999, host="127.0.0.1"
        )

    mock_show.assert_called_once()

    call_args = mock_show.call_args
    spotlight_df = call_args[0][0]

    assert isinstance(spotlight_df, pd.DataFrame)
    assert list(spotlight_df.columns) == [
        "file_name",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "bbox_width",
        "bbox_height",
        "label",
        "score",
    ]
    assert len(spotlight_df) == 1
    assert spotlight_df["label"].iloc[0] == "Tree"
    assert spotlight_df["score"].iloc[0] == 0.95
    assert call_args[1]["port"] == 9999
    assert call_args[1]["host"] == "127.0.0.1"

# Gallery/Packaging Tests
def test_prepare_spotlight_package_valid_gallery(tmp_path):
    """Test packaging a valid gallery for Spotlight with CSV metadata."""

    gallery_dir = tmp_path / "gallery"
    gallery_dir.mkdir()

    metadata_df = pd.DataFrame({
        "image_path": [REAL_IMAGE_PATH],
        "xmin": [10], "ymin": [10], "xmax": [50], "ymax": [50],
        "label": ["Tree"], "score": [0.95]
    })
    metadata_file = gallery_dir / "detections.csv"
    metadata_df.to_csv(metadata_file, index=False)

    # Call prepare_spotlight_package
    out_dir = tmp_path / "spotlight_output"
    result = prepare_spotlight_package(gallery_dir, out_dir=out_dir)

    assert result["gallery_dir"] == str(gallery_dir)
    assert result["out_dir"] == str(out_dir)
    assert result["metadata_file"] == str(metadata_file)
    assert result["num_images"] > 0
    assert result["format"] == "lightly"


    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()

    with open(manifest_path, encoding="utf8") as f:
        manifest = json.load(f)

    assert "samples" in manifest
    assert manifest["version"] == "1.0"
    assert manifest["bbox_format"] == "pixels"


def test_prepare_spotlight_package_missing_dir():
    """Test FileNotFoundError when gallery directory doesn't exist."""
    with pytest.raises(FileNotFoundError, match="Gallery directory not found"):
        prepare_spotlight_package("/nonexistent/gallery", out_dir="/tmp/out")


def test_prepare_spotlight_package_no_metadata(tmp_path):
    """Test error when gallery has no metadata files (CSV/JSON)."""

    gallery_dir = tmp_path / "empty_gallery"
    gallery_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No metadata files"):
        prepare_spotlight_package(gallery_dir, out_dir=tmp_path / "output")

def test_prepare_spotlight_package_csv_input(tmp_path):
    """Test reading and converting CSV metadata from gallery."""
    gallery_dir = tmp_path / "gallery"
    gallery_dir.mkdir()

    metadata_df = pd.DataFrame({
        "image_path": [REAL_IMAGE_PATH, REAL_IMAGE_PATH],
        "xmin": [10, 100], "ymin": [10, 100], "xmax": [50, 150], "ymax": [50, 150],
        "label": ["Tree", "Tree"], "score": [0.95, 0.87]
    })
    metadata_file = gallery_dir / "predictions.csv"
    metadata_df.to_csv(metadata_file, index=False)

    out_dir = tmp_path / "output"
    result = prepare_spotlight_package(gallery_dir, out_dir=out_dir)

    assert result["metadata_file"] == str(metadata_file)
    assert result["num_images"] == 1


    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, encoding="utf8") as f:
        manifest = json.load(f)


    assert len(manifest["samples"]) == 1
    assert len(manifest["samples"][0]["annotations"]) == 2
    first_annotation = manifest["samples"][0]["annotations"][0]
    assert first_annotation["score"] == 0.95


def test_prepare_spotlight_package_json_input(tmp_path):
    """Test reading and converting JSON metadata from gallery."""
    gallery_dir = tmp_path / "gallery"
    gallery_dir.mkdir()

    metadata = [
        {
            "image_path": REAL_IMAGE_PATH,
            "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50,
            "label": "Tree", "score": 0.92
        }
    ]
    metadata_file = gallery_dir / "annotations.json"
    with open(metadata_file, "w", encoding="utf8") as f:
        json.dump(metadata, f)

    out_dir = tmp_path / "output"
    result = prepare_spotlight_package(gallery_dir, out_dir=out_dir)


    assert result["metadata_file"] == str(metadata_file)
    assert result["num_images"] == 1  # One annotation

    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()

    with open(manifest_path, encoding="utf8") as f:
        manifest = json.load(f)
    assert len(manifest["samples"]) == 1


def test_prepare_spotlight_package_multiple_images(tmp_path):
    """Test packaging gallery with multiple annotations grouped correctly."""
    gallery_dir = tmp_path / "gallery"
    gallery_dir.mkdir()

    metadata_df = pd.DataFrame({
        "image_path": [REAL_IMAGE_PATH, REAL_IMAGE_PATH, REAL_IMAGE_PATH],
        "xmin": [10, 100, 200],
        "ymin": [10, 100, 200],
        "xmax": [50, 150, 250],
        "ymax": [50, 150, 250],
        "label": ["Tree", "Tree", "Tree"],
        "score": [0.95, 0.87, 0.91]
    })
    metadata_file = gallery_dir / "detections.csv"
    metadata_df.to_csv(metadata_file, index=False)

    out_dir = tmp_path / "output"
    result = prepare_spotlight_package(gallery_dir, out_dir=out_dir)

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, encoding="utf8") as f:
        manifest = json.load(f)

    samples = manifest["samples"]
    assert len(samples) == 1
    assert len(samples[0]["annotations"]) == 3


# Format Conversion Tests
def test_view_with_spotlight_format_lightly():
    """Test conversion to Lightly format."""
    df = pd.DataFrame({
        "image_path": [REAL_IMAGE_PATH],
        "xmin": [10], "ymin": [10], "xmax": [50], "ymax": [50],
        "label": ["Tree"], "score": [0.95]
    })

    result = view_with_spotlight(df, format="lightly")

    assert "samples" in result
    assert "version" in result
    assert "bbox_format" in result

    sample = result["samples"][0]
    assert sample["file_name"] == REAL_IMAGE_PATH
    assert "metadata" in sample
    assert "annotations" in sample

    annotation = sample["annotations"][0]
    assert annotation["bbox"] == [10.0, 10.0, 50.0, 50.0]
    assert annotation["category_id"] == "Tree"
    assert annotation["label"] == "Tree"
    assert annotation["score"] == 0.95


def test_view_with_spotlight_with_optional_columns():
    """Test handling of optional width/height columns."""
    df = pd.DataFrame({
        "image_path": [REAL_IMAGE_PATH],
        "xmin": [10], "ymin": [10], "xmax": [50], "ymax": [50],
        "label": ["Tree"], "score": [0.95],
        "width": [500], "height": [400]
    })

    result = view_with_spotlight(df, format="objects")

    image = result["images"][0]
    assert image["width"] == 500
    assert image["height"] == 400


def test_view_with_spotlight_missing_label():
    """Test handling of missing label column (should use default or omit)."""
    df = pd.DataFrame({
        "image_path": [REAL_IMAGE_PATH],
        "xmin": [10], "ymin": [10], "xmax": [50], "ymax": [50],
        "score": [0.95]
    })

    result = view_with_spotlight(df, format="objects")

    annotation = result["images"][0]["annotations"][0]
    assert "label" not in annotation  # Label is optional


def test_view_with_spotlight_multiple_images():
    """Test handling multiple images in single DataFrame."""
    df = pd.DataFrame({
        "image_path": [REAL_IMAGE_PATH, REAL_IMAGE_PATH],
        "xmin": [10, 100], "ymin": [10, 100], "xmax": [50, 150], "ymax": [50, 150],
        "label": ["Tree", "Tree"], "score": [0.95, 0.88]
    })

    result = view_with_spotlight(df, format="objects")

    assert len(result["images"]) == 1
    assert len(result["images"][0]["annotations"]) == 2


def test_view_with_spotlight_unsupported_format():
    """Test error handling for unsupported format."""
    df = pd.DataFrame({
        "image_path": [REAL_IMAGE_PATH],
        "xmin": [10], "ymin": [10], "xmax": [50], "ymax": [50]
    })

    with pytest.raises(ValueError, match="Unsupported format"):
        view_with_spotlight(df, format="unsupported_format")


def test_dataframe_accessor_with_file_output(tmp_path):
    """Test DataFrame accessor with file output."""
    df = pd.DataFrame({
        "image_path": [REAL_IMAGE_PATH],
        "xmin": [10], "ymin": [10], "xmax": [50], "ymax": [50],
        "label": ["Tree"]
    })

    out_dir = tmp_path / "output"
    result = df.spotlight(format="objects", out_dir=str(out_dir))

    assert "version" in result
    manifest_file = out_dir / "manifest.json"
    assert manifest_file.exists()
