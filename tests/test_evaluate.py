# Test evaluate
# Test IoU
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from deepforest import IoU
from deepforest import evaluate
from deepforest import get_data
from deepforest import main
from deepforest.utilities import read_file

from PIL import Image
from shapely import maximum_inscribed_circle
from shapely.geometry import Point, box



def test_evaluate_image(m):
    csv_file = get_data("OSBS_029.csv")
    predictions = m.predict_file(csv_file=csv_file, root_dir=os.path.dirname(csv_file))
    ground_truth = read_file(csv_file)
    predictions.label = 0  # Model outputs numeric class IDs
    # Use wrapper to handle label conversion (ground_truth has "Tree", predictions have 0)
    result = evaluate.__evaluate_wrapper__(
        predictions=predictions,
        ground_df=ground_truth,
        numeric_to_label_dict={0: "Tree"},
        iou_threshold=0.4,
        geometry_type="box"
    )

    assert result["results"].shape[0] == ground_truth.shape[0]
    assert sum(result["results"].IoU) > 10

    # Verify class-level metrics are computed correctly after label conversion
    assert result["class_recall"] is not None
    assert result["class_recall"].shape[0] == 1  # Single class "Tree"
    assert result["class_recall"]["label"].iloc[0] == "Tree"
    # With matching labels and good IoU, recall and precision should be high
    assert result["class_recall"]["recall"].iloc[0] > 0.5
    assert result["class_recall"]["precision"].iloc[0] > 0.5


def test_evaluate_boxes(m):
    csv_file = get_data("OSBS_029.csv")
    predictions = m.predict_file(csv_file=csv_file, root_dir=os.path.dirname(csv_file))
    predictions.label = "Tree"
    ground_truth = read_file(csv_file)
    predictions = predictions.loc[range(10)]
    results = evaluate.evaluate_geometry(predictions=predictions,
                                      ground_df=ground_truth)

    assert results["results"].shape[0] == ground_truth.shape[0]
    assert results["box_recall"] > 0.1
    assert results["class_recall"].shape == (1, 4)
    assert results["class_recall"].recall.values == 1
    assert results["class_recall"].precision.values == 1
    assert "score" in results["results"].columns
    assert results["results"].true_label.unique() == "Tree"


def test_evaluate_boxes_multiclass():
    csv_file = get_data("testfile_multi.csv")
    ground_truth = read_file(csv_file)
    ground_truth["label"] = ground_truth.label.astype("category").cat.codes

    # Manipulate the data to create some false positives
    predictions = ground_truth.copy()
    predictions["score"] = 1
    predictions.iloc[[36, 35, 34], predictions.columns.get_indexer(['label'])]
    results = evaluate.evaluate_geometry(predictions=predictions,
                                      ground_df=ground_truth)

    assert results["results"].shape[0] == ground_truth.shape[0]
    assert results["class_recall"].shape == (2, 4)


def test_evaluate_boxes_save_images():
    csv_file = get_data("testfile_multi.csv")
    ground_truth = read_file(csv_file)
    ground_truth["label"] = ground_truth.label.astype("category").cat.codes

    # Manipulate the data to create some false positives
    predictions = ground_truth.copy()
    predictions["score"] = 1
    predictions.iloc[[36, 35, 34], predictions.columns.get_indexer(['label'])]
    results = evaluate.evaluate_geometry(predictions=predictions,
                                      ground_df=ground_truth)


def test_evaluate_empty(m, tmp_path):
    # Evaluate with an empty model which should return no predictions.
    m = main.deepforest(config_args={"model": {"name": None},
                                     "label_dict": {"Tree": 0},
                                     "num_classes": 1})
    csv_file = get_data("OSBS_029.csv")
    results = m.evaluate(csv_file, iou_threshold=0.4, root_dir=os.path.dirname(csv_file))

    # Does this make reasonable predictions, we know the model works.
    assert np.isnan(results["box_precision"])
    assert results["box_recall"] == 0


@pytest.fixture
def sample_results():
    # Create a sample DataFrame for testing
    data = {'true_label': [1, 1, 2], 'predicted_label': [1, 2, 1]}
    return pd.DataFrame(data)


def test_compute_class_recall(sample_results):
    # Test case with sample data
    expected_recall = pd.DataFrame({
        'label': [1, 2],
        'recall': [0.5, 0],
        'precision': [0.5, 0],
        'size': [2, 1]
    }).reset_index(drop=True)

    assert evaluate.compute_class_recall(sample_results).equals(expected_recall)


def test_point_recall():
    # Sample dataframe with one correct prediction
    predictions = pd.DataFrame({
        "image_path": ["OSBS_029.png"],
        "x": [1],
        "y": [25],
        "label": ["A"],
    })
    ground_df = pd.DataFrame({
        "image_path": ["OSBS_029.png", "OSBS_029.png"],
        "x": [1.5, 20],
        "y": [30, 300],
        "label": ["A", "B"],
    })

    results = evaluate.evaluate_geometry(ground_df=ground_df,
                                            predictions=predictions,
                                            geometry_type="point")

    # We predicted one of the two points correctly
    assert results["point_recall"] == 0.5
    # The single prediction is correct
    assert results["point_precision"] == 1

    assert isinstance(results["results"], gpd.GeoDataFrame)
    assert "predicted_label" in results["results"].columns
    assert "true_label" in results["results"].columns
    assert "geometry" in results["results"].columns


def test_evaluate_boxes_no_predictions_for_image():
    """Test evaluate_boxes when ground truth exists but no predictions for that image."""

    # Create ground truth with non-default index
    ground_truth = gpd.GeoDataFrame(
        {
            "image_path": ["image1.jpg", "image1.jpg"],
            "label": ["Tree", "Tree"],
            "xmin": [10, 50],
            "ymin": [10, 50],
            "xmax": [30, 70],
            "ymax": [30, 70],
        },
        index=[100, 200]  # Non-default index to trigger the issue
    )
    ground_truth["geometry"] = ground_truth.apply(
        lambda row: box(row["xmin"], row["ymin"], row["xmax"], row["ymax"]), axis=1
    )

    # Create predictions for a different image
    predictions = gpd.GeoDataFrame(
        {
            "image_path": ["image2.jpg"],
            "label": ["Tree"],
            "xmin": [10],
            "ymin": [10],
            "xmax": [30],
            "ymax": [30],
            "score": [0.9],
        }
    )
    predictions["geometry"] = predictions.apply(
        lambda row: box(row["xmin"], row["ymin"], row["xmax"], row["ymax"]), axis=1
    )

    # Should not raise TypeError
    results = evaluate.evaluate_geometry(
        predictions=predictions,
        ground_df=ground_truth,
        iou_threshold=0.4
    )

    # Verify results structure
    assert results["results"].shape[0] == 2  # Two ground truth boxes
    assert results["box_recall"] == 0  # No predictions for image1.jpg
    assert all(results["results"].match == False)  # No matches
    assert all(results["results"].prediction_id.isna())  # No prediction IDs


def test_match_points_box():
    # Check point matching against box ground truth
    # First point falls inside the first box; second point is outside all boxes.
    predictions = gpd.GeoDataFrame(
        {
            "image_path": ["OSBS_029.png", "OSBS_029.png"],
            "label": ["A", "B"],
            "geometry": [box(1, 1, 100, 50), box(150, 75, 200, 100)],
        },
        geometry="geometry",
    )
    ground_df = gpd.GeoDataFrame(
        {
            "image_path": ["OSBS_029.png", "OSBS_029.png"],
            "label": ["A", "B"],
            "geometry": [Point(5, 30), Point(20, 300)],
        },
        geometry="geometry",
    )

    results = IoU.match_points_box(ground_truth=ground_df, submission=predictions)

    assert results.shape[0] == ground_df.shape[0]
    assert results.loc[results.truth_id == 0, "prediction_id"].notna().all()
    assert results.loc[results.truth_id == 1, "prediction_id"].isna().all()
    assert (results.loc[results.truth_id == 0, "predicted_label"] == "A").all()
    assert (results.loc[results.truth_id == 0, "true_label"] == "A").all()
    assert "predicted_label" in results.columns
    assert "true_label" in results.columns


# These thresholds are high, as the image is simple and only contains 4 trees.
@pytest.mark.parametrize(
    "geometry_type,ground_truth_file,thresholds",
    [
        (
            "point",
            "2019_BLAN_3_751000_4330000_image_crop_keypoints.csv",
            {"distance_threshold": 10.0, "min_recall": 0.9, "min_precision": 0.7},
        ),
        (
            "polygon",
            "2019_BLAN_3_751000_4330000_image_crop_polygon.csv",
            {"iou_threshold": 0.2, "min_recall": 0.9, "min_precision": 0.7},
        ),
        (
            "box",
            "2019_BLAN_3_751000_4330000_image_crop_bbox.csv",
            {"iou_threshold": 0.7, "min_recall": 0.9, "min_precision": 0.7},
        ),
    ],
)
def test_evaluate_boxes_as_geometry(m, geometry_type, ground_truth_file, thresholds):
    """Predict boxes then convert to different geometry types and evaluate."""
    image_path = get_data("2019_BLAN_3_751000_4330000_image_crop.jpg")
    predictions = m.predict_image(path=image_path)

    # Coerce geometries to desired type
    if geometry_type == "point":
        predictions["x"] = (predictions["xmin"] + predictions["xmax"]) / 2
        predictions["y"] = (predictions["ymin"] + predictions["ymax"]) / 2
        predictions = predictions[["image_path", "x", "y", "label", "score"]]
    elif geometry_type == "polygon":
        predictions["geometry"] = predictions.apply(
            lambda row: Point(
                (row["xmin"] + row["xmax"]) / 2,
                (row["ymin"] + row["ymax"]) / 2,
            ).buffer(
                min(row["xmax"] - row["xmin"], row["ymax"] - row["ymin"]) / 2
            ),
            axis=1,
        )

    ground_df = read_file(get_data(ground_truth_file))


    results = evaluate.evaluate_geometry(
        predictions=predictions,
        ground_df=ground_df,
        geometry_type=geometry_type,
        iou_threshold=thresholds.get("iou_threshold", 0.4),
        distance_threshold=thresholds.get("distance_threshold", 10.0),
    )

    assert results[f"{geometry_type}_recall"] > thresholds["min_recall"]
    assert results[f"{geometry_type}_precision"] > thresholds["min_precision"]
    assert isinstance(results["results"], gpd.GeoDataFrame)
    assert "predicted_label" in results["results"].columns
    assert "true_label" in results["results"].columns
    assert results["class_recall"] is not None


def test_validate_predictions_match_predict_file(m):
    """trainer.validate should populate predictions equivalent to main.predict_file."""
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)

    # Run validation to populate m.predictions
    m.create_trainer()
    m.trainer.validate(m)

    # Concatenate predictions collected during validation
    if len(m.predictions) > 0:
        val_preds = pd.concat(m.predictions, ignore_index=True)
    else:
        val_preds = pd.DataFrame(
            columns=["image_path", "xmin", "ymin", "xmax", "ymax", "label"])

    # Predict through inference path
    infer_preds = m.predict_file(csv_file=csv_file, root_dir=root_dir)

    assert infer_preds.empty == val_preds.empty
    assert infer_preds.shape == val_preds.shape

    # Compare the first row of the predictions
    assert infer_preds.iloc[0].xmin == val_preds.iloc[0].xmin
    assert infer_preds.iloc[0].ymin == val_preds.iloc[0].ymin
    assert infer_preds.iloc[0].xmax == val_preds.iloc[0].xmax
    assert infer_preds.iloc[0].ymax == val_preds.iloc[0].ymax
    assert infer_preds.iloc[0].label == val_preds.iloc[0].label


def test_validate_predictions_match_predict_file_mixed_sizes(m, tmp_path):
    """trainer.validate should populate predictions equivalent to main.predict_file for mixed-size images."""
    # Prepare two images at different sizes
    src_path = get_data("OSBS_029.tif")
    img = Image.open(src_path).convert("RGB")

    # Create a smaller and a larger variant (bounded to avoid extreme sizes)
    w, h = img.size
    small = img.resize((max(64, w // 2), max(64, h // 2)))
    large = img.resize((min(w * 2, 2 * w), min(h * 2, 2 * h)))

    # Save both to tmp directory as PNGs
    small_name = "mixed_small.png"
    large_name = "mixed_large.png"
    small_path = os.path.join(tmp_path, small_name)
    large_path = os.path.join(tmp_path, large_name)
    small.save(small_path)
    large.save(large_path)

    # Build a CSV with empty annotations (0,0,0,0) for validation compatibility
    # predict_file only needs image_path, but validation requires full annotation format
    csv_path = os.path.join(tmp_path, "mixed_images.csv")
    df = pd.DataFrame({
        "image_path": [small_name, large_name],
        "xmin": [0, 0],
        "ymin": [0, 0],
        "xmax": [0, 0],
        "ymax": [0, 0],
        "label": ["Tree", "Tree"]
    })
    df.to_csv(csv_path, index=False)

    # Configure validation to use the mixed-size images
    m.config.validation.csv_file = csv_path
    m.config.validation.root_dir = str(tmp_path)
    # Don't set validation.size to avoid resizing - use batch_size=1 to handle mixed sizes
    m.config.batch_size = 1

    # Run validation to populate m.predictions
    m.create_trainer()
    m.trainer.validate(m)

    # Concatenate predictions collected during validation
    if len(m.predictions) > 0:
        val_preds = pd.concat(m.predictions, ignore_index=True)
    else:
        val_preds = pd.DataFrame(
            columns=["image_path", "xmin", "ymin", "xmax", "ymax", "label"])

    # Predict through inference path
    # Don't pass size to avoid resizing - coordinates should be in original image space
    infer_preds = m.predict_file(csv_file=csv_path, root_dir=str(tmp_path))

    assert infer_preds.empty == val_preds.empty
    assert infer_preds.shape == val_preds.shape

    # Compare predictions by image_path to ensure matching
    # Sort both dataframes for comparison
    val_preds_sorted = val_preds.sort_values(by=["image_path", "xmin", "ymin"]).reset_index(drop=True)
    infer_preds_sorted = infer_preds.sort_values(by=["image_path", "xmin", "ymin"]).reset_index(drop=True)

    # Compare the first row of the predictions if both are non-empty
    if not val_preds_sorted.empty and not infer_preds_sorted.empty:
        assert infer_preds_sorted.iloc[0].xmin == val_preds_sorted.iloc[0].xmin
        assert infer_preds_sorted.iloc[0].ymin == val_preds_sorted.iloc[0].ymin
        assert infer_preds_sorted.iloc[0].xmax == val_preds_sorted.iloc[0].xmax
        assert infer_preds_sorted.iloc[0].ymax == val_preds_sorted.iloc[0].ymax
        assert infer_preds_sorted.iloc[0].label == val_preds_sorted.iloc[0].label
