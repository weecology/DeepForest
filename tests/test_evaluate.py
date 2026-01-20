# Test evaluate
# Test IoU
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from deepforest import evaluate
from deepforest import get_data
from deepforest import main
from deepforest.utilities import read_file

from PIL import Image
from shapely.geometry import box


def test_evaluate_image(m):
    csv_file = get_data("OSBS_029.csv")
    predictions = m.predict_file(csv_file=csv_file, root_dir=os.path.dirname(csv_file))
    ground_truth = read_file(csv_file)
    predictions.label = 0
    result = evaluate.evaluate_boxes(predictions=predictions,
                                     ground_df=ground_truth)

    assert result["results"].shape[0] == ground_truth.shape[0]
    assert sum(result["results"].IoU) > 10


def test_evaluate_boxes(m):
    csv_file = get_data("OSBS_029.csv")
    predictions = m.predict_file(csv_file=csv_file, root_dir=os.path.dirname(csv_file))
    predictions.label = "Tree"
    ground_truth = read_file(csv_file)
    predictions = predictions.loc[range(10)]
    results = evaluate.evaluate_boxes(predictions=predictions,
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
    results = evaluate.evaluate_boxes(predictions=predictions,
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
    results = evaluate.evaluate_boxes(predictions=predictions,
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


def test_point_recall_image():
    # create sample dataframes
    predictions = pd.DataFrame({
        "image_path": ["OSBS_029.png", "OSBS_029.png"],
        "xmin": [1, 150],
        "xmax": [100, 200],
        "ymin": [1, 75],
        "ymax": [50, 100],
        "label": ["A", "B"],
    })
    ground_df = pd.DataFrame({
        "image_path": ["OSBS_029.png", "OSBS_029.png"],
        "x": [5, 20],
        "y": [30, 300],
        "label": ["A", "B"],
    })

    # run the function
    result = evaluate._point_recall_image_(predictions, ground_df)

    # check the output, 1 match of 2 ground truth
    assert all(result.predicted_label.isnull().values == [False, True])  # First point inside box, second point outside
    assert isinstance(result, gpd.GeoDataFrame)
    assert "predicted_label" in result.columns
    assert "true_label" in result.columns
    assert "geometry" in result.columns


def test_point_recall():
    # create sample dataframes
    predictions = pd.DataFrame({
        "image_path": ["OSBS_029.png", "OSBS_029.png"],
        "xmin": [1, 150],
        "xmax": [100, 200],
        "ymin": [1, 75],
        "ymax": [50, 100],
        "label": ["A", "B"],
    })
    ground_df = pd.DataFrame({
        "image_path": ["OSBS_029.png", "OSBS_029.png"],
        "x": [5, 20],
        "y": [30, 300],
        "label": ["A", "B"],
    })

    results = evaluate.point_recall(ground_df=ground_df, predictions=predictions)
    assert results["box_recall"] == 0.5
    assert results["class_recall"].recall[0] == 1


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
    results = evaluate.evaluate_boxes(
        predictions=predictions,
        ground_df=ground_truth,
        iou_threshold=0.4
    )

    # Verify results structure
    assert results["results"].shape[0] == 2  # Two ground truth boxes
    assert results["box_recall"] == 0  # No predictions for image1.jpg
    assert all(results["results"].match == False)  # No matches
    assert all(results["results"].prediction_id.isna())  # No prediction IDs


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
