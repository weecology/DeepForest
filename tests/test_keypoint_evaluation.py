"""Focused tests for keypoint distance metrics and evaluation."""
import numpy as np
import pandas as pd
import pytest
import geopandas as gpd
from shapely.geometry import Point

from deepforest import keypoint_distance, evaluate


# Helper to reduce Point(x, y) repetition
def pts(coords):
    """Convert [(x, y), ...] to [Point(x, y), ...]."""
    return [Point(x, y) for x, y in coords]


# ============================================================================
# Distance Computation Tests
# ============================================================================

def test_compute_distances_perfect_match():
    """Test distance computation with perfectly matching keypoints."""
    points = [(100, 100), (200, 200)]
    predictions = gpd.GeoDataFrame({"geometry": pts(points), "score": [0.9, 0.8]})
    ground_truth = gpd.GeoDataFrame({"geometry": pts(points)})

    result = keypoint_distance.compute_distances(ground_truth, predictions)

    assert len(result) == 2
    assert all(result["distance"] < 0.01)
    assert all(result["prediction_id"].notna())
    assert list(result["score"]) == [0.9, 0.8]


def test_compute_distances_known_distances():
    """Test distance computation returns correct Euclidean distances (3-4-5 triangle)."""
    result = keypoint_distance.compute_distances(
        gpd.GeoDataFrame({"geometry": pts([(3, 4)])}),
        gpd.GeoDataFrame({"geometry": pts([(0, 0)]), "score": [0.9]})
    )
    assert result["distance"].iloc[0] == pytest.approx(5.0)


def test_compute_distances_optimal_matching():
    """Test Hungarian algorithm finds optimal assignment, not greedy."""
    result = keypoint_distance.compute_distances(
        gpd.GeoDataFrame({"geometry": pts([(200, 201), (100, 101)])}),
        gpd.GeoDataFrame({"geometry": pts([(100, 100), (200, 200)]), "score": [0.9, 0.8]})
    )
    assert all(result["distance"] < 2.0)


def test_compute_distances_more_predictions():
    """Test matching when there are more predictions than ground truth."""
    result = keypoint_distance.compute_distances(
        gpd.GeoDataFrame({"geometry": pts([(100, 100), (200, 200)])}),
        gpd.GeoDataFrame({"geometry": pts([(i*100, i*100) for i in range(4)]), "score": [0.9, 0.8, 0.7, 0.6]})
    )
    assert len(result) == 2
    assert result["prediction_id"].notna().sum() == 2


def test_compute_distances_more_ground_truth():
    """Test matching when there are more ground truth than predictions."""
    result = keypoint_distance.compute_distances(
        gpd.GeoDataFrame({"geometry": pts([(i*100, i*100) for i in range(4)])}),
        gpd.GeoDataFrame({"geometry": pts([(0, 0), (100, 100)]), "score": [0.9, 0.8]})
    )
    assert len(result) == 4
    assert result["prediction_id"].notna().sum() == 2
    assert result["prediction_id"].isna().sum() == 2
    assert all(result[result["prediction_id"].isna()]["distance"] == np.inf)


def test_compute_distances_empty_predictions():
    """Test handling of empty predictions."""
    result = keypoint_distance.compute_distances(
        gpd.GeoDataFrame({"geometry": pts([(100, 100), (200, 200)])}),
        gpd.GeoDataFrame({"geometry": pts([]), "score": []})
    )
    assert len(result) == 0


def test_compute_distances_empty_ground_truth():
    """Test handling of empty ground truth."""
    result = keypoint_distance.compute_distances(
        gpd.GeoDataFrame({"geometry": pts([])}),
        gpd.GeoDataFrame({"geometry": pts([(100, 100)]), "score": [0.9]})
    )
    assert len(result) == 0


def test_compute_distances_without_scores():
    """Test matching works even without score column."""
    result = keypoint_distance.compute_distances(
        gpd.GeoDataFrame({"geometry": pts([(100, 100), (200, 200)])}),
        gpd.GeoDataFrame({"geometry": pts([(100, 100), (200, 200)])})
    )
    assert len(result) == 2
    assert all(result["score"].isna())


# ============================================================================
# Image-level Keypoint Evaluation Tests
# ============================================================================

def test_evaluate_image_keypoints_perfect_match():
    """Test image-level evaluation with perfect matches."""
    points = [(100, 100), (200, 200)]
    predictions = gpd.GeoDataFrame({
        "geometry": pts(points),
        "score": [0.9, 0.8],
        "label": ["Tree", "Bird"],
        "image_path": ["img1.jpg", "img1.jpg"]
    })
    ground_truth = gpd.GeoDataFrame({
        "geometry": pts(points),
        "label": ["Tree", "Bird"],
        "image_path": ["img1.jpg", "img1.jpg"]
    })

    result = evaluate.evaluate_image_keypoints(predictions, ground_truth)

    assert len(result) == 2
    assert all(result["distance"] < 0.01)
    assert list(result["predicted_label"]) == ["Tree", "Bird"]
    assert list(result["true_label"]) == ["Tree", "Bird"]


def test_evaluate_image_keypoints_label_mapping():
    """Test that labels are correctly mapped from indices."""
    result = evaluate.evaluate_image_keypoints(
        gpd.GeoDataFrame({"geometry": pts([(100, 100)]), "score": [0.9], "label": ["Tree"], "image_path": ["img1.jpg"]}),
        gpd.GeoDataFrame({"geometry": pts([(100, 100)]), "label": ["Bird"], "image_path": ["img1.jpg"]})
    )
    assert result["predicted_label"].iloc[0] == "Tree"
    assert result["true_label"].iloc[0] == "Bird"


def test_evaluate_image_keypoints_multiple_images_error():
    """Test that function raises error with multiple images."""
    with pytest.raises(ValueError, match="More than one plot"):
        evaluate.evaluate_image_keypoints(
            gpd.GeoDataFrame({"geometry": pts([(100, 100), (200, 200)]), "label": ["Tree", "Tree"], "image_path": ["img1.jpg", "img2.jpg"]}),
            gpd.GeoDataFrame({"geometry": pts([(100, 100)]), "label": ["Tree"], "image_path": ["img1.jpg"]})
        )


# ============================================================================
# Full Keypoint Evaluation Tests
# ============================================================================

def test_evaluate_keypoints_recall_precision():
    """Test recall and precision calculation with pixel threshold."""
    predictions = gpd.GeoDataFrame({
        "geometry": pts([(100, 100), (205, 205), (350, 350)]),
        "score": [0.9, 0.8, 0.7],
        "label": ["Tree", "Tree", "Tree"],
        "image_path": ["img1.jpg", "img1.jpg", "img1.jpg"]
    })
    ground_truth = gpd.GeoDataFrame({
        "geometry": pts([(100, 100), (200, 200)]),
        "label": ["Tree", "Tree"],
        "image_path": ["img1.jpg", "img1.jpg"]
    })

    result = evaluate.evaluate_keypoints(predictions, ground_truth, pixel_threshold=10.0)

    assert result["recall"] == 1.0
    assert result["precision"] == pytest.approx(2/3)
    assert result["results"]["match"].sum() == 2


def test_evaluate_keypoints_threshold_filtering():
    """Test that pixel threshold correctly filters matches."""
    predictions = gpd.GeoDataFrame({
        "geometry": pts([(100, 100), (250, 250)]),
        "score": [0.9, 0.8],
        "label": ["Tree", "Tree"],
        "image_path": ["img1.jpg", "img1.jpg"]
    })
    ground_truth = gpd.GeoDataFrame({
        "geometry": pts([(100, 100), (200, 200)]),
        "label": ["Tree", "Tree"],
        "image_path": ["img1.jpg", "img1.jpg"]
    })

    assert evaluate.evaluate_keypoints(predictions, ground_truth, pixel_threshold=10.0)["results"]["match"].sum() == 1
    assert evaluate.evaluate_keypoints(predictions, ground_truth, pixel_threshold=100.0)["results"]["match"].sum() == 2


def test_evaluate_keypoints_empty_predictions():
    """Test evaluation with no predictions."""
    result = evaluate.evaluate_keypoints(
        gpd.GeoDataFrame({"geometry": pts([]), "label": [], "image_path": []}),
        gpd.GeoDataFrame({"geometry": pts([(100, 100)]), "label": ["Tree"], "image_path": ["img1.jpg"]})
    )
    assert result["recall"] == 0.0
    assert pd.isna(result["precision"])
    assert result["class_recall"] is None


def test_evaluate_keypoints_empty_ground_truth():
    """Test evaluation with no ground truth."""
    result = evaluate.evaluate_keypoints(
        gpd.GeoDataFrame({"geometry": pts([(100, 100)]), "score": [0.9], "label": ["Tree"], "image_path": ["img1.jpg"]}),
        gpd.GeoDataFrame({"geometry": pts([]), "label": [], "image_path": []})
    )
    assert result["results"] is None
    assert result["recall"] is None
    assert result["precision"] == 0.0


def test_evaluate_keypoints_multi_image():
    """Test evaluation across multiple images."""
    predictions = gpd.GeoDataFrame({
        "geometry": pts([(100, 100), (200, 200), (300, 300)]),
        "score": [0.9, 0.8, 0.7],
        "label": ["Tree", "Tree", "Bird"],
        "image_path": ["img1.jpg", "img1.jpg", "img2.jpg"]
    })
    ground_truth = predictions.copy()

    result = evaluate.evaluate_keypoints(predictions, ground_truth, pixel_threshold=5.0)

    assert result["recall"] == 1.0
    assert result["precision"] == 1.0
    assert len(result["results"]) == 3
    assert set(result["results"]["image_path"]) == {"img1.jpg", "img2.jpg"}


def test_evaluate_keypoints_class_recall():
    """Test per-class recall and precision calculation."""
    predictions = gpd.GeoDataFrame({
        "geometry": pts([(100, 100), (205, 205), (300, 300)]),
        "score": [0.9, 0.8, 0.7],
        "label": ["Tree", "Tree", "Bird"],
        "image_path": ["img1.jpg", "img1.jpg", "img1.jpg"]
    })
    ground_truth = gpd.GeoDataFrame({
        "geometry": pts([(100, 100), (200, 200), (300, 300)]),
        "label": ["Tree", "Tree", "Bird"],
        "image_path": ["img1.jpg", "img1.jpg", "img1.jpg"]
    })

    result = evaluate.evaluate_keypoints(predictions, ground_truth, pixel_threshold=10.0)
    class_recall = result["class_recall"]

    assert class_recall[class_recall["label"] == "Tree"]["recall"].iloc[0] == 1.0
    assert class_recall[class_recall["label"] == "Tree"]["precision"].iloc[0] == 1.0
    assert class_recall[class_recall["label"] == "Bird"]["recall"].iloc[0] == 1.0
    assert class_recall[class_recall["label"] == "Bird"]["precision"].iloc[0] == 1.0


def test_evaluate_keypoints_wrong_labels():
    """Test evaluation when predicted labels don't match ground truth."""
    result = evaluate.evaluate_keypoints(
        gpd.GeoDataFrame({"geometry": pts([(100, 100), (200, 200)]), "score": [0.9, 0.8], "label": ["Tree", "Bird"], "image_path": ["img1.jpg", "img1.jpg"]}),
        gpd.GeoDataFrame({"geometry": pts([(100, 100), (200, 200)]), "label": ["Bird", "Tree"], "image_path": ["img1.jpg", "img1.jpg"]}),
        pixel_threshold=5.0
    )
    assert result["results"]["match"].sum() == 2
    assert all(result["class_recall"]["recall"] == 0.0)


def test_evaluate_keypoints_partial_image_matches():
    """Test evaluation where images have different match rates."""
    predictions = gpd.GeoDataFrame({
        "geometry": pts([(100, 100), (150, 150), (200, 200)]),
        "score": [0.9, 0.85, 0.8],
        "label": ["Tree", "Tree", "Tree"],
        "image_path": ["img1.jpg", "img1.jpg", "img2.jpg"]
    })
    ground_truth = gpd.GeoDataFrame({
        "geometry": pts([(100, 100), (150, 150), (200, 200), (250, 250)]),
        "label": ["Tree", "Tree", "Tree", "Tree"],
        "image_path": ["img1.jpg", "img1.jpg", "img2.jpg", "img2.jpg"]
    })

    result = evaluate.evaluate_keypoints(predictions, ground_truth, pixel_threshold=5.0)

    assert result["recall"] == 0.75
    assert result["precision"] == 1.0
