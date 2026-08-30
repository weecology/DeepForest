import math

import pytest
import torch

from deepforest.metrics import RecallPrecision


def _mask(xmin, ymin, xmax, ymax, size=100):
    """A single square instance mask in a size x size frame."""
    mask = torch.zeros((1, size, size), dtype=torch.uint8)
    mask[0, ymin:ymax, xmin:xmax] = 1
    return mask


def _perfect_polygon_pair(label=0):
    """Prediction and target with a single matching mask."""
    mask = _mask(10, 10, 50, 50)
    box = torch.tensor([[10.0, 10.0, 50.0, 50.0]], dtype=torch.float32)
    labels = torch.tensor([label], dtype=torch.int64)
    pred = {
        "masks": mask,
        "boxes": box,
        "labels": labels,
        "scores": torch.tensor([0.9]),
    }
    target = {"masks": mask, "boxes": box, "labels": labels}
    return pred, target


def _empty_polygon_pair():
    """Empty ground truth with no predictions."""
    empty = {
        "masks": torch.zeros((0, 100, 100), dtype=torch.uint8),
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.tensor([], dtype=torch.int64),
    }
    return {**empty, "scores": torch.tensor([], dtype=torch.float32)}, empty


def _perfect_match_pair():
    """Prediction and target with a single matching box."""
    box = torch.tensor([[10.0, 10.0, 50.0, 50.0]], dtype=torch.float32)
    pred = {
        "boxes": box,
        "labels": torch.tensor([0], dtype=torch.int64),
        "scores": torch.tensor([0.9]),
    }
    target = {"boxes": box, "labels": torch.tensor([0], dtype=torch.int64)}
    return pred, target


def _empty_frame_pair():
    """Empty ground truth with no predictions."""
    return (
        {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor([], dtype=torch.int64),
            "scores": torch.tensor([], dtype=torch.float32),
        },
        {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor([], dtype=torch.int64),
        },
    )


def _gt_without_predictions_pair():
    """Ground truth with boxes but no predictions (per-image recall should be 0)."""
    target = {
        "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]], dtype=torch.float32),
        "labels": torch.tensor([0], dtype=torch.int64),
    }
    pred = {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.tensor([], dtype=torch.int64),
        "scores": torch.tensor([], dtype=torch.float32),
    }
    return pred, target


def test_box_recall_excludes_empty_frames_from_average():
    """Empty frames should not lower mean recall.

    One image with perfect recall and one empty frame should average to 1.0,
    not 0.5 from dividing by all images passed to update.
    """
    metric = RecallPrecision(label_dict={"Tree": 0})
    pred, target = _perfect_match_pair()
    empty_pred, empty_target = _empty_frame_pair()

    metric.update(
        [pred, empty_pred],
        [target, empty_target],
        ["with_trees.jpg", "empty.jpg"],
    )
    results = metric.compute()

    assert metric.num_images == 2
    assert metric.num_empty_frames == 1
    assert math.isclose(results["box_recall"], 1.0, rel_tol=1e-5), (
        f"box_recall={results['box_recall']:.2f}, expected 1.0 when averaging "
        "only over images with ground truth"
    )


def test_box_recall_counts_gt_images_without_predictions_as_zero():
    """Images with ground truth but no predictions should count as recall 0.

    Mean recall should be over images with non-empty ground truth. One perfect
    image and one image with missed detections should average to 0.5 even when
    an empty frame is also in the batch (empty frames must not enter the mean).
    """
    metric = RecallPrecision(label_dict={"Tree": 0})
    pred, target = _perfect_match_pair()
    missed_pred, missed_target = _gt_without_predictions_pair()
    empty_pred, empty_target = _empty_frame_pair()

    metric.update(
        [pred, missed_pred, empty_pred],
        [target, missed_target, empty_target],
        ["matched.jpg", "missed.jpg", "empty.jpg"],
    )
    results = metric.compute()

    assert metric.num_images == 3
    assert metric.num_empty_frames == 1
    assert math.isclose(results["box_recall"], 0.5, rel_tol=1e-5), (
        f"box_recall={results['box_recall']:.2f}, expected 0.5 (mean of 1.0 and 0.0 "
        "over two images with ground truth, excluding the empty frame)"
    )


def test_unsupported_task_raises():
    """Only box, point and polygon tasks are supported."""
    with pytest.raises(ValueError, match="Unsupported task"):
        RecallPrecision(task="mask")


def test_polygon_perfect_match():
    """Matching masks give perfect precision and recall."""
    metric = RecallPrecision(task="polygon", label_dict={"Tree": 0})
    pred, target = _perfect_polygon_pair()

    metric.update([pred], [target], ["trees.jpg"])
    results = metric.compute()

    assert math.isclose(results["polygon_precision"], 1.0, rel_tol=1e-5)
    assert math.isclose(results["polygon_recall"], 1.0, rel_tol=1e-5)


def test_polygon_recall_excludes_empty_frames_from_average():
    """Empty frames are counted but do not enter the recall average."""
    metric = RecallPrecision(task="polygon", label_dict={"Tree": 0})
    pred, target = _perfect_polygon_pair()
    empty_pred, empty_target = _empty_polygon_pair()

    metric.update(
        [pred, empty_pred],
        [target, empty_target],
        ["with_trees.jpg", "empty.jpg"],
    )
    results = metric.compute()

    assert metric.num_images == 2
    assert metric.num_empty_frames == 1
    assert metric.correct_empty_predictions == 1
    assert math.isclose(results["polygon_recall"], 1.0, rel_tol=1e-5)


def test_polygon_multiclass_reports_per_class_metrics():
    """Multi-class polygon targets report per-class recall and precision."""
    metric = RecallPrecision(task="polygon", label_dict={"Tree": 0, "Bird": 1})
    tree_pred, tree_target = _perfect_polygon_pair(label=0)
    bird_pred, bird_target = _perfect_polygon_pair(label=1)

    metric.update(
        [tree_pred, bird_pred],
        [tree_target, bird_target],
        ["tree.jpg", "bird.jpg"],
    )
    results = metric.compute()

    assert math.isclose(results["Tree_Recall"], 1.0, rel_tol=1e-5)
    assert math.isclose(results["Bird_Recall"], 1.0, rel_tol=1e-5)
    assert "Tree_Precision" in results
    assert "Bird_Precision" in results
