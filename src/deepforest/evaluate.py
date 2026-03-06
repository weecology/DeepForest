"""Evaluation module."""

import geopandas as gpd
import numpy as np
import pandas as pd

from deepforest import IoU
from deepforest.utilities import __pandas_to_geodataframe__


def _empty_result_dataframe_(group, image_path, task="box"):
    """Create an empty result dataframe for images with no predictions."""

    result_dict = {
        "truth_id": group.index.values,
        "prediction_id": pd.Series([None] * len(group), dtype="object"),
        "geometry": group.geometry,
        "image_path": image_path,
        "match": pd.Series([False] * len(group), dtype="bool"),
        "score": pd.Series([None] * len(group), dtype="float64"),
        "predicted_label": pd.Series([None] * len(group), dtype="object"),
        "true_label": group.label,
    }

    if task == "box" or task == "polygon":
        result_dict.update(
            {
                "IoU": pd.Series([0.0] * len(group), dtype="float64"),
            }
        )

    return pd.DataFrame(result_dict)


def match_predictions(predictions, ground_df, task="box"):
    """Compute intersection-over-union matching among prediction and ground
    truth geometries for one image. The returned results are guaranteed to be
    at most one-to-one, but are not filtered for "quality" of match (i.e. IoU
    threshold).

    Args:
        predictions: a geopandas dataframe with geometry columns
        ground_df: a geopandas dataframe with geometry columns

    Returns:
        result: pandas dataframe with crown ids of prediction and ground truth and the IoU score.
    """
    plot_names = predictions["image_path"].unique()
    if len(plot_names) > 1:
        raise ValueError(f"More than one plot passed to image crown: {plot_names}")

    # match
    if task in ["box", "polygon"]:
        result = IoU.match_polygons(ground_df, predictions)
    elif task == "point":
        result = IoU.match_points(ground_df, predictions, norm="l2")
    else:
        raise NotImplementedError(f"Geometry type {task} not implemented")

    # Map prediction/truth IDs back to their original labels from input dataframes
    pred_label_dict = predictions.label.to_dict()
    ground_label_dict = ground_df.label.to_dict()
    result["predicted_label"] = result.prediction_id.map(pred_label_dict)
    result["true_label"] = result.truth_id.map(ground_label_dict)

    return result


def compute_class_recall(results):
    """Given a set of evaluations, what proportion of predicted boxes match.

    True boxes which are not matched to predictions do not count against
    accuracy.
    """
    # Per class recall and precision
    class_recall_dict = {}
    class_precision_dict = {}
    class_size = {}

    box_results = results[results.predicted_label.notna()]
    if box_results.empty:
        print("No predictions made")
        class_recall = None
        return class_recall

    # Get all labels from both predictions and ground truth
    predicted_labels = set(box_results["predicted_label"].dropna())
    true_labels = set(box_results["true_label"].dropna())
    all_labels = predicted_labels.union(true_labels)

    for label in all_labels:
        # Recall: of all ground truth boxes with this label, how many were correctly predicted?
        ground_df = box_results[box_results["true_label"] == label]
        n_ground_boxes = ground_df.shape[0]
        if n_ground_boxes > 0:
            class_recall_dict[label] = (
                sum(ground_df.true_label == ground_df.predicted_label) / n_ground_boxes
            )

        # Precision: of all predictions with this label, how many were correct?
        pred_df = box_results[box_results["predicted_label"] == label]
        n_pred_boxes = pred_df.shape[0]
        if n_pred_boxes > 0:
            class_precision_dict[label] = (
                sum(pred_df.true_label == pred_df.predicted_label) / n_pred_boxes
            )

        class_size[label] = n_ground_boxes

    # fillna(0) handles labels with no ground truth (recall=0) or no predictions (precision=0)
    class_recall = (
        pd.DataFrame(
            {
                "recall": pd.Series(class_recall_dict),
                "precision": pd.Series(class_precision_dict),
                "size": pd.Series(class_size),
            }
        )
        .reset_index(names="label")
        .fillna(0)
        .sort_values("label")
    )

    return class_recall


def __evaluate_wrapper__(
    predictions: pd.DataFrame | gpd.GeoDataFrame,
    ground_df: pd.DataFrame | gpd.GeoDataFrame,
    numeric_to_label_dict: dict,
    iou_threshold: float = 0.4,
    l2_threshold: float = 10.0,
    geometry_type: str | None = "box",
) -> dict:
    """Evaluate a set of predictions against ground truth
    Args:
        predictions: a pandas dataframe with a root dir attribute is needed to give the relative path of files in df.name. The labels in ground truth and predictions must match. If one is numeric, the other must be numeric.
        ground_df: a pandas dataframe with a root dir attribute is needed to give the relative path of files in df.name
        iou_threshold: intersection-over-union threshold, see deepforest.evaluate
        numeric_to_label_dict: mapping from numeric class codes to string labels
        geometry_type: 'box', 'polygon' or 'point'
    Returns:
        results: a dictionary of results with keys, results, box_recall, box_precision, class_recall
    """

    # Convert labels to consistent types prior to eval
    # Use shallow copy to avoid duplicating large data arrays
    predictions = predictions.copy(deep=False)
    ground_df = ground_df.copy(deep=False)

    # Apply numeric_to_label_dict mapping to ensure type consistency. Checking
    # for labels guards against empty frames.
    if not predictions.empty and "label" in predictions.columns:
        predictions["label"] = predictions["label"].map(
            lambda x: numeric_to_label_dict.get(x, x) if pd.notnull(x) else x
        )
    if not ground_df.empty and "label" in ground_df.columns:
        ground_df["label"] = ground_df["label"].map(
            lambda x: numeric_to_label_dict.get(x, x) if pd.notnull(x) else x
        )

    results = evaluate_geometry(
        predictions=predictions,
        ground_df=ground_df,
        iou_threshold=iou_threshold,
        distance_threshold=l2_threshold,
        geometry_type=geometry_type,
    )

    # Store the converted predictions for reference
    if results["results"] is not None:
        results["predictions"] = predictions

    return results


def evaluate_boxes(
    predictions: pd.DataFrame | gpd.GeoDataFrame,
    ground_df: pd.DataFrame | gpd.GeoDataFrame,
    iou_threshold: float = 0.4,
) -> dict:
    """Evaluate bounding box predictions against ground truth. Calls
    evaluate_geometry.

    Args:
        predictions: a pandas dataframe with geometry columns. The labels in ground truth and predictions must match. If one is numeric, the other must be numeric.
        ground_df: a pandas dataframe with geometry columns
        iou_threshold: intersection-over-union threshold, see deepforest.evaluate

    Returns:
        results: a dictionary of results with keys, results, box_recall, box_precision, class_recall
    """
    return evaluate_geometry(
        predictions=predictions,
        ground_df=ground_df,
        iou_threshold=iou_threshold,
        geometry_type="box",
    )


def evaluate_geometry(
    predictions: pd.DataFrame | gpd.GeoDataFrame,
    ground_df: pd.DataFrame | gpd.GeoDataFrame,
    iou_threshold: float = 0.4,
    distance_threshold: float = 10.0,
    geometry_type: str = "box",
) -> dict:
    """Image annotated crown evaluation routine submission can be submitted as
    a .shp, existing pandas dataframe or .csv path.

    Args:
        predictions: a pandas dataframe with geometry columns. The labels in ground truth and predictions must match. If one is numeric, the other must be numeric.
        ground_df: a pandas dataframe with geometry columns
        iou_threshold: intersection-over-union threshold, see deepforest.evaluate
        l2_threshold: L2 distance threshold for point matching
        geometry_type: 'box', 'polygon' or 'point'

    Returns:
        results: a dataframe of match bounding boxes
        box_recall: proportion of true positives of box position, regardless of class
        box_precision: proportion of predictions that are true positive, regardless of class
        class_recall: a pandas dataframe of class level recall and precision with class sizes
    """

    if geometry_type not in ["box", "polygon", "point"]:
        raise ValueError(
            f"Unknown geometry type {geometry_type}. Must be one of 'box', 'polygon' or 'point'."
        )

    # If no predictions, return 0 recall, NaN precision
    if predictions.empty:
        return {
            "results": None,
            f"{geometry_type}_recall": 0,
            f"{geometry_type}_precision": np.nan,
            "class_recall": None,
            "predictions": predictions,
            "ground_df": ground_df,
        }
    elif not isinstance(predictions, gpd.GeoDataFrame):
        predictions = __pandas_to_geodataframe__(predictions)

    # Remove empty ground truth boxes
    if geometry_type == "box":
        ground_df = ground_df[
            ~(
                (ground_df.xmin == 0)
                & (ground_df.xmax == 0)
                & (ground_df.ymin == 0)
                & (ground_df.ymax == 0)
            )
        ]
    elif geometry_type == "polygon":
        ground_df = ground_df[~ground_df.geometry.is_empty]
    elif geometry_type == "point":
        ground_df = ground_df[~((ground_df.x == 0) & (ground_df.y == 0))]

    # If all empty ground truth, return 0 recall and precision
    if ground_df.empty:
        return {
            "results": None,
            f"{geometry_type}_recall": None,
            f"{geometry_type}_precision": 0,
            "class_recall": None,
            "predictions": predictions,
            "ground_df": ground_df,
        }

    if not isinstance(ground_df, gpd.GeoDataFrame):
        ground_df = __pandas_to_geodataframe__(ground_df)

    # Pre-group predictions by image
    predictions_by_image = {
        name: group.reset_index(drop=True)
        for name, group in predictions.groupby("image_path")
    }

    # Run evaluation on all plots
    results = []
    per_image_recalls = []
    per_image_precisions = []
    for image_path, group in ground_df.groupby("image_path"):
        # Predictions for this image
        image_predictions = predictions_by_image.get(image_path, pd.DataFrame())

        # If empty, add to list without computing IoU
        if image_predictions.empty:
            # Reset index
            group = group.reset_index(drop=True)
            result = _empty_result_dataframe_(group, image_path, task=geometry_type)
            # An empty prediction set has recall of 0, precision of NA.
            per_image_recalls.append(0)
            results.append(result)
            continue
        else:
            group = group.reset_index(drop=True)
            result = match_predictions(
                predictions=image_predictions, ground_df=group, task=geometry_type
            )

        result["image_path"] = image_path

        # Determine matches based on IoU or distance thresholds
        if geometry_type == "box" or geometry_type == "polygon":
            result["match"] = result.IoU > iou_threshold
        elif geometry_type == "point":
            result["match"] = result.distance < distance_threshold

        # Convert None to False for boolean consistency
        result["match"] = result["match"].fillna(False)
        true_positive = sum(result["match"])
        recall = true_positive / result.shape[0]
        precision = true_positive / image_predictions.shape[0]

        per_image_recalls.append(recall)
        per_image_precisions.append(precision)
        results.append(result)

    # Concatenate results
    if results:
        results = pd.concat(results, ignore_index=True)
        # Convert back to GeoDataFrame if it has geometry column
        if "geometry" in results.columns:
            results = gpd.GeoDataFrame(results, geometry="geometry")
    else:
        columns = [
            "truth_id",
            "prediction_id",
            "predicted_label",
            "score",
            "match",
            "true_label",
            "geometry",
            "image_path",
        ]

        if geometry_type == "box" or geometry_type == "polygon":
            columns.append("IoU")
        elif geometry_type == "point":
            columns.append("distance")

        results = gpd.GeoDataFrame(columns=columns)

    mean_precision = np.mean(per_image_precisions)
    mean_recall = np.mean(per_image_recalls)

    # Only matching boxes are considered in class recall
    matched_results = results[results.match]
    class_recall = compute_class_recall(matched_results)

    return {
        "results": results,
        f"{geometry_type}_precision": mean_precision,
        f"{geometry_type}_recall": mean_recall,
        "class_recall": class_recall,
        "predictions": predictions,
        "ground_df": ground_df,
    }
