"""Evaluation module."""

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from deepforest import IoU
from deepforest.utilities import determine_geometry_type


def evaluate_image_boxes(predictions, ground_df):
    """Compute intersection-over-union matching among prediction and ground
    truth boxes for one image.

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
    result = IoU.compute_IoU(ground_df, predictions)

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

    for name, group in box_results.groupby("true_label"):
        class_recall_dict[name] = (
            sum(group.true_label == group.predicted_label) / group.shape[0]
        )
        number_of_predictions = box_results[box_results.predicted_label == name].shape[0]
        if number_of_predictions == 0:
            class_precision_dict[name] = 0
        else:
            class_precision_dict[name] = (
                sum(group.true_label == group.predicted_label) / number_of_predictions
            )
        class_size[name] = group.shape[0]

    class_recall = pd.DataFrame(
        {
            "label": class_recall_dict.keys(),
            "recall": pd.Series(class_recall_dict),
            "precision": pd.Series(class_precision_dict),
            "size": pd.Series(class_size),
        }
    ).reset_index(drop=True)

    return class_recall


def __evaluate_wrapper__(predictions, ground_df, iou_threshold, numeric_to_label_dict):
    """Evaluate a set of predictions against a ground truth csv file
    Args:
        predictions: a pandas dataframe, if supplied a root dir is needed to give the relative path of files in df.name. The labels in ground truth and predictions must match. If one is numeric, the other must be numeric.
        ground_df: a pandas dataframe, if supplied a root dir is needed to give the relative path of files in df.name
        iou_threshold: intersection-over-union threshold, see deepforest.evaluate
    Returns:
        results: a dictionary of results with keys, results, box_recall, box_precision, class_recall
    """
    # remove empty samples from ground truth
    ground_df = ground_df[~((ground_df.xmin == 0) & (ground_df.xmax == 0))]

    # Default results for blank predictions
    if predictions.empty:
        results = {
            "results": None,
            "box_recall": 0,
            "box_precision": np.nan,
            "class_recall": None,
            "predictions": predictions,
            "ground_df": ground_df,
        }
        return results

    # Convert pandas to geopandas if needed
    if not isinstance(predictions, gpd.GeoDataFrame):
        warnings.warn(
            "Converting predictions to GeoDataFrame using geometry column", stacklevel=2
        )
        # Check if we have bounding box columns and need to create geometry
        if "geometry" not in predictions.columns and all(
            col in predictions.columns for col in ["xmin", "ymin", "xmax", "ymax"]
        ):
            # Create geometry from bounding box columns
            predictions = predictions.copy()
            predictions["geometry"] = shapely.box(
                predictions["xmin"],
                predictions["ymin"],
                predictions["xmax"],
                predictions["ymax"],
            )
        predictions = gpd.GeoDataFrame(predictions, geometry="geometry")

    # Also ensure ground_df is a GeoDataFrame
    if not isinstance(ground_df, gpd.GeoDataFrame):
        # Check if we have bounding box columns and need to create geometry
        if "geometry" not in ground_df.columns and all(
            col in ground_df.columns for col in ["xmin", "ymin", "xmax", "ymax"]
        ):
            # Create geometry from bounding box columns
            ground_df = ground_df.copy()
            ground_df["geometry"] = shapely.box(
                ground_df["xmin"], ground_df["ymin"], ground_df["xmax"], ground_df["ymax"]
            )
        ground_df = gpd.GeoDataFrame(ground_df, geometry="geometry")

    prediction_geometry = determine_geometry_type(predictions)
    if prediction_geometry == "point":
        raise NotImplementedError("Point evaluation is not yet implemented")
    elif prediction_geometry == "box":
        results = evaluate_boxes(
            predictions=predictions, ground_df=ground_df, iou_threshold=iou_threshold
        )
    else:
        raise NotImplementedError(f"Geometry type {prediction_geometry} not implemented")

    if results["results"] is not None:
        # Convert numeric class codes to string labels for results
        results["results"]["predicted_label"] = results["results"]["predicted_label"].map(
            lambda x: numeric_to_label_dict.get(x, x) if pd.notnull(x) else x
        )
        results["results"]["true_label"] = results["results"]["true_label"].map(
            numeric_to_label_dict
        )
        results["predictions"] = predictions
        # Also convert labels in the original predictions for consistency
        results["predictions"]["label"] = results["predictions"]["label"].map(
            numeric_to_label_dict
        )

    return results


def evaluate_boxes(predictions, ground_df, iou_threshold=0.4):
    """Image annotated crown evaluation routine submission can be submitted as
    a .shp, existing pandas dataframe or .csv path.

    Args:
        predictions: a pandas dataframe with geometry columns. The labels in ground truth and predictions must match. If one is numeric, the other must be numeric.
        ground_df: a pandas dataframe with geometry columns
        iou_threshold: intersection-over-union threshold, see deepforest.evaluate

    Returns:
        results: a dataframe of match bounding boxes
        box_recall: proportion of true positives of box position, regardless of class
        box_precision: proportion of predictions that are true positive, regardless of class
        class_recall: a pandas dataframe of class level recall and precision with class sizes
    """

    # If all empty ground truth, return 0 recall and precision
    if ground_df.empty:
        return {
            "results": None,
            "box_recall": None,
            "box_precision": 0,
            "class_recall": None,
            "predictions": predictions,
            "ground_df": ground_df,
        }

    # Convert pandas to geopandas if needed
    if not isinstance(predictions, gpd.GeoDataFrame):
        # Check if we have bounding box columns and need to create geometry
        if "geometry" not in predictions.columns and all(
            col in predictions.columns for col in ["xmin", "ymin", "xmax", "ymax"]
        ):
            # Create geometry from bounding box columns
            predictions = predictions.copy()
            predictions["geometry"] = shapely.box(
                predictions["xmin"],
                predictions["ymin"],
                predictions["xmax"],
                predictions["ymax"],
            )
        predictions = gpd.GeoDataFrame(predictions, geometry="geometry")

    if not isinstance(ground_df, gpd.GeoDataFrame):
        # Check if we have bounding box columns and need to create geometry
        if "geometry" not in ground_df.columns and all(
            col in ground_df.columns for col in ["xmin", "ymin", "xmax", "ymax"]
        ):
            # Create geometry from bounding box columns
            ground_df = ground_df.copy()
            ground_df["geometry"] = shapely.box(
                ground_df["xmin"], ground_df["ymin"], ground_df["xmax"], ground_df["ymax"]
            )
        ground_df = gpd.GeoDataFrame(ground_df, geometry="geometry")

    # Pre-group predictions by image
    predictions_by_image = {
        name: group.reset_index(drop=True)
        for name, group in predictions.groupby("image_path")
    }

    # Run evaluation on all plots
    results = []
    box_recalls = []
    box_precisions = []
    for image_path, group in ground_df.groupby("image_path"):
        # Predictions for this image
        image_predictions = predictions_by_image.get(image_path, pd.DataFrame())
        if not isinstance(image_predictions, pd.DataFrame) or image_predictions.empty:
            image_predictions = pd.DataFrame()

        # If empty, add to list without computing IoU
        if image_predictions.empty:
            # Reset index
            group = group.reset_index(drop=True)
            result = pd.DataFrame(
                {
                    "truth_id": group.index.values,
                    "prediction_id": pd.Series([None] * len(group), dtype="object"),
                    "IoU": pd.Series([0.0] * len(group), dtype="float64"),
                    "predicted_label": pd.Series([None] * len(group), dtype="object"),
                    "score": pd.Series([None] * len(group), dtype="float64"),
                    "match": pd.Series([False] * len(group), dtype="bool"),
                    "true_label": group.label.astype("object"),
                    "geometry": group.geometry,
                }
            )
            # An empty prediction set has recall of 0, precision of NA.
            box_recalls.append(0)
            results.append(result)
            continue
        else:
            group = group.reset_index(drop=True)
            result = evaluate_image_boxes(predictions=image_predictions, ground_df=group)

        result["image_path"] = image_path
        result["match"] = result.IoU > iou_threshold
        # Convert None to False for boolean consistency
        result["match"] = result["match"].fillna(False)
        true_positive = sum(result["match"])
        recall = true_positive / result.shape[0]
        precision = true_positive / image_predictions.shape[0]

        box_recalls.append(recall)
        box_precisions.append(precision)
        results.append(result)

    # Concatenate results
    if results:
        results = pd.concat(results, ignore_index=True)
    else:
        columns = [
            "truth_id",
            "prediction_id",
            "IoU",
            "predicted_label",
            "score",
            "match",
            "true_label",
            "geometry",
            "image_path",
        ]
        results = pd.DataFrame(columns=columns)

    box_precision = np.mean(box_precisions)
    box_recall = np.mean(box_recalls)

    # Only matching boxes are considered in class recall
    matched_results = results[results.match]
    class_recall = compute_class_recall(matched_results)

    return {
        "results": results,
        "box_precision": box_precision,
        "box_recall": box_recall,
        "class_recall": class_recall,
        "predictions": predictions,
        "ground_df": ground_df,
    }


def _point_recall_image_(predictions, ground_df):
    """Compute intersection-over-union matching among prediction and ground
    truth boxes for one image.

    Args:
        predictions: a pandas dataframe. The labels in ground truth and predictions must match. For example, if one is numeric, the other must be numeric.
        ground_df: a pandas dataframe

    Returns:
        result: pandas dataframe with crown ids of prediciton and ground truth and the IoU score.
    """
    plot_names = predictions["image_path"].unique()
    if len(plot_names) > 1:
        raise ValueError(f"More than one image passed to function: {plot_names[0]}")

    predictions["geometry"] = predictions.apply(
        lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1
    )
    predictions = gpd.GeoDataFrame(predictions, geometry="geometry")

    ground_df["geometry"] = ground_df.apply(
        lambda x: shapely.geometry.Point(x.x, x.y), axis=1
    )
    ground_df = gpd.GeoDataFrame(ground_df, geometry="geometry")

    # Which points in boxes
    result = gpd.sjoin(ground_df, predictions, predicate="within", how="left")
    result = result.rename(
        columns={
            "label_left": "true_label",
            "label_right": "predicted_label",
            "image_path_left": "image_path",
        }
    )
    result = result.drop(columns=["index_right"])

    return result


def point_recall(predictions, ground_df):
    """Evaluate the proportion on ground truth points overlap with predictions
    submission can be submitted as a .shp, existing pandas dataframe or .csv
    path For bounding box recall, see evaluate().

    Args:
        predictions: a pandas dataframe, if supplied a root dir is needed to give the relative path of files in df.name. The labels in ground truth and predictions must match. If one is numeric, the other must be numeric.
        ground_df: a pandas dataframe, if supplied a root dir is needed to give the relative path of files in df.name

    Returns:
        results: a dataframe of matched bounding boxes and ground truth labels
        box_recall: proportion of true positives between predicted boxes and ground truth points, regardless of class
        class_recall: a pandas dataframe of class level recall and precision with class sizes
    """
    # Run evaluation on all images
    results = []
    box_recalls = []
    for image_path, group in ground_df.groupby("image_path"):
        image_predictions = predictions[
            predictions["image_path"] == image_path
        ].reset_index(drop=True)

        # If empty, add to list without computing recall
        if image_predictions.empty:
            result = pd.DataFrame(
                {
                    "recall": 0,
                    "predicted_label": None,
                    "score": None,
                    "true_label": group.label,
                }
            )
            # An empty prediction set has recall of 0, precision of NA.
            box_recalls.append(0)
            results.append(result)
            continue
        else:
            group = group.reset_index(drop=True)
            result = _point_recall_image_(predictions=image_predictions, ground_df=group)

        result["image_path"] = image_path

        # What proportion of boxes match? Regardless of label
        true_positive = sum(result.predicted_label.notnull())
        recall = true_positive / result.shape[0]

        box_recalls.append(recall)
        results.append(result)

    results = pd.concat(results)
    box_recall = np.mean(box_recalls)

    # Only matching boxes are considered in class recall
    matched_results = results[results.predicted_label.notnull()]
    class_recall = compute_class_recall(matched_results)

    return {"results": results, "box_recall": box_recall, "class_recall": class_recall}
