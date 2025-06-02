"""Evaluation module."""

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import torch
from scipy import optimize
from torchmetrics.detection.mean_ap import MeanAveragePrecision

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
        raise ValueError(
            "More than one plot passed to image crown: {}".format(plot_names)
        )

    # match
    result = IoU.compute_IoU(ground_df, predictions)

    # add the label classes
    result["predicted_label"] = result.prediction_id.apply(
        lambda x: predictions.label.loc[x] if pd.notnull(x) else x
    )
    result["true_label"] = result.truth_id.apply(lambda x: ground_df.label.loc[x])

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
        number_of_predictions = box_results[box_results.predicted_label == name].shape[
            0
        ]
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


def _torch_dict_from_df(df, label_dict):
    d = {}
    d["boxes"] = torch.tensor(
        df[["xmin", "ymin", "xmax", "ymax"]].astype(int).values,
    )
    if "score" in df.columns:
        d["scores"] = torch.tensor(df["score"].values)
    d["labels"] = torch.tensor(df["label"].map(label_dict).astype(int).values)
    return d


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
        }
        return results

    # Convert pandas to geopandas if needed
    if not isinstance(predictions, gpd.GeoDataFrame):
        warnings.warn("Converting predictions to GeoDataFrame using geometry column")
        predictions = gpd.GeoDataFrame(predictions, geometry="geometry")

    prediction_geometry = determine_geometry_type(predictions)
    if prediction_geometry == "point":
        raise NotImplementedError("Point evaluation is not yet implemented")
    elif prediction_geometry == "box":
        results = evaluate_boxes(
            predictions=predictions, ground_df=ground_df, iou_threshold=iou_threshold
        )

    else:
        raise NotImplementedError(
            "Geometry type {} not implemented".format(prediction_geometry)
        )

    # replace classes if not NUll
    if results is not None:
        results["results"]["predicted_label"] = results["results"][
            "predicted_label"
        ].apply(lambda x: numeric_to_label_dict[x] if not pd.isnull(x) else x)
        results["results"]["true_label"] = results["results"]["true_label"].apply(
            lambda x: numeric_to_label_dict[x]
        )
        results["predictions"] = predictions
        results["predictions"]["label"] = results["predictions"]["label"].apply(
            lambda x: numeric_to_label_dict[x]
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
    # TODO: add `label_dict`, `rec_sep`/`rec_thresholds` and `max_detection_threshold`/
    # `max_detection_thresholds` as keyword arguments?

    # get a label dictionary mapping the integer ids to the class names, from the
    # "label" column of `predictions` and `ground_df`
    label_dict = {
        label: i
        for i, label in enumerate(set(predictions["label"]).union(ground_df["label"]))
    }

    # torchmetrics will compute the metrics at multiple recall thresholds, so in order
    # to emulate the previous deepforest metrics, we need it to be small enough so that
    # we can get the precision for the recall threshold closest ot the actual recall
    # value
    rec_sep = 0.01
    rec_thresholds = np.arange(0, 1 + rec_sep, rec_sep).tolist()

    max_detection_threshold = 1000
    max_detection_thresholds = [0, 0, max_detection_threshold]
    mean_ap_kwargs = dict(
        iou_type="bbox",
        iou_thresholds=[iou_threshold],
        rec_thresholds=rec_thresholds,
        max_detection_thresholds=max_detection_thresholds,
        extended_summary=True,
    )

    metric = MeanAveragePrecision(
        **mean_ap_kwargs,
    )
    metric.update(
        [
            _torch_dict_from_df(_pred_df, label_dict)
            for _, _pred_df in predictions.groupby("image_path")
        ],
        [
            _torch_dict_from_df(_annot_df, label_dict)
            for _, _annot_df in ground_df.groupby("image_path")
        ],
    )
    mean_ap_dict = metric.compute()

    # ious
    # shape: dict with keys of the form (image, class) and values with the corresponding
    # ious for each detection (row) and ground truth (column)
    # TODO: support multi-class
    # TODO: support multi-image predictions and annotations
    # ious = mean_ap_dict["ious"]  # [(0, 0)].max(dim=0).values

    # recall
    # shape: n_iou_thresholds, n_classes, n_areas, n_max_detections
    # selection:
    # - one iou_threshold
    # - all classes
    # - first area, i.e., 'all' https://github.com/cocodataset/cocoapi/blob/master/
    #   PythonAPI/pycocotools/cocoeval.py#L509
    # - last max_detection_threshold
    recall = mean_ap_dict["recall"][0, :, 0, -1].item()

    # precision
    # shape: n_iou_thresholds, n_recall_thresholds, n_classes, n_areas, n_max_detections
    # selection:
    # - one iou_threshold
    # - all recall_thresholds
    # - all classes
    # - first area, i.e., 'all'
    # - last max_detection_threshold
    precision = mean_ap_dict["precision"][
        0,
        :,
        :,
        0,
        -1,
    ]
    # idea: emulate deepforest and
    # three options:
    # 1. get the recall_threshold closest to the actual recall using argmin
    # note that we'd need to ensure that rec_thresholds is a numpy array
    # np.argmin(
    #     np.abs(np.array(rec_thresholds)[:, np.newaxis] - recall.numpy()), axis=0
    # ),
    # 2. same but using np.searchsorted, which would be quicker than argmin, but
    # performance should not be an issue here
    # see https://stackoverflow.com/questions/44526121/
    # finding-closest-values-in-two-numpy-arrays
    # 3. get the precision as the latest non-zero value of the precision-recall curve:
    precision = precision[np.nonzero(precision)[-1]][0].item()

    # build the evaluation data frame
    # TODO: are keys in `ious` sorted based on preds or targets? does that depend on the
    # backend used (i.e., pycocotools or faster_coco_eval)?
    image_path_dict = {
        image_key: image_path
        for image_key, (image_path, _) in enumerate(ground_df.groupby("image_path"))
    }
    numeric_to_label_dict = {val: label for label, val in label_dict.items()}
    iou_dict = mean_ap_dict["ious"]

    def process_image_class(image_key, class_val):
        iou_tensor = iou_dict[(image_key, class_val)]
        # check that `iou_tensor` is a two-dimensional tensor
        if iou_tensor != [] and len(iou_tensor.shape) == 2:
            # create cost matrix for assignment, with rows and columns respectively
            # representing predictions and ground truths and the cost being the area of
            # the intersection
            # ACHTUNG: since in most cases images are tiles (hence are rather small),
            # using the spatial index is slower
            # annot_sindex = annot_df.sindex
            # cost_arr = pred_df.geometry.apply(
            #     lambda pred_geom: annot_df.iloc[
            #         annot_sindex.intersection(pred_geom.bounds)
            #     ]
            #     .intersection(pred_geom)
            #     .area
            # ).fillna(0)
            # TODO: can we get the ground truth-prediction matches from torchmetrics?
            # we'd need to access the `coco_eval` variable within the
            # `MeanAveragePrecision.compute` method (see https://github.com/
            # Lightning-AI/torchmetrics/blob/master/src/torchmetrics/detection/
            # mean_ap.py#L522), but this would require modifying torchmetrics.
            # Additionally, how to access the matches would depend on the backend used
            # (i.e., pycocotools or faster_coco_eval).
            cost_arr = (
                predictions[
                    (predictions["image_path"] == image_path_dict[image_key])
                    & (predictions["label"] == numeric_to_label_dict[class_val])
                ]
                .geometry.apply(
                    lambda pred_geom: ground_df[
                        (ground_df["image_path"] == image_path_dict[image_key])
                        & (ground_df["label"] == numeric_to_label_dict[class_val])
                    ]
                    .intersection(pred_geom)
                    .area
                )
                .values
            )

            row_ind, col_ind = optimize.linear_sum_assignment(
                cost_arr,
                maximize=True,
            )
            return pd.DataFrame(
                {
                    "prediction_id": row_ind,
                    "truth_id": col_ind,
                    "IoU": iou_tensor[row_ind, col_ind],
                },
            ).sort_values("truth_id", ascending=True)
        else:
            # when iou_dict[(image_key, class_val)] is [], aka, no predictions
            # except ValueError, IndexError:
            return None

    results_df = pd.concat(
        [
            process_image_class(image_key, class_val)
            for image_key, class_val in iou_dict.keys()
        ],
        axis="rows",
        ignore_index=True,
    )

    # TODO: does this work even if we shuffle the data frames?
    results_df["image_path"] = ground_df["image_path"]
    # set true labels
    results_df["true_label"] = ground_df["label"]  # .map(label_dict)
    # set the geometry
    results_df["geometry"] = ground_df["geometry"]
    # set the score and predicted label
    # TODO: how to best manage the prediction_id-to-index of `predictions` mapping?
    results_df = results_df.merge(
        predictions[["score", "label"]].reset_index(drop=True),
        left_on="prediction_id",
        right_index=True,
    )
    results_df = results_df.rename(columns={"label": "predicted_label"})
    # set whether it is a match
    results_df["match"] = results_df["IoU"] >= iou_threshold

    # only matching boxes are considered in class recall
    class_recall = compute_class_recall(results_df[results_df["match"] == True])

    return {
        "results": results_df,
        "box_precision": precision,
        "box_recall": recall,
        "class_recall": class_recall,
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
        raise ValueError("More than one image passed to function: {}".format(plot_name))
    else:
        plot_name = plot_names[0]

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
            result = _point_recall_image_(
                predictions=image_predictions, ground_df=group
            )

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
