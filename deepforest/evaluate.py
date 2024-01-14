"""
Evaluation module
"""
import pandas as pd
import geopandas as gpd
import shapely
import numpy as np
import cv2
from PIL import Image

from deepforest import IoU
from deepforest.utilities import check_file
from deepforest import visualize
import warnings


def evaluate_image(predictions, ground_df, root_dir, savedir=None):
    """
    Compute intersection-over-union matching among prediction and ground truth boxes for one image
    Args:
        df: a pandas dataframe with columns name, xmin, xmax, ymin, ymax, label. The 'name' column should be the path relative to the location of the file.
        summarize: Whether to group statistics by plot and overall score
        image_coordinates: Whether the current boxes are in coordinate system of the image, e.g. origin (0,0) upper left.
        root_dir: Where to search for image names in df
        savedir: optional directory to save image with overlaid predictions and annotations
    Returns:
        result: pandas dataframe with crown ids of prediciton and ground truth and the IoU score.
    """
    plot_names = predictions["image_path"].unique()
    if len(plot_names) > 1:
        raise ValueError("More than one plot passed to image crown: {}".format(plot_name))
    else:
        plot_name = plot_names[0]

    predictions['geometry'] = predictions.apply(
        lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
    predictions = gpd.GeoDataFrame(predictions, geometry='geometry')

    ground_df['geometry'] = ground_df.apply(
        lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
    ground_df = gpd.GeoDataFrame(ground_df, geometry='geometry')

    # match
    result = IoU.compute_IoU(ground_df, predictions)

    # add the label classes
    result["predicted_label"] = result.prediction_id.apply(
        lambda x: predictions.label.loc[x] if pd.notnull(x) else x)
    result["true_label"] = result.truth_id.apply(lambda x: ground_df.label.loc[x])

    if savedir:
        image = np.array(Image.open("{}/{}".format(root_dir, plot_name)))[:, :, ::-1]
        image = visualize.plot_predictions(image, df=predictions)
        image = visualize.plot_predictions(image, df=ground_df, color=(0, 165, 255))
        cv2.imwrite("{}/{}".format(savedir, plot_name), image)

    return result


def compute_class_recall(results):
    """Given a set of evaluations, what proportion of predicted boxes match. True boxes which are not matched to predictions do not count against accuracy."""
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
        class_recall_dict[name] = sum(
            group.true_label == group.predicted_label) / group.shape[0]
        number_of_predictions = box_results[box_results.predicted_label == name].shape[0]
        if number_of_predictions == 0:
            class_precision_dict[name] = 0
        else:
            class_precision_dict[name] = sum(
                group.true_label == group.predicted_label) / number_of_predictions
        class_size[name] = group.shape[0]

    class_recall = pd.DataFrame({
        "label": class_recall_dict.keys(),
        "recall": pd.Series(class_recall_dict),
        "precision": pd.Series(class_precision_dict),
        "size": pd.Series(class_size)
    }).reset_index(drop=True)

    return class_recall


def __evaluate_wrapper__(predictions,
                         ground_df,
                         root_dir,
                         iou_threshold,
                         numeric_to_label_dict,
                         savedir=None):
    """Evaluate a set of predictions against a ground truth csv file
        Args:   
            predictions: a pandas dataframe, if supplied a root dir is needed to give the relative path of files in df.name. The labels in ground truth and predictions must match. If one is numeric, the other must be numeric.
            csv_file: a csv file with columns xmin, ymin, xmax, ymax, label, image_path
            root_dir: location of files in the dataframe 'name' column.
            iou_threshold: intersection-over-union threshold, see deepforest.evaluate
            savedir: optional directory to save image with overlaid predictions and annotations
        Returns:
            results: a dictionary of results with keys, results, box_recall, box_precision, class_recall
        """
    # remove empty samples from ground truth
    ground_df = ground_df[~((ground_df.xmin == 0) & (ground_df.xmax == 0))]

    results = evaluate(predictions=predictions,
                       ground_df=ground_df,
                       root_dir=root_dir,
                       iou_threshold=iou_threshold,
                       savedir=savedir)

    # replace classes if not NUll
    if not results is None:
        results["results"]["predicted_label"] = results["results"][
            "predicted_label"].apply(lambda x: numeric_to_label_dict[x]
                                     if not pd.isnull(x) else x)
        results["results"]["true_label"] = results["results"]["true_label"].apply(
            lambda x: numeric_to_label_dict[x])
        results["predictions"] = predictions
        results["predictions"]["label"] = results["predictions"]["label"].apply(
            lambda x: numeric_to_label_dict[x])

    return results


def evaluate(predictions, ground_df, root_dir, iou_threshold=0.4, savedir=None):
    """Image annotated crown evaluation routine
    submission can be submitted as a .shp, existing pandas dataframe or .csv path

    Args:
        predictions: a pandas dataframe, if supplied a root dir is needed to give the relative path of files in df.name. The labels in ground truth and predictions must match. If one is numeric, the other must be numeric.
        ground_df: a pandas dataframe, if supplied a root dir is needed to give the relative path of files in df.name
        root_dir: location of files in the dataframe 'name' column.
        savedir: optional directory to save image with overlaid predictions and annotations
    Returns:
        results: a dataframe of match bounding boxes
        box_recall: proportion of true positives of box position, regardless of class
        box_precision: proportion of predictions that are true positive, regardless of class
        class_recall: a pandas dataframe of class level recall and precision with class sizes
    """

    check_file(ground_df)
    check_file(predictions)

    # Run evaluation on all plots
    results = []
    box_recalls = []
    box_precisions = []
    for image_path, group in ground_df.groupby("image_path"):
        # clean indices
        image_predictions = predictions[predictions["image_path"] ==
                                        image_path].reset_index(drop=True)

        # If empty, add to list without computing IoU
        if image_predictions.empty:
            result = pd.DataFrame({
                "truth_id": group.index.values,
                "prediction_id": None,
                "IoU": 0,
                "predicted_label": None,
                "score": None,
                "match": None,
                "true_label": group.label
            })
            # An empty prediction set has recall of 0, precision of NA.
            box_recalls.append(0)
            results.append(result)
            continue
        else:
            group = group.reset_index(drop=True)
            result = evaluate_image(predictions=image_predictions,
                                    ground_df=group,
                                    root_dir=root_dir,
                                    savedir=savedir)

        result["image_path"] = image_path
        result["match"] = result.IoU > iou_threshold
        true_positive = sum(result["match"])
        recall = true_positive / result.shape[0]
        precision = true_positive / image_predictions.shape[0]

        box_recalls.append(recall)
        box_precisions.append(precision)
        results.append(result)

    results = pd.concat(results)
    box_precision = np.mean(box_precisions)
    box_recall = np.mean(box_recalls)

    # Only matching boxes are considered in class recall
    matched_results = results[results.match == True]
    class_recall = compute_class_recall(matched_results)

    return {
        "results": results,
        "box_precision": box_precision,
        "box_recall": box_recall,
        "class_recall": class_recall
    }


def _point_recall_image_(predictions, ground_df, root_dir=None, savedir=None):
    """
    Compute intersection-over-union matching among prediction and ground truth boxes for one image
    Args:
        predictions: a pandas dataframe. The labels in ground truth and predictions must match. For example, if one is numeric, the other must be numeric.
        ground_df: a pandas dataframe
        root_dir: location of files in the dataframe 'name' column, only needed if savedir is supplied
        savedir: optional directory to save image with overlaid predictions and annotations
    Returns:
        result: pandas dataframe with crown ids of prediciton and ground truth and the IoU score.
    """
    plot_names = predictions["image_path"].unique()
    if len(plot_names) > 1:
        raise ValueError("More than one image passed to function: {}".format(plot_name))
    else:
        plot_name = plot_names[0]

    predictions['geometry'] = predictions.apply(
        lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
    predictions = gpd.GeoDataFrame(predictions, geometry='geometry')

    ground_df['geometry'] = ground_df.apply(lambda x: shapely.geometry.Point(x.x, x.y),
                                            axis=1)
    ground_df = gpd.GeoDataFrame(ground_df, geometry='geometry')

    # Which points in boxes
    result = gpd.sjoin(ground_df, predictions, op='within', how="left")
    result = result.rename(
        columns={
            "label_left": "true_label",
            "label_right": "predicted_label",
            "image_path_left": "image_path"
        })
    result = result.drop(columns=["index_right"])

    if savedir:
        if root_dir is None:
            raise AttributeError("savedir is {}, but root dir is None".format(savedir))
        image = np.array(Image.open("{}/{}".format(root_dir, plot_name)))[:, :, ::-1]
        image = visualize.plot_predictions(image, df=predictions)
        image = visualize.plot_points(image, df=ground_df, color=(0, 165, 255))
        cv2.imwrite("{}/{}".format(savedir, plot_name), image)

    return result


def point_recall(predictions, ground_df, root_dir=None, savedir=None):
    """Evaluate the proportion on ground truth points overlap with predictions
    submission can be submitted as a .shp, existing pandas dataframe or .csv path
    For bounding box recall, see evaluate(). 
    Args:
        predictions: a pandas dataframe, if supplied a root dir is needed to give the relative path of files in df.name. The labels in ground truth and predictions must match. If one is numeric, the other must be numeric.
        ground_df: a pandas dataframe, if supplied a root dir is needed to give the relative path of files in df.name
        root_dir: location of files in the dataframe 'name' column.
        savedir: optional directory to save image with overlaid predictions and annotations
    Returns:
        results: a dataframe of matched bounding boxes and ground truth labels
        box_recall: proportion of true positives between predicted boxes and ground truth points, regardless of class
        class_recall: a pandas dataframe of class level recall and precision with class sizes
    """
    check_file(predictions)
    if savedir:
        if root_dir is None:
            raise AttributeError("savedir is {}, but root dir is None".format(savedir))

    # Run evaluation on all images
    results = []
    box_recalls = []
    for image_path, group in ground_df.groupby("image_path"):
        image_predictions = predictions[predictions["image_path"] ==
                                        image_path].reset_index(drop=True)

        # If empty, add to list without computing recall
        if image_predictions.empty:
            result = pd.DataFrame({
                "recall": 0,
                "predicted_label": None,
                "score": None,
                "true_label": group.label
            })
            # An empty prediction set has recall of 0, precision of NA.
            box_recalls.append(0)
            results.append(result)
            continue
        else:
            group = group.reset_index(drop=True)
            result = _point_recall_image_(predictions=image_predictions,
                                          ground_df=group,
                                          root_dir=root_dir,
                                          savedir=savedir)

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
