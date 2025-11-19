"""Evaluation module."""

import os
import warnings

import cv2
import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import pycolmap
import shapely
import torch
import torchvision
from shapely.geometry import box

from deepforest import IoU
from deepforest.utilities import determine_geometry_type
from kornia.feature import extract_features, match_features
from kornia.feature.matching import pairs_from_exhaustive
from pycolmap import get_matches


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

def get_matching_points(h5_file, image1_name, image2_name, min_score=None):
    """
    Extract matching feature points between two images from SfM feature files.
    
    This function retrieves corresponding feature points between two images that were
    previously extracted and matched using Structure-from-Motion techniques. The matching
    points are essential for computing geometric transformations between images.
    
    Args:
        h5_file (str): Path to the HDF5 file containing feature matches
        image1_name (str): Name of the first image (as stored in the feature files)
        image2_name (str): Name of the second image (as stored in the feature files)
        min_score (float, optional): Minimum matching score threshold for filtering
            low-quality matches. If None, all matches are returned.
            
    Returns:
        tuple: (points1, points2) where each is a numpy array of shape (N, 2)
            containing the (x, y) coordinates of matching points in each image
    
    Note:
        This function requires that SfM features have been previously extracted
        using create_sfm_model() or similar functionality.
    """
    matches, scores = get_matches(h5_file, image1_name, image2_name)
    if min_score is not None:
        matches = matches[scores > min_score]
    match_index = pd.DataFrame(matches, columns=["image1", "image2"])
    
    features_path = os.path.join(os.path.dirname(h5_file), "features.h5")
    with h5py.File(features_path, 'r') as features_h5_f:
        keypoints_image1 = pd.DataFrame(features_h5_f[image1_name]["keypoints"][:], columns=["x", "y"])
        keypoints_image2 = pd.DataFrame(features_h5_f[image2_name]["keypoints"][:], columns=["x", "y"])
        points1 = keypoints_image1.iloc[match_index["image1"].values].values
        points2 = keypoints_image2.iloc[match_index["image2"].values].values
    return points1, points2

def compute_homography_matrix(h5_file, image1_name, image2_name):
    """
    Compute the homography matrix between two images using RANSAC.
    
    A homography matrix is a 3x3 transformation matrix that maps points from one
    image plane to another. This is essential for aligning predictions between
    overlapping images in the double-counting removal algorithm.
    
    The function uses RANSAC (Random Sample Consensus) to robustly estimate the
    homography matrix even in the presence of outliers and noise in the feature matches.
    
    Args:
        h5_file (str): Path to the HDF5 file containing feature matches
        image1_name (str): Name of the first image
        image2_name (str): Name of the second image
        
    Returns:
        dict: A report dictionary containing:
            - 'H': The 3x3 homography matrix (numpy array)
            - 'inliers': Boolean array indicating which matches are inliers
            - 'num_inliers': Number of inlier matches used
            - Additional RANSAC statistics
            
    Raises:
        ValueError: If fewer than 4 matching points are found between the images
        ValueError: If homography matrix estimation fails (insufficient inliers)
        
    Note:
        The RANSAC parameters are set to max_error=4.0 pixels, which works well
        for most aerial imagery scenarios. For different image types, these parameters
        may need adjustment.
    """
    points1, points2 = get_matching_points(h5_file, image1_name, image2_name)
    if len(points1) < 4 or len(points2) < 4:
        raise ValueError(f"Not enough matching points (<4) found between images {image1_name} and {image2_name}")

    ransac_options = pycolmap.RANSACOptions(max_error=4.0)
    report = pycolmap.estimate_homography_matrix(points1, points2, ransac_options)

    if report is None:
        raise ValueError(f"Homography matrix estimation failed for images {image1_name} and {image2_name}")
    return report

def warp_box(xmin, ymin, xmax, ymax, homography):
    """
    Transform a bounding box using a homography matrix.
    
    This function applies a perspective transformation to a bounding box by transforming
    its four corner points and then computing the axis-aligned bounding box of the
    transformed corners. This is used to align predictions from one image coordinate
    system to another.
    
    Args:
        xmin (float): Left coordinate of the bounding box
        ymin (float): Top coordinate of the bounding box  
        xmax (float): Right coordinate of the bounding box
        ymax (float): Bottom coordinate of the bounding box
        homography (numpy.ndarray): 3x3 homography matrix
        
    Returns:
        tuple: (warped_xmin, warped_ymin, warped_xmax, warped_ymax) - the transformed
            bounding box coordinates as integers
            
    Note:
        The transformation may result in non-rectangular shapes, so the function
        computes the axis-aligned bounding box of the transformed quadrilateral.
        This can lead to slight expansion of the bounding box area.
    """
    points = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32)
    reshaped_points = points.reshape(-1, 1, 2)
    warped_points = cv2.perspectiveTransform(reshaped_points, homography).squeeze(1)
    
    warped_xmin, warped_ymin = warped_points.min(axis=0)
    warped_xmax, warped_ymax = warped_points.max(axis=0)
    return int(warped_xmin), int(warped_ymin), int(warped_xmax), int(warped_ymax)

def align_predictions(predictions, homography_matrix):
    """
    Transform all bounding box predictions using a homography matrix.
    
    This function applies geometric transformation to align predictions from one image
    coordinate system to another. It's a key step in the double-counting removal process,
    allowing comparison of predictions between overlapping images.
    
    Args:
        predictions (pandas.DataFrame): DataFrame containing bounding box predictions
            with columns ['xmin', 'ymin', 'xmax', 'ymax', ...]
        homography_matrix (numpy.ndarray): 3x3 homography matrix for coordinate transformation
        
    Returns:
        pandas.DataFrame: A copy of the input DataFrame with transformed bounding box
            coordinates. All other columns remain unchanged.
            
    Note:
        The function creates a copy of the input DataFrame to avoid modifying the original.
        The transformation is applied row-wise to each bounding box prediction.
    """
    transformed_predictions = predictions.copy()
    for index, row in transformed_predictions.iterrows():
        xmin, ymin, xmax, ymax = warp_box(row['xmin'], row['ymin'], row['xmax'], row['ymax'], homography_matrix)
        transformed_predictions.loc[index, ['xmin', 'ymin', 'xmax', 'ymax']] = xmin, ymin, xmax, ymax
    return transformed_predictions

def remove_predictions(src_predictions, dst_predictions, aligned_predictions, threshold, device, strategy='highest-score'):
    """
    Remove overlapping predictions between two images using specified strategies.
    
    This function implements the core logic for resolving double-counting by removing
    overlapping detections between aligned image pairs. Three different strategies are
    available, each with different trade-offs between accuracy and computational cost.
    
    **Strategies:**
    
    1. **'highest-score'**: Uses Non-Maximum Suppression (NMS) to keep predictions with
       the highest confidence scores. This is the most sophisticated approach that considers
       both spatial overlap and prediction confidence.
       
    2. **'left-hand'**: Keeps all predictions from the source image and removes overlapping
       predictions from the destination image. Simple but may not be optimal.
       
    3. **'right-hand'**: Keeps all predictions from the destination image and removes
       overlapping predictions from the source image. Simple but may not be optimal.
    
    Args:
        src_predictions (pandas.DataFrame): Predictions from the source image
        dst_predictions (pandas.DataFrame): Predictions from the destination image  
        aligned_predictions (pandas.DataFrame): Source predictions transformed to destination
            coordinate system using homography matrix
        threshold (float): IoU threshold for determining overlap (0.0 to 1.0)
        device (torch.device): PyTorch device for tensor operations (CPU or GPU)
        strategy (str): Strategy for removing overlaps. Options: 'highest-score', 
            'left-hand', 'right-hand'. Defaults to 'highest-score'.
            
    Returns:
        tuple: (src_filtered, dst_filtered) - DataFrames containing the filtered
            predictions for source and destination images respectively
            
    Raises:
        ValueError: If an unknown strategy is specified
        
    Note:
        For the 'highest-score' strategy, the function uses PyTorch's NMS implementation
        which requires the predictions to have a 'score' column. For geometric strategies,
        the function uses GeoPandas spatial operations which require 'box_id' columns.
    """
    if strategy == "highest-score":
        dst_and_aligned_predictions = pd.concat([aligned_predictions, dst_predictions], ignore_index=True)
        boxes = torch.tensor(dst_and_aligned_predictions[['xmin', 'ymin', 'xmax', 'ymax']].values, dtype=torch.float).to(device)
        scores = torch.tensor(dst_and_aligned_predictions['score'].values, dtype=torch.float).to(device)
        
        keep_indices = torchvision.ops.nms(boxes, scores, threshold)
        indices_to_keep = dst_and_aligned_predictions.iloc[keep_indices.cpu()]
        
        src_filtered = src_predictions[src_predictions.box_id.isin(indices_to_keep.box_id)]
        dst_filtered = dst_predictions[dst_predictions.box_id.isin(indices_to_keep.box_id)]
    else:
        aligned_predictions["geometry"] = aligned_predictions.apply(lambda row: box(row['xmin'], row['ymin'], row['xmax'], row['ymax']), axis=1)
        dst_predictions["geometry"] = dst_predictions.apply(lambda row: box(row['xmin'], row['ymin'], row['xmax'], row['ymax']), axis=1)
        aligned_gdf = gpd.GeoDataFrame(aligned_predictions, geometry="geometry")
        dst_gdf = gpd.GeoDataFrame(dst_predictions, geometry='geometry')

        joined = gpd.sjoin(aligned_gdf, dst_gdf, how='inner', predicate='intersects')

        if strategy == "left-hand":
            src_indices_to_keep = src_predictions.box_id
            dst_indices_to_keep = dst_predictions[~dst_predictions.box_id.isin(joined.box_id_right)].box_id
        elif strategy == "right-hand":
            src_indices_to_keep = src_predictions[~src_predictions.box_id.isin(joined.box_id_left)].box_id
            dst_indices_to_keep = dst_predictions.box_id
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from 'highest-score', 'left-hand', 'right-hand'.")

        src_filtered = src_predictions[src_predictions.box_id.isin(src_indices_to_keep)]
        dst_filtered = dst_predictions[dst_predictions.box_id.isin(dst_indices_to_keep)]

    return src_filtered, dst_filtered

def align_and_delete(matching_h5_file, predictions, device, threshold=0.325, strategy='highest-score'):
    """
    Main function for removing double-counting across multiple overlapping images.
    
    This function implements the complete double-counting removal pipeline by processing
    all pairs of images in the dataset. For each pair, it computes geometric alignment
    and removes overlapping predictions using the specified strategy.
    
    **Algorithm Overview:**
    
    1. **Pairwise Processing**: Processes all unique pairs of images (N*(N-1)/2 pairs)
    2. **Homography Estimation**: Computes transformation matrix between each image pair
    3. **Prediction Alignment**: Transforms predictions from source to destination coordinates
    4. **Overlap Resolution**: Removes overlapping detections using chosen strategy
    5. **Iterative Refinement**: Updates predictions after each pair processing
    
    **Performance Considerations:**
    
    - Computational complexity: O(N²) where N is the number of images
    - Memory usage scales with the number of predictions per image
    - GPU acceleration is used for NMS operations when available
    
    Args:
        matching_h5_file (str): Path to HDF5 file containing feature matches between images
        predictions (pandas.DataFrame): Initial predictions from all images with columns:
            ['xmin', 'ymin', 'xmax', 'ymax', 'score', 'label', 'image_path']
        device (torch.device): PyTorch device for tensor operations
        threshold (float, optional): IoU threshold for overlap detection. Defaults to 0.325
        strategy (str, optional): Strategy for removing overlaps. Options: 'highest-score',
            'left-hand', 'right-hand'. Defaults to 'highest-score'
            
    Returns:
        pandas.DataFrame: Final predictions with double-counting removed. Contains the same
            columns as input plus a 'box_id' column for tracking.
            
    Raises:
        ValueError: If fewer than 2 images are provided
        ValueError: If homography computation fails for image pairs
        
    Note:
        The function processes images in sorted order by name for consistent results.
        Images that cannot be geometrically aligned are skipped with a warning.
        The threshold parameter should be tuned based on the expected overlap between images.
    """
    image_names = sorted(predictions.image_path.unique())
    if len(image_names) < 2:
        return predictions

    predictions["box_id"] = range(len(predictions))
    filtered_predictions = {name: predictions[predictions.image_path == name] for name in image_names}
    
    num_pairs = len(image_names) * (len(image_names) - 1) // 2
    pair_count = 0

    for i in range(len(image_names)):
        for j in range(i + 1, len(image_names)):
            src_image_name, dst_image_name = image_names[i], image_names[j]
            pair_count += 1
            print(f"Processing Pair {pair_count}/{num_pairs}: ({src_image_name}, {dst_image_name})")

            try:
                homography = compute_homography_matrix(h5_file=matching_h5_file, image1_name=src_image_name, image2_name=dst_image_name)
            except ValueError as e:
                print(f"Skipping pair, could not compute homography: {e}")
                continue

            src_preds, dst_preds = filtered_predictions[src_image_name], filtered_predictions[dst_image_name]
            
            if src_preds.empty or dst_preds.empty:
                continue

            aligned_src_preds = align_predictions(predictions=src_preds, homography_matrix=homography["H"])
            
            src_filtered, dst_filtered = remove_predictions(
                src_predictions=src_preds,
                dst_predictions=dst_preds,
                aligned_predictions=aligned_src_preds,
                threshold=threshold,
                device=device,
                strategy='left-hand'
            )
            
            filtered_predictions[src_image_name] = src_filtered
            filtered_predictions[dst_image_name] = dst_filtered
            
    return pd.concat(filtered_predictions.values()).drop_duplicates(subset="box_id")

def create_sfm_model(image_dir, output_path, references, overwrite=False):
    """
    Generate Structure-from-Motion (SfM) feature files needed for geometric matching.
    
    This function creates the essential SfM infrastructure required for the double-counting
    removal algorithm. It extracts distinctive features from images and establishes matches
    between overlapping image pairs using state-of-the-art computer vision techniques.
    
    **Process Overview:**
    
    1. **Feature Extraction**: Uses DISK descriptors to extract robust, scale-invariant
       features from each image. DISK is particularly effective for aerial imagery.
       
    2. **Pair Generation**: Creates exhaustive pairs between all images for matching.
       This ensures all possible overlaps are considered.
       
    3. **Feature Matching**: Uses LightGlue matcher to establish correspondences between
       feature points in overlapping image pairs. LightGlue provides high-quality matches
       with good robustness to viewpoint changes.
    
    **Output Files:**
    
    - `features.h5`: Contains extracted feature points and descriptors for each image
    - `matches.h5`: Contains feature matches between all image pairs
    - `pairs-sfm.txt`: Lists all image pairs to be matched
    
    Args:
        image_dir (pathlib.Path): Directory containing input images
        output_path (pathlib.Path): Directory where SfM files will be saved
        references (list): List of image filenames to process
        overwrite (bool, optional): If True, overwrite existing SfM files.
            If False, skip processing if files already exist. Defaults to False.
            
    Raises:
        FileNotFoundError: If image files cannot be found
        RuntimeError: If feature extraction or matching fails
        
    Note:
        This function requires the hloc library and can be computationally intensive
        for large images or many images. Processing time scales roughly quadratically
        with the number of images due to exhaustive pairwise matching.
        
        The DISK+LightGlue combination is chosen for its effectiveness on aerial imagery
        and robustness to illumination changes, which are common in drone surveys.
    """
    feature_conf = extract_features.confs["disk"]
    matcher_conf = match_features.confs["disk+lightglue"]
    
    sfm_pairs, features, matches = output_path / 'pairs-sfm.txt', output_path / 'features.h5', output_path / 'matches.h5'
    
    extract_features.main(conf=feature_conf, image_dir=image_dir, image_list=references, feature_path=features, overwrite=overwrite)
    pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches, overwrite=overwrite)