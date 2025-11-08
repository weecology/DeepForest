"""Keypoint Distance Module for matching predicted keypoints to ground truth.

Similar to IoU.py but uses Euclidean pixel distance instead of
intersection-over-union.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def _compute_distances(predictions: "gpd.GeoDataFrame", ground_truth: "gpd.GeoDataFrame"):
    """Computes pairwise Euclidean distances between all predicted and ground
    truth keypoints.

    Args:
        predictions: GeoDataFrame with Point geometry (predicted keypoints)
        ground_truth: GeoDataFrame with Point geometry (ground truth keypoints)

    Returns:
        distances: (n_truth, n_pred) array of Euclidean distances in pixels
        truth_ids: (n_truth,) truth index values
        pred_ids: (n_pred,) prediction index values
    """
    # Extract coordinates from Point geometries
    pred_coords = np.array([[p.x, p.y] for p in predictions.geometry])
    truth_coords = np.array([[p.x, p.y] for p in ground_truth.geometry])

    pred_ids = predictions.index.to_numpy()
    truth_ids = ground_truth.index.to_numpy()

    n_pred = len(pred_coords)
    n_truth = len(truth_coords)

    # Handle empty cases
    if n_pred == 0 or n_truth == 0:
        return (
            np.full((n_truth, n_pred), np.inf, dtype=float),
            truth_ids,
            pred_ids,
        )

    # Compute pairwise Euclidean distances
    # Broadcasting: (n_truth, 1, 2) - (1, n_pred, 2) = (n_truth, n_pred, 2)
    distances = np.sqrt(
        ((truth_coords[:, np.newaxis, :] - pred_coords[np.newaxis, :, :]) ** 2).sum(
            axis=2
        )
    )

    return distances, truth_ids, pred_ids


# TODO - consider making this a shared/generic function with IoU where we can pass in
# indices + costs.
def compute_distances(ground_truth: "gpd.GeoDataFrame", predictions: "gpd.GeoDataFrame"):
    """Match predicted keypoints to ground truth using Hungarian algorithm with
    pixel distance.

    This function performs matching between ground truth and predicted keypoints.
    For each ground truth keypoint, we compute the Euclidean pixel distance to all
    predictions. These distances are used as the cost matrix for Hungarian matching,
    which ensures that each ground truth is matched to at most one prediction, and
    each prediction is used at most once, minimizing the total distance.

    No filtering on distance threshold or score is performed - that happens downstream.

    Args:
        ground_truth: a geopandas dataframe with Point geometry
        predictions: a geopandas dataframe with Point geometry

    Returns:
        distance_df: dataframe with columns:
            - prediction_id: matched prediction ID (or None if no match)
            - truth_id: ground truth ID
            - distance: Euclidean pixel distance
            - score: prediction confidence score (if available)
            - geometry: ground truth geometry
    """
    # Compute pairwise distances
    distance_matrix, truth_ids, pred_ids = _compute_distances(
        predictions=predictions, ground_truth=ground_truth
    )

    if distance_matrix.size == 0:
        # No matches, early exit
        return pd.DataFrame(
            {
                "prediction_id": pd.Series(dtype="float64"),
                "truth_id": pd.Series(dtype=truth_ids.dtype),
                "distance": pd.Series(dtype="float64"),
                "score": pd.Series(dtype="float64"),
                "geometry": pd.Series(dtype=object),
            }
        )

    # Linear sum assignment (minimizes total distance)
    # We want to MINIMIZE distance, so no need for maximize=True
    row_ind, col_ind = linear_sum_assignment(distance_matrix, maximize=False)
    match_for_truth = dict(zip(row_ind, col_ind, strict=False))

    # Score lookup
    pred_scores = predictions["score"].to_dict() if "score" in predictions.columns else {}

    # Build rows for every truth element (unmatched => None, distance inf)
    records = []
    for t_idx, truth_id in enumerate(truth_ids):
        # If we matched this truth keypoint
        if t_idx in match_for_truth:
            # Look up matching prediction and corresponding distance and score
            p_idx = match_for_truth[t_idx]
            matched_id = pred_ids[p_idx]
            distance = float(distance_matrix[t_idx, p_idx])
            score = pred_scores.get(matched_id, None)
        else:
            matched_id = None
            distance = np.inf
            score = None

        records.append(
            {
                "prediction_id": matched_id,
                "truth_id": truth_id,
                "distance": distance,
                "score": score,
            }
        )

    # Output dataframe
    distance_df = pd.DataFrame.from_records(records)
    distance_df = distance_df.merge(
        ground_truth.assign(truth_id=truth_ids)[["truth_id", "geometry"]],
        on="truth_id",
        how="left",
    )

    return distance_df
