"""
IoU Module, with help from https://github.com/SpaceNetChallenge/utilities/blob/spacenetV3/spacenetutilities/evalTools.py
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.optimize import linear_sum_assignment
from shapely import STRtree


def _overlap_all(test_polys: "gpd.GeoDataFrame", truth_polys: "gpd.GeoDataFrame"):
    """Computes intersection and union areas for all polygons in the test/truth
    dataframes.

    For efficient querying, truth polygons are stored in a spatial R-Tree
    and we only compute intersections/unions for matching pairs. The output from the
    function are Numpy arrays containing the all-to-all intersection and union areas and
    the indices of intersecting ground truth and prediction polygons.

    This method works with any Shapely polygon, but may have
    reduced performance for the polygon case where bounding box intersection does
    not necessarily mean the vertices intersect. For rectangles, it's efficient
    as the an r-tree hit is usually a true intersection, depending on how touching
    edge cases are handled.

    Returns:
      intersections  : (n_truth, n_pred) intersection areas
      unions : (n_truth, n_pred) union areas
      truth_ids : (n_truth,) truth index values (order matches rows of areas/unions)
      pred_ids  : (n_pred,) prediction index values (order matches cols of areas/unions)
    """
    # geometry arrays
    pred_geoms = np.asarray(test_polys.geometry.values, dtype=object)
    truth_geoms = np.asarray(truth_polys.geometry.values, dtype=object)

    pred_ids = test_polys.index.to_numpy()
    truth_ids = truth_polys.index.to_numpy()

    n_pred = pred_geoms.size
    n_truth = truth_geoms.size

    # empty cases
    if n_pred == 0 or n_truth == 0:
        return (
            np.zeros((n_truth, n_pred), dtype=float),
            np.zeros((n_truth, n_pred), dtype=float),
            truth_ids,
            pred_ids,
        )

    # spatial index on truth
    tree = STRtree(truth_geoms)
    p_idx, t_idx = tree.query(pred_geoms, predicate="intersects")  # shape (2, M)

    intersections = np.zeros((n_truth, n_pred), dtype=float)
    unions = np.zeros((n_truth, n_pred), dtype=float)

    if p_idx.size:
        inter = shapely.intersection(truth_geoms[t_idx], pred_geoms[p_idx])
        uni = shapely.union(truth_geoms[t_idx], pred_geoms[p_idx])
        intersections[t_idx, p_idx] = shapely.area(inter)
        unions[t_idx, p_idx] = shapely.area(uni)

    return intersections, unions, truth_ids, pred_ids


def compute_IoU(ground_truth: "gpd.GeoDataFrame", submission: "gpd.GeoDataFrame"):
    """Find area of overlap among all sets of ground truth and prediction.

    This function performs matching between a ground truth dataset and a
    submission or prediction dataset, typically the output from a validation or
    test run. In order to compute IoU, we must know which boxes correspond
    between the datasets. This is performed by Hungarian matching, or linear
    sum assignment.

    For each ground truth polygon, we compute the IoUs of all
    overlapping polygons. Intersection areas are used as the input cost matrix for the assignment and the
    algorithm is such that at most one prediction is assigned to each ground truth,
    and each prediction is only used at most once, with the solver aiming to
    maximise the total area of intersection. The matching indices are then returned,
    along with their IoUs and scores, to be used in downstream metrics like recall
    and precision.

    No filtering on IoU or score is performed.

    Args:
        ground_truth: a projected geopandas dataframe with geoemtry
        submission: a projected geopandas dataframe with geometry
    Returns:
        iou_df: dataframe of IoU scores
    """
    # Compute truth <> prediction overlaps
    intersections, unions, truth_ids, pred_ids = _overlap_all(
        test_polys=submission, truth_polys=ground_truth
    )

    # Cost matrix is the intersection area
    matrix = intersections

    if matrix.size == 0:
        # No matches, early exit
        return pd.DataFrame(
            {
                "prediction_id": pd.Series(dtype="float64"),
                "truth_id": pd.Series(dtype=truth_ids.dtype),
                "IoU": pd.Series(dtype="float64"),
                "score": pd.Series(dtype="float64"),
                "geometry": pd.Series(dtype=object),
            }
        )

    # Linear sum assignment + match lookup
    row_ind, col_ind = linear_sum_assignment(matrix, maximize=True)
    match_for_truth = dict(zip(row_ind, col_ind, strict=False))

    # Score lookup
    pred_scores = submission["score"].to_dict() if "score" in submission.columns else {}

    # IoU matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        iou_mat = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=float),
            where=unions > 0,
        )

    # build rows for every truth element (unmatched => None, IoU 0)
    records = []
    for t_idx, truth_id in enumerate(truth_ids):
        # If we matched this truth box
        if t_idx in match_for_truth:
            # Look up matching prediction and corresponding IoU and score
            p_idx = match_for_truth[t_idx]
            matched_id = pred_ids[p_idx]
            iou = float(iou_mat[t_idx, p_idx])
            score = pred_scores.get(matched_id, None)
        else:
            matched_id = None
            iou = 0.0
            score = None
        records.append(
            {
                "prediction_id": matched_id,
                "truth_id": truth_id,
                "IoU": iou,
                "score": score,
            }
        )

    # Output dataframe
    iou_df = pd.DataFrame.from_records(records)
    iou_df = iou_df.merge(
        ground_truth.assign(truth_id=truth_ids)[["truth_id", "geometry"]],
        on="truth_id",
        how="left",
    )
    return iou_df
