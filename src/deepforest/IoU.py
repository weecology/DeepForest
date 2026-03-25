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
      t_idx       : (M,) truth row indices for each overlapping pair
      p_idx       : (M,) pred col indices for each overlapping pair
      inter_areas : (M,) intersection area for each overlapping pair
      union_areas : (M,) union area for each overlapping pair
      truth_ids   : (n_truth,) truth index values
      pred_ids    : (n_pred,) prediction index values
    """
    # geometry arrays
    pred_geoms = np.asarray(test_polys.geometry.values, dtype=object)
    truth_geoms = np.asarray(truth_polys.geometry.values, dtype=object)

    pred_ids = test_polys.index.to_numpy()
    truth_ids = truth_polys.index.to_numpy()

    n_pred = pred_geoms.size
    n_truth = truth_geoms.size

    _empty = np.array([], dtype=np.intp)
    _empty_f = np.array([], dtype=float)

    if n_pred == 0 or n_truth == 0:
        return _empty, _empty, _empty_f, _empty_f, truth_ids, pred_ids

    # spatial index on truth
    tree = STRtree(truth_geoms)
    p_idx, t_idx = tree.query(pred_geoms, predicate="intersects")

    if p_idx.size == 0:
        return _empty, _empty, _empty_f, _empty_f, truth_ids, pred_ids

    inter_areas = shapely.area(shapely.intersection(truth_geoms[t_idx], pred_geoms[p_idx]))
    union_areas = shapely.area(shapely.union(truth_geoms[t_idx], pred_geoms[p_idx]))

    return t_idx, p_idx, inter_areas, union_areas, truth_ids, pred_ids


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
    # Compute truth --- prediction overlaps (sparse)
    t_idx, p_idx, inter_areas, union_areas, truth_ids, pred_ids = _overlap_all(
        test_polys=submission, truth_polys=ground_truth
    )

    if truth_ids.size == 0 or pred_ids.size == 0 or t_idx.size == 0:
        records = [
            {"prediction_id": None, "truth_id": tid, "IoU": 0.0, "score": None}
            for tid in truth_ids
        ]
        iou_df = pd.DataFrame.from_records(records) if records else pd.DataFrame(
            {
                "prediction_id": pd.Series(dtype="float64"),
                "truth_id": pd.Series(dtype=truth_ids.dtype),
                "IoU": pd.Series(dtype="float64"),
                "score": pd.Series(dtype="float64"),
                "geometry": pd.Series(dtype=object),
            }
        )
        if "geometry" not in iou_df.columns:
            iou_df = iou_df.merge(
                ground_truth.assign(truth_id=truth_ids)[["truth_id", "geometry"]],
                on="truth_id", how="left",
            )
        return iou_df

    # Sparse IoU for each overlapping pair
    with np.errstate(divide="ignore", invalid="ignore"):
        iou_values = np.where(union_areas > 0, inter_areas / union_areas, 0.0)
    iou_lookup = dict(zip(zip(t_idx.tolist(), p_idx.tolist()), iou_values.tolist()))

    # Score lookup
    pred_scores = submission["score"].to_dict() if "score" in submission.columns else {}

    # Stage 1: resolve unambiguous 1:1 matches directly
    truth_counts = np.bincount(t_idx, minlength=len(truth_ids))
    pred_counts = np.bincount(p_idx, minlength=len(pred_ids))
    is_unique = (truth_counts[t_idx] == 1) & (pred_counts[p_idx] == 1)

    match_for_truth = {}
    if is_unique.any():
        for ti, pi in zip(t_idx[is_unique].tolist(), p_idx[is_unique].tolist()):
            match_for_truth[ti] = pi

    # Stage 2: Hungarian matching on ambiguous remainder only
    ambig = ~is_unique
    if ambig.any():
        amb_t = t_idx[ambig]
        amb_p = p_idx[ambig]
        amb_inter = inter_areas[ambig]

        active_t = np.unique(amb_t)
        active_p = np.unique(amb_p)
        t_map = {int(v): i for i, v in enumerate(active_t)}
        p_map = {int(v): i for i, v in enumerate(active_p)}

        sub = np.zeros((len(active_t), len(active_p)), dtype=float)
        for ti, pi, ia in zip(amb_t.tolist(), amb_p.tolist(), amb_inter.tolist()):
            sub[t_map[ti], p_map[pi]] = ia

        sub_r, sub_c = linear_sum_assignment(sub, maximize=True)
        for r, c in zip(sub_r.tolist(), sub_c.tolist()):
            if sub[r, c] > 0:
                match_for_truth.setdefault(int(active_t[r]), int(active_p[c]))

    # Build rows for every truth element (unmatched ==> None, IoU 0)
    records = []
    for ti, truth_id in enumerate(truth_ids):
        if ti in match_for_truth:
            pi = match_for_truth[ti]
            matched_id = pred_ids[pi]
            iou = iou_lookup.get((ti, pi), 0.0)
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
