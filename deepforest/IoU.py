"""
IoU Module, with help from https://github.com/SpaceNetChallenge/utilities/blob/spacenetV3/spacenetutilities/evalTools.py
"""
import numpy as np
import rtree
import pandas as pd
from scipy.optimize import linear_sum_assignment


def create_rtree_from_poly(poly_list):
    # create index
    index = rtree.index.Index(interleaved=True)
    for idx, geom in enumerate(poly_list):
        index.insert(idx, geom.bounds)

    return index


def _overlap_(test_poly, truth_polys, rtree_index):
    """Calculate overlap between one polygon and all ground truth by area"""
    prediction_id = []
    truth_id = []
    area = []
    matched_list = list(rtree_index.intersection(test_poly.geometry.bounds))
    for index in truth_polys.index:
        if index in matched_list:
            # get the original index just to be sure
            intersection_result = test_poly.geometry.intersection(
                truth_polys.loc[index].geometry)
            intersection_area = intersection_result.area
        else:
            intersection_area = 0
        
        prediction_id.append(test_poly.prediction_id)
        truth_id.append(truth_polys.loc[index].truth_id) 
        area.append(intersection_area)

    results = pd.DataFrame({
        "prediction_id": prediction_id,
        "truth_id": truth_id,
        "area": area
    })
    return results


def _overlap_all(test_polys, truth_polys, rtree_index):
    """Find area of overlap among all sets of ground truth and prediction"""
    results = []
    for index, row in test_polys.iterrows():
        result = _overlap_(test_poly=row,
                           truth_polys=truth_polys,
                           rtree_index=rtree_index)
        results.append(result)
    results = pd.concat(results, ignore_index=True)

    return results


def _iou_(test_poly, truth_poly):
    """Intersection over union"""
    intersection_result = test_poly.intersection(truth_poly.geometry)
    intersection_area = intersection_result.area
    union_area = test_poly.union(truth_poly.geometry).area
    return (intersection_area / union_area)


def compute_IoU(ground_truth, submission):
    """
    Args:
        ground_truth: a projected geopandas dataframe with geoemtry
        submission: a projected geopandas dataframe with geometry
    Returns:
        iou_df: dataframe of IoU scores
        """
    # Create index columns for ease
    ground_truth["truth_id"] = ground_truth.index.values
    submission["prediction_id"] =  submission.index.values

    # rtree_index
    rtree_index = create_rtree_from_poly(ground_truth.geometry)

    # find overlap among all sets
    overlap_df = _overlap_all(test_polys=submission,
                              truth_polys=ground_truth,
                              rtree_index=rtree_index)

    # Create cost matrix for assignment
    matrix = overlap_df.pivot("truth_id", "prediction_id", "area").values
    row_ind, col_ind = linear_sum_assignment(matrix, maximize=True)

    # Create IoU dataframe, match those predictions and ground truth, IoU = 0
    # for all others, they will get filtered out
    iou_df = []
    for index, row in ground_truth.iterrows():
        if index in row_ind:
            matched_id = col_ind[np.where(index == row_ind)[0][0]]
            iou = _iou_(submission[submission.prediction_id == matched_id],
                          ground_truth.loc[index])
            score = submission[submission.prediction_id == matched_id].score.values[0]
        else:
            iou = 0
            matched_id = None
            score = None
        iou_df.append(
            pd.DataFrame({
                "prediction_id": [matched_id],
                "truth_id": [index],
                "IoU": iou,
                "score": score
            }))

    iou_df = pd.concat(iou_df)
    iou_df = iou_df.merge(ground_truth[["truth_id","xmin","xmax","ymin","ymax"]])
    
    return iou_df
