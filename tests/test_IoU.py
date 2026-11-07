# Test IoU
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from deepforest import IoU
from deepforest import get_data


def test_compute_IoU(m):
    csv_file = get_data("OSBS_029.csv")
    predictions = m.predict_file(csv_file=csv_file, root_dir=os.path.dirname(csv_file))
    ground_truth = pd.read_csv(csv_file)

    predictions['geometry'] = predictions.apply(lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
    predictions = gpd.GeoDataFrame(predictions, geometry='geometry')

    ground_truth['geometry'] = ground_truth.apply(lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax),
                                                  axis=1)
    ground_truth = gpd.GeoDataFrame(ground_truth, geometry='geometry')

    ground_truth.label = 0
    predictions.label = 0

    result = IoU.match_polygons(ground_truth, predictions)
    assert result.shape[0] == ground_truth.shape[0]
    assert sum(result.IoU) > 10


def test_match_points():
    ground_truth = gpd.GeoDataFrame(
        {
            "image_path": ["OSBS_029.tif", "OSBS_029.tif"],
            "geometry": [
                shapely.geometry.Point(0, 0),
                shapely.geometry.Point(10, 10),
            ],
        },
        index=[10, 20],
    )
    submission = gpd.GeoDataFrame(
        {
            "image_path": ["OSBS_029.tif", "OSBS_029.tif"],
            "score": [0.9, 0.8],
            "geometry": [
                shapely.geometry.Point(3, 4),
                shapely.geometry.Point(10, 12),
            ],
        },
        index=[1, 2],
    )

    res_l2 = IoU.match_points(ground_truth, submission, norm="l2")
    assert list(res_l2["prediction_id"]) == [1, 2]
    assert np.allclose(res_l2["distance"].to_numpy(), [5.0, 2.0])

    res_l1 = IoU.match_points(ground_truth, submission, norm="l1")
    assert list(res_l1["prediction_id"]) == [1, 2]
    assert np.allclose(res_l1["distance"].to_numpy(), [7.0, 2.0])

    res_empty = IoU.match_points(ground_truth.iloc[0:0], submission)
    assert res_empty.empty

    res_unmatched = IoU.match_points(
        ground_truth, submission.iloc[:1].drop(columns=["score"])
    )
    unmatched_row = res_unmatched.loc[res_unmatched["truth_id"] == 20].iloc[0]
    assert unmatched_row["distance"] == float("inf")
    assert pd.isna(unmatched_row["prediction_id"])

    multi_image_sub = submission.assign(
        image_path=["OSBS_029.tif", "other.tif"]
    )
    with pytest.raises(ValueError, match="More than one image"):
        IoU.match_points(ground_truth, multi_image_sub)

    with pytest.raises(ValueError, match="Unknown norm type"):
        IoU.match_points(ground_truth, submission, norm="invalid")
