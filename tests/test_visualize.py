# Test visualize
from deepforest import visualize
from deepforest.utilities import read_file
from deepforest import get_data
import os
import pytest
import numpy as np
import pandas as pd

import geopandas as gpd
from shapely import geometry
import cv2


def test_format_boxes(m):
    ds = m.val_dataloader()
    batch = next(iter(ds))
    paths, images, targets = batch
    for path, image, target in zip(paths, images, targets):
        target_df = visualize.format_boxes(target, scores=False)
        assert list(target_df.columns.values) == ["xmin", "ymin", "xmax", "ymax", "label"]
        assert not target_df.empty


# Test different color labels
@pytest.mark.parametrize("label", [0, 1, 20])
def test_plot_predictions(m, tmpdir, label):
    ds = m.val_dataloader()
    batch = next(iter(ds))
    paths, images, targets = batch
    for path, image, target in zip(paths, images, targets):
        target_df = visualize.format_boxes(target, scores=False)
        target_df["image_path"] = path
        image = np.array(image)[:, :, ::-1]
        image = np.rollaxis(image, 0, 3)
        target_df.label = label
        image = visualize.plot_predictions(image, target_df)

        assert image.dtype == "uint8"


def test_plot_prediction_dataframe(m, tmpdir):
    ds = m.val_dataloader()
    batch = next(iter(ds))
    paths, images, targets = batch
    for path, image, target in zip(paths, images, targets):
        target_df = visualize.format_boxes(target, scores=False)
        target_df["image_path"] = path
        filenames = visualize.plot_prediction_dataframe(
            df=target_df, savedir=tmpdir, root_dir=m.config["validation"]["root_dir"])

    assert all([os.path.exists(x) for x in filenames])


def test_plot_predictions_and_targets(m, tmpdir):
    ds = m.val_dataloader()
    batch = next(iter(ds))
    paths, images, targets = batch
    m.model.eval()
    predictions = m.model(images)
    for path, image, target, prediction in zip(paths, images, targets, predictions):
        image = image.permute(1, 2, 0)
        save_figure_path = visualize.plot_prediction_and_targets(
            image, prediction, target, image_name=os.path.basename(path), savedir=tmpdir)
        assert os.path.exists(save_figure_path)

def test_predict_image_and_plot(m, tmpdir):
    sample_image_path = get_data("OSBS_029.png")
    results = m.predict_image(path=sample_image_path)
    visualize.plot_results(results, savedir=tmpdir)

    assert os.path.exists(os.path.join(tmpdir, "OSBS_029.png"))

def test_predict_tile_and_plot(m, tmpdir):
    sample_image_path = get_data("OSBS_029.png")
    results = m.predict_tile(path=sample_image_path)
    visualize.plot_results(results, savedir=tmpdir)

    assert os.path.exists(os.path.join(tmpdir, "OSBS_029.png"))


def test_multi_class_plot( tmpdir):
    results = pd.read_csv(get_data("testfile_multi.csv"))
    results = read_file(results, root_dir=os.path.dirname(get_data("SOAP_061.png")))
    visualize.plot_results(results, savedir=tmpdir)

    assert os.path.exists(os.path.join(tmpdir, "SOAP_061.png"))

def test_convert_to_sv_format():
    # Create a mock DataFrame
    data = {
        'xmin': [0, 10],
        'ymin': [0, 20],
        'xmax': [5, 15],
        'ymax': [5, 25],
        'label': ['Tree', 'Tree'],
        'score': [0.9, 0.8],
        'image_path': ['image1.jpg', 'image1.jpg']
    }
    df = pd.DataFrame(data)
    df = read_file(df, root_dir=os.path.dirname(get_data("OSBS_029.tif")))

    # Call the function
    detections = visualize.convert_to_sv_format(df)

    # Expected values
    expected_boxes = np.array([[0, 0, 5, 5], [10, 20, 15, 25]], dtype=np.float32)
    expected_labels = np.array([0, 0])
    expected_scores = np.array([0.9, 0.8])

    # Assertions
    np.testing.assert_array_equal(detections.xyxy, expected_boxes)
    np.testing.assert_array_equal(detections.class_id, expected_labels)
    np.testing.assert_array_equal(detections.confidence, expected_scores)
    assert detections['class_name'] == ['Tree', 'Tree']

def test_plot_annotations(tmpdir):
    # Create a mock DataFrame with box annotations
    data = {
        'xmin': [10, 20],
        'ymin': [10, 20],
        'xmax': [30, 40],
        'ymax': [30, 40],
        'label': ['Tree', 'Tree'],
        'image_path': [get_data("OSBS_029.tif"), get_data("OSBS_029.tif")],
        "score": [0.9, 0.8]
    }
    df = pd.DataFrame(data)
    gdf = read_file(df, root_dir=os.path.dirname(get_data("OSBS_029.tif")))
    gdf.root_dir = os.path.dirname(get_data("OSBS_029.tif"))

    # Call the function
    visualize.plot_annotations(gdf, savedir=tmpdir)

    # Assertions
    assert os.path.exists(os.path.join(tmpdir, "OSBS_029.png"))


def test_plot_results_box(m, tmpdir):
    # Create a mock DataFrame with box annotations
    data = {
        'xmin': [10, 20],
        'ymin': [10, 20],
        'xmax': [30, 40],
        'ymax': [30, 40],
        'label': ['Tree', 'Tree'],
        'image_path': [get_data("OSBS_029.tif"), get_data("OSBS_029.tif")],
        "score": [0.9, 0.8]
    }
    df = pd.DataFrame(data)
    gdf = read_file(df, root_dir=os.path.dirname(get_data("OSBS_029.tif")))
    gdf.root_dir = os.path.dirname(get_data("OSBS_029.tif"))

    # Call the function
    visualize.plot_results(gdf, savedir=tmpdir)

    # Assertions
    assert os.path.exists(os.path.join(tmpdir, "OSBS_029.png"))

def test_plot_results_point(m, tmpdir):
    # Create a mock DataFrame with point annotations
    data = {
        'x': [15, 25],
        'y': [15, 25],
        'label': ['Tree', 'Tree'],
        'image_path': [get_data("OSBS_029.tif"), get_data("OSBS_029.tif")],
        'score': [0.9, 0.8]
    }
    df = pd.DataFrame(data)
    gdf = read_file(df, root_dir=os.path.dirname(get_data("OSBS_029.tif")))
    gdf.root_dir = os.path.dirname(get_data("OSBS_029.tif"))

    # Call the function
    visualize.plot_results(gdf, savedir=tmpdir)

    # Assertions
    assert os.path.exists(os.path.join(tmpdir, "OSBS_029.png"))

def test_plot_results_point_no_label(m, tmpdir):
    # Create a mock DataFrame with point annotations
    data = {
        'x': [15, 25],
        'y': [15, 25],
        'label': ['Tree', 'Tree'],
        'image_path': [get_data("OSBS_029.tif"), get_data("OSBS_029.tif")],
    }
    df = pd.DataFrame(data)
    gdf = read_file(df, root_dir=os.path.dirname(get_data("OSBS_029.tif")))
    gdf.root_dir = os.path.dirname(get_data("OSBS_029.tif"))

    # Call the function
    visualize.plot_results(gdf, savedir=tmpdir)

    # Assertions
    assert os.path.exists(os.path.join(tmpdir, "OSBS_029.png"))

def test_plot_results_polygon(m, tmpdir):
    # Create a mock DataFrame with polygon annotations
    data = {
        'geometry': [geometry.Polygon([(10, 10), (20, 10), (20, 20), (10, 20), (15, 25)]),
                        geometry.Polygon([(30, 30), (40, 30), (40, 40), (30, 40), (35, 35)])],
        'label': ['Tree', 'Tree'],
        'image_path': [get_data("OSBS_029.tif"), get_data("OSBS_029.tif")],
        'score': [0.9, 0.8]
    }
    gdf = gpd.GeoDataFrame(data)

    #Read in image and get height
    image = cv2.imread(get_data("OSBS_029.tif"))
    height = image.shape[0]
    width = image.shape[1]
    gdf.root_dir = os.path.dirname(get_data("OSBS_029.tif"))

    # Call the function
    visualize.plot_results(gdf, savedir=tmpdir,height=height, width=width)

    # Assertions
    assert os.path.exists(os.path.join(tmpdir, "OSBS_029.png"))
