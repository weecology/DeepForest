#Test visualize
from deepforest import visualize
from deepforest import main
from deepforest import get_data
import os
import pytest
import numpy as np

import pandas as pd
import geopandas as gpd
from shapely import geometry


def test_format_boxes(m):
    ds = m.val_dataloader()
    batch = next(iter(ds))
    paths, images, targets = batch
    for path, image, target in zip(paths, images, targets):
        target_df = visualize.format_boxes(target, scores=False)
        assert list(target_df.columns.values) == ["xmin","ymin","xmax","ymax","label"]
        assert not target_df.empty
        

#Test different color labels
@pytest.mark.parametrize("label",[0,1,20])
def test_plot_predictions(m, tmpdir,label):
    ds = m.val_dataloader()
    batch = next(iter(ds))
    paths, images, targets = batch
    for path, image, target in zip(paths, images, targets):
        target_df = visualize.format_boxes(target, scores=False)
        target_df["image_path"] = path
        image = np.array(image)[:,:,::-1]
        image = np.rollaxis(image,0,3)
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
        filenames = visualize.plot_prediction_dataframe(df=target_df,savedir=tmpdir, root_dir=m.config["validation"]["root_dir"])
        
    assert all([os.path.exists(x) for x in filenames])
        
def test_plot_predictions_and_targets(m, tmpdir):
    ds = m.val_dataloader()
    batch = next(iter(ds))
    paths, images, targets = batch
    m.model.eval()    
    predictions = m.model(images)    
    for path, image, target, prediction in zip(paths, images, targets, predictions):
        image = image.permute(1,2,0)
        save_figure_path = visualize.plot_prediction_and_targets(image, prediction, target, image_name=os.path.basename(path), savedir=tmpdir)
        assert os.path.exists(save_figure_path)

def test_plot_predictions_points():
    # Create a pandas dataframe with point annotations
    df = pd.DataFrame({
        "x": [100, 200, 300],
        "y": [100, 200, 300],
        "label": [0, 1, 2]
    })

    # Create a blank image
    image = np.zeros((500, 500, 3), dtype=np.uint8)

    # Call the plot_predictions function
    result = visualize.plot_predictions(image, df)

    # Assert that the result is an image with the points plotted
    assert isinstance(result, np.ndarray)
    assert result.shape == (500, 500, 3)

def test_plot_predictions_boxes():
    # Create a pandas dataframe with box annotations
    df = pd.DataFrame({
        "xmin": [100, 200, 300],
        "ymin": [100, 200, 300],
        "xmax": [150, 250, 350],
        "ymax": [150, 250, 350],
        "label": [0, 1, 2]
    })

    # Create a blank image
    image = np.zeros((500, 500, 3), dtype=np.uint8)

    # Call the plot_predictions function
    result = visualize.plot_predictions(image, df)

    # Assert that the result is an image with the boxes plotted
    assert isinstance(result, np.ndarray)
    assert result.shape == (500, 500, 3)

def test_plot_predictions_polygons():
    # Create a geopandas geodataframe with polygon annotations
    gdf = gpd.GeoDataFrame({
        "geometry": [
            geometry.Polygon(((100, 100), (200, 100), (200, 200), (100, 200))),
            geometry.Polygon(((300, 300), (400, 300), (400, 400), (300, 400)))
        ],
        "label": [0, 1]
    })

    # Create a blank image
    image = np.zeros((700, 700, 3), dtype=np.float32)

    # Call the plot_predictions function
    result = visualize.plot_predictions(image, gdf)

    # Assert that the result is an image with the polygons plotted
    assert isinstance(result, np.ndarray)
    assert result.shape == (700, 700, 3)


def test_plot_predictions_geometry_points():
    # Create a geopandas geodataframe with point annotations
    gdf = gpd.GeoDataFrame({
        "geometry": [
            geometry.Point((100, 100)),
            geometry.Point((300, 300))
        ],
        "label": [0, 1]
    })

    # Create a blank image
    image = np.zeros((700, 700, 3), dtype=np.float32)

    # Call the plot_predictions function
    result = visualize.plot_predictions(image, gdf)

    # Assert that the result is an image with the polygons plotted
    assert isinstance(result, np.ndarray)
    assert result.shape == (700, 700, 3)

def test_plot_prediction_dataframe_from_predict_image(m, tmpdir):
    # Load an image
    image_path = get_data("OSBS_029.tif")
    boxes = m.predict_image(path=image_path)

    # Call the plot_prediction_dataframe function
    result = visualize.plot_prediction_dataframe(boxes, savedir=tmpdir, root_dir=os.path.dirname(image_path))

    # Assert that the result is a list of filenames
    assert isinstance(result, list)
    assert all([isinstance(x, str) for x in result])
    assert all([os.path.exists(x) for x in result])