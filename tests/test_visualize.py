# Test visualize
from deepforest import visualize
from deepforest import utilities
from deepforest import get_data
import os
import pytest
import numpy as np
import pandas as pd
from PIL import Image
import geopandas as gpd
from shapely import geometry
import cv2

@pytest.fixture
def gdf():
    data = {
        'geometry': [geometry.Polygon([(10, 10), (20, 10), (20, 20), (10, 20), (15, 25)]),
                        geometry.Polygon([(30, 30), (40, 30), (40, 40), (30, 40), (35, 35)])],
        'label': ['Tree', 'Tree'],
        'image_path': [get_data("OSBS_029.tif"), get_data("OSBS_029.tif")],
        'score': [0.9, 0.8]
    }
    gdf = gpd.GeoDataFrame(data)
    gdf.root_dir = os.path.dirname(get_data("OSBS_029.tif"))
    return gdf


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

def test_multi_class_plot(tmpdir):
    results = pd.read_csv(get_data("testfile_multi.csv"))
    results = utilities.read_file(results, root_dir=os.path.dirname(get_data("SOAP_061.png")))
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
    df = utilities.read_file(df, root_dir=os.path.dirname(get_data("OSBS_029.tif")))

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
    gdf = utilities.read_file(df, root_dir=os.path.dirname(get_data("OSBS_029.tif")))
    gdf.root_dir = os.path.dirname(get_data("OSBS_029.tif"))

    # Call the function
    visualize.plot_annotations(gdf, savedir=tmpdir)

    # Assertions
    assert os.path.exists(os.path.join(tmpdir, "OSBS_029.png"))


def test_plot_results_box(tmpdir):
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
    gdf = utilities.read_file(df, root_dir=os.path.dirname(get_data("OSBS_029.tif")))
    gdf.root_dir = os.path.dirname(get_data("OSBS_029.tif"))

    # Call the function
    visualize.plot_results(gdf, savedir=tmpdir)

    # Assertions
    assert os.path.exists(os.path.join(tmpdir, "OSBS_029.png"))

def test_plot_results_point(tmpdir):
    # Create a mock DataFrame with point annotations
    data = {
        'x': [15, 25],
        'y': [15, 25],
        'label': ['Tree', 'Tree'],
        'image_path': [get_data("OSBS_029.tif"), get_data("OSBS_029.tif")],
        'score': [0.9, 0.8]
    }
    df = pd.DataFrame(data)
    gdf = utilities.read_file(df, root_dir=os.path.dirname(get_data("OSBS_029.tif")))
    gdf.root_dir = os.path.dirname(get_data("OSBS_029.tif"))

    # Call the function
    visualize.plot_results(gdf, savedir=tmpdir)

    # Assertions
    assert os.path.exists(os.path.join(tmpdir, "OSBS_029.png"))

def test_plot_results_point_no_label(tmpdir):
    # Create a mock DataFrame with point annotations
    data = {
        'x': [15, 25],
        'y': [15, 25],
        'label': ['Tree', 'Tree'],
        'image_path': [get_data("OSBS_029.tif"), get_data("OSBS_029.tif")],
    }
    df = pd.DataFrame(data)
    gdf = utilities.read_file(df, root_dir=os.path.dirname(get_data("OSBS_029.tif")))
    gdf.root_dir = os.path.dirname(get_data("OSBS_029.tif"))

    # Call the function
    visualize.plot_results(gdf, savedir=tmpdir)

    # Assertions
    assert os.path.exists(os.path.join(tmpdir, "OSBS_029.png"))

def test_plot_results_polygon(tmpdir):
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

def test_draw_points():
    image = visualize._load_image(get_data("OSBS_029.tif"))
    points = np.array([[10, 10], [20, 20]])

    image = visualize.draw_points(image, points)
    assert image is not None

def test_draw_objects(gdf):
    image = visualize._load_image(get_data("OSBS_029.tif"))
    image = visualize.draw_objects(image, gdf)
    assert image is not None

def test_image_from_path_or_array():
    image_path = visualize._load_image(np.array(Image.open(get_data("OSBS_029.tif"))))
    image_array = visualize._load_image(get_data("OSBS_029.tif"))
    assert np.allclose(image_path, image_array)

def test_image_from_pil():
    image = visualize._load_image(Image.open(get_data("OSBS_029.tif")))
    assert image is not None

def test_load_chw_to_hwc():
    image = np.random.randint(0, 255, size=(3, 100, 100), dtype=np.uint8)
    image = visualize._load_image(image)
    assert image.shape == (100, 100, 3)

def test_load_drop_alpha():
    image = np.random.randint(0, 255, size=(100, 100, 4), dtype=np.uint8)
    image = visualize._load_image(image)
    assert image.shape == (100, 100, 3)

def test_image_from_gdf(gdf):
    image = visualize._load_image(df=gdf)
    assert image is not None

def test_check_dtype_rescale():
    image = np.random.random((100, 100, 3))
    assert image.dtype != np.uint8
    image = visualize._load_image(image)
    assert image.dtype == np.uint8
    assert image.max() > 1 and image.max() <= 255

@pytest.mark.xfail
def test_image_empty():
    image = visualize._load_image()
    assert image is not None
