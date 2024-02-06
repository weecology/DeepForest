# test_utilities
import numpy as np
import os
import pytest
import pandas as pd
import rasterio as rio
from shapely import geometry
import geopandas as gpd

from deepforest import get_data
from deepforest import utilities
from deepforest import main

#import general model fixture
from .conftest import download_release

@pytest.fixture()
def config():
    config = utilities.read_config("deepforest_config.yml")
    return config


def test_xml_to_annotations():
    annotations = utilities.xml_to_annotations(
        xml_path=get_data("OSBS_029.xml"))
    print(annotations.shape)
    assert annotations.shape == (61, 6)

    # bounding box extents should be int
    assert annotations["xmin"].dtype == np.int64

def test_use_release(download_release):
    # Download latest model from github release
    release_tag, state_dict = utilities.use_release(check_release=False)

def test_use_bird_release(download_release):
    # Download latest model from github release
    release_tag, state_dict = utilities.use_bird_release()
    assert os.path.exists(get_data("bird.pt"))    
    
def test_float_warning(config):
    """Users should get a rounding warning when adding annotations with floats"""
    float_annotations = "tests/data/float_annotations.txt"
    annotations = utilities.xml_to_annotations(float_annotations)
    assert annotations.xmin.dtype is np.dtype('int64')

def test_project_boxes():
    csv_file = get_data("OSBS_029.csv")
    df = pd.read_csv(csv_file)
    gdf = utilities.project_boxes(df, root_dir=os.path.dirname(csv_file))
    
    assert df.shape[0] == gdf.shape[0]

def test_shapefile_to_annotations_convert_to_boxes(tmpdir):
    sample_geometry = [geometry.Point(404211.9 + 10,3285102 + 20),geometry.Point(404211.9 + 20,3285102 + 20)]
    labels = ["Tree","Tree"]
    df = pd.DataFrame({"geometry":sample_geometry,"label":labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32617")
    gdf.to_file("{}/annotations.shp".format(tmpdir))
    image_path = get_data("OSBS_029.tif")
    shp = utilities.shapefile_to_annotations(shapefile="{}/annotations.shp".format(tmpdir), rgb=image_path, savedir=tmpdir, geometry_type="point")
    assert shp.shape[0] == 2
    
def test_shapefile_to_annotations(tmpdir):
    sample_geometry = [geometry.Point(404211.9 + 10,3285102 + 20),geometry.Point(404211.9 + 20,3285102 + 20)]
    labels = ["Tree","Tree"]
    df = pd.DataFrame({"geometry":sample_geometry,"label":labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32617")
    gdf["geometry"] = [geometry.box(left, bottom, right, top) for left, bottom, right, top in gdf.geometry.buffer(0.5).bounds.values]
    
    gdf.to_file("{}/annotations.shp".format(tmpdir))
    image_path = get_data("OSBS_029.tif")
    shp = utilities.shapefile_to_annotations(shapefile="{}/annotations.shp".format(tmpdir), rgb=image_path, savedir=tmpdir, geometry_type="bbox")
    assert shp.shape[0] == 2

def test_shapefile_to_annotations_incorrect_crs(tmpdir):
    sample_geometry = [geometry.Point(404211.9 + 10,3285102 + 20),geometry.Point(404211.9 + 20,3285102 + 20)]
    labels = ["Tree","Tree"]
    df = pd.DataFrame({"geometry":sample_geometry,"label":labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32618")
    gdf["geometry"] = [geometry.box(left, bottom, right, top) for left, bottom, right, top in gdf.geometry.buffer(0.5).bounds.values]
    
    gdf.to_file("{}/annotations.shp".format(tmpdir))
    image_path = get_data("OSBS_029.tif")
    with pytest.raises(ValueError):
        shp = utilities.shapefile_to_annotations(shapefile="{}/annotations.shp".format(tmpdir), rgb=image_path, savedir=tmpdir, geometry_type="bbox")
def test_boxes_to_shapefile_projected(m):
    img = get_data("OSBS_029.tif")
    r = rio.open(img)
    df = m.predict_image(path=img)
    gdf = utilities.boxes_to_shapefile(df, root_dir=os.path.dirname(img), projected=True)
    
    #Confirm that each boxes within image bounds
    geom = geometry.box(*r.bounds)
    assert all(gdf.geometry.apply(lambda x: geom.intersects(geom)).values)
    
    #Edge case, only one row in predictions
    gdf = utilities.boxes_to_shapefile(df.iloc[:1,], root_dir=os.path.dirname(img), projected=True)
    assert gdf.shape[0] == 1

def test_boxes_to_shapefile_projected_from_predict_tile(m):
    img = get_data("OSBS_029.tif")
    r = rio.open(img)
    df = m.predict_tile(raster_path=img)
    gdf = utilities.boxes_to_shapefile(df, root_dir=os.path.dirname(img), projected=True)
    
    #Confirm that each boxes within image bounds
    geom = geometry.box(*r.bounds)
    assert all(gdf.geometry.apply(lambda x: geom.intersects(geom)).values)
    
    #Edge case, only one row in predictions
    gdf = utilities.boxes_to_shapefile(df.iloc[:1,], root_dir=os.path.dirname(img), projected=True)
    assert gdf.shape[0] == 1
    
# Test unprojected data, including warning if flip_y_axis is set to True, but projected is False
@pytest.mark.parametrize("flip_y_axis", [True, False])
@pytest.mark.parametrize("projected", [True, False])
def test_boxes_to_shapefile_unprojected(m, flip_y_axis, projected):
    img = get_data("OSBS_029.png")
    r = rio.open(img)
    df = m.predict_image(path=img)

    with pytest.warns(UserWarning):
        gdf = utilities.boxes_to_shapefile(
            df,
            root_dir=os.path.dirname(img),
            projected=projected,
            flip_y_axis=flip_y_axis)
    
    # Confirm that each boxes within image bounds
    geom = geometry.box(*r.bounds)
    assert all(gdf.geometry.apply(lambda x: geom.intersects(geom)).values)