# test_utilities
import numpy as np
import os
import pytest
import pandas as pd
import rasterio as rio
from rasterio.plot import show
from rasterio import warp
from shapely import geometry
import geopandas as gpd
from matplotlib import pyplot as plt

from deepforest import get_data
from deepforest import utilities, visualize
from deepforest import main


#import general model fixture
from .conftest import download_release
import requests
import pytest
import matplotlib.pyplot as plt
import cv2
import asyncio
from aiolimiter import AsyncLimiter


from PIL import Image

@pytest.fixture()
def config():
    config = utilities.read_config("deepforest_config.yml")
    return config

def test_read_pascal_voc():
    annotations = utilities.read_pascal_voc(
        xml_path=get_data("OSBS_029.xml"))
    print(annotations.shape)
    assert annotations.shape[0] == 61

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
    annotations = utilities.read_pascal_voc(float_annotations)
    assert annotations.xmin.dtype is np.dtype('int64')

def test_project_boxes():
    csv_file = get_data("OSBS_029.csv")
    df = pd.read_csv(csv_file)
    gdf = utilities.project_boxes(df, root_dir=os.path.dirname(csv_file))
    
def test_read_file(tmpdir):
    sample_geometry = [geometry.Point(404211.9 + 10,3285102 + 20),geometry.Point(404211.9 + 20,3285102 + 20)]
    labels = ["Tree","Tree"]
    df = pd.DataFrame({"geometry":sample_geometry,"label":labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32617")
    gdf["geometry"] = [geometry.box(left, bottom, right, top) for left, bottom, right, top in gdf.geometry.buffer(0.5).bounds.values]
    gdf["image_path"] = get_data("OSBS_029.tif")
    gdf.to_file("{}/annotations.shp".format(tmpdir))
    shp = utilities.read_file(input="{}/annotations.shp".format(tmpdir))
    assert shp.shape[0] == 2

def test_shapefile_to_annotations_convert_unprojected_to_boxes(tmpdir):
    sample_geometry = [geometry.Point(10,20),geometry.Point(20,40)]
    labels = ["Tree","Tree"]
    df = pd.DataFrame({"geometry":sample_geometry,"label":labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    gdf.to_file("{}/annotations.shp".format(tmpdir))
    image_path = get_data("OSBS_029.png")
    shp = utilities.shapefile_to_annotations(shapefile="{}/annotations.shp".format(tmpdir), rgb=image_path, savedir=tmpdir, geometry_type="point")
    assert shp.shape[0] == 2

def test_shapefile_to_annotations_invalid_epsg(tmpdir):
    sample_geometry = [geometry.Point(404211.9 + 10, 3285102 + 20), geometry.Point(404211.9 + 20, 3285102 + 20)]
    labels = ["Tree", "Tree"]
    df = pd.DataFrame({"geometry": sample_geometry, "label": labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    gdf.to_file("{}/annotations.shp".format(tmpdir))
    assert gdf.crs.to_string() == "EPSG:4326"
    image_path = get_data("OSBS_029.tif")
    with pytest.raises(ValueError):
        shp = utilities.shapefile_to_annotations(shapefile="{}/annotations.shp".format(tmpdir), rgb=image_path, savedir=tmpdir, geometry_type="bbox")
        
def test_read_file_boxes_projected(tmpdir):
    sample_geometry = [geometry.Point(404211.9 + 10,3285102 + 20),geometry.Point(404211.9 + 20,3285102 + 20)]
    labels = ["Tree","Tree"]
    df = pd.DataFrame({"geometry":sample_geometry,"label":labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32617")
    gdf["geometry"] = [geometry.box(left, bottom, right, top) for left, bottom, right, top in gdf.geometry.buffer(0.5).bounds.values]
    image_path = get_data("OSBS_029.tif")
    gdf["image_path"] = image_path
    gdf.to_file("{}/test_read_file_boxes_projected.shp".format(tmpdir))
    image_path = get_data("OSBS_029.tif")

    shp = utilities.read_file(input="{}/test_read_file_boxes_projected.shp".format(tmpdir))
    assert shp.shape[0] == 2

def test_read_file_polygons_projected(tmpdir):
    sample_geometry = [geometry.Point(404211.9 + 10,3285102 + 20),geometry.Point(404211.9 + 20,3285102 + 20)]
    labels = ["Tree", "Tree"]
    df = pd.DataFrame({"geometry": sample_geometry, "label": labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32617")
    gdf["geometry"] = [geometry.Polygon([(left, bottom), (left, top), (right, top), (right, bottom)]) for left, bottom, right, top in gdf.geometry.buffer(0.5).bounds.values]
    image_path = get_data("OSBS_029.tif")
    gdf["image_path"] = image_path
    gdf.to_file("{}/test_read_file_polygons_projected.shp".format(tmpdir))
    shp = utilities.read_file(input="{}/test_read_file_polygons_projected.shp".format(tmpdir))
    assert shp.shape[0] == 2

def test_read_file_points_projected(tmpdir):
    sample_geometry = [geometry.Point(404211.9 + 10,3285102 + 20),geometry.Point(404211.9 + 20,3285102 + 20)]
    labels = ["Tree", "Tree"]
    df = pd.DataFrame({"geometry": sample_geometry, "label": labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32617")
    image_path = get_data("OSBS_029.tif")
    gdf["image_path"] = image_path
    gdf.to_file("{}/test_read_file_points_projected.shp".format(tmpdir))
    shp = utilities.read_file(input="{}/test_read_file_points_projected.shp".format(tmpdir))
    assert shp.shape[0] == 2
    assert shp.geometry.iloc[0].type == "Point"

def test_read_file_boxes_unprojected(tmpdir):
    # Create a sample GeoDataFrame with box geometries
    sample_geometry = [geometry.box(0, 0, 1, 1), geometry.box(2, 2, 3, 3)]
    labels = ["Tree", "Tree"]
    df = pd.DataFrame({"geometry": sample_geometry, "label": labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    image_path = get_data("OSBS_029.png")
    gdf["image_path"] = image_path
    gdf.to_file("{}/test_read_file_boxes_unprojected.shp".format(tmpdir))
    annotations = utilities.read_file(input="{}/test_read_file_boxes_unprojected.shp".format(tmpdir))

    # Assert the expected number of annotations and geometry type
    assert annotations.shape[0] == 2
    assert annotations.geometry.iloc[0].type == "Polygon"

def test_read_file_points_unprojected(tmpdir):
    # Create a sample GeoDataFrame with point geometries
    sample_geometry = [geometry.Point(0.5, 0.5), geometry.Point(2.5, 2.5)]
    labels = ["Tree", "Tree"]
    df = pd.DataFrame({"geometry": sample_geometry, "label": labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    image_path = get_data("OSBS_029.png")
    gdf["image_path"] = image_path
    gdf.to_file("{}/test_read_file_points_unprojected.shp".format(tmpdir))

    annotations = utilities.read_file(input="{}/test_read_file_points_unprojected.shp".format(tmpdir))

    # Assert the expected number of annotations
    assert annotations.shape[0] == 2
    assert annotations.geometry.iloc[0].type == "Point"

def test_read_file_polygons_unprojected(tmpdir):
    # Create a sample GeoDataFrame with polygon geometries with 6 points
    sample_geometry = [geometry.Polygon([(0, 0), (0, 2), (1, 1), (1, 0), (0, 0)]), geometry.Polygon([(2, 2), (2, 4), (3, 3), (3, 2), (2, 2)])]
    
    labels = ["Tree", "Tree"]
    df = pd.DataFrame({"geometry": sample_geometry, "label": labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    image_path = get_data("OSBS_029.png")
    gdf["image_path"] = image_path
    gdf.to_file("{}/test_read_file_polygons_unprojected.shp".format(tmpdir))

    # Call the function under test
    annotations = utilities.read_file(input="{}/test_read_file_polygons_unprojected.shp".format(tmpdir))

    # Assert the expected number of annotations
    assert annotations.shape[0] == 2
    assert annotations.geometry.iloc[0].type == "Polygon"

def test_crop_raster_valid_crop(tmpdir):
    rgb_path = get_data("2018_SJER_3_252000_4107000_image_477.tif")
    raster_bounds = rio.open(rgb_path).bounds
    
    # Define the bounds for cropping
    bounds = (raster_bounds[0] + 10, raster_bounds[1] + 10, raster_bounds[0] + 30, raster_bounds[1] + 30)

    # Call the function under test
    result = utilities.crop_raster(bounds, rgb_path=rgb_path, savedir=tmpdir, filename="crop")

    # Assert the output filename
    expected_filename = str(tmpdir.join("crop.tif"))
    assert result == expected_filename

    # Assert the saved crop
    with rio.open(result) as src:
        # Round to nearest integer to avoid floating point errors
        assert np.round(src.bounds[2] - src.bounds[0]) == 20
        assert np.round(src.bounds[3] - src.bounds[1]) == 20
        assert src.count == 3
        assert src.dtypes == ("uint8", "uint8", "uint8")

def test_crop_raster_invalid_crop(tmpdir):
    rgb_path = get_data("2018_SJER_3_252000_4107000_image_477.tif")
    raster_bounds = rio.open(rgb_path).bounds
    
    # Define the bounds for cropping
    bounds = (raster_bounds[0] - 100, raster_bounds[1] - 100, raster_bounds[0] - 30, raster_bounds[1] - 30)

    # Call the function under test
    with pytest.raises(ValueError):
        result = utilities.crop_raster(bounds, rgb_path=rgb_path, savedir=tmpdir, filename="crop")

def test_crop_raster_no_savedir(tmpdir):
    rgb_path = get_data("2018_SJER_3_252000_4107000_image_477.tif")
    raster_bounds = rio.open(rgb_path).bounds
    
    # Define the bounds for cropping
    bounds = (int(raster_bounds[0] + 10), int(raster_bounds[1] + 10), int(raster_bounds[0] + 20), int(raster_bounds[1] + 20))

    # Call the function under test
    result = utilities.crop_raster(bounds, rgb_path=rgb_path)

    # Assert out is a output numpy array
    assert isinstance(result, np.ndarray)

def test_crop_raster_png_unprojected(tmpdir):
    # Define the bounds for cropping
    bounds = (0, 0, 100, 100)

    # Set the paths
    rgb_path = get_data("OSBS_029.png")
    savedir = str(tmpdir)
    filename = "crop"

    # Call the function under test
    result = utilities.crop_raster(bounds, rgb_path=rgb_path, savedir=savedir, filename=filename, driver="PNG")

    # Assert the output filename
    expected_filename = os.path.join(savedir, "crop.png")
    assert result == expected_filename

    # Assert the saved crop
    with rio.open(result) as src:
        # Assert the driver is PNG
        assert src.driver == "PNG"

        # Assert the crs is not present
        assert src.crs is None

def test_geo_to_image_coordinates_UTM_N(tmpdir):
    """Read in a csv file, make a projected shapefile, convert to image coordinates and view the results"""
    annotations = get_data("2018_SJER_3_252000_4107000_image_477.csv")
    path_to_raster = get_data("2018_SJER_3_252000_4107000_image_477.tif")
    src = rio.open(path_to_raster)
    original = utilities.read_file(annotations)
    assert original.crs is None

    geo_coords = utilities.image_to_geo_coordinates(original, root_dir=os.path.dirname(path_to_raster))
    assert geo_coords.crs == src.crs
    src_window = geometry.box(*src.bounds)

    # fig, ax = plt.subplots(figsize=(10, 10))
    # gpd.GeoSeries(src_window).plot(ax=ax, color="blue", alpha=0.5)
    # geo_coords.plot(ax=ax, color="red")
    # plt.show()

    assert geo_coords[geo_coords.intersects(src_window)].shape[0] == pd.read_csv(annotations).shape[0]  

    # Convert to image coordinates
    image_coords = utilities.geo_to_image_coordinates(geo_coords, image_bounds=src.bounds, image_resolution=src.res[0])   
    assert image_coords.crs is None

    #Confirm overlap
    numpy_image = src.read()
    channels, height, width = numpy_image.shape
    numpy_window = geometry.box(0, 0, width, height)
    assert image_coords[image_coords.intersects(numpy_window)].shape[0] == pd.read_csv(annotations).shape[0]

    images = visualize.plot_prediction_dataframe(image_coords, root_dir=os.path.dirname(path_to_raster), savedir=tmpdir)
    # Confirm the image coordinates are correct
    for image in images:
        im = Image.open(image)
        im.show()

def test_geo_to_image_coordinates_UTM_S(tmpdir):
    """Read in a csv file, make a projected shapefile, convert to image coordinates and view the results"""
    annotations = get_data("australia.shp")
    path_to_raster = get_data("australia.tif")
    src = rio.open(path_to_raster)
    
    geo_coords = gpd.read_file(annotations)
    src_window = geometry.box(*src.bounds)
    #fig, ax = plt.subplots(figsize=(10, 10))
    #gpd.GeoSeries(src_window).plot(ax=ax, color="blue", alpha=0.5)
    #geo_coords.plot(ax=ax, color="red")
    #plt.show()
    
    assert geo_coords[geo_coords.intersects(src_window)].shape[0] == gpd.read_file(annotations).shape[0]  

    # Convert to image coordinates
    image_coords = utilities.geo_to_image_coordinates(geo_coords, image_bounds=src.bounds, image_resolution=src.res[0])   
    assert image_coords.crs is None

    #Confirm overlap
    numpy_image = src.read()
    channels, height, width = numpy_image.shape
    numpy_window = geometry.box(0, 0, width, height)
    assert image_coords[image_coords.intersects(numpy_window)].shape[0] == gpd.read_file(annotations).shape[0] 

    images = visualize.plot_prediction_dataframe(image_coords, root_dir=os.path.dirname(path_to_raster), savedir=tmpdir)
    # Confirm the image coordinates are correct
    for image in images:
        im = Image.open(image)
        im.show()

def test_image_to_geo_coordinates(tmpdir):
    annotations = get_data("2018_SJER_3_252000_4107000_image_477.csv")
    path_to_raster = get_data("2018_SJER_3_252000_4107000_image_477.tif")

    # Convert to image coordinates
    gdf = utilities.read_file(annotations)   
    images = visualize.plot_prediction_dataframe(gdf, root_dir=os.path.dirname(path_to_raster), savedir=tmpdir)

    # Confirm it has no crs
    assert gdf.crs is None

    # Confirm the image coordinates are correct
    for image in images:
        im = Image.open(image)
        im.show(title="before")
    
    # Convert to geo coordinates
    src = rio.open(path_to_raster)
    geo_coords = utilities.image_to_geo_coordinates(gdf, root_dir=os.path.dirname(path_to_raster))
    src_window = geometry.box(*src.bounds)
    assert geo_coords[geo_coords.intersects(src_window)].shape[0] == pd.read_csv(annotations).shape[0]  

    # Plot using geopandas
    fig, ax = plt.subplots(figsize=(10, 10))
    gpd.GeoSeries(src_window).plot(ax=ax, color="blue", alpha=0.5)
    geo_coords.plot(ax=ax, color="red", alpha=0.2)
    show(src, ax=ax)
    plt.show()

def test_image_to_geo_coordinates_boxes(tmpdir):
    annotations = get_data("2018_SJER_3_252000_4107000_image_477.csv")
    path_to_raster = get_data("2018_SJER_3_252000_4107000_image_477.tif")

    # Convert to image coordinates
    gdf = utilities.read_file(annotations)   
    images = visualize.plot_prediction_dataframe(gdf, root_dir=os.path.dirname(path_to_raster), savedir=tmpdir)

    # Confirm it has no crs
    assert gdf.crs is None

    # Confirm the image coordinates are correct
    for image in images:
        im = Image.open(image)
        im.show(title="before")
    
    # Convert to geo coordinates
    src = rio.open(path_to_raster)
    geo_coords = utilities.image_to_geo_coordinates(gdf, root_dir=os.path.dirname(path_to_raster))
    src_window = geometry.box(*src.bounds)
    assert geo_coords[geo_coords.intersects(src_window)].shape[0] == pd.read_csv(annotations).shape[0]  

    # Plot using geopandas
    fig, ax = plt.subplots(figsize=(10, 10))
    gpd.GeoSeries(src_window).plot(ax=ax, color="blue", alpha=0.5)
    geo_coords.plot(ax=ax, color="red", alpha=0.2)
    show(src, ax=ax)
    plt.show()

def test_image_to_geo_coordinates_points(tmpdir):
    annotations = get_data("2018_SJER_3_252000_4107000_image_477.csv")
    path_to_raster = get_data("2018_SJER_3_252000_4107000_image_477.tif")

    # Convert to image coordinates
    gdf = utilities.read_file(annotations)   
    gdf["geometry"] = gdf.geometry.centroid
    images = visualize.plot_prediction_dataframe(gdf, root_dir=os.path.dirname(path_to_raster), savedir=tmpdir)

    # Confirm it has no crs
    assert gdf.crs is None

    # Confirm the image coordinates are correct
    for image in images:
        im = Image.open(image)
        im.show(title="before")
    
    # Convert to geo coordinates
    src = rio.open(path_to_raster)
    geo_coords = utilities.image_to_geo_coordinates(gdf, root_dir=os.path.dirname(path_to_raster))
    src_window = geometry.box(*src.bounds)
    assert geo_coords[geo_coords.intersects(src_window)].shape[0] == pd.read_csv(annotations).shape[0]  

    # Plot using geopandas
    fig, ax = plt.subplots(figsize=(10, 10))
    gpd.GeoSeries(src_window).plot(ax=ax, color="blue", alpha=0.5)
    geo_coords.plot(ax=ax, color="red", alpha=0.2)
    show(src, ax=ax)
    plt.show()

def test_image_to_geo_coordinates_polygons(tmpdir):
    annotations = get_data("2018_SJER_3_252000_4107000_image_477.csv")
    path_to_raster = get_data("2018_SJER_3_252000_4107000_image_477.tif")

    # Convert to image coordinates
    gdf = utilities.read_file(annotations)   
    # Skew boxes to make them polygons
    gdf["geometry"] = gdf.geometry.skew(7,7)
    images = visualize.plot_prediction_dataframe(gdf, root_dir=os.path.dirname(path_to_raster), savedir=tmpdir)

    # Confirm it has no crs
    assert gdf.crs is None

    # Confirm the image coordinates are correct
    for image in images:
        im = Image.open(image)
        im.show(title="before")
    
    # Convert to geo coordinates
    src = rio.open(path_to_raster)
    geo_coords = utilities.image_to_geo_coordinates(gdf, root_dir=os.path.dirname(path_to_raster))
    src_window = geometry.box(*src.bounds)
    assert geo_coords[geo_coords.intersects(src_window)].shape[0] == pd.read_csv(annotations).shape[0]  

    # Plot using geopandas
    fig, ax = plt.subplots(figsize=(10, 10))
    gpd.GeoSeries(src_window).plot(ax=ax, color="blue", alpha=0.5)
    geo_coords.plot(ax=ax, color="red", alpha=0.2)
    show(src, ax=ax)
    plt.show()


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
    
def url():
    return [
        "https://map.dfg.ca.gov/arcgis/rest/services/Base_Remote_Sensing/NAIP_2020_CIR/ImageServer/",
        "https://gis.calgary.ca/arcgis/rest/services/pub_Orthophotos/CurrentOrthophoto/ImageServer/",
        "https://orthos.its.ny.gov/arcgis/rest/services/wms/Latest/MapServer"
    ]

def boxes():
    return [
        (-124.112622, 40.493891, -124.111536, 40.49457),
        (-114.12529, 51.072134, -114.12117, 51.07332),
        (-73.763941, 41.111032, -73.763447, 41.111626)
    ]

def additional_params():
    return [
        None,
        None,
        {"format":"png"}
    ]

def download_service():
    return [
        "exportImage",
        "exportImage",
        "export"
    ]

# Pair each URL with its corresponding box
url_box_pairs = list(zip(["CA.tif","MA.tif","NY.png"],url(), boxes(), additional_params(), download_service()))
@pytest.mark.parametrize("image_name, url, box, params, download_service_name", url_box_pairs)
def test_download_ArcGIS_REST(tmpdir, image_name, url, box, params, download_service_name):
    async def run_test():
        semaphore = asyncio.Semaphore(20)
        limiter = AsyncLimiter(1,0.05)
        xmin, ymin, xmax, ymax = box
        bbox_crs = "EPSG:4326"  # Assuming WGS84 for bounding box CRS
        savedir = tmpdir
        filename = await utilities.download_ArcGIS_REST(semaphore, limiter, url, xmin, ymin, xmax, ymax, bbox_crs, savedir, additional_params=params, image_name=image_name, download_service=download_service_name)
        
        # Check the saved file
        assert os.path.exists(filename)
        
        # Confirm file has CRS
        with rio.open(filename) as src:
            if image_name.endswith('.tif'):
                assert src.crs is not None
                plt.imshow(src.read().transpose(1, 2, 0))
            else:
                assert src.crs is None
                plt.imshow(cv2.imread(filename)[:, :, ::-1])
    
    asyncio.run(run_test())
