# test_utilities
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio as rio
# import general model fixture
import shapely
import torch
from shapely import geometry

from deepforest import get_data
from deepforest import utilities


@pytest.fixture()
def config():
    config = utilities.load_config()
    return config

def test_nonexistant_data():
    filename = "does_not_exist.png"

    # Check filename exists in error string:
    with pytest.raises(FileNotFoundError, match=filename):
        get_data(filename)

def test_read_pascal_voc():
    annotations = utilities.read_pascal_voc(xml_path=get_data("OSBS_029.xml"))
    print(annotations.shape)
    assert annotations.shape[0] == 61


def test_float_warning(config):
    """Users should get a rounding warning when adding annotations with floats"""
    float_annotations = "tests/data/float_annotations.txt"
    with pytest.warns(UserWarning, match="Annotations file contained non-integer coordinates"):
        annotations = utilities.read_pascal_voc(float_annotations)
    assert annotations.xmin.dtype is np.dtype('int64')


def test_read_file(tmpdir):
    sample_geometry = [geometry.Point(404211.9 + 10, 3285102 + 20), geometry.Point(404211.9 + 20, 3285102 + 20)]
    labels = ["Tree", "Tree"]
    df = pd.DataFrame({"geometry": sample_geometry, "label": labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32617")
    gdf["geometry"] = [geometry.box(left, bottom, right, top) for left, bottom, right, top in
                       gdf.geometry.buffer(0.5).bounds.values]
    gdf["image_path"] = get_data("OSBS_029.tif")
    gdf.to_file("{}/annotations.shp".format(tmpdir))
    shp = utilities.read_file(input="{}/annotations.shp".format(tmpdir))
    assert shp.shape[0] == 2


def test_read_file_in_memory_geodataframe():
    """Test reading an in-memory GeoDataFrame"""
    sample_geometry = [geometry.Point(404211.9 + 10, 3285102 + 20), geometry.Point(404211.9 + 20, 3285102 + 20)]
    labels = ["Tree", "Tree"]
    df = pd.DataFrame({"geometry": sample_geometry, "label": labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32617")
    gdf["image_path"] = get_data("OSBS_029.tif")

    # Process through read_file
    result = utilities.read_file(input=gdf)

    # Verify coordinate conversion happened
    original_coords = gdf.geometry.iloc[0].coords[0]
    result_coords = result.geometry.iloc[0].coords[0]

    # Coordinates should change after geo_to_image conversion
    assert original_coords != result_coords

    # Verify output
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 2
    assert "geometry" in result.columns


def test_read_file_in_memory_dataframe():
    """Test reading an in-memory DataFrame with box coordinates"""
    # Create DataFrame with box columns
    test_df = pd.DataFrame({
        'xmin': [0, 10], 'ymin': [0, 10],
        'xmax': [5, 15], 'ymax': [5, 15],
        'label': ['Tree', 'Tree']
    })

    # Process through read_file
    result = utilities.read_file(input=test_df)

    # Verify output
    assert isinstance(result, gpd.GeoDataFrame)
    assert 'geometry' in result.columns
    assert all(result.geometry.geom_type == 'Polygon')
    assert len(result) == 2


def test_shapefile_to_annotations_convert_unprojected_to_boxes(tmpdir):
    sample_geometry = [geometry.Point(10, 20), geometry.Point(20, 40)]
    labels = ["Tree", "Tree"]
    df = pd.DataFrame({"geometry": sample_geometry, "label": labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    gdf.to_file("{}/annotations.shp".format(tmpdir))
    image_path = get_data("OSBS_029.png")
    shp = utilities.shapefile_to_annotations(shapefile="{}/annotations.shp".format(tmpdir), rgb=image_path)
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
        shp = utilities.shapefile_to_annotations(shapefile="{}/annotations.shp".format(tmpdir), rgb=image_path)


def test_read_file_boxes_projected(tmpdir):
    sample_geometry = [geometry.Point(404211.9 + 10, 3285102 + 20), geometry.Point(404211.9 + 20, 3285102 + 20)]
    labels = ["Tree", "Tree"]
    df = pd.DataFrame({"geometry": sample_geometry, "label": labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32617")
    gdf["geometry"] = [geometry.box(left, bottom, right, top) for left, bottom, right, top in
                       gdf.geometry.buffer(0.5).bounds.values]
    image_path = get_data("OSBS_029.tif")
    gdf["image_path"] = image_path
    gdf.to_file("{}/test_read_file_boxes_projected.shp".format(tmpdir))
    image_path = get_data("OSBS_029.tif")

    shp = utilities.read_file(input="{}/test_read_file_boxes_projected.shp".format(tmpdir))
    assert shp.shape[0] == 2


def test_read_file_points_csv(tmpdir):
    x = [10, 20]
    y = [20, 20]
    labels = ["Tree", "Tree"]
    image_path = [get_data("OSBS_029.tif"), get_data("OSBS_029.tif")]
    df = pd.DataFrame({"x": x, "y": y, "label": labels})
    df.to_csv("{}/test_read_file_points.csv".format(tmpdir), index=False)
    read_df = utilities.read_file(input="{}/test_read_file_points.csv".format(tmpdir))
    assert read_df.shape[0] == 2


def test_read_file_polygons_csv(tmpdir):
    # Create a sample GeoDataFrame with polygon geometries with 6 points
    sample_geometry = [geometry.Polygon([(0, 0), (0, 2), (1, 1), (1, 0), (0, 0)]),
                       geometry.Polygon([(2, 2), (2, 4), (3, 3), (3, 2), (2, 2)])]

    labels = ["Tree", "Tree"]
    image_path = get_data("OSBS_029.png")
    df = pd.DataFrame({"geometry": sample_geometry, "label": labels, "image_path": os.path.basename(image_path)})
    df.to_csv("{}/test_read_file_polygons.csv".format(tmpdir), index=False)

    # Call the function under test
    annotations = utilities.read_file(input="{}/test_read_file_polygons.csv".format(tmpdir))

    # Assert the expected number of annotations
    assert annotations.shape[0] == 2
    assert annotations.geometry.iloc[0].type == "Polygon"


def test_read_file_polygons_projected(tmpdir):
    sample_geometry = [geometry.Point(404211.9 + 10, 3285102 + 20), geometry.Point(404211.9 + 20, 3285102 + 20)]
    labels = ["Tree", "Tree"]
    df = pd.DataFrame({"geometry": sample_geometry, "label": labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32617")
    gdf["geometry"] = [geometry.Polygon([(left, bottom), (left, top), (right, top), (right, bottom)]) for
                       left, bottom, right, top in gdf.geometry.buffer(0.5).bounds.values]
    image_path = get_data("OSBS_029.tif")
    gdf["image_path"] = image_path
    gdf.to_file("{}/test_read_file_polygons_projected.shp".format(tmpdir))
    shp = utilities.read_file(input="{}/test_read_file_polygons_projected.shp".format(tmpdir))
    assert shp.shape[0] == 2


def test_read_file_points_projected(tmpdir):
    sample_geometry = [geometry.Point(404211.9 + 10, 3285102 + 20), geometry.Point(404211.9 + 20, 3285102 + 20)]
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
    sample_geometry = [geometry.Polygon([(0, 0), (0, 2), (1, 1), (1, 0), (0, 0)]),
                       geometry.Polygon([(2, 2), (2, 4), (3, 3), (3, 2), (2, 2)])]

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

    # Assert the output filename (normalize both paths)
    expected_filename = str(tmpdir.join("crop.tif"))
    assert os.path.normpath(result) == os.path.normpath(expected_filename)

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
    bounds = (int(raster_bounds[0] + 10), int(raster_bounds[1] + 10),
              int(raster_bounds[0] + 20), int(raster_bounds[1] + 20))

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

    # Assert the output filename (normalize both paths)
    expected_filename = os.path.join(savedir, "crop.png")
    assert os.path.normpath(result) == os.path.normpath(expected_filename)

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

    geo_coords = utilities.image_to_geo_coordinates(original)
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

    # Confirm overlap
    numpy_image = src.read()
    channels, height, width = numpy_image.shape
    numpy_window = geometry.box(0, 0, width, height)
    assert image_coords[image_coords.intersects(numpy_window)].shape[0] == pd.read_csv(
        annotations).shape[0]


def test_geo_to_image_coordinates_UTM_S(tmpdir):
    """Read in a csv file, make a projected shapefile, convert to image coordinates and view the results"""
    annotations = get_data("australia.shp")
    path_to_raster = get_data("australia.tif")
    src = rio.open(path_to_raster)

    geo_coords = gpd.read_file(annotations)
    src_window = geometry.box(*src.bounds)
    # fig, ax = plt.subplots(figsize=(10, 10))
    # gpd.GeoSeries(src_window).plot(ax=ax, color="blue", alpha=0.5)
    # geo_coords.plot(ax=ax, color="red")
    # plt.show()

    assert geo_coords[geo_coords.intersects(src_window)].shape[0] == gpd.read_file(
        annotations).shape[0]

    # Convert to image coordinates
    image_coords = utilities.geo_to_image_coordinates(geo_coords, image_bounds=src.bounds, image_resolution=src.res[0])
    assert image_coords.crs is None

    # Confirm overlap
    numpy_image = src.read()
    channels, height, width = numpy_image.shape
    numpy_window = geometry.box(0, 0, width, height)
    assert image_coords[image_coords.intersects(numpy_window)].shape[0] == gpd.read_file(annotations).shape[0]


def test_image_to_geo_coordinates(tmpdir):
    annotations = get_data("2018_SJER_3_252000_4107000_image_477.csv")
    path_to_raster = get_data("2018_SJER_3_252000_4107000_image_477.tif")

    # Convert to image coordinates
    gdf = utilities.read_file(annotations)

    # Confirm it has no crs
    assert gdf.crs is None

    # Convert to geo coordinates
    src = rio.open(path_to_raster)
    geo_coords = utilities.image_to_geo_coordinates(gdf)
    src_window = geometry.box(*src.bounds)
    assert geo_coords[geo_coords.intersects(src_window)].shape[0] == pd.read_csv(annotations).shape[0]

    # Plot using geopandas
    # fig, ax = plt.subplots(figsize=(10, 10))
    # gpd.GeoSeries(src_window).plot(ax=ax, color="blue", alpha=0.5)
    # geo_coords.plot(ax=ax, color="red", alpha=0.2)
    # show(src, ax=ax)
    # plt.show()

    # Check that geo coordiates are also reflected in box coordinate columns
    bounds = geo_coords.geometry.bounds
    assert (geo_coords["xmin"] == bounds.minx).all()
    assert (geo_coords["xmax"] == bounds.maxx).all()
    assert (geo_coords["ymin"] == bounds.miny).all()
    assert (geo_coords["ymax"] == bounds.maxy).all()


def test_image_to_geo_coordinates_boxes(tmpdir):
    annotations = get_data("2018_SJER_3_252000_4107000_image_477.csv")
    path_to_raster = get_data("2018_SJER_3_252000_4107000_image_477.tif")

    # Convert to image coordinates
    gdf = utilities.read_file(annotations)

    # Confirm it has no crs
    assert gdf.crs is None

    # Convert to geo coordinates
    src = rio.open(path_to_raster)
    geo_coords = utilities.image_to_geo_coordinates(gdf)
    src_window = geometry.box(*src.bounds)
    assert geo_coords[geo_coords.intersects(src_window)].shape[0] == pd.read_csv(annotations).shape[0]

    # Plot using geopandas
    # fig, ax = plt.subplots(figsize=(10, 10))
    # gpd.GeoSeries(src_window).plot(ax=ax, color="blue", alpha=0.5)
    # geo_coords.plot(ax=ax, color="red", alpha=0.2)
    # show(src, ax=ax)
    # plt.show()


def test_image_to_geo_coordinates_points(tmpdir):
    annotations = get_data("2018_SJER_3_252000_4107000_image_477.csv")
    path_to_raster = get_data("2018_SJER_3_252000_4107000_image_477.tif")

    # Convert to image coordinates
    gdf = utilities.read_file(annotations)
    gdf["geometry"] = gdf.geometry.centroid

    # Confirm it has no crs
    assert gdf.crs is None

    # Convert to geo coordinates
    src = rio.open(path_to_raster)
    geo_coords = utilities.image_to_geo_coordinates(gdf)
    src_window = geometry.box(*src.bounds)
    assert geo_coords[geo_coords.intersects(src_window)].shape[0] == pd.read_csv(annotations).shape[0]

    # Plot using geopandas
    # fig, ax = plt.subplots(figsize=(10, 10))
    # gpd.GeoSeries(src_window).plot(ax=ax, color="blue", alpha=0.5)
    # geo_coords.plot(ax=ax, color="red", alpha=0.2)
    # show(src, ax=ax)
    # plt.show()


def test_image_to_geo_coordinates_polygons(tmpdir):
    annotations = get_data("2018_SJER_3_252000_4107000_image_477.csv")
    path_to_raster = get_data("2018_SJER_3_252000_4107000_image_477.tif")

    # Convert to image coordinates
    gdf = utilities.read_file(annotations)
    # Skew boxes to make them polygons
    gdf["geometry"] = gdf.geometry.skew(7, 7)

    # Confirm it has no crs
    assert gdf.crs is None

    # Convert to geo coordinates
    src = rio.open(path_to_raster)
    geo_coords = utilities.image_to_geo_coordinates(gdf)
    src_window = geometry.box(*src.bounds)
    assert geo_coords[geo_coords.intersects(src_window)].shape[0] == pd.read_csv(annotations).shape[0]

    # Plot using geopandas
    # fig, ax = plt.subplots(figsize=(10, 10))
    # gpd.GeoSeries(src_window).plot(ax=ax, color="blue", alpha=0.5)
    # geo_coords.plot(ax=ax, color="red", alpha=0.2)
    # show(src, ax=ax)
    # plt.show()




def test_read_coco_json(tmpdir):
    """Test reading a COCO format JSON file"""
    # Create a sample COCO JSON structure
    coco_data = {
        "images": [
            {"id": 1, "file_name": "OSBS_029.png"},
            {"id": 2, "file_name": "OSBS_029.tif"}
        ],
        "annotations": [
            {
                "image_id": 1,
                "segmentation": [[0, 0, 0, 10, 10, 10, 10, 0]]  # Simple square
            },
            {
                "image_id": 2,
                "segmentation": [[5, 5, 5, 15, 15, 15, 15, 5]]  # Another square
            }
        ]
    }

    # Write the sample JSON to a temporary file
    json_path = tmpdir.join("test_coco.json")
    with open(json_path, "w") as f:
        json.dump(coco_data, f)

    # Read the file using our utility
    df = utilities.read_file(str(json_path))

    # Assert the dataframe has the expected structure
    assert df.shape[0] == 2  # Two annotations
    assert "image_path" in df.columns
    assert "geometry" in df.columns

    # Check the image paths are correct
    assert "OSBS_029.png" in df.image_path.values
    assert "OSBS_029.tif" in df.image_path.values

    # Verify the geometries are valid polygons
    for geom in df.geometry:
        assert geom.is_valid
        assert isinstance(geom, shapely.geometry.Polygon)


def test_format_geometry_box():
    """Test formatting box geometry from model predictions"""
    # Create a mock prediction with box coordinates
    prediction = {
        "boxes": torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]]),
        "labels": torch.tensor([0, 0]),
        "scores": torch.tensor([1.0, 0.8])
    }

    # Format geometry
    result = utilities.format_geometry(prediction)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["xmin", "ymin", "xmax", "ymax", "label", "score", "geometry"]
    assert len(result) == 2

    # Check values
    assert result.iloc[0]["xmin"] == 10
    assert result.iloc[0]["ymin"] == 20
    assert result.iloc[0]["xmax"] == 30
    assert result.iloc[0]["ymax"] == 40
    assert result.iloc[0]["label"] == 0
    assert result.iloc[0]["score"] == 1.0


def test_format_geometry_empty():
    """Test formatting empty predictions"""
    # Create empty prediction
    prediction = {
        "boxes": torch.tensor([]),
        "labels": torch.tensor([]),
        "scores": torch.tensor([])
    }

    # Format geometry
    result = utilities.format_geometry(prediction)

    # Check output format
    assert result is None

def test_format_geometry_multi_class():
    """Test formatting predictions with multiple classes"""
    # Create predictions with different classes
    prediction = {
        "boxes": torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]]),
        "labels": torch.tensor([0, 1]),  # Different classes
        "scores": torch.tensor([0.9, 0.8])
    }

    # Format geometry
    result = utilities.format_geometry(prediction)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["xmin", "ymin", "xmax", "ymax", "label", "score", "geometry"]
    assert len(result) == 2

    # Check values
    assert result.iloc[0]["label"] == 0
    assert result.iloc[1]["label"] == 1


def test_format_geometry_invalid_input():
    """Test handling of invalid input"""
    # Test with missing required keys
    prediction = {
        "boxes": torch.tensor([[10, 20, 30, 40]]),
        "labels": torch.tensor([0])
        # Missing scores
    }

    with pytest.raises(KeyError):
        utilities.format_geometry(prediction)

    # Test with mismatched lengths
    prediction = {
        "boxes": torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]]),
        "labels": torch.tensor([0]),  # Only one label
        "scores": torch.tensor([0.9, 0.8])
    }

    with pytest.raises(ValueError):
        utilities.format_geometry(prediction)


def test_format_geometry_with_geometry_column():
    """Test formatting predictions and adding geometry column"""
    # Create predictions
    prediction = {
        "boxes": torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]]),
        "labels": torch.tensor([0, 0]),
        "scores": torch.tensor([0.9, 0.8])
    }

    # Format geometry
    result = utilities.format_geometry(prediction)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    assert "geometry" in result.columns
    assert len(result) == 2

    # Check geometry values
    assert isinstance(result.iloc[0]["geometry"], geometry.Polygon)
    assert result.iloc[0]["geometry"].bounds == (10, 20, 30, 40)


def test_format_geometry_point():
    """Test formatting point predictions"""
    # Create a mock prediction with point coordinates
    prediction = {
        "points": torch.tensor([[10, 20], [50, 60]]),
        "labels": torch.tensor([0, 0]),
        "scores": torch.tensor([0.9, 0.8])
    }

    # Format geometry should raise ValueError since point predictions are not supported
    with pytest.raises(ValueError, match="Point predictions are not yet supported for formatting"):
        utilities.format_geometry(prediction, geom_type="point")


def test_format_geometry_polygon():
    """Test formatting polygon predictions"""
    # Create a mock prediction with polygon coordinates
    prediction = {
        "polygon": torch.tensor([[[10, 20], [30, 20], [30, 40], [10, 40], [10, 20]],
                               [[50, 60], [70, 60], [70, 80], [50, 80], [50, 60]]]),
        "labels": torch.tensor([0, 0]),
        "scores": torch.tensor([0.9, 0.8])
    }

    # Format geometry should raise ValueError since polygon predictions are not supported
    with pytest.raises(ValueError, match="Polygon predictions are not yet supported for formatting"):
        utilities.format_geometry(prediction, geom_type="polygon")


def test_read_file_column_names():
    """Ensure read_file does not change incoming DataFrame columns (siteID -> siteID)."""
    # Create minimal DataFrame with box geometry columns and a capitalized siteID
    df = pd.DataFrame({
        'xmin': [0],
        'ymin': [0],
        'xmax': [10],
        'ymax': [10],
        'label': ['Tree'],
        'siteID': ['TEST_SITE']
    })

    result = utilities.read_file(df)

    # Column names should not be changed
    assert 'siteID' in df.columns
    assert 'siteID' in result.columns

    # Value should be preserved under the lowercased column
    assert result.loc[0, 'siteID'] == 'TEST_SITE'
