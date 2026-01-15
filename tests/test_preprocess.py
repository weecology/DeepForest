# test preprocessing
import os

import numpy as np
import pandas as pd
import pytest
import rasterio
from PIL import Image
from shapely import geometry

from deepforest import get_data
from deepforest import preprocess
from deepforest import utilities


@pytest.fixture()
def config():
    config = utilities.load_config()
    config.patch_size = 300
    config.patch_overlap = 0.25
    config.annotations_xml = get_data("OSBS_029.xml")
    config.rgb_dir = "data"
    config.path_to_raster = get_data("OSBS_029.tif")

    # Create a clean config test data
    annotations = utilities.read_pascal_voc(xml_path=config.annotations_xml)
    annotations.to_csv("tests/data/OSBS_029.csv", index=False)

    return config


@pytest.fixture()
def geodataframe():
    csv_file = get_data("OSBS_029.csv")
    annotations = utilities.read_file(csv_file)
    return annotations


@pytest.fixture()
def image(config):
    raster = Image.open(config.path_to_raster)

    # Convert to channels first
    raster = np.array(raster)
    raster = np.moveaxis(raster, 2, 0)
    return raster


def test_compute_windows(config, image):
    windows = preprocess.compute_windows(image, config.patch_size,
                                         config.patch_overlap)
    assert len(windows) == 4


def test_select_annotations(config, image):
    windows = preprocess.compute_windows(image, patch_size=300, patch_overlap=0.5)
    csv_file = get_data("OSBS_029.csv")
    image_annotations = utilities.read_file(csv_file)

    selected_annotations = preprocess.select_annotations(image_annotations,
                                                         window=windows[0])

    # The largest box cannot be off the edge of the window
    assert selected_annotations.geometry.bounds.minx.min() >= 0
    assert selected_annotations.geometry.bounds.miny.min() >= 0
    assert selected_annotations.geometry.bounds.maxx.max() <= 300
    assert selected_annotations.geometry.bounds.maxy.max() <= 300


@pytest.mark.parametrize("input_type", ["path", "dataframe"])
def test_split_raster(config, tmp_path, input_type, geodataframe):
    """Split raster into crops with overlaps to maintain all annotations"""
    raster = get_data("2019_YELL_2_528000_4978000_image_crop2.png")
    annotations = utilities.read_pascal_voc(
        get_data("2019_YELL_2_528000_4978000_image_crop2.xml"))
    annotations.to_csv(tmp_path / "example.csv", index=False)

    if input_type == "path":
        annotations_file = get_data("OSBS_029.csv")
    else:
        annotations_file = geodataframe

    output_annotations = preprocess.split_raster(path_to_raster=get_data("OSBS_029.tif"),
                                                 annotations_file=annotations_file,
                                                 save_dir=tmp_path,
                                                 patch_size=300,
                                                 patch_overlap=0)

    # Returns a 7 column pandas array
    assert output_annotations.shape[1] == 7
    assert not output_annotations.empty

def test_split_raster_from_pd_dataframe(tmp_path):
    """Split raster into crops with overlaps to maintain all annotations"""
    annotations_file = pd.read_csv(get_data("OSBS_029.csv"))
    output_annotations = preprocess.split_raster(path_to_raster=get_data("OSBS_029.tif"),
                                                 annotations_file=annotations_file,
                                                 save_dir=tmp_path,
                                                 root_dir=os.path.dirname(get_data("OSBS_029.tif")),
                                                 patch_size=300,
                                                 patch_overlap=0)

    # Returns a 7 column pandas array
    assert output_annotations.shape[1] == 7
    assert not output_annotations.empty

def test_split_raster_no_annotations(config, tmp_path):
    """Split raster into crops with overlaps to maintain all annotations"""
    raster = get_data("2019_YELL_2_528000_4978000_image_crop2.png")

    output_crops = preprocess.split_raster(path_to_raster=raster,
                                           annotations_file=None,
                                           save_dir=tmp_path,
                                           patch_size=500,
                                           patch_overlap=0)

    # Returns a list of crops.
    assert len(output_crops) == 25

    # Assert that all output_crops exist
    for crop in output_crops:
        assert os.path.exists(crop)


def test_split_raster_from_image(config, tmp_path, geodataframe):
    r = rasterio.open(config.path_to_raster).read()
    r = np.rollaxis(r, 0, 3)
    annotations_file = preprocess.split_raster(numpy_image=r,
                                               annotations_file=geodataframe,
                                               save_dir=tmp_path,
                                               patch_size=config.patch_size,
                                               patch_overlap=config.patch_overlap,
                                               image_name="OSBS_029.tif")

    assert not annotations_file.empty


@pytest.mark.parametrize("allow_empty", [True, False])
def test_split_raster_empty(tmp_path, config, allow_empty):
    # Blank annotations file
    blank_annotations = pd.DataFrame({
        "image_path": "OSBS_029.tif",
        "xmin": [0],
        "ymin": [0],
        "xmax": [0],
        "ymax": [0],
        "label": ["Tree"]
    })
    blank_annotations.to_csv(tmp_path / "blank_annotations.csv", index=False)

    # Ignore blanks
    if not allow_empty:
        with pytest.raises(ValueError):
            annotations_file = preprocess.split_raster(
                path_to_raster=config.path_to_raster,
                annotations_file=str(tmp_path / "blank_annotations.csv"),
                save_dir=tmp_path,
                patch_size=config.patch_size,
                patch_overlap=config.patch_overlap,
                allow_empty=allow_empty)
    else:
        annotations_file = preprocess.split_raster(
            path_to_raster=config.path_to_raster,
            annotations_file=str(tmp_path / "blank_annotations.csv"),
            save_dir=tmp_path,
            patch_size=config.patch_size,
            patch_overlap=config.patch_overlap,
            allow_empty=allow_empty)
        assert annotations_file.shape[0] == 4
        assert annotations_file["xmin"].sum() == 0
        assert annotations_file["ymin"].sum() == 0
        assert annotations_file["xmax"].sum() == 0
        assert annotations_file["ymax"].sum() == 0
        assert (tmp_path / "OSBS_029_1.png").exists()


def test_split_size_error(config, tmp_path, geodataframe):
    with pytest.raises(ValueError):
        annotations_file = preprocess.split_raster(
            path_to_raster=config.path_to_raster,
            annotations_file=geodataframe,
            save_dir=tmp_path,
            patch_size=2000,
            patch_overlap=config.patch_overlap)


@pytest.mark.parametrize("orders", [(4, 400, 400), (400, 400, 4)])
def test_split_raster_4_band_warns(config, tmp_path, orders, geodataframe):
    """Test rasterio channel order
    (400, 400, 4) C x H x W
    (4, 400, 400) wrong channel order, H x W x C
    """

    # Confirm that the rasterio channel order is C x H x W
    assert rasterio.open(get_data("OSBS_029.tif")).read().shape[0] == 3
    numpy_image = np.zeros(orders, dtype=np.uint8)

    with pytest.warns(UserWarning):
        preprocess.split_raster(numpy_image=numpy_image,
                                annotations_file=geodataframe,
                                save_dir=tmp_path,
                                patch_size=config.patch_size,
                                patch_overlap=config.patch_overlap,
                                image_name="OSBS_029.tif")


# Test split_raster with point annotations file
def test_split_raster_with_point_annotations(tmp_path, config):
    # Create a temporary point annotations file
    annotations = pd.DataFrame({
        "image_path": ["OSBS_029.tif", "OSBS_029.tif"],
        "x": [100, 200],
        "y": [100, 200],
        "label": ["Tree", "Tree"]
    })
    annotations_file = str(tmp_path / "point_annotations.csv")
    annotations.to_csv(annotations_file, index=False)

    # Call split_raster function
    preprocess.split_raster(annotations_file=annotations_file,
                            path_to_raster=config.path_to_raster,
                            save_dir=tmp_path)

    # Assert that the output annotations file is created
    assert (tmp_path / "OSBS_029_0.png").exists()


# Test split_raster with box annotations file
def test_split_raster_with_box_annotations(tmp_path, config):
    # Create a temporary box annotations file
    annotations = pd.DataFrame({
        "image_path": ["OSBS_029.tif", "OSBS_029.tif"],
        "xmin": [100, 200],
        "ymin": [100, 200],
        "xmax": [200, 300],
        "ymax": [200, 300],
        "label": ["Tree", "Tree"]
    })
    annotations_file = str(tmp_path / "box_annotations.csv")
    annotations.to_csv(annotations_file, index=False)

    # Call split_raster function
    preprocess.split_raster(annotations_file=annotations_file,
                            path_to_raster=config.path_to_raster,
                            save_dir=tmp_path)

    # Assert that the output annotations file is created
    assert (tmp_path / "OSBS_029_0.png").exists()


# Test split_raster with polygon annotations file
def test_split_raster_with_polygon_annotations(tmp_path, config):
    # Create a temporary polygon annotations file with a polygon in WKT format
    sample_geometry = [
        geometry.Polygon([(0, 0), (0, 2), (1, 1), (1, 0), (0, 0)]),
        geometry.Polygon([(2, 2), (2, 4), (3, 3), (3, 2), (2, 2)])
    ]
    annotations = pd.DataFrame({
        "image_path": ["OSBS_029.tif", "OSBS_029.tif"],
        "polygon": [sample_geometry[0].wkt, sample_geometry[1].wkt],
        "label": ["Tree", "Tree"]
    })
    annotations_file = str(tmp_path / "polygon_annotations.csv")
    annotations.to_csv(annotations_file, index=False)

    # Call split_raster function
    split_annotations = preprocess.split_raster(annotations_file=annotations_file,
                                                path_to_raster=config.path_to_raster,
                                                save_dir=tmp_path)

    assert not split_annotations.empty

    # Assert that the output annotations file is created
    assert (tmp_path / "OSBS_029_0.png").exists()


def test_split_raster_points_translate_and_bounds(tmp_path):
    """Ensure split_raster keeps point geometry, translates to window coords,
    preserves count, and points lie within patch window when patch_overlap=0."""
    path_to_raster = get_data("OSBS_029.tif")
    # Choose points well inside expected windows to avoid edge duplication
    df = pd.DataFrame(
        {
            "image_path": ["OSBS_029.tif", "OSBS_029.tif"],
            "x": [50, 350],
            "y": [50, 350],
            "label": ["Tree", "Tree"],
        }
    )
    csv_path = str(tmp_path / "point_ann.csv")
    df.to_csv(csv_path, index=False)

    # Split with no overlap
    split_df = preprocess.split_raster(
        annotations_file=csv_path,
        path_to_raster=path_to_raster,
        save_dir=tmp_path,
        patch_size=300,
        patch_overlap=0,
    )

    # Geometry should be points
    assert utilities.determine_geometry_type(split_df) == "point"

    # Count should be preserved
    assert split_df.shape[0] == df.shape[0]

    # Coordinates should be relative to window and within [0, patch_size]
    # Use geometry.x/y as the source of truth
    assert (split_df.geometry.x >= 0).all()
    assert (split_df.geometry.y >= 0).all()
    assert (split_df.geometry.x <= 300).all()
    assert (split_df.geometry.y <= 300).all()


def test_split_raster_from_csv(tmp_path):
    """Read in annotations, convert to a projected shapefile, read back in and crop,
    the output annotations should still be maintained in logical place"""
    annotations = get_data("2018_SJER_3_252000_4107000_image_477.csv")
    path_to_raster = get_data("2018_SJER_3_252000_4107000_image_477.tif")

    # Check original data
    split_annotations = preprocess.split_raster(annotations_file=annotations,
                                                path_to_raster=path_to_raster,
                                                save_dir=tmp_path,
                                                root_dir=os.path.dirname(path_to_raster),
                                                patch_size=300)
    assert not split_annotations.empty


def test_split_raster_from_shp(tmp_path):
    annotations = get_data("2018_SJER_3_252000_4107000_image_477.csv")
    path_to_raster = get_data("2018_SJER_3_252000_4107000_image_477.tif")
    gdf = utilities.read_file(annotations)
    geo_coords = utilities.image_to_geo_coordinates(gdf)
    annotations_file = str(tmp_path / "projected_annotations.shp")
    geo_coords.to_file(annotations_file)

    # Call split_raster function
    split_annotations = preprocess.split_raster(annotations_file=annotations_file,
                                                path_to_raster=path_to_raster,
                                                save_dir=tmp_path,
                                                root_dir=os.path.dirname(path_to_raster),
                                                patch_size=300)

    assert not split_annotations.empty
