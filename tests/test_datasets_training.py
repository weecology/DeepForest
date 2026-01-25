# test dataset model
import os
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from shapely import geometry

from deepforest import get_data, main, utilities
from deepforest.datasets.training import BoxDataset


def single_class():
    csv_file = get_data("example.csv")

    return csv_file


def multi_class():
    csv_file = get_data("testfile_multi.csv")

    return csv_file


@pytest.fixture()
def raster_path():
    return get_data(path="OSBS_029.tif")


@pytest.mark.parametrize(
    "csv_file,label_dict",
    [(single_class(), {"Tree": 0}), (multi_class(), {"Alive": 0, "Dead": 1})],
)
def test_BoxDataset(csv_file, label_dict):
    root_dir = os.path.dirname(get_data("OSBS_029.png"))
    ds = BoxDataset(csv_file=csv_file, root_dir=root_dir, label_dict=label_dict)
    raw_data = pd.read_csv(csv_file)

    assert len(ds) == len(raw_data.image_path.unique())

    for i in range(len(ds)):
        # Between 0 and 1
        image, targets, paths = ds[i]
        assert image.max() <= 1
        assert image.min() >= 0
        assert targets["boxes"].shape == (raw_data.shape[0], 4)
        assert targets["labels"].shape == (raw_data.shape[0],)
        assert targets["labels"].dtype == torch.int64
        assert len(np.unique(targets["labels"])) == len(raw_data.label.unique())


def test_single_class_with_empty(tmp_path):
    """Add fake empty annotations to test parsing"""
    csv_file1 = get_data("example.csv")
    csv_file2 = get_data("OSBS_029.csv")

    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    df = pd.concat([df1, df2])

    df.loc[df.image_path == "OSBS_029.tif", "xmin"] = 0
    df.loc[df.image_path == "OSBS_029.tif", "ymin"] = 0
    df.loc[df.image_path == "OSBS_029.tif", "xmax"] = 0
    df.loc[df.image_path == "OSBS_029.tif", "ymax"] = 0

    df.to_csv(f"{tmp_path}_test_empty.csv")

    root_dir = os.path.dirname(get_data("OSBS_029.png"))
    ds = BoxDataset(
        csv_file=f"{tmp_path}_test_empty.csv", root_dir=root_dir, label_dict={"Tree": 0}
    )
    assert len(ds) == 2
    # First image has annotations
    assert not torch.sum(ds[0][1]["boxes"]) == 0
    # Second image has no annotations
    assert torch.sum(ds[1][1]["boxes"]) == 0


@pytest.mark.parametrize("augment", [True, False])
def test_BoxDataset_transform(augment):
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    ds = BoxDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        augmentations=["HorizontalFlip"] if augment else None,
    )

    for i in range(len(ds)):
        # Between 0 and 1
        image, targets, path = ds[i]
        assert image.max() <= 1
        assert image.min() >= 0
        assert targets["boxes"].shape == (79, 4)
        assert targets["labels"].shape == (79,)

        assert torch.is_tensor(targets["boxes"])
        assert torch.is_tensor(targets["labels"])
        assert targets["labels"].dtype == torch.int64
        assert torch.is_tensor(image)


def test_collate():
    """Due to data augmentations the dataset class may yield empty bounding box annotations"""
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    ds = BoxDataset(csv_file=csv_file, root_dir=root_dir)

    for i in range(len(ds)):
        # Between 0 and 1
        batch = ds[i]
        collated_batch = utilities.collate_fn(batch)
        assert len(collated_batch) == 2


def test_empty_collate():
    """Due to data augmentations the dataset class may yield empty bounding box annotations"""
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    ds = BoxDataset(csv_file=csv_file, root_dir=root_dir)

    for i in range(len(ds)):
        # Between 0 and 1
        batch = ds[i]
        collated_batch = utilities.collate_fn([None, batch, batch])
        len(collated_batch[0]) == 2


def test_BoxDataset_format():
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    ds = BoxDataset(csv_file=csv_file, root_dir=root_dir)
    image, targets, path = next(iter(ds))

    # Assert image is channels first format
    assert image.shape[0] == 3


def test_multi_image_warning(tmp_path):
    csv_file1 = get_data("example.csv")
    csv_file2 = get_data("OSBS_029.csv")
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    df = pd.concat([df1, df2])
    csv_file = str(tmp_path.joinpath("multiple.csv"))
    df.to_csv(csv_file)

    root_dir = os.path.dirname(csv_file1)
    ds = BoxDataset(csv_file=csv_file, root_dir=root_dir)

    for i in range(len(ds)):
        # Between 0 and 1
        batch = ds[i]
        collated_batch = utilities.collate_fn([None, batch, batch])
        len(collated_batch[0]) == 2


def test_label_validation__training_csv(tmp_path):
    """Test training CSV labels are validated against label_dict"""
    m = main.deepforest(config_args={"num_classes": 1, "label_dict": {"Bird": 0}, "log_root": tmp_path})
    m.config.train.csv_file = get_data("example.csv")  # contains 'Tree' label
    m.config.train.root_dir = os.path.dirname(get_data("example.csv"))
    m.create_trainer()

    with pytest.raises(
        ValueError, match="Labels \\['Tree'\\] are missing from label_dict"
    ):
        m.trainer.fit(m)


def test_csv_label_validation__validation_csv(m, tmp_path):
    """Test validation CSV labels are validated against label_dict"""
    m = main.deepforest(config_args={"num_classes": 1, "label_dict": {"Tree": 0}, "log_root": tmp_path})
    m.config.train.csv_file = get_data("example.csv")  # contains 'Tree' label
    m.config.train.root_dir = os.path.dirname(get_data("example.csv"))
    m.config.validation.csv_file = get_data(
        "testfile_multi.csv"
    )  # contains 'Dead', 'Alive' labels
    m.config.validation.root_dir = os.path.dirname(get_data("testfile_multi.csv"))
    m.create_trainer()

    with pytest.raises(
        ValueError, match="Labels \\['Dead', 'Alive'\\] are missing from label_dict"
    ):
        m.trainer.fit(m)


def test_BoxDataset_validate_labels():
    """Test that BoxDataset validates labels correctly"""
    from deepforest.datasets.training import BoxDataset

    csv_file = get_data("example.csv")  # contains 'Tree' label
    root_dir = os.path.dirname(csv_file)

    # Valid case: CSV labels are in label_dict
    ds = BoxDataset(csv_file=csv_file, root_dir=root_dir, label_dict={"Tree": 0})
    # Should not raise an error

    # Invalid case: CSV labels are not in label_dict
    with pytest.raises(
        ValueError, match="Labels \\['Tree'\\] are missing from label_dict"
    ):
        BoxDataset(csv_file=csv_file, root_dir=root_dir, label_dict={"Bird": 0})


def test_validate_BoxDataset_missing_image(tmp_path, raster_path):
    csv_path = str(tmp_path / "test.csv")
    df = pd.DataFrame(
        {
            "image_path": ["missing.tif"],
            "xmin": 0,
            "ymin": 0,
            "xmax": 10,
            "ymax": 10,
            "label": ["Tree"],
        }
    )
    df.to_csv(csv_path, index=False)
    root_dir = os.path.dirname(raster_path)
    with pytest.raises(ValueError, match="Failed to open image"):
        _ = BoxDataset(csv_file=csv_path, root_dir=root_dir)


def test_BoxDataset_validate_coordinates(tmp_path, raster_path):
    # Valid case: uses example.csv with all valid boxes
    csv_path = get_data("example.csv")
    root_dir = os.path.dirname(csv_path)
    _ = BoxDataset(csv_file=csv_path, root_dir=root_dir)

    # Test various invalid box coordinates
    with Image.open(raster_path) as image:
        width, height = image.size

    invalid_boxes = [
        (width - 5, 0, width + 10, 10),  # xmax exceeds width
        (0, height - 5, 10, height + 10),  # ymax exceeds height
        (-5, 0, 10, 10),  # negative xmin
        (0, -5, 10, 10),  # negative ymin
    ]

    for box in invalid_boxes:
        csv_path = str(tmp_path / "test.csv")
        df = pd.DataFrame(
            {
                "image_path": ["OSBS_029.tif"],
                "xmin": [box[0]],
                "ymin": [box[1]],
                "xmax": [box[2]],
                "ymax": [box[3]],
                "label": ["Tree"],
            }
        )
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="exceeds image dimensions"):
            BoxDataset(csv_file=csv_path, root_dir=root_dir)


def test_BoxDataset_validate_non_rectangular_polygon(tmp_path, raster_path):
    # Create a non-rectangular polygon (triangle)
    non_rect_polygon = geometry.Polygon([(10, 10), (50, 10), (30, 40)])

    # Create dataframe with the non-rectangular geometry
    df = pd.DataFrame(
        {
            "geometry": [non_rect_polygon],
            "label": ["Tree"],
            "image_path": ["OSBS_029.tif"],
        }
    )

    # Convert to GeoDataFrame and save
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    csv_path = str(tmp_path / "non_rect.csv")
    gdf.to_csv(csv_path, index=False)

    root_dir = os.path.dirname(raster_path)

    # Should raise an error because the geometry is not a valid bounding box
    with pytest.raises(ValueError, match="is not a valid bounding box"):
        BoxDataset(csv_file=csv_path, root_dir=root_dir)


def test_BoxDataset_with_projected_shapefile(tmp_path, raster_path):
    """Test that BoxDataset can load a shapefile with projected coordinates and converts to pixel coordinates"""
    import geopandas as gpd

    # Get the raster to extract CRS and bounds
    import rasterio
    from shapely import geometry

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        bounds = src.bounds
        left, bottom, right, top = bounds
        resolution = src.res[0]

    # Create sample geometry in projected coordinates (within raster bounds)
    # Use coordinates that are well within the raster bounds to avoid edge issues
    sample_x = left + (right - left) * 0.3
    sample_y = bottom + (top - bottom) * 0.3
    box_size = resolution * 5  # 5 pixels in projected coordinates

    sample_geometry = [
        geometry.box(sample_x, sample_y, sample_x + box_size, sample_y + box_size),
        geometry.box(
            sample_x + box_size * 2,
            sample_y + box_size * 2,
            sample_x + box_size * 3,
            sample_y + box_size * 3,
        ),
    ]
    labels = ["Tree", "Tree"]
    image_path = os.path.basename(raster_path)

    df = pd.DataFrame(
        {"geometry": sample_geometry, "label": labels, "image_path": image_path}
    )
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=raster_crs)

    # Save as shapefile
    shapefile_path = str(tmp_path / "annotations.shp")
    gdf.to_file(shapefile_path)

    # Load with BoxDataset
    root_dir = os.path.dirname(raster_path)
    ds = BoxDataset(csv_file=shapefile_path, root_dir=root_dir, label_dict={"Tree": 0})

    # Verify dataset loaded successfully
    assert len(ds) == 1  # One unique image

    # Get one sample
    image, targets, path = ds[0]

    # Verify boxes are in pixel coordinates (should be positive and reasonable)
    # After geo_to_image_coordinates conversion, values should be in pixel space
    boxes = targets["boxes"]
    assert torch.all(boxes >= 0), (
        "Boxes should have non-negative coordinates in pixel space"
    )
    assert torch.all(boxes[:, 2] > boxes[:, 0]), "xmax should be greater than xmin"
    assert torch.all(boxes[:, 3] > boxes[:, 1]), "ymax should be greater than ymin"

def test_validate_coordinates_negative(tmpdir):
    """
    Ensure vectorized validation catches negative coordinates
    """
    img_path = os.path.join(tmpdir, "test_neg.jpg")
    Image.new('RGB', (100, 100), color='white').save(img_path)

    df = pd.DataFrame({
        'image_path': ["test_neg.jpg"],
        'xmin': [-5], 'ymin': [10],
        'xmax': [50], 'ymax': [50],
        'label': ["Tree"]
    })

    with pytest.raises(ValueError, match="negative coordinates"):
        training.BoxDataset(annotation_dict=df, root_dir=str(tmpdir))

def test_validate_coordinates_out_of_bounds(tmpdir):
    """
    Ensure vectorized validation catches OOB coordinates
    """
    img_path = os.path.join(tmpdir, "test_oob.jpg")
    Image.new('RGB', (100, 100), color='white').save(img_path)

    df = pd.DataFrame({
        'image_path': ["test_oob.jpg"],
        'xmin': [10], 'ymin': [10],
        'xmax': [150], 'ymax': [50],
        'label': ["Tree"]
    })

    with pytest.raises(ValueError, match="exceeding image dimensions"):
        training.BoxDataset(annotation_dict=df, root_dir=str(tmpdir))
