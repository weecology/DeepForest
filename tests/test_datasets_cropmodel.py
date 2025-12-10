import os

import pandas as pd
from PIL import Image

from deepforest import get_data
from deepforest.datasets.cropmodel import BoundingBoxDataset


def test_bounding_box_dataset():
    # Create a sample dataframe
    df = pd.read_csv(get_data("OSBS_029.csv"))

    # Create the BoundingBoxDataset object
    ds = BoundingBoxDataset(df, root_dir=os.path.dirname(get_data("OSBS_029.png")))

    # Check the length of the dataset
    assert len(ds) == df.shape[0]

    # Get an item from the dataset
    item = ds[0]

    # Check the shape of the RGB tensor
    assert item.shape == (3, 224, 224)


def test_bounding_box_dataset_expand_increases_window():
    # Load annotations and corresponding image
    df = pd.read_csv(get_data("OSBS_029.csv"))
    img_path = get_data("OSBS_029.png")
    root_dir = os.path.dirname(img_path)

    # Read image size
    width, height = Image.open(img_path).size

    # Select a box well inside the image to avoid clipping during expansion
    margin = 30
    safe = df[
        (df["xmin"] > margin)
        & (df["ymin"] > margin)
        & (df["xmax"] < (width - margin))
        & (df["ymax"] < (height - margin))
    ]
    # Use a single-row dataframe for deterministic comparison
    row = safe.iloc[0:1].copy()

    # Use a no-op transform so we can compare raw window sizes directly (H, W, 3)
    noop = lambda x: x
    ds_no_expand = BoundingBoxDataset(row, root_dir=root_dir, transform=noop, expand=0)
    ds_expand_10 = BoundingBoxDataset(row, root_dir=root_dir, transform=noop, expand=10)

    crop_no_expand = ds_no_expand[0]
    crop_expand_10 = ds_expand_10[0]

    # Expansion adds 10 pixels to each side -> +20 px in both height and width
    assert crop_expand_10.shape[0] == crop_no_expand.shape[0] + 20
    assert crop_expand_10.shape[1] == crop_no_expand.shape[1] + 20
