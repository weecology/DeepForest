import os

import pandas as pd

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
