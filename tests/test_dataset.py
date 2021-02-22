#test dataset model
from deepforest import get_data
from deepforest import dataset
import os
import pytest
import pandas as pd

def test_TreeDataset():
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    ds = dataset.TreeDataset(csv_file=csv_file,
                             root_dir=root_dir,
                             transforms=None)
    raw_data = pd.read_csv(csv_file)
    
    assert len(ds) == len(raw_data.image_path.unique())
    
    for i in range(len(ds)):
        #Between 0 and 1
        path, image, targets = ds[i]
        assert image.max() <= 1
        assert image.min() >= 0
        assert targets["boxes"].shape == (79, 4)
        assert targets["labels"].shape == (79,)
        

@pytest.mark.parametrize("augment",[True,False])
def test_TreeDataset_transform(augment):
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    ds = dataset.TreeDataset(csv_file=csv_file,
                             root_dir=root_dir,
                             transforms=dataset.get_transform(augment=augment))

    for i in range(len(ds)):
        #Between 0 and 1
        path, image, targets = ds[i]
        assert image.max() <= 1
        assert image.min() >= 0
        assert targets["boxes"].shape == (79, 4)
        assert targets["labels"].shape == (79,)
        
