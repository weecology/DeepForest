#test dataset model
from deepforest import get_data
from deepforest import dataset
from torch.utils.data import DataLoader
import os
import pytest

def test_TreeDataset():
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    ds = dataset.TreeDataset(csv_file=csv_file,
                             root_dir=root_dir,
                             transforms=None)

    for i in range(len(ds)):
        #Between 0 and 1
        image, targets = ds[i]
        assert image.max() <= 1
        assert image.min() >= 0
        assert targets["boxes"].shape == (79, 4)
        assert targets["labels"].shape == (79,)
        

@pytest.mark.parametrize("train",[True,False])
def test_TreeDataset_transform(train):
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    ds = dataset.TreeDataset(csv_file=csv_file,
                             root_dir=root_dir,
                             transforms=dataset.get_transform(train=train))

    for i in range(len(ds)):
        #Between 0 and 1
        image, targets = ds[i]
        assert image.max() <= 1
        assert image.min() >= 0
        assert targets["boxes"].shape == (79, 4)
        assert targets["labels"].shape == (79,)
        
