#test dataset model
from deepforest import get_data
from deepforest import dataset
from deepforest import utilities
import os
import pytest
import torch
import pandas as pd
import numpy as np

def single_class():
    csv_file = get_data("example.csv")
    
    return csv_file

def multi_class():
    csv_file = get_data("testfile_multi.csv")
    
    return csv_file

@pytest.mark.parametrize("csv_file,label_dict",[(single_class(), {"Tree":0}),(multi_class(),{"Alive":0,"Dead":1})])
def test_TreeDataset(csv_file, label_dict):
    root_dir = os.path.dirname(csv_file)
    ds = dataset.TreeDataset(csv_file=csv_file,
                             root_dir=root_dir,
                             label_dict=label_dict)
    raw_data = pd.read_csv(csv_file)
    
    assert len(ds) == len(raw_data.image_path.unique())
    
    for i in range(len(ds)):
        #Between 0 and 1
        path, image, targets = ds[i]
        assert image.max() <= 1
        assert image.min() >= 0
        assert targets["boxes"].shape == (raw_data.shape[0],4)
        assert targets["labels"].shape == (raw_data.shape[0],)
        assert len(np.unique(targets["labels"])) == len(raw_data.label.unique())
        

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
        
        assert torch.is_tensor(targets["boxes"])
        assert torch.is_tensor(targets["labels"])
        assert torch.is_tensor(image)

def test_collate():
    """Due to data augmentations the dataset class may yield empty bounding box annotations"""
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    ds = dataset.TreeDataset(csv_file=csv_file,
                             root_dir=root_dir,
                             transforms=dataset.get_transform(augment=False))

    for i in range(len(ds)):
        #Between 0 and 1
        batch = ds[i]
        collated_batch = utilities.collate_fn(batch)
        assert len(collated_batch) == 2
        
def test_empty_collate():
    """Due to data augmentations the dataset class may yield empty bounding box annotations"""
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    ds = dataset.TreeDataset(csv_file=csv_file,
                             root_dir=root_dir,
                             transforms=dataset.get_transform(augment=False))

    for i in range(len(ds)):
        #Between 0 and 1
        batch = ds[i]
        collated_batch = utilities.collate_fn([None, batch, batch])
        len(collated_batch[0]) == 2

def test_predict_dataloader():
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    ds = dataset.TreeDataset(csv_file=csv_file,
                             root_dir=root_dir,
                             train=False)
    image = next(iter(ds))
    #Assert image is channels first format
    assert image.shape[0] == 3
    