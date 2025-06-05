# Ensure that multiprocessing is behaving as expected.
from deepforest import main, get_data
from deepforest.datasets import prediction

import pytest
import os

@pytest.mark.parametrize("num_workers", [2])
def test_predict_tile_workers(m, num_workers):
    # Default workers is 0
    original_workers = m.config.workers
    assert original_workers == 0

    m.config.workers = num_workers
    csv_file = get_data("OSBS_029.csv")
    # make a dataset
    ds = prediction.FromCSVFile(csv_file=csv_file,
                             root_dir=os.path.dirname(csv_file))
    dataloader = m.predict_dataloader(ds)
    assert dataloader.num_workers == num_workers


@pytest.mark.parametrize("num_workers", [2])
@pytest.mark.parametrize("dataset_class", [
    prediction.FromCSVFile,
    prediction.SingleImage,
    prediction.MultiImage,
    prediction.TiledRaster,
])
def test_dataset_tile_workers_config(m, num_workers, dataset_class):
    csv_file = get_data("OSBS_029.csv")
    root_dir = os.path.dirname(csv_file)
    
    # Create dataset based on class
    if dataset_class == prediction.FromCSVFile:
        ds = dataset_class(csv_file=csv_file, root_dir=root_dir)
    elif dataset_class == prediction.SingleImage:
        image_path = os.path.join(root_dir, "OSBS_029.png")
        ds = dataset_class(path=image_path)
    elif dataset_class == prediction.MultiImage:
        image_path = os.path.join(root_dir, "OSBS_029.png")
        ds = dataset_class(paths=[image_path], patch_size=400, patch_overlap=0.1)
    else:  # TiledRaster
        image_path = os.path.join(root_dir, "test_tiled.tif")
        ds = dataset_class(path=image_path, patch_size=400, patch_overlap=0.1)
        
    dataloader = m.predict_dataloader(ds)
    assert dataloader.num_workers == num_workers


def test_multi_process_dataloader_strategy_single(m):
    root_dir = os.path.dirname(get_data("OSBS_029.csv"))
    image_path = os.path.join(root_dir, "OSBS_029.png")
    
    results = m.predict_tile(
        path=image_path,
        dataloader_strategy="single",
        patch_size=400,
        patch_overlap=0,
    )
    assert len(results) > 0

def test_multi_process_dataloader_strategy_batch(m):
    root_dir = os.path.dirname(get_data("OSBS_029.csv"))
    image_path = os.path.join(root_dir, "OSBS_029.png")
    
    results = m.predict_tile(
        path=[image_path],
        dataloader_strategy="batch",
        patch_size=400,
        patch_overlap=0,
    )
    assert len(results) > 0

def test_multi_process_dataloader_strategy_window(m):
    root_dir = os.path.dirname(get_data("OSBS_029.csv"))
    image_path = os.path.join(root_dir, "test_tiled.tif")
    
    with pytest.raises(ValueError):
        results = m.predict_tile(
            path=image_path,
            dataloader_strategy="window", 
            patch_size=400,
            patch_overlap=0,
        )
