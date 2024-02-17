# Ensure that multiprocessing is behaving as expected. 
from deepforest import main, get_data
from deepforest import dataset
import pytest
import os

@pytest.mark.parametrize("num_workers",[0, 1 ,2])
def test_predict_tile_workers(m, num_workers):
    # Default workers is 0
    original_workers = m.config["workers"]
    assert original_workers == 0

    m.config["workers"] = num_workers
    csv_file = get_data("OSBS_029.csv")
    # make a dataset
    ds = dataset.TreeDataset(csv_file=csv_file,
                                 root_dir=os.path.dirname(csv_file),
                                 transforms=None,
                                 train=False)
    dataloader = m.predict_dataloader(ds)
    assert dataloader.num_workers == num_workers
    

    

