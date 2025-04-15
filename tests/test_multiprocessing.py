# Ensure that multiprocessing is behaving as expected.
from deepforest import main, get_data
from deepforest import dataset
from deepforest import utilities

import pytest
import os
import shutil
import yaml


@pytest.mark.parametrize("num_workers", [0, 2])
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

#TODO: Fix this test to reflect hydra
@pytest.mark.xfail
def test_predict_tile_workers_config(tmpdir):
    # Open config file and change workers to 1, save to tmpdir
    config_file = get_data("config.yml")
    tmp_config_file = os.path.join(tmpdir, "config.yml")

    shutil.copyfile(config_file, tmp_config_file)
    x = utilities.load_config(tmp_config_file)
    x["workers"] = 1
    with open(tmp_config_file, "w+") as f:
        f.write(yaml.dump(x))

    m = main.deepforest(config_file=tmp_config_file)
    csv_file = get_data("OSBS_029.csv")
    # make a dataset
    ds = dataset.TreeDataset(csv_file=csv_file,
                             root_dir=os.path.dirname(csv_file),
                             transforms=None,
                             train=False)
    dataloader = m.predict_dataloader(ds)
    assert dataloader.num_workers == 1
