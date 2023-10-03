#Fixtures model to only download model once
# download latest release
import pytest
from deepforest import utilities, main
from deepforest import get_data
import os

collect_ignore = ['setup.py']

@pytest.fixture(scope="session")
def download_release():
    print("running fixtures")
    utilities.use_release()
    assert os.path.exists(get_data("NEON.pt"))

@pytest.fixture()
def two_class_m():
    m = main.deepforest(num_classes=2,label_dict={"Alive":0,"Dead":1})
    m.config["train"]["csv_file"] = get_data("testfile_multi.csv") 
    m.config["train"]["root_dir"] = os.path.dirname(get_data("testfile_multi.csv"))
    m.config["train"]["fast_dev_run"] = True
    m.config["batch_size"] = 2
        
    m.config["validation"]["csv_file"] = get_data("testfile_multi.csv") 
    m.config["validation"]["root_dir"] = os.path.dirname(get_data("testfile_multi.csv"))
    m.config["validation"]["val_accuracy_interval"] = 1

    m.create_trainer()
    
    return m

@pytest.fixture()
def m(download_release):
    m = main.deepforest()
    m.config["train"]["csv_file"] = get_data("example.csv") 
    m.config["train"]["root_dir"] = os.path.dirname(get_data("example.csv"))
    m.config["train"]["fast_dev_run"] = True
    m.config["batch_size"] = 2
        
    m.config["validation"]["csv_file"] = get_data("example.csv") 
    m.config["validation"]["root_dir"] = os.path.dirname(get_data("example.csv"))
    m.config["workers"] = 0 
    m.config["validation"]["val_accuracy_interval"] = 1
    m.config["train"]["epochs"] = 2
    
    m.create_trainer()
    m.use_release(check_release=False)
    
    return m