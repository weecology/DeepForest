#Fixtures model to only download model once
# download latest release
import pytest
from deepforest import get_data
from deepforest import main
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
@pytest.fixture(scope="session")
def m():
    print("Creating conftest deepforest object")
    m = main.deepforest()
    m.use_release()    
    m.config["train"]["csv_file"] = get_data("example.csv") 
    m.config["train"]["root_dir"] = os.path.dirname(get_data("example.csv"))
    m.config["train"]["fast_dev_run"] = False
    m.config["batch_size"] = 2
    m.config["workers"] = 0
    
    m.config["validation"]["csv_file"] = get_data("example.csv") 
    m.config["validation"]["root_dir"] = os.path.dirname(get_data("example.csv"))
    
    m.create_trainer()
    #save the prediction dataframe after training and compare with prediction after reload checkpoint 
    
    return m

@pytest.fixture(scope="session")
def two_class_m():
    m = main.deepforest(num_classes=2,label_dict={"Alive":0,"Dead":1})
    m.config["train"]["csv_file"] = get_data("testfile_multi.csv") 
    m.config["train"]["root_dir"] = os.path.dirname(get_data("testfile_multi.csv"))
    m.config["train"]["fast_dev_run"] = True
    m.config["batch_size"] = 2
    m.config["workers"] = 0
        
    m.config["validation"]["csv_file"] = get_data("testfile_multi.csv") 
    m.config["validation"]["root_dir"] = os.path.dirname(get_data("testfile_multi.csv"))
    m.config["validation"]["val_accuracy_interval"] = 1

    m.create_trainer()
    
    return m
