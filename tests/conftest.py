#Fixtures model to only download model once
# download latest release
import pytest
from deepforest import get_data
from deepforest import utilities
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
def pytest_sessionstart(session):
    release_tag, state_dict = utilities.use_bird_release()
    release_tag, state_dict = utilities.use_release()
    
@pytest.fixture(scope="session",autouse=True)
def config():
    config = utilities.read_config("deepforest_config.yml")
    config["train"]["csv_file"] = get_data("example.csv") 
    config["train"]["root_dir"] = os.path.dirname(get_data("example.csv"))
    config["train"]["fast_dev_run"] = True
    config["batch_size"] = 2
    config["workers"] = 0
    
    config["validation"]["csv_file"] = get_data("example.csv") 
    config["validation"]["root_dir"] = os.path.dirname(get_data("example.csv"))
        
    return config
