#Fixtures model to only download model once
# download latest release
import pytest
from deepforest import utilities
from deepforest import get_data
import os
from deepforest import _ROOT

collect_ignore = ['setup.py']

@pytest.fixture()
def raster_path():
    return get_data(path='OSBS_029.tif')    
    
@pytest.fixture(scope="session")
def download_release():
    print("running fixtures")
    utilities.use_release()
    assert os.path.exists(get_data("NEON.pt"))

@pytest.fixture(scope="session")
def config():
    config = utilities.read_config("{}/deepforest_config.yml".format(os.path.dirname(_ROOT)))
    config["fast_dev_run"] = True
    config["batch_size"] = True

    return config

@pytest.fixture(scope="session")
def ROOT():
    from deepforest import _ROOT

    return _ROOT