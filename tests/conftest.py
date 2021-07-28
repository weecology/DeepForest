#Fixtures model to only download model once
# download latest release
import pytest
from deepforest import utilities
from deepforest import get_data
import os

@pytest.fixture(scope="session")
def download_release():
    print("running fixtures")
    utilities.use_release()
    assert os.path.exists(get_data("NEON.pt"))
    
