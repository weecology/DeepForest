#Fixtures model to only download model once
# download latest release
import pytest
from deepforest import utilities


@pytest.fixture(scope="session")
def download_release():
    print("running fixtures")
    utilities.use_release()
