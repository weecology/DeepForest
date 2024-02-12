# Tree model tests
from backbones.tree import train
from deepforest import get_data
import os
import pytest

@pytest.fixture()
def backbone_config():
    backbone_config = train.read_config("{}/backbone_config.yml".format(os.path.dirname(train.__file__)))
    backbone_config["train"] = get_data("OSBS_029.csv")
    backbone_config["test"]  = get_data("OSBS_029.csv")

    return backbone_config

def test_wrapper(backbone_config):
    train.wrapper(backbone_config)