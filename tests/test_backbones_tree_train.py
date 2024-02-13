# Tree model tests
from backbones.tree import train
from deepforest import get_data
from deepforest.utilities import read_config
import os
import pytest

@pytest.fixture()
def backbone_config(tmpdir):
    backbone_config = train.read_config("{}/backbone_config.yml".format(os.path.dirname(train.__file__)))
    backbone_config["train"] = get_data("OSBS_029.csv")
    backbone_config["test"]  = get_data("OSBS_029.csv")
    backbone_config["savedir"] = tmpdir.strpath

    return backbone_config

def test_wrapper(backbone_config):
    m = train.wrapper(backbone_config, deepforest_config_kwargs={"train":{"fast_dev_run":True}})
    assert os.path.exists(m.model_path)

def test_wrapper_pretrain(backbone_config):
    m = train.wrapper(backbone_config, deepforest_config_kwargs={"train":{"fast_dev_run":True}})
    assert os.path.exists(m.model_path)