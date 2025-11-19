from omegaconf import OmegaConf
import os
import pytest
import yaml

from deepforest import main, get_data
from deepforest.conf import schema


# Test that the default configuration can be correctly loaded into the schema template.
def test_default_config_schema():

    # Schema + config
    base = OmegaConf.structured(schema.Config)
    config_path = get_data("config.yaml")

    default_config = OmegaConf.load(config_path)
    config = OmegaConf.merge(base, default_config)

    assert isinstance(config, type(base))

def test_config_main_dictconfig():
    base = OmegaConf.structured(schema.Config)
    m = main.deepforest(config=base)
    assert m.config.model.name == "weecology/deepforest-tree"

def test_config_main_empty_config():
    m = main.deepforest(config=None)
    assert m.config.model.name == "weecology/deepforest-tree"

def test_custom_config_file(tmpdir):
    # Verify that we can use a custom config file which overrides the default
    config = {
        'label_dict': {
            'foo': 0
        }
    }

    config_path = os.path.join(tmpdir, "custom-config.yaml")
    with open(config_path, 'w') as fp:
        yaml.dump(config, fp)

    m = main.deepforest(config = config_path)
    assert 'foo' in m.config.label_dict
    assert 'Tree' not in m.config.label_dict
