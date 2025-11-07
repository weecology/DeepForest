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

@pytest.mark.parametrize("config, expected_model, classes", [("tree", "weecology/deepforest-tree", ["Tree"]),
                                           ("livestock", "weecology/deepforest-livestock", ["Animal"]),
                                           ("bird", "weecology/deepforest-bird", ["Bird"])])
def test_config_built_in(config, expected_model, classes):
    # Check that derived configs correctly set the expected models with the right classes
    m = main.deepforest(config=config)
    assert m.config.model.name == expected_model
    assert set(m.config.label_dict.keys()) == set(classes)

def test_config_derived_with_override():
    # Check two levels of override (derived config + config_arg from user)
    m = main.deepforest(config="bird", config_args={'label_dict': {'foo': 0}})
    assert m.config.model.name == "weecology/deepforest-bird"
    assert 'foo' in m.config.label_dict
    assert 'Bird' not in m.config.label_dict

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
