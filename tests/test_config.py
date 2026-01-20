from omegaconf import OmegaConf
import os
import pytest
import torch
import yaml

from deepforest import main, get_data, utilities
from deepforest.conf import schema
from deepforest.model import CropModel


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

def test_custom_config_file(tmp_path):
    # Verify that we can use a custom config file which overrides the default
    config = {
        'label_dict': {
            'foo': 0
        }
    }

    config_path = tmp_path / "custom-config.yaml"
    with open(config_path, 'w') as fp:
        yaml.dump(config, fp)

    m = main.deepforest(config = str(config_path))
    assert 'foo' in m.config.label_dict
    assert 'Tree' not in m.config.label_dict


def test_load_config_missing_field(m, tmp_path):
    """Test that checkpoints load successfully when they're
    missing fields added to the schema
    """
    # Calling fit/predict etc. is required before save_model
    m.predict_tile(get_data("SOAP_061.png"))
    checkpoint_path = tmp_path / "checkpoint.pl"
    m.save_model(checkpoint_path)

    # Load the checkpoint and remove "batch size"
    checkpoint = torch.load(checkpoint_path)
    saved_config = checkpoint['hyper_parameters']['config']
    del saved_config['batch_size']
    torch.save(checkpoint, checkpoint_path)

    # Load the checkpoint
    loaded = main.deepforest.load_from_checkpoint(checkpoint_path)
    assert isinstance(loaded.config.batch_size, int)

def test_config_additional_field(m, tmp_path):
    """Test that checkpoints with extra fields not in schema still load successfully,
    such as added config items in newer versions of DeepForest.
    """
    m.predict_tile(get_data("SOAP_061.png"))
    checkpoint_path = tmp_path / "checkpoint.pl"
    m.save_model(checkpoint_path)

    # Load the checkpoint and add a field not in the current schema
    checkpoint = torch.load(checkpoint_path)
    saved_config = checkpoint['hyper_parameters']['config']
    saved_config['new_flag'] = 'test_value'
    torch.save(checkpoint, checkpoint_path)

    # Load the checkpoint
    loaded = main.deepforest.load_from_checkpoint(checkpoint_path)
    assert loaded.config.new_flag == 'test_value'

def test_strict_mode_rejects_unknown_fields():
    """Test that strict mode rejects configs with unknown fields."""
    from omegaconf import errors as omegaconf_errors

    # Try to load a config with an unknown field in strict mode
    config_with_unknown_field = {"unknown_field": "value"}

    with pytest.raises(omegaconf_errors.ConfigKeyError):
        utilities.load_config(overrides=config_with_unknown_field, strict=True)

def test_load_checkpoint_with_dictconfig(m, tmp_path):
    """Test that we can load an older checkpoint with a DictConfig in it.
    """
    m.predict_tile(get_data("SOAP_061.png"))
    checkpoint_path = tmp_path / "checkpoint.pl"
    m.save_model(checkpoint_path)

    # Load the checkpoint and replace the config dict with a DictConfig
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    saved_config = checkpoint['hyper_parameters']['config']
    checkpoint['hyper_parameters']['config'] = OmegaConf.create(saved_config)
    torch.save(checkpoint, checkpoint_path)

    # Load the checkpoint with "unsafe" weights_only
    loaded = main.deepforest.load_from_checkpoint(checkpoint_path, weights_only=False)
    assert loaded.config == m.config

    # Check if we save + reload, this gets fixed.
    loaded.predict_tile(get_data("SOAP_061.png"))
    loaded.save_model(checkpoint_path)
    fixed = main.deepforest.load_from_checkpoint(checkpoint_path)
    assert fixed.config == m.config
