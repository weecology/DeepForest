import os
from omegaconf import OmegaConf
from deepforest.conf import schema
from deepforest import get_data


# Test that the default configuration can be correctly loaded into the schema template.
def test_default_config_schema():

    # Schema + config
    base = OmegaConf.structured(schema.Config)
    config_path = get_data("config.yaml")

    default_config = OmegaConf.load(config_path)
    config = OmegaConf.merge(base, default_config)

    assert isinstance(config, type(base))
