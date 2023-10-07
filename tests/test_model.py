import pytest
import torch
from deepforest import model

# The model object is achitecture agnostic container.
def test_model_no_args(config):
    with pytest.raises(ValueError):
        model.Model(config)

    