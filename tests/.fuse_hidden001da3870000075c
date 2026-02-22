import pytest
from deepforest.main import deepforest

MODEL_NAMES = [
    "weecology/deepforest-bird",
    "weecology/everglades-bird-species-detector",
    "weecology/deepforest-tree",
    "weecology/deepforest-livestock",
    "weecology/cropmodel-deadtrees",
    "weecology/everglades-nest-detection",
]

@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_huggingface_model_labels(model_name):
    # Initialize the model
    model = deepforest()
    
    # Load the specified model from HuggingFace
    model.load_model(model_name=model_name)
    
    # We assert that the numeric_to_label_dict is populated correctly.
    # Prior to deepforest PR #1263, this would incorrectly default to {0: "Tree"} for all models.
    # While the exact label dictionaries differ per model, we can safely assert:
    # 1. It is a non-empty dictionary
    # 2. It has integer keys and string values
    
    assert isinstance(model.numeric_to_label_dict, dict)
    assert len(model.numeric_to_label_dict) > 0
    
    for numeric, label in model.numeric_to_label_dict.items():
        assert isinstance(numeric, int)
        assert isinstance(label, str)
        assert len(label) > 0

    # For the default tree model, it should be {0: "Tree"}
    if model_name == "weecology/deepforest-tree":
        assert model.numeric_to_label_dict == {0: "Tree"}
        
    # For the bird model, we specifically know it's "Bird"
    if model_name == "weecology/deepforest-bird":
        assert 0 in model.numeric_to_label_dict
        assert "Bird" in model.numeric_to_label_dict.values()
