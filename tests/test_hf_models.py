import pytest
from deepforest import main
from deepforest.model import CropModel


BOX_MODELS = [
    "weecology/deepforest-bird",
    "weecology/deepforest-tree",
    "weecology/deepforest-livestock",
    "weecology/everglades-bird-species-detector",
    "weecology/everglades-nest-detection",
]

CROP_MODELS = [
    "weecology/cropmodel-tree-genus",
    "weecology/cropmodel-tree-species",
    "weecology/cropmodel-neon-resnet18-genus",
    "weecology/cropmodel-neon-resnet18-species",
    "weecology/cropmodel-deadtrees",
]


@pytest.mark.parametrize("repo_id", CROP_MODELS)
def test_load_crop_models(repo_id):
    """Load all public models in the weecology org from the HF Hub.
    """
    model = CropModel.load_model(repo_id=repo_id)
    assert model is not None
    # light sanity: has label_dict and num_classes after load
    assert getattr(model, "label_dict", None) is not None
    assert getattr(model, "num_classes", None) is not None

@pytest.mark.parametrize("repo_id", BOX_MODELS)
def test_load_box_models(repo_id):
        df = main.deepforest()
        df.load_model(model_name=repo_id)
        assert df.model is not None
        # detection models should have label_dict on the underlying model
        assert getattr(df.model, "label_dict", None) is not None
