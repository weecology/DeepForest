import pytest
import numpy as np
from deepforest import main

MODELS_TO_TEST = [
    ("weecology/deepforest-bird", "Bird"),
    # ("weecology/everglades-nest-detection", "Nest"),
    ("weecology/deepforest-tree", "Tree"),
    ("weecology/deepforest-livestock", "Livestock")
]

@pytest.mark.parametrize("model_name, expected_label", MODELS_TO_TEST)
def test_hub_model_labels(model_name, expected_label):
    """
    Regression test for #1276: Ensure models loaded from Hub
    return their specific labels, not the default 'Tree'.
    """
    m = main.deepforest()

    m.load_model(model_name=model_name)

    assert expected_label in m.label_dict.keys(), \
        f"Model {model_name} label_dict {m.label_dict} does not contain '{expected_label}'"

    white_img = np.full((2048, 2048, 3), 255, dtype=np.uint8)

    m.config.score_thresh = 0.01
    m.model.score_thresh = 0.01

    res = m.predict_tile(
        image=white_img,
        patch_size=128,
        patch_overlap=0,
        iou_threshold=0.1
    )

    if res is not None and not res.empty:
        assert (res["label"] == expected_label).all(), \
            f"Model {model_name} predicted wrong labels: {res['label'].unique()}"
