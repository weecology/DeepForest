import numpy as np
import pandas as pd
import pytest

from deepforest import main


@pytest.mark.parametrize(
    "model_name",
    [
        "weecology/deepforest-bird",
        # "weecology/deepforest-everglades-bird-species-detector",
        # "weecology/deepforest-tree",
        # "weecology/deepforest-livestock",
        # "weecology/cropmodel-deadtrees",
    ],
)
def test_white_image_predict_tile_no_predictions_bird_model(model_name):
    """All-white image should yield no detections with various models."""
    m = main.deepforest()
    m.create_trainer()
    m.load_model(model_name)
    # Create a white image (uint8 RGB)
    white = np.full((2048, 2048, 3), 255, dtype=np.uint8)
    res = m.predict_tile(
        image=white,
        patch_size=128,
        patch_overlap=0.0,
        iou_threshold=m.config.nms_thresh,
    )
    assert len(res) == 0
    #assert (res is None) or (isinstance(res, pd.DataFrame) and res.empty)
