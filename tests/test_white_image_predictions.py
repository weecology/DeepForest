import numpy as np
import pandas as pd
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

WHITE_IMAGE_SIZE = (2048, 2048, 3)
PATCH_SIZE = 400
PATCH_OVERLAP = 0.0
SCORE_THRESH = 0.3
IOU_THRESH = 0.0


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_white_image_no_predictions(model_name):
    model = deepforest()
    model.load_model(model_name=model_name)
    model.config.score_thresh = SCORE_THRESH
    if hasattr(model, "model") and hasattr(model.model, "score_thresh"):
        model.model.score_thresh = SCORE_THRESH

    white = np.full(WHITE_IMAGE_SIZE, 255, dtype=np.uint8)
    results = model.predict_tile(
        image=white,
        patch_size=PATCH_SIZE,
        patch_overlap=PATCH_OVERLAP,
        iou_threshold=IOU_THRESH,
    )

    if isinstance(results, tuple):
        results = results[0]

    assert results is None or (isinstance(results, pd.DataFrame) and results.empty), (
        f"{model_name} produced {len(results)} predictions"
    )
