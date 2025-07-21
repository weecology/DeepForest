import pytest
from deepforest import get_data, utilities
from deepforest.main import deepforest

def run_inference(config):
    model = deepforest(config=config)
    model.load_model()
    results = model.predict_image(path=get_data("OSBS_029.png"))
    assert len(results) > 0

    return results

@pytest.mark.parametrize("overrides", [
    {"architecture": "retinanet",
     "model": {"name": "weecology/deepforest-tree"}},

    {"architecture": "DeformableDetr",
     "model": {"name": "joshvm/milliontrees-detr"}},
])
def test_model_inference(overrides):
    config = utilities.load_config(overrides=overrides)
    run_inference(config)

def test_score_threshold():
    config = utilities.load_config(overrides={"score_thresh": 0})
    results_zero_conf = run_inference(config)

    config = utilities.load_config(overrides={"score_thresh": 0.75})
    results_high_conf = run_inference(config)

    assert len(results_zero_conf) > 0

    # Expect a non-zero result to be smaller than allowing all preds
    if results_high_conf is not None:
        assert len(results_high_conf) > 0
        assert len(results_high_conf) < len(results_zero_conf)


def test_nms_threshold():
    config = utilities.load_config(overrides={"nms_thresh": 0})
    results_zero_nms_thresh = run_inference(config)

    config = utilities.load_config(overrides={"nms_thresh": 0.75})
    results_high_nms_thresh = run_inference(config)

    assert len(results_high_nms_thresh) > 0, "Model failed to produce any results with high NMS threshold"
    assert len(results_zero_nms_thresh) > 0, "Model should produce at least one result"
    assert len(results_high_nms_thresh) >= len(results_zero_nms_thresh), (
        "Higher NMS threshold should retain at least as many boxes as a lower threshold"
    )
