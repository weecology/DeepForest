# Create a multi-test gpu module that tests predict_image and predict_tile in the presense of 1 and more GPUs. If no GPU present, skip tests
import pytest
import torch
from deepforest import get_data

def is_multi_gpu():
    gpus = torch.cuda.device_count()
    if gpus > 1:
        return True
    else:
        return False

pytest.mark.skipif(is_multi_gpu(), reason="No GPU found")
def test_multi_gpu(m):
    gpus = torch.cuda.device_count()
    print(f"Found {gpus} GPUs")
    img_path = get_data("OSBS_029.png")

    # Assert that pytorch lightning will detect GPUS available
    assert m.config["devices"] == "auto"

    # predict_image will only be passed to one GPU, nothing to parallelize
    boxes = m.predict_image(path=img_path)

    # Predict tile will be sent to multi-gpu
    boxes = m.predict_tile(raster_path = img_path, patch_size=100, patch_overlap=0.1)

    assert not boxes.empty
  

# Asset that 1 and 2 gpu produce the same results
def test_multi_gpu_equivalence(m):
    gpus = torch.cuda.device_count()
    print(f"Found {gpus} GPUs")
    img_path = get_data("OSBS_029.png")

    # Assert that pytorch lightning will detect GPUS available
    assert m.trainer.num_devices == gpus
    m.config["devices"] = 2
    m.create_trainer()
    assert m.trainer.num_devices == 2
    # Predict tile will be sent to multi-gpu
    boxes_2_gpu = m.predict_tile(raster_path = img_path, patch_size=100, patch_overlap=0.1)

    # Assert that pytorch lightning will detect GPUS available
    m.config["devices"] = 1
    m.create_trainer()
    assert m.trainer.num_devices == 1
    # Predict tile will be sent to multi-gpu
    boxes_1_gpu = m.predict_tile(raster_path = img_path, patch_size=100, patch_overlap=0.1)

    # Assert that the pandas dataframes are the same
    assert boxes_1_gpu.equals(boxes_2_gpu)
