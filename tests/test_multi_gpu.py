# Create a multi-test gpu module that tests predict_image and predict_tile in the presense of 1 and more GPUs. If no GPU present, skip tests
import pytest
import torch
from deepforest import get_data
from matplotlib import pyplot as plt    

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
    assert m.config["devices"] == "Auto"

    # predict_image will only be passed to one GPU, nothing to parallelize
    boxes = m.predict_image()

    # Predict tile will be sent to multi-gpu
    img = m.predict_tile(return_plot=True)
    plt.imshow(img)
    plt.show()