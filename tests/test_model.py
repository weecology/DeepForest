#test model
from deepforest import model
import pytest
import numpy as np
import torch

def test_load_backbone():
    retinanet = model.load_backbone()
    retinanet.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    prediction = retinanet(x)    

@pytest.mark.parametrize("num_classes",[1,2,10])
def test_create_model(num_classes):
    retinanet_model = model.create_model(num_classes=2,nms_thresh=0.1, score_thresh=0.2)
    
    retinanet_model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = retinanet_model(x)    
