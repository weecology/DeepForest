#test model
from deepforest import model
import torch


def test_load_backbone():
    retinanet = model.load_backbone()
    retinanet.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    prediction = retinanet(x)
    
    prediction["classes"] == 1
    

def test_create_model():
    retinanet_model = model.create_model(num_classes=2)
    assert retinanet_model.parameters