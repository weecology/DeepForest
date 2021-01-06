#test model
from deepforest import model
import torch


def test_load_backbone():
    retinanet = model.load_backbone()
    retinanet.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    prediction = retinanet(x)
