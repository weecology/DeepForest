#test model
from deepforest import model
import torch


def test_load_backbone():
    retinanet = model.load_backbone()
    retinanet.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    prediction = retinanet(x)    

def test_create_model():
    retinanet_model = model.create_model(num_classes=2,nms_thresh=0.1, score_thresh=0.2)
    
    #retinanet_model.eval()
    #x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    #predictions = retinanet_model(x)    
    #for prediction in predictions:
        #assert [x == 1 for x in prediction["labels"]]
    
    #retinanet_model = model.create_model(num_classes=3)
    
    #retinanet_model.eval()
    #x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    #predictions = retinanet_model(x)    
    #for prediction in predictions:
        #assert [x in [1,2] for x in prediction["labels"]]    
