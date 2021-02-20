#Model
import torchvision
import torch


def load_backbone():
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        pretrained=True)

    # load the model onto the computation device
    return model

def create_model():
    pass

def load_model():
    pass
