#Model
import torchvision
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection.retinanet import AnchorGenerator

def load_backbone():
    backbone = torchvision.models.detection.retinanet_resnet50_fpn(
        pretrained=True)

    # load the model onto the computation device
    return backbone

def create_anchor_generator():
     # let's make the network generate 5 x 3 anchors per spatial
     # location, with 5 different sizes and 3 different aspect
     # ratios. We have a Tuple[Tuple[int]] because each feature
     # map could potentially have different sizes and
     # aspect ratios
    #Documented https://github.com/pytorch/vision/blob/67b25288ca202d027e8b06e17111f1bcebd2046c/torchvision/models/detection/anchor_utils.py#L9
    
    anchor_generator = AnchorGenerator(sizes=((8,16,32,64,128,256),),aspect_ratios=((0.5, 1.0, 2.0),))
    
    return anchor_generator
    
def create_model(num_classes):
    backbone = load_backbone()
    anchor_generator = create_anchor_generator()
    
    model = RetinaNet(backbone, num_classes=num_classes, anchor_generator=anchor_generator)
    
    return model