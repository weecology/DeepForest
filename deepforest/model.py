# Model
import torchvision
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection.retinanet import AnchorGenerator


def load_backbone():
    """A torch vision retinanet model"""
    backbone = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)

    # load the model onto the computation device
    return backbone


def create_anchor_generator(sizes=((8, 16, 32, 64, 128, 256, 400),),
                            aspect_ratios=((0.5, 1.0, 2.0),)):
    """
    Create anchor box generator as a function of sizes and aspect ratios
    Documented https://github.com/pytorch/vision/blob/67b25288ca202d027e8b06e17111f1bcebd2046c/torchvision/models/detection/anchor_utils.py#L9
    let's make the network generate 5 x 3 anchors per spatial
    location, with 5 different sizes and 3 different aspect
    ratios. We have a Tuple[Tuple[int]] because each feature
    map could potentially have different sizes and
    aspect ratios
    Args:
        sizes:
        aspect_ratios:

    Returns: anchor_generator, a pytorch module

    """
    anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)

    return anchor_generator


def create_model(num_classes, nms_thresh, score_thresh, backbone = None):
    """Create a retinanet model
    Args:
        num_classes (int): number of classes in the model
        nms_thresh (float): non-max suppression threshold for intersection-over-union [0,1]
        score_thresh (float): minimum prediction score to keep during prediction  [0,1]
    Returns:
        model: a pytorch nn module
    """
    if not backbone:
        resnet = load_backbone()
        backbone = resnet.backbone
        
    model = RetinaNet(backbone=backbone, num_classes=num_classes)
    model.nms_thresh = nms_thresh
    model.score_thresh = score_thresh

    # Optionally allow anchor generator parameters to be created here
    # https://pytorch.org/vision/stable/_modules/torchvision/models/detection/retinanet.html

    return model
