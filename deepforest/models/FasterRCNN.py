import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
import torch
from deepforest.model import Model


class Model(Model):

    def __init__(self, config, **kwargs):
        super().__init__(config)

    def load_backbone(self):
        """A torch vision retinanet model"""
        backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        return backbone

    def create_model(self, backbone=None):
        """Create a FasterRCNN model
        Args:
            backbone: a compatible torchvision backbone, e.g. torchvision.models.detection.fasterrcnn_resnet50_fpn
        Returns:
            model: a pytorch nn module
        """
        # load Faster RCNN pre-trained model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # get the number of input features
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # define a new head for the detector with required number of classes
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes=self.config["num_classes"])

        return model
