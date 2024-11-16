import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from deepforest.model import Model


class Model(Model):

    def __init__(self, config, **kwargs):
        super().__init__(config)

    def load_backbone(self):
        backbone = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')

        return backbone

    def create_model(self, backbone=None):
        model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')

        # Modify the box predictor for the desired number of classes
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features_box, num_classes=self.config["num_classes"])

        # Modify the mask predictor for the desired number of classes
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, dim_reduced, num_classes=self.config["num_classes"])

        return model

    def check_model(self):
        """Ensure that model follows deepforest guidelines, see ##### If fails,
        raise ValueError."""
        # This assumes model creation is not expensive
        test_model = self.create_model()
        test_model.eval()

        # Create a dummy batch of 3 band data.
        x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]

        predictions = test_model(x)
        # Model takes in a batch of images
        assert len(predictions) == 2

        # Returns a list equal to number of images with proper keys per image
        model_keys = list(predictions[1].keys())
        model_keys.sort()
        assert model_keys == ['boxes', 'masks', 'labels', 'scores']
