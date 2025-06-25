# Model
from pathlib import Path
import torch
import torchvision
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection.retinanet import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights
from deepforest.model import BaseModel
from huggingface_hub import PyTorchModelHubMixin


class RetinaNetHub(RetinaNet, PyTorchModelHubMixin):
    """RetinaNet extension that allows the use of push_to_hub."""

    def __init__(self,
                 backbone_weights: str | None = None,
                 num_classes: int = 1,
                 nms_thresh: float = 0.05,
                 score_thresh: float = 0.1,
                 label_dict: dict = {"Tree": 0},
                 **kwargs):

        if len(label_dict) != num_classes:
            raise ValueError("The length of label_dict must match the number of classes.")

        backbone = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=backbone_weights).backbone

        super().__init__(backbone=backbone,
                         num_classes=num_classes,
                         score_thresh=score_thresh,
                         nms_thresh=nms_thresh,
                         **kwargs)

        self.label_dict = label_dict

        # Stored as config on HF
        self.config = {
            "num_classes": num_classes,
            "nms_thresh": nms_thresh,
            "score_thresh": score_thresh,
            "label_dict": label_dict,
            **kwargs
        }


class Model(BaseModel):

    def __init__(self, config, **kwargs):
        super().__init__(config)

    def create_anchor_generator(self,
                                sizes=((8, 16, 32, 64, 128, 256, 400),),
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

    def create_model(self,
                     pretrained: str | Path | None = None,
                     *,
                     revision: str | None = None,
                     map_location: str | torch.device | None = None,
                     **hf_args) -> RetinaNetHub:
        """Create a retinanet model
        Args:
            pretrained (str | Path | None): If supplied, specifies repository ID for weight download, otherwise use default COCO weights
            revision (str | None): Repository revision
            map_location (str | torch.device | None): Device to load weights onto
            **hf_args: Any other arguments to load_pretrained
        Returns:
            model: a pytorch nn module
        """

        if pretrained is None:
            model = RetinaNetHub(backbone_weights="COCO_V1",
                                 num_classes=self.config.num_classes,
                                 nms_thresh=self.config.nms_thresh,
                                 score_thresh=self.config.retinanet.score_thresh,
                                 label_dict=self.config.label_dict)
        else:
            model = RetinaNetHub.from_pretrained(pretrained,
                                                 revision=revision,
                                                 map_location=map_location,
                                                 **hf_args)

        return model
