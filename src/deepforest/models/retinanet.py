# Model
from pathlib import Path
import warnings

import torch
import torchvision
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection.retinanet import AnchorGenerator
from deepforest.model import BaseModel
from deepforest.models.dinov3 import Dinov3Model
from huggingface_hub import PyTorchModelHubMixin


class RetinaNetHub(RetinaNet, PyTorchModelHubMixin):
    """RetinaNet extension that allows the use of the HF Hub API."""

    def __init__(self,
                 backbone_weights: str | None = None,
                 num_classes: int = 1,
                 nms_thresh: float = 0.05,
                 score_thresh: float = 0.5,
                 label_dict: dict = None,
                 use_conv_pyramid: bool = True,
                 fpn_out_channels: int = 256,
                 freeze_backbone: bool = True,
                 **kwargs):

        if backbone_weights == "dinov3":
            backbone = Dinov3Model(
                use_conv_pyramid=use_conv_pyramid,
                fpn_out_channels=fpn_out_channels,
                frozen=freeze_backbone,
            )
            anchor_sizes = tuple((x, int(x * 2**(1.0 / 3)), int(x * 2**(2.0 / 3)))
                                 for x in [32, 64, 128, 256])
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

            # Can vary with model, e.g. sat does not use ImageNet
            image_mean = backbone.image_mean
            image_std = backbone.image_std
        else:
            if freeze_backbone:
                warnings.warn(
                    "Frozen backbone is currently not enabled for ResNet, but you can set the learning rate to zero."
                )
            backbone = torchvision.models.detection.retinanet_resnet50_fpn(
                weights=backbone_weights).backbone
            anchor_generator = None  # Use default

            # Explicitly use ImageNet
            image_mean = torch.tensor([0.485, 0.456, 0.406])
            image_std = torch.tensor([0.229, 0.224, 0.225])

        super().__init__(
            backbone=backbone,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            image_mean=image_mean,
            image_std=image_std,
            **kwargs,
        )

        self.num_classes = num_classes
        self.label_dict = label_dict
        self.use_conv_pyramid = use_conv_pyramid
        self.fpn_out_channels = fpn_out_channels
        self.kwargs = kwargs

        # Store normalization parameters for denormalization
        self.image_mean = image_mean
        self.image_std = image_std

        self.update_config()

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        *,
                        num_classes=None,
                        label_dict=None,
                        **kwargs):
        """This function overrides the default from_pretrained method to better
        support overriding the number of classes in a pretrained model.

        If the target num_classes differs from the model's num_classes,
        then the model heads are reinitialized to compensate.
        """
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Override class info if specified
        if num_classes is not None and label_dict is not None:
            if num_classes != model.num_classes:
                warnings.warn(
                    f"The number of classes in your config differs compared to the model checkpoint ({model.num_classes}-class)."
                    f" If you are fine-tuning on a new dataset that has {num_classes} then this is expected."
                )

                model._adjust_classes(num_classes)

            model.label_dict = label_dict
            model.update_config()

        return model

    def _adjust_classes(self, num_classes):

        self.num_classes = num_classes

        self.head.classification_head = torchvision.models.detection.retinanet.RetinaNetClassificationHead(
            in_channels=self.backbone.out_channels,
            num_classes=num_classes,
            num_anchors=self.head.classification_head.num_anchors)
        self.head.regression_head = torchvision.models.detection.retinanet.RetinaNetRegressionHead(
            in_channels=self.backbone.out_channels,
            num_anchors=self.head.classification_head.num_anchors)

    def update_config(self):

        # Stored as config on HF
        self.config = {
            "num_classes": self.num_classes,
            "nms_thresh": self.nms_thresh,
            "score_thresh": self.score_thresh,
            "label_dict": self.label_dict,
            **self.kwargs
        }

    @staticmethod
    def _strip_legacy_prefix(module, state_dict, prefix, local_metadata, strict,
                             missing_keys, unexpected_keys, error_msgs):
        """Static method to fixup state dict keys from older DeepForest
        checkpoints. The method simply renames keys that start with "model."
        and the hook is called before from_pretrained (and load_state_dict) is
        called.

        The function signature is required by PyTorch but most of the
        arguments are undocumented and we don't use them.
        """

        if prefix:
            return

        to_add = {}
        to_delete = []
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_k = k.replace("model.", "", 1)  # -> "backbone.*"
                to_add[new_k] = v
                to_delete.append(k)

        for k in to_delete:
            del state_dict[k]
        state_dict.update(to_add)


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
                                 score_thresh=self.config.score_thresh,
                                 label_dict=self.config.label_dict)
        elif pretrained == "dinov3":
            model = RetinaNetHub(backbone_weights="dinov3",
                                 num_classes=self.config.num_classes,
                                 nms_thresh=self.config.nms_thresh,
                                 score_thresh=self.config.score_thresh,
                                 label_dict=self.config.label_dict,
                                 freeze_backbone=True)
        else:
            model = RetinaNetHub.from_pretrained(pretrained,
                                                 revision=revision,
                                                 num_classes=self.config.num_classes,
                                                 label_dict=self.config.label_dict,
                                                 nms_thresh=self.config.nms_thresh,
                                                 score_thresh=self.config.score_thresh,
                                                 **hf_args)

        return model.to(map_location)
