import os
import warnings
from pathlib import Path

import torch
import torchvision
from huggingface_hub import PyTorchModelHubMixin
from torchvision.models.detection.retinanet import (
    AnchorGenerator,
    ResNet50_Weights,
    RetinaNet,
    RetinaNet_ResNet50_FPN_Weights,
)

from deepforest.model import BaseModel
from deepforest.models.dinov3 import Dinov3Model


class RetinaNetHub(RetinaNet, PyTorchModelHubMixin):
    """RetinaNet extension that allows the use of the HF Hub API."""

    def __init__(
        self,
        weights: str | None = None,
        backbone_weights: str | None = None,
        backbone="resnet50",
        num_classes: int = 1,
        nms_thresh: float = 0.05,
        score_thresh: float = 0.5,
        label_dict: dict = None,
        use_conv_pyramid: bool = True,
        fpn_out_channels: int = 256,
        freeze_backbone: bool = False,
        **kwargs,
    ):
        if backbone == "dinov3":
            # Only pass repo_id if weights is not None to avoid overriding the default
            dinov3_kwargs = {
                "use_conv_pyramid": use_conv_pyramid,
                "fpn_out_channels": fpn_out_channels,
                "frozen": freeze_backbone,
            }
            if weights is not None:
                dinov3_kwargs["repo_id"] = weights

            backbone = Dinov3Model(**dinov3_kwargs)
            anchor_sizes = tuple(
                (x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3)))
                for x in [32, 64, 128, 256]
            )
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

            # Can vary with model, e.g. sat does not use ImageNet
            image_mean = backbone.image_mean
            image_std = backbone.image_std
        elif backbone == "resnet50":
            backbone = torchvision.models.detection.retinanet_resnet50_fpn(
                weights=weights, backbone_weights=backbone_weights
            ).backbone
            anchor_generator = None  # Use default

            if freeze_backbone:
                for param in backbone.parameters():
                    param.requires_grad = False

            # Explicitly use ImageNet
            image_mean = torch.tensor([0.485, 0.456, 0.406])
            image_std = torch.tensor([0.229, 0.224, 0.225])
        else:
            raise NotImplementedError(
                f"Backbone {backbone} is unknown, or not supported. Use 'dinov3' or 'resnet50'"
            )

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

        # See docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_load_state_dict_pre_hook
        # For backwards compatibility with earlier versions of torch, call the _ method
        self._register_load_state_dict_pre_hook(
            RetinaNetHub._strip_legacy_prefix, with_module=True
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
    def from_pretrained(
        cls, pretrained_model_name_or_path, *, num_classes=None, label_dict=None, **kwargs
    ):
        """This function overrides the default from_pretrained method to better
        support overriding the number of classes in a pretrained model.

        If the target num_classes differs from the model's num_classes,
        then the model heads are reinitialized to compensate.

        Also handles PyTorch Lightning .ckpt files directly.
        """

        # Handle PyTorch Lightning checkpoint files (.ckpt)
        # TODO: Use from_pretrained and generate HF compatible config from ckpt?
        if (
            isinstance(pretrained_model_name_or_path, (str, Path))
            and str(pretrained_model_name_or_path).endswith(".ckpt")
            and os.path.exists(pretrained_model_name_or_path)
        ):
            # Always load to CPU first to avoid device mapping issues later
            checkpoint = torch.load(
                pretrained_model_name_or_path, map_location="cpu", weights_only=False
            )

            config = checkpoint["hyper_parameters"]["config"]

            if config["architecture"] != "retinanet":
                raise ValueError(
                    f"Checkpoint architecture is {config['architecture']}, should be retinanet."
                )
            elif config["backbone"] != kwargs.get("backbone"):
                raise ValueError(
                    f"You are trying to instantiate a RetinaNet with a {kwargs.get('backbone')}, backbone, but your checkpoint contains {config['backbone']}. Check your config is correct."
                )

            # Instantiate model with config as provided and load in the state
            model = cls(**kwargs)
            model.load_state_dict(checkpoint["state_dict"])

        else:
            # Otherwise use HuggingFace Hub path, which requires weights + config
            model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Override class info if specified
        if num_classes is not None and label_dict is not None:
            if num_classes != model.num_classes:
                warnings.warn(
                    f"The number of classes in your config differs "
                    f"compared to the model checkpoint ({model.num_classes}-class)."
                    f" If you are fine-tuning on a new dataset that "
                    f"has {num_classes} then this is expected.",
                    stacklevel=2,
                )

                model._adjust_classes(num_classes)

            model.label_dict = label_dict
            model.update_config()

        return model

    def _adjust_classes(self, num_classes):
        self.num_classes = num_classes

        self.head.classification_head = (
            torchvision.models.detection.retinanet.RetinaNetClassificationHead(
                in_channels=self.backbone.out_channels,
                num_classes=num_classes,
                num_anchors=self.head.classification_head.num_anchors,
            )
        )
        self.head.regression_head = (
            torchvision.models.detection.retinanet.RetinaNetRegressionHead(
                in_channels=self.backbone.out_channels,
                num_anchors=self.head.classification_head.num_anchors,
            )
        )

    def update_config(self):
        # Stored as config on HF
        self.config = {
            "num_classes": self.num_classes,
            "nms_thresh": self.nms_thresh,
            "score_thresh": self.score_thresh,
            "label_dict": self.label_dict,
            **self.kwargs,
        }

    @staticmethod
    def _strip_legacy_prefix(
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
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

    def create_anchor_generator(
        self, sizes=((8, 16, 32, 64, 128, 256, 400),), aspect_ratios=((0.5, 1.0, 2.0),)
    ):
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

    def create_model(
        self,
        pretrained: str | Path = "resnet50-mscoco",
        *,
        revision: str | None = None,
        map_location: str | torch.device | None = None,
        **hf_args,
    ) -> RetinaNetHub:
        """Create a retinanet model
        Args:
            pretrained (str | Path): Specifies repository ID for weight download or predefined model type. Defaults to "resnet50-mscoco"
            revision (str | None): Repository revision
            map_location (str | torch.device | None): Device to load weights onto
            **hf_args: Any other arguments to load_pretrained
        Returns:
            model: a pytorch nn module
        """

        if pretrained == "resnet50-imagenet":
            if revision is not None:
                warnings.warn(
                    "Ignoring revision and using an un-initialized RetinaNet head, ImageNet backbone.",
                    stacklevel=2,
                )
            model = RetinaNetHub(
                weights=None,
                backbone_weights=ResNet50_Weights.IMAGENET1K_V2,
                num_classes=self.config.num_classes,
                nms_thresh=self.config.nms_thresh,
                score_thresh=self.config.score_thresh,
                label_dict=self.config.label_dict,
                freeze_backbone=self.config.train.freeze_backbone,
            )
        elif pretrained == "resnet50-mscoco":
            if revision is not None:
                warnings.warn(
                    "Ignoring revision and fine-tuning from ResNet50 MS-COCO checkpoint.",
                    stacklevel=2,
                )
            model = RetinaNetHub(
                weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1,
                num_classes=self.config.num_classes,
                nms_thresh=self.config.nms_thresh,
                score_thresh=self.config.score_thresh,
                label_dict=self.config.label_dict,
                freeze_backbone=self.config.train.freeze_backbone,
            )
        elif pretrained is None:
            warnings.warn(
                "Using a randomly initialized model. You probably don't want to do this unless you have a very large dataset to pretrain on..",
                stacklevel=2,
            )
            model = RetinaNetHub(
                weights=None,
                backbone_weights=None,
                backbone=self.config.backbone,
                num_classes=self.config.num_classes,
                nms_thresh=self.config.nms_thresh,
                score_thresh=self.config.score_thresh,
                label_dict=self.config.label_dict,
                freeze_backbone=self.config.train.freeze_backbone,
            )
        # Deepforest/tree, fine-tune from user, etc.
        else:
            model = RetinaNetHub.from_pretrained(
                pretrained,
                revision=revision,
                backbone=self.config.backbone,
                num_classes=self.config.num_classes,
                label_dict=self.config.label_dict,
                nms_thresh=self.config.nms_thresh,
                score_thresh=self.config.score_thresh,
                freeze_backbone=self.config.train.freeze_backbone,
                **hf_args,
            )

        return model.to(map_location)
