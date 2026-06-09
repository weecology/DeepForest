import warnings
from pathlib import Path

import torch
import torchvision
from huggingface_hub import PyTorchModelHubMixin
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN as _TorchvisionMaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from deepforest.model import BaseModel


class MaskRCNN(_TorchvisionMaskRCNN, PyTorchModelHubMixin):
    """Mask R-CNN extension that allows the use of the HF Hub API.

    DeepForest labels are zero-indexed foreground classes (e.g.
    ``{"Tree": 0}``). torchvision detection models reserve class ``0`` for
    background, so this wrapper builds the underlying model with
    ``num_classes + 1`` outputs and transparently shifts labels by one:
    targets are shifted up before training and predictions are shifted back
    down. Callers therefore always see zero-indexed labels, matching the box
    and point workflows.
    """

    task: str = "polygon"

    def __init__(
        self,
        backbone_weights: str | None = None,
        num_classes: int = 1,
        nms_thresh: float = 0.05,
        score_thresh: float = 0.5,
        label_dict: dict = None,
        **kwargs,
    ):
        backbone = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=backbone_weights
        ).backbone

        # torchvision reserves class 0 for background, so add one class.
        super().__init__(
            backbone=backbone,
            num_classes=num_classes + 1,
            box_nms_thresh=nms_thresh,
            box_score_thresh=score_thresh,
            **kwargs,
        )

        self.num_classes = num_classes
        self.label_dict = label_dict
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.kwargs = kwargs

        self.update_config()

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *, num_classes=None, label_dict=None, **kwargs
    ):
        """Override default from_pretrained to support changing the number of
        classes in a pretrained model.

        If the target num_classes differs from the model's num_classes,
        the box and mask heads are reinitialized to compensate.
        """
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Override class info if specified
        if num_classes is not None and label_dict is not None:
            if len(label_dict) != num_classes:
                raise ValueError(
                    f"num_classes ({num_classes}) does not match the number of labels "
                    f"in label_dict ({len(label_dict)})."
                )

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
        """Rebuild the box and mask predictor heads for ``num_classes``
        foreground classes (``num_classes + 1`` with background)."""
        self.num_classes = num_classes

        in_features = self.roi_heads.box_predictor.cls_score.in_features
        self.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

        in_features_mask = self.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = self.roi_heads.mask_predictor.conv5_mask.out_channels
        self.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes + 1
        )

    def update_config(self):
        # Stored as config on HF
        self._hub_mixin_config = {
            "num_classes": self.num_classes,
            "nms_thresh": self.nms_thresh,
            "score_thresh": self.score_thresh,
            "label_dict": self.label_dict,
            **self.kwargs,
        }

    def forward(self, images, targets=None):
        """Shift labels between DeepForest (0-indexed) and torchvision
        (background=0) conventions.

        In training mode, target labels are shifted up by one. In eval
        mode, predicted labels are shifted back down by one.
        """
        if self.training:
            if targets is None:
                raise ValueError("targets must be provided in training mode")
            shifted_targets = []
            for target in targets:
                shifted = dict(target)
                shifted["labels"] = target["labels"] + 1
                shifted_targets.append(shifted)
            return super().forward(images, shifted_targets)

        outputs = super().forward(images)
        for output in outputs:
            output["labels"] = output["labels"] - 1
        return outputs


class Model(BaseModel):
    """DeepForest model wrapper for Mask R-CNN instance segmentation.

    Selected via ``config.architecture = "maskrcnn"`` with
    ``config.model.task`` resolving to ``"polygon"``.
    """

    def create_model(
        self,
        pretrained: str | Path | None = None,
        *,
        revision: str | None = None,
        map_location: str | torch.device | None = None,
        **hf_args,
    ) -> MaskRCNN:
        """Create a Mask R-CNN model.

        Args:
            pretrained: If supplied, repository ID for weight download, otherwise use default COCO backbone weights
            revision: Repository revision
            map_location: Device to load weights onto
            **hf_args: Any other arguments to load_pretrained
        Returns:
            model: a pytorch nn module
        """
        label_dict = dict(self.config.label_dict) if self.config.label_dict else None

        if pretrained is None:
            model = MaskRCNN(
                backbone_weights="COCO_V1",
                num_classes=self.config.num_classes,
                nms_thresh=self.config.nms_thresh,
                score_thresh=self.config.score_thresh,
                label_dict=label_dict,
                box_detections_per_img=self.config.detections_per_img,
            )
        else:
            model = MaskRCNN.from_pretrained(
                pretrained,
                revision=revision,
                num_classes=self.config.num_classes,
                label_dict=label_dict,
                nms_thresh=self.config.nms_thresh,
                score_thresh=self.config.score_thresh,
                box_detections_per_img=self.config.detections_per_img,
                **hf_args,
            )

        return model.to(map_location)

    def check_model(self) -> None:
        """Validate the model returns instance-segmentation outputs."""
        test_model = self.create_model()
        test_model.eval()

        x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        predictions = test_model(x)
        assert len(predictions) == 2

        model_keys = sorted(predictions[1].keys())
        assert model_keys == ["boxes", "labels", "masks", "scores"]
