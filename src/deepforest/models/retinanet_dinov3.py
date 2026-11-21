import json
import logging
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torchvision.models.detection.retinanet import AnchorGenerator, RetinaNet

from deepforest.model import BaseModel
from deepforest.models.dinov3 import Dinov3Model
from deepforest.models.retinanet import RetinaNetHub

logger = logging.getLogger(__name__)


def _read_hub_config(name: str | Path, revision: str | None = None) -> dict:
    """Read config.json from a Hugging Face repository or local directory.

    Args:
        name: Repository ID or local directory.
        revision: Repository revision.
    """
    if Path(name).is_dir():
        config_file = Path(name) / "config.json"
    else:
        config_file = hf_hub_download(str(name), "config.json", revision=revision)
    with open(config_file) as f:
        return json.load(f)


class RetinaNetDinov3Hub(RetinaNetHub):
    """RetinaNet with a DINOv3 vision transformer backbone.

    The backbone weights are loaded from a DINOv3 repository on Hugging
    Face. Note that Facebook's base weights are gated and require users
    to request access to the model on Hugging Face before use.
    """

    def __init__(
        self,
        backbone_weights: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        num_classes: int = 1,
        nms_thresh: float = 0.05,
        score_thresh: float = 0.5,
        label_dict: dict = None,
        freeze_backbone: bool = False,
        **kwargs,
    ):
        """
        Args:
            backbone_weights: Hugging Face repository ID of the DINOv3 model.
            num_classes: Number of classes.
            nms_thresh: IoU threshold for non-maximum suppression.
            score_thresh: Minimum score for a prediction to be returned.
            label_dict: Mapping from label name to class ID.
            freeze_backbone: If True, the DINOv3 transformer is not trained.
                The feature pyramid layers are always trained.
            **kwargs: Passed to torchvision RetinaNet.
        """
        backbone = Dinov3Model(repo_id=backbone_weights)
        if freeze_backbone:
            backbone.model.requires_grad_(False)

        # Three anchor scales for each of the four pyramid levels, as in torchvision
        # Note: It may be sensible to add an additional smaller scale at 16px for smaller
        # objects, but this has not been experimentally validated.
        anchor_sizes = tuple(
            (x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3)))
            for x in [32, 64, 128, 256]
        )
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

        # DINOv3 models don't necessarily use ImageNet statistics. Values
        # passed in, for example from a saved hub config, take priority.
        detector_args = {
            "anchor_generator": AnchorGenerator(anchor_sizes, aspect_ratios),
            "image_mean": backbone.image_mean,
            "image_std": backbone.image_std,
        }

        # Skip RetinaNetHub.__init__, which builds a ResNet50 backbone
        RetinaNet.__init__(
            self,
            backbone=backbone,
            num_classes=num_classes,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            **{**detector_args, **kwargs},
        )

        self.num_classes = num_classes
        self.label_dict = label_dict
        self.backbone_weights = backbone_weights
        self.kwargs = kwargs

        self.update_config()

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *, num_classes=None, label_dict=None, **kwargs
    ):
        """Load a trained model, or create a new one on a DINOv3 encoder.

        A DINOv3 encoder repository (model type "dinov3_vit") has no
        detection head. As with torchvision, we add this head on top of the
        encoder and rebuild if the number of classes/labels differ from the 
        requested checkpoint. A saved RetinaNetDinov3Hub is loaded with its
        weights.
        """
        config = _read_hub_config(pretrained_model_name_or_path, kwargs.get("revision"))

        # Arguments of PyTorchModelHubMixin.from_pretrained that control the download
        # and are dropped when creating a new model on a DINOv3 encoder.
        _HUB_ARGS = (
            "revision",
            "cache_dir",
            "force_download",
            "proxies",
            "resume_download",
            "local_files_only",
            "token",
        )

        # Users must provide sufficient information to construct a detection
        # head.
        if config.get("model_type") == "dinov3_vit":
            if num_classes is None or label_dict is None:
                raise ValueError(
                    f"{pretrained_model_name_or_path} is a DINOv3 encoder. Set "
                    "num_classes and label_dict to create a new model on it."
                )

            model_kwargs = {k: v for k, v in kwargs.items() if k not in _HUB_ARGS}
            return cls(
                backbone_weights=str(pretrained_model_name_or_path),
                num_classes=num_classes,
                label_dict=label_dict,
                **model_kwargs,
            )

        # Fail if we were given an incompatible model type.
        if "backbone_weights" not in config:
            raise ValueError(
                f"{pretrained_model_name_or_path} is not a DINOv3 encoder or a "
                "retinanet_dinov3 checkpoint. Check config.architecture."
            )

        return super().from_pretrained(
            pretrained_model_name_or_path,
            num_classes=num_classes,
            label_dict=label_dict,
            **kwargs,
        )

    def update_config(self):
        super().update_config()
        self._hub_mixin_config["backbone_weights"] = self.backbone_weights


class Model(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)

    def create_model(
        self,
        pretrained: str | Path | None = None,
        *,
        revision: str | None = None,
        map_location: str | torch.device | None = None,
        **hf_args,
    ) -> RetinaNetDinov3Hub:
        """Create a RetinaNet with a DINOv3 backbone
        Args:
            pretrained (str | Path | None): A DINOv3 encoder repository, or a trained retinanet_dinov3 model, see RetinaNetDinov3Hub.from_pretrained. None creates a new model on the default encoder
            revision (str | None): Repository revision
            map_location (str | torch.device | None): Device to load weights onto
            **hf_args: Any other arguments to from_pretrained
        Returns:
            model: a pytorch nn module
        """
        model_args = {
            "num_classes": self.config.num_classes,
            "label_dict": dict(self.config.label_dict)
            if self.config.label_dict
            else None,
            "nms_thresh": self.config.nms_thresh,
            "score_thresh": self.config.score_thresh,
            "freeze_backbone": self.config.train.freeze_backbone,
            "detections_per_img": self.config.detections_per_img,
            "topk_candidates": self.config.topk_candidates,
        }

        if pretrained is None:
            model = RetinaNetDinov3Hub(**model_args)
        else:
            model = RetinaNetDinov3Hub.from_pretrained(
                pretrained, revision=revision, **model_args, **hf_args
            )

        return model.to(map_location)
