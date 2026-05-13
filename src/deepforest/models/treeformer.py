"""Point density prediction model based on TreeFormer.

PvT-V2 backbone + multi-scale Regression head producing a density map at
1/4 input resolution.

See: 10.1109/TGRS.2023.3295802
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoConfig, AutoImageProcessor, PvtV2Model

from deepforest.model import BaseModel
from deepforest.models.treeformer_decoder import Regression
from deepforest.utilities import density_to_points


class TreeFormerModel(nn.Module, PyTorchModelHubMixin):
    """PvT-V2 backbone + Regression head for density estimation."""

    task = "point"

    # Native output channel dims for each PvtV2 variant.
    HIDDEN_SIZES = {
        "pvt_v2_b0": [32, 64, 160, 256],
        "pvt_v2_b1": [64, 128, 320, 512],
        "pvt_v2_b2": [64, 128, 320, 512],
        "pvt_v2_b3": [64, 128, 320, 512],
        "pvt_v2_b4": [64, 128, 320, 512],
        "pvt_v2_b5": [64, 128, 320, 512],
    }

    # Fixed dims Regression expects
    REG_DIMS = [128, 256, 512, 1024]

    def __init__(
        self,
        backbone: str = "pvt_v2_b3",
        num_classes: int = 1,
        label_dict: dict | None = None,
        num_of_iter_in_ot: int = 100,
        sinkhorn_reg: float = 1.0,
        density_sigma: float = 5.0,
        mae_weight: float = 1.0,
        ot_weight: float = 0.1,
        density_l1_weight: float = 0.01,
        count_cls_weight: float = 1.0,
        losses: list | None = None,
        norm_cood: bool = False,
        enforce_count: bool = True,
        score_thresh: float = 0.5,
        score_integration_radius: int = 5,
        **kwargs,
    ):
        """Initialize TreeFormerModel."""
        super().__init__()
        if "/" not in backbone:
            backbone = f"OpenGVLab/{backbone}"
        self.backbone_name = backbone

        # Processor handles ImageNet normalization
        self.processor = AutoImageProcessor.from_pretrained(
            backbone,
            use_fast=True,
            do_normalize=True,
            do_rescale=False,
            do_resize=False,
        )

        # Instaniate architecture but don't pull weights
        self.backbone = PvtV2Model(AutoConfig.from_pretrained(backbone))

        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Suppress some noisy warnings that show in DDP.
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
        for module in self.backbone.modules():
            if (
                isinstance(module, nn.Conv2d)
                and module.groups > 1
                and module.groups == module.in_channels
            ):
                module.weight.register_hook(lambda grad: grad.contiguous())

        variant = backbone.split("/")[-1]
        src = self.HIDDEN_SIZES.get(variant, None)
        if src is None:
            raise ValueError(
                f"Backbone variant {variant} isn't supported. Please use one of {list(self.HIDDEN_SIZES.keys())}"
            )

        self.proj = nn.ModuleList(
            [nn.Conv2d(s, d, 1) for s, d in zip(src, self.REG_DIMS, strict=True)]
        )
        self.num_classes = num_classes
        self.label_dict = label_dict
        self.regression = Regression(num_classes=num_classes)

        # Fixed output stride for PvtV2
        self.downsample_ratio = 4

        self.enforce_count = enforce_count
        self.norm_cood = norm_cood

        # Losses
        self.density_l1 = nn.L1Loss(reduction="none")
        self.cls_l1 = nn.L1Loss()

        # Training params
        self.ot_iter = num_of_iter_in_ot
        self.sinkhorn_reg = sinkhorn_reg
        self.density_sigma = density_sigma
        self.mae_weight = mae_weight
        self.ot_weight = ot_weight
        self.density_l1_weight = density_l1_weight
        self.count_cls_weight = count_cls_weight
        self.score_thresh = score_thresh
        self.score_integration_radius = score_integration_radius
        self.losses = (
            list(losses)
            if losses is not None
            else ["count", "ot", "density_l1", "count_cls"]
        )

        self.kwargs = kwargs
        self.update_config()

    def update_config(self):
        # Stored as config on HF
        self._hub_mixin_config = {
            "backbone": self.backbone_name,
            "num_classes": self.num_classes,
            "label_dict": self.label_dict,
            "num_of_iter_in_ot": self.ot_iter,
            "sinkhorn_reg": self.sinkhorn_reg,
            "density_sigma": self.density_sigma,
            "mae_weight": self.mae_weight,
            "ot_weight": self.ot_weight,
            "density_l1_weight": self.density_l1_weight,
            "count_cls_weight": self.count_cls_weight,
            "losses": self.losses,
            "norm_cood": self.norm_cood,
            "enforce_count": self.enforce_count,
            "score_thresh": self.score_thresh,
            "score_integration_radius": self.score_integration_radius,
            **self.kwargs,
        }

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _normalize_density(
        self, score_map: torch.Tensor, cls_count: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (density_map, normed_density) from raw score map and count
        scalar.

        If the model is in enforce_count mode, the raw density map is
        scaled to match the count prediction; otherwise it's returned
        as-is.
        """
        B = score_map.size(0)
        score_sum = score_map.view(B, -1).sum(1).view(B, 1, 1, 1)
        normed = score_map / (score_sum + 1e-4)
        if self.enforce_count:
            count = cls_count.view(B, 1, 1, 1).abs().clamp(min=1e-4)
            return normed * count, normed
        return score_map, normed

    def _output_shapes(
        self, image_shapes: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Return valid output-map extent (H//4, W//4) for each input image."""
        return [
            (
                max(image_h // self.downsample_ratio, 1),
                max(image_w // self.downsample_ratio, 1),
            )
            for image_h, image_w in image_shapes
        ]

    def _cls_outputs_to_count(
        self,
        cls_output: torch.Tensor,
        image_shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Convert CLS-head count-density prediction to absolute count by
        multiplying by the area of each image."""
        count_density = cls_output.reshape(len(image_shapes))
        count_areas = torch.tensor(
            [image_h * image_w for image_h, image_w in image_shapes],
            dtype=torch.float32,
            device=self.device,
        )
        return count_density * count_areas

    def _forward_features(self, x: torch.Tensor):
        """Run backbone and project stage outputs to REG_DIMS.

        Returns:
            feats: list of 4 spatial tensors (B, REG_DIMS[i], H/4^i, W/4^i)
            cls:   list of 3 vectors (B, REG_DIMS[1..3]) — GAP of stages 1-3
        """
        out = self.backbone(x, output_hidden_states=True)
        feats = [p(h) for p, h in zip(self.proj, out.hidden_states, strict=False)]
        cls = [feats[i].mean(dim=[2, 3]) for i in range(1, 4)]
        return feats, cls

    def postprocess_density(self, density_map, images) -> list[dict]:
        """Convert TreeFormer density map outputs to per-image point dicts.

        Scales point coordinates from density-map space to image-pixel space.
        Handles both batched tensor inputs and list inputs (from the default
        prediction dataloader which returns a list via its collate_fn).

        Args:
            density_outputs: tuple ``(density_map, normed_density)`` from
                ``TreeFormerModel.forward``.  When ``images`` is a list,
                ``density_map`` is also a list of per-image tensors.
            images: batch tensor ``(B, C, H_img, W_img)`` or list of
                ``(C, H_img, W_img)`` tensors.

        Returns:
            List of dicts with ``"points"`` (N, 2), ``"scores"`` (N,),
            ``"labels"`` (N,) per image, with (x, y) in image-pixel coordinates.
        """
        preds = density_to_points(
            density_map,
            score_thresh=self.score_thresh,
            score_integration_radius=self.score_integration_radius,
        )

        if not isinstance(images, (list, tuple)):
            images = [images[i] for i in range(images.shape[0])]

        # Per-image density maps; scale each individually.
        for pred, dm, img in zip(preds, density_map, images, strict=False):
            H_img, W_img = img.shape[-2], img.shape[-1]
            H_dm, W_dm = dm.shape[-2], dm.shape[-1]
            if pred["points"].shape[0] > 0:
                pts = pred["points"].clone()
                pts[:, 0] = pts[:, 0] * (W_img / W_dm)
                pts[:, 1] = pts[:, 1] * (H_img / H_dm)
                pred["points"] = pts

        return preds

    def compute_loss(
        self,
        density_maps: list[torch.Tensor],
        normed_density: list[torch.Tensor],
        cls_outputs: list,
        targets: list,
        image_shapes: list[tuple[int, int]],
    ) -> dict:
        """Compute training loss: L1 between density-map count and GT count.

        Args:
            density_maps: list of multi-scale density maps; index 0 is primary.
            normed_density: spatially normalised version of density_maps[0].
            cls_outputs: CLS-head count predictions (unused here).
            targets: list of target dicts with ``"points"`` tensors.
            image_shapes: original ``(H, W)`` for each image.

        Returns:
            dict with ``"loss"`` scalar tensor (supports backprop).
        """
        true_counts = torch.tensor(
            [float(len(t["points"])) for t in targets],
            dtype=torch.float32,
            device=self.device,
        )

        # TODO: This is a placeholder to verify integration, full loss
        # computation as used with the model releases will be implemented
        # in a future PR.
        pred_counts = self._cls_outputs_to_count(cls_outputs[0], image_shapes)
        count_loss = self.cls_l1(pred_counts, true_counts) * self.mae_weight

        return {"loss": count_loss}

    def forward(
        self, inputs: torch.Tensor | list[torch.Tensor], targets: list | None = None
    ):
        """Forward pass.

        In training mode returns a loss dict; in eval mode returns a list of
        per-image prediction dicts (see ``postprocess_density``).

        Args:
            inputs: ``(B, C, H, W)`` tensor or list of ``(C, H, W)`` tensors.
            targets: list of target dicts (required during training).
        """
        # Batch-pad variable-size images; record original sizes.
        if isinstance(inputs, list):
            shapes = [(img.shape[-2], img.shape[-1]) for img in inputs]
            H = max(h for h, _ in shapes)
            W = max(w for _, w in shapes)
            batch = inputs[0].new_zeros(len(inputs), inputs[0].shape[0], H, W)
            for i, img in enumerate(inputs):
                batch[i, :, : shapes[i][0], : shapes[i][1]] = img
        else:
            shapes = [(inputs.shape[2], inputs.shape[3])] * inputs.shape[0]
            batch = inputs

        # Pad to next multiple of 32 for PvT stride compatibility.
        H, W = batch.shape[2:]
        batch = F.pad(batch, (0, (32 - W % 32) % 32, 0, (32 - H % 32) % 32))

        encoded = self.processor.preprocess(
            images=batch,
            return_tensors="pt",
            do_rescale=False,
            do_resize=False,
        )["pixel_values"].to(self.device)

        label_feats, l_cls = self._forward_features(encoded)
        out_L, out_cls_l = self.regression(label_feats, l_cls)

        # Crop each output to its valid spatial extent and normalise.
        # Cropping naturally excludes batch padding, which is correct for both
        # training losses and inference peak-finding.
        density_list, normed_list = [], []
        for i, (valid_h, valid_w) in enumerate(self._output_shapes(shapes)):
            crop = out_L[0][i : i + 1, :, :valid_h, :valid_w].contiguous()
            cls_count = self._cls_outputs_to_count(out_cls_l[0][i : i + 1], [shapes[i]])
            dm, nd = self._normalize_density(crop, cls_count)
            density_list.append(dm)
            normed_list.append(nd)

        if self.training:
            if targets is None:
                raise ValueError("targets must be provided in training mode")
            return self.compute_loss(
                density_list, normed_list, out_cls_l, targets, image_shapes=shapes
            )

        return self.postprocess_density(density_list, batch)


class Model(BaseModel):
    """DeepForest model wrapper for TreeFormer.

    Selected via ``config.architecture = "treeformer"``.
    """

    def create_model(
        self,
        pretrained: str | None = None,
        revision: str | None = None,
        map_location: str | torch.device | None = None,
        **hf_args,
    ) -> TreeFormerModel:
        """Create or load a TreeFormerModel.

        Args:
            pretrained: HuggingFace repo ID to load weights from, or None.
            revision: Model revision/tag on the Hub.
            map_location: Device to move model to after loading.

        Returns:
            Configured TreeFormerModel instance.
        """
        label_dict = dict(self.config.label_dict) if self.config.label_dict else None
        num_classes = (
            len(label_dict) if label_dict is not None else self.config.num_classes or 1
        )
        backbone = self.config.point.backbone

        # Load fully trained backbone + head from hub
        if pretrained:
            if label_dict is not None:
                hf_args["label_dict"] = label_dict

            model = TreeFormerModel.from_pretrained(
                pretrained,
                revision=revision,
                num_classes=num_classes,
                score_thresh=self.config.score_thresh,
                score_integration_radius=self.config.point.score_integration_radius,
                **hf_args,
            )
        # Architecture from config, backbone weights from ImageNet.
        elif backbone:
            model = TreeFormerModel(
                backbone=backbone,
                num_classes=num_classes,
                label_dict=label_dict,
                score_thresh=self.config.score_thresh,
                score_integration_radius=self.config.point.score_integration_radius,
                **hf_args,
            )
            model.backbone = PvtV2Model.from_pretrained(
                model.backbone_name, ignore_mismatched_sizes=True
            )
        # Random init
        else:
            model = TreeFormerModel(
                num_classes=num_classes,
                label_dict=label_dict,
                score_thresh=self.config.score_thresh,
                score_integration_radius=self.config.point.score_integration_radius,
                **hf_args,
            )

        if map_location is not None:
            model = model.to(map_location)
        return model
