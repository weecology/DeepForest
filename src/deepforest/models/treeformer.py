"""TreeFormer with PvT-V2 backbone (HuggingFace transformers).

Forward signatures:
  training:  model(inputs, points)           -> loss_dict
  eval:      model(inputs)                   -> (density_map, normed_density)

density_map:    (B, 1, H', W') raw density predictions
normed_density: density_map normalised to sum to 1 per image
loss_dict keys: 'loss' (total), 'count_loss', 'ot_loss', 'density_l1_loss', 'count_cls_loss'
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from scipy.ndimage import gaussian_filter
from transformers import AutoConfig, AutoImageProcessor, PvtV2Config, PvtV2Model

from deepforest.losses.ot_loss import OT_Loss
from deepforest.model import BaseModel
from deepforest.models.treeformer_decoder import Regression


class TreeFormerModel(nn.Module, PyTorchModelHubMixin):
    """PvT-V2 backbone + Regression head for tree density estimation.

    1x1 projection layers adapt the backbone's native channel dims to the fixed
    dims Regression expects, so swapping backbone size (b2/b3/b4) requires only
    changing the ``backbone`` constructor argument.
    """

    task = "keypoint"

    # Native output channel dims for each PvtV2 variant.
    HIDDEN_SIZES = {
        "pvt_v2_b0": [32, 64, 160, 256],
        "pvt_v2_b1": [64, 128, 320, 512],
        "pvt_v2_b2": [64, 128, 320, 512],
        "pvt_v2_b3": [64, 128, 320, 512],
        "pvt_v2_b4": [64, 128, 320, 512],
        "pvt_v2_b5": [64, 128, 320, 512],
    }

    # Fixed dims Regression expects (matches pvt_treeformer embed_dims).
    REG_DIMS = [128, 256, 512, 1024]

    def __init__(
        self,
        backbone: str = "OpenGVLab/pvt_v2_b3",
        pretrained: bool = True,
        num_classes: int = 1,
        label_dict: dict | None = None,
        num_of_iter_in_ot: int = 100,
        norm_coord: bool = False,
        sinkhorn_reg: float = 1.0,
        density_sigma: float = 5.0,
        density_sigma_start: float | None = None,
        density_sigma_end: float | None = None,
        density_sigma_schedule_epochs: int | None = None,
        mae_weight: float = 1.0,
        ot_weight: float = 0.1,
        density_l1_weight: float = 0.01,
        count_cls_weight: float = 1.0,
        losses: list | None = None,
        norm_cood: bool = False,
        enforce_count: bool = True,
        **kwargs,
    ):
        """Initialize TreeFormerModel."""
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(
            backbone,
            use_fast=True,
            do_normalize=True,
            do_rescale=False,
            do_resize=False,
        )

        if pretrained:
            self.backbone = PvtV2Model.from_pretrained(backbone)
        else:
            config = AutoConfig.from_pretrained(backbone)
            if not isinstance(config, PvtV2Config):
                raise TypeError(
                    f"Expected PvtV2Config for backbone {backbone}, got {type(config).__name__}"
                )
            self.backbone = PvtV2Model(config)

        variant = backbone.split("/")[-1]
        src = self.HIDDEN_SIZES[variant]
        self.proj = nn.ModuleList(
            [nn.Conv2d(s, d, 1) for s, d in zip(src, self.REG_DIMS, strict=True)]
        )
        self.num_classes = num_classes
        self.label_dict = label_dict
        self.regression = Regression(num_classes=num_classes)

        self.ot_iter = num_of_iter_in_ot

        # This is the output stride of the model and is
        # fixed for PvtV2. We store it for convenience
        # since it's used by various loss functions as a
        # scaling factor.
        self.downsample_ratio = 4

        self.sinkhorn_reg = sinkhorn_reg

        # If start/end provided, use start as initial sigma; else use the single density_sigma.
        self.density_sigma_start = (
            density_sigma_start if density_sigma_start is not None else density_sigma
        )
        self.density_sigma_end = (
            density_sigma_end if density_sigma_end is not None else density_sigma
        )
        self.density_sigma = self.density_sigma_start
        self.density_sigma_schedule_epochs = density_sigma_schedule_epochs

        self.mae_weight = mae_weight
        self.ot_weight = ot_weight
        self.density_l1_weight = density_l1_weight
        self.count_cls_weight = count_cls_weight
        self.enforce_count = enforce_count
        self.norm_cood = norm_cood

        if losses is None:
            losses = ["count", "ot", "density_l1", "count_cls"]
        self.active_losses = set(losses)

        # Losses that don't require a device are set up eagerly.
        self.density_l1 = nn.L1Loss(reduction="none")
        self.cls_l1 = nn.L1Loss()

        # OT_Loss creates device-bound tensors; initialised lazily on first
        # forward call once the model's device is known.
        self._ot_loss: OT_Loss | None = None

        self.kwargs = kwargs
        self.update_config()

    def update_config(self):
        # Stored as config on HF
        self._hub_mixin_config = {
            "num_classes": self.num_classes,
            "label_dict": self.label_dict,
            **self.kwargs,
        }

    def set_epoch(self, epoch: int, total_epochs: int) -> None:
        """Update ``density_sigma`` using a cosine anneal from start to end.

        Args:
            epoch: Current zero-based epoch index.
            total_epochs: Total number of training epochs. Overridden by
                ``self.density_sigma_schedule_epochs`` if set.
        """
        schedule_epochs = self.density_sigma_schedule_epochs or total_epochs
        t = np.pi * epoch / max(schedule_epochs - 1, 1)
        self.density_sigma = float(
            self.density_sigma_end
            + 0.5 * (self.density_sigma_start - self.density_sigma_end) * (1 + np.cos(t))
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _get_ot_loss(self) -> OT_Loss:
        """Return OT_Loss, creating it on first call with the current
        device."""
        if self._ot_loss is None:
            self._ot_loss = OT_Loss(
                self.norm_cood,
                self.device,
                self.ot_iter,
                self.sinkhorn_reg,
            )
        return self._ot_loss

    def _normalize_density(
        self, score_map: torch.Tensor, cls_count: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (density_map, normed_density) from raw score map and count
        scalar.

        Behaviour is controlled by ``self.enforce_count`` (set via the
        ``enforce_count`` constructor argument):

        * ``False`` (default, v1 behaviour): density_map = score_map (unconstrained
          scale). count_loss trains the spatial head directly.
        * ``True`` (physically-consistent, v5 behaviour): density_map =
          score_normed * cls_count, so density_map.sum() == cls_count by
          construction. OT/TV losses train the spatial distribution; count_loss
          trains the CLS branch.

        Args:
            score_map:  (B, 1, H, W) raw non-negative output of the density head.
            cls_count:  (B,) count prediction from the CLS branch.

        Returns:
            density_map:    (B, 1, H, W) density used for count_loss and output.
            normed_density: (B, 1, H, W) spatially normalized map (sums to 1),
                            used for OT and TV losses.
        """
        B = score_map.size(0)
        score_sum = score_map.view(B, -1).sum(1).view(B, 1, 1, 1)
        normed = score_map / (score_sum + 1e-6)
        if self.enforce_count:
            count = cls_count.view(B, 1, 1, 1).clamp(min=0)
            return normed * count, normed
        return score_map, normed

    def _scale_points_to_output(
        self,
        points: list,
        image_h: int,
        image_w: int,
        out_h: int,
        out_w: int,
    ) -> list:
        """Scale image-space point coordinates into output-map coordinates."""
        scale = torch.tensor(
            [out_w / image_w, out_h / image_h],
            dtype=torch.float32,
            device=self.device,
        )
        scaled_points = []
        for p in points:
            if len(p) == 0:
                scaled_points.append(p.clone())
            else:
                scaled_points.append(p.to(dtype=torch.float32) * scale)
        return scaled_points

    def _coerce_points(self, targets: list) -> list[torch.Tensor]:
        """Normalize training targets to a list of point tensors."""
        points = []
        for target in targets:
            if isinstance(target, dict):
                point_tensor = target.get("points")
                if point_tensor is None:
                    raise ValueError("Each target dict must include a 'points' entry")
            else:
                point_tensor = target

            if not isinstance(point_tensor, torch.Tensor):
                point_tensor = torch.as_tensor(point_tensor, dtype=torch.float32)

            points.append(point_tensor.to(device=self.device, dtype=torch.float32))

        return points

    def forward_features(self, x: torch.Tensor):
        """Run backbone and project stage outputs to REG_DIMS.

        Returns:
            feats: list of 4 spatial tensors (B, REG_DIMS[i], H/4^i, W/4^i)
            cls:   list of 3 vectors (B, REG_DIMS[1..3]) - GAP of stages 1-3,
                   substituting the cls tokens from PvT-v1.
        """
        out = self.backbone(x, output_hidden_states=True)
        feats = [p(h) for p, h in zip(self.proj, out.hidden_states, strict=False)]
        cls = [feats[i].mean(dim=[2, 3]) for i in range(1, 4)]
        return feats, cls

    def gen_discrete_map(
        self, im_height: int, im_width: int, points: np.ndarray
    ) -> np.ndarray:
        """Generate a discrete density map for a single image.

        Args:
            im_height: output map height
            im_width:  output map width
            points:    (N, 2) float array of (x, y) keypoint coordinates in
                       output-map space

        Returns:
            (im_height, im_width) float32 array with 1.0 at each keypoint location
        """
        discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
        num_gt = len(points)
        if num_gt == 0:
            return discrete_map

        points_np = np.array(points).round().astype(int)
        p_h = np.clip(points_np[:, 1], 0, im_height - 1)
        p_w = np.clip(points_np[:, 0], 0, im_width - 1)
        np.add.at(discrete_map, (p_h, p_w), 1)
        assert discrete_map.sum() == num_gt
        return discrete_map

    def _make_gt_density(
        self,
        points: list,
        out_h: int,
        out_w: int,
        image_h: int,
        image_w: int,
    ) -> torch.Tensor:
        """Build a batched Gaussian density map (B, 1, out_h, out_w) from point
        lists.

        Points are expected in output-map coordinates. A Gaussian is
        applied with sigma rescaled from full-image pixels into output-
        map pixels, then count-normalized so each map sums to the number
        of ground-truth points.
        """
        # sigma is applied directly in output-map space (matching the reference
        # which uses sigma=5.0 on the 64x64 output map, not image-space pixels).
        sigma = self.density_sigma
        maps = []
        for p in points:
            p_np = p.cpu().numpy()
            discrete = self.gen_discrete_map(out_h, out_w, p_np)
            if discrete.sum() > 0:
                smoothed = gaussian_filter(discrete, sigma=sigma)
                smoothed = smoothed * (discrete.sum() / smoothed.sum())  # count-normalize
            else:
                smoothed = discrete
            maps.append(smoothed)
        return torch.from_numpy(np.stack(maps)).unsqueeze(1).float().to(self.device)

    def compute_loss(
        self,
        density_maps: list,
        normed_density: torch.Tensor,
        cls_outputs: list,
        targets: list,
        image_h: int,
        image_w: int,
        gt_discrete: torch.Tensor | None = None,
    ) -> dict:
        """Compute supervised losses.

        Args:
            density_maps:   [y0, y1, y2] multi-scale outputs from Regression head
            normed_density: density_maps[0] normalised to sum to 1, (B, 1, H', W')
            cls_outputs:    [yc0, yc1, yc2] count scalars from CLS-token pathway
            targets:        list of B target dicts or point tensors in image space
            image_h:        original image height
            image_w:        original image width
            gt_discrete:    optional pre-computed (B, 1, H', W') discrete map;
                            computed from points if None

        Returns:
            dict with 'loss' (total scalar) plus individual named terms
        """
        density_map = density_maps[0]  # primary output, (B, 1, H', W')
        B, _, H, W = density_map.shape
        points = self._coerce_points(targets)

        point_counts = torch.tensor(
            [len(p) for p in points], dtype=torch.float32, device=self.device
        )
        scaled_points = self._scale_points_to_output(points, image_h, image_w, H, W)

        if gt_discrete is None:
            gt_discrete = self._make_gt_density(scaled_points, H, W, image_h, image_w)

        active = self.active_losses
        zero = density_map.new_zeros(1)

        # ---- MAE count loss -----------------------------------------------
        count_loss = (
            self.cls_l1(density_map.view(B, -1).sum(1), point_counts) * self.mae_weight
            if "count" in active
            else zero
        )

        # ---- Optimal transport loss ----------------------------------------
        if "ot" in active:
            ot_raw, _, _ = self._get_ot_loss()(normed_density, density_map, scaled_points)
            ot_loss = (ot_raw / B) * self.ot_weight
        else:
            ot_loss = zero

        # ---- Density L1 loss (pixel-wise L1 between normalized density maps) ----
        if "density_l1" in active:
            gt_discrete_normed = gt_discrete / (point_counts.view(B, 1, 1, 1) + 1e-6)
            density_l1_loss = (
                self.density_l1(normed_density, gt_discrete_normed)
                .sum(dim=[1, 2, 3])
                .mul(point_counts)
                .mean()
                * self.density_l1_weight
            )
        else:
            density_l1_loss = zero

        # ---- Stage GAP count regression ----
        # cls_outputs entries may collapse to a 0-dim tensor when B=1 due to
        # .squeeze() in Regression; reshape to (B,) for safe stacking.
        if "count_cls" in active:
            cls_preds = torch.stack([c.reshape(B) for c in cls_outputs])  # (3, B)
            gt_counts = point_counts.unsqueeze(0).expand(3, -1)  # (3, B)
            count_cls_loss = self.cls_l1(cls_preds, gt_counts) * self.count_cls_weight
        else:
            count_cls_loss = zero

        total = count_loss + ot_loss + density_l1_loss + count_cls_loss
        return {
            "loss": total,
            "count_loss": count_loss,
            "ot_loss": ot_loss,
            "density_l1_loss": density_l1_loss,
            "count_cls_loss": count_cls_loss,
        }

    def forward(
        self,
        inputs: torch.Tensor | list[torch.Tensor],
        targets: list | None = None,
        gt_discrete: torch.Tensor | None = None,
    ):
        """Forward pass.

        Train mode: targets must be provided; returns a loss dict.
        Eval mode:  returns (density_map, normed_density).

        inputs: (B, C, H, W) tensor or list of (C, H, W) tensors (variable sizes).
        """
        # Batch-pad variable-size images; record original sizes for output crop.
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
        padded_h, padded_w = batch.shape[2:]

        encoded = self.processor.preprocess(
            images=batch,
            return_tensors="pt",
            do_rescale=False,
            do_resize=False,
        )["pixel_values"].to(self.device)

        label_feats, l_cls = self.forward_features(encoded)
        out_L, out_cls_l = self.regression(label_feats, l_cls)

        if self.training:
            if targets is None:
                raise ValueError("targets must be provided in training mode")
            density_map, label_normed = self._normalize_density(out_L[0], out_cls_l[0])
            return self.compute_loss(
                [density_map] + out_L[1:],
                label_normed,
                out_cls_l,
                targets,
                image_h=padded_h,
                image_w=padded_w,
                gt_discrete=gt_discrete,
            )

        # Eval: crop each output back to its original spatial extent.
        dr = self.downsample_ratio
        density_list, normed_list = [], []
        for i, (h, w) in enumerate(shapes):
            crop = out_L[0][i : i + 1, :, : h // dr, : w // dr].contiguous()
            dm, nd = self._normalize_density(crop, out_cls_l[0][i : i + 1])
            density_list.append(dm)
            normed_list.append(nd)

        if isinstance(inputs, torch.Tensor):
            return torch.cat(density_list), torch.cat(normed_list)
        return density_list, normed_list


class Model(BaseModel):
    """DeepForest model wrapper for TreeFormer.

    Follows the same pattern as retinanet.Model and DeformableDetr.Model.
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

        Returns:
            Configured TreeFormerModel instance.
        """
        cfg = self.config.keypoint
        label_dict = dict(self.config.label_dict) if self.config.label_dict else None
        num_classes = len(label_dict)

        if pretrained:
            model = TreeFormerModel.from_pretrained(
                pretrained, revision=revision, **hf_args
            )
        else:
            model = TreeFormerModel(
                num_classes=num_classes,
                label_dict=label_dict,
                density_sigma_start=cfg.density_sigma_start,
                density_sigma_end=cfg.density_sigma_end,
                density_sigma_schedule_epochs=cfg.density_sigma_schedule_epochs,
                mae_weight=cfg.mae_weight,
                count_cls_weight=cfg.count_cls_weight,
                losses=list(cfg.losses) if cfg.losses is not None else None,
                norm_cood=cfg.norm_cood,
                enforce_count=True,
                **hf_args,
            )

        return model.to(map_location)
