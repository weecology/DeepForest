"""TreeFormer with PvT-V2 backbone (HuggingFace transformers).

Forward signatures:
  training:  model(inputs, points)           -> loss_dict
  eval:      model(inputs)                   -> (density_map, normed_density)

density_map:    (B, 1, H', W') raw density predictions
normed_density: density_map normalised to sum to 1 per image
loss_dict keys: 'loss' (total), 'count_loss', 'ot_loss', 'tv_loss', 'count_cls_loss'
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from scipy.ndimage import gaussian_filter
from transformers import AutoConfig, AutoImageProcessor, PvtV2Config, PvtV2Model

from deepforest.losses.ot_loss import OT_Loss
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
        num_of_iter_in_ot: int = 100,
        norm_coord: bool = False,
        sinkhorn_reg: float = 1.0,
        density_sigma: float = 5.0,
        mae_weight: float = 1.0,
        ot_weight: float = 0.1,
        tv_weight: float = 0.01,
        count_cls_weight: float = 1.0,
        losses: list | None = None,
    ):
        """
        Args:
            mae_weight: Scalar applied to the MAE count loss (α₁ in the paper,
                default 1.0 per the original TreeFormer, calibrated for KCL-London
                with 256×256 training crops). The default is dataset- and
                crop-size-specific: count loss magnitude scales linearly with the
                number of trees per crop. For portability across datasets or crop
                sizes, set ``mae_weight = 1 / (tree_density * crop_size**2)`` where
                ``tree_density`` is trees per pixel in the training set. This keeps
                the expected count loss ≈ 1 at initialisation regardless of scene
                density or crop size, and simplifies relative tuning of the other
                loss weights.
            count_cls_weight: Scalar applied to the CLS-token count consistency
                loss. Measures the same quantity as ``mae_weight`` (tree count MAE)
                so should be set to the same value:
                ``count_cls_weight = 1 / (tree_density * crop_size**2)``.
            density_sigma: Gaussian sigma (pixels) applied in *output-map* space
                (stride-4 downsampled). The default of 5.0 was chosen for 256×256
                crops (64×64 output maps). Scale proportionally with crop size to
                maintain the same blob-to-map ratio:
                ``density_sigma = 5.0 * (crop_size / 256)``.
        """
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
        self.regression = Regression()

        self.ot_iter = num_of_iter_in_ot

        # This is the output stride of the model and is
        # fixed for PvtV2. We store it for convenience
        # since it's used by various loss functions as a
        # scaling factor.
        self.downsample_ratio = 4

        self.norm_cood = norm_coord
        self.sinkhorn_reg = sinkhorn_reg

        self.density_sigma = density_sigma

        self.mae_weight = mae_weight
        self.ot_weight = ot_weight
        self.tv_weight = tv_weight
        self.count_cls_weight = count_cls_weight

        if losses is None:
            losses = ["count", "ot", "tv", "count_cls"]
        self.active_losses = set(losses)

        # Losses that don't require a device are set up eagerly.
        self.tvloss = nn.L1Loss(reduction="none")
        self.mae = nn.L1Loss()

        # OT_Loss creates device-bound tensors; initialised lazily on first
        # forward call once the model's device is known.
        self._ot_loss: OT_Loss | None = None

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
        p_index = torch.from_numpy(p_h * im_width + p_w).to(torch.int64)
        discrete_map = (
            torch.zeros(im_width * im_height)
            .scatter_add_(0, p_index, torch.ones(im_width * im_height))
            .view(im_height, im_width)
            .numpy()
        )
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
        points: list,
        image_h: int,
        image_w: int,
        gt_discrete: torch.Tensor | None = None,
    ) -> dict:
        """Compute supervised losses.

        Args:
            density_maps:   [y0, y1, y2] multi-scale outputs from Regression head
            normed_density: density_maps[0] normalised to sum to 1, (B, 1, H', W')
            cls_outputs:    [yc0, yc1, yc2] count scalars from CLS-token pathway
            points:         list of B tensors, each (N_i, 2) keypoints in image space
            image_h:        original image height
            image_w:        original image width
            gt_discrete:    optional pre-computed (B, 1, H', W') discrete map;
                            computed from points if None

        Returns:
            dict with 'loss' (total scalar) plus individual named terms
        """
        density_map = density_maps[0]  # primary output, (B, 1, H', W')
        B, _, H, W = density_map.shape

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
            self.mae(density_map.view(B, -1).sum(1), point_counts) * self.mae_weight
            if "count" in active
            else zero
        )

        # ---- Optimal transport loss ----------------------------------------
        if "ot" in active:
            ot_raw, _, _ = self._get_ot_loss()(normed_density, density_map, scaled_points)
            ot_loss = ot_raw * self.ot_weight
        else:
            ot_loss = zero

        # ---- TV loss (predicted vs GT discrete density) --------------------
        if "tv" in active:
            gt_discrete_normed = gt_discrete / (point_counts.view(B, 1, 1, 1) + 1e-6)
            tv_loss = (
                self.tvloss(normed_density, gt_discrete_normed)
                .sum(dim=[1, 2, 3])
                .mul(point_counts)
                .mean()
                * self.tv_weight
            )
        else:
            tv_loss = zero

        # ---- CLS-token count consistency (labeled) -------------------------
        # cls_outputs entries may collapse to a 0-dim tensor when B=1 due to
        # .squeeze() in Regression; reshape to (B,) for safe stacking.
        if "count_cls" in active:
            cls_preds = torch.stack([c.reshape(B) for c in cls_outputs])  # (3, B)
            gt_counts = point_counts.unsqueeze(0).expand(3, -1)  # (3, B)
            count_cls_loss = self.mae(cls_preds, gt_counts) * self.count_cls_weight
        else:
            count_cls_loss = zero

        total = count_loss + ot_loss + tv_loss + count_cls_loss
        return {
            "loss": total,
            "count_loss": count_loss,
            "ot_loss": ot_loss,
            "tv_loss": tv_loss,
            "count_cls_loss": count_cls_loss,
        }

    def forward(
        self,
        inputs: torch.Tensor,
        points: list | None = None,
        gt_discrete: torch.Tensor | None = None,
    ):
        """Forward pass.

        Train mode: ``points`` must be provided; returns a loss dict.
        Eval mode:  returns ``(density_map, normed_density)``.

        Args:
            inputs:      (B, C, H, W) image batch
            points:      list of B tensors, each (N_i, 2) keypoints in image
                         space - required in training mode
            gt_discrete: optional pre-computed (B, 1, H', W') discrete GT map
                         for the TV loss; computed from points if None
        """
        inputs = self.processor.preprocess(
            images=inputs,
            return_tensors="pt",
            do_rescale=False,
            do_resize=False,
            device=inputs.device,
        )["pixel_values"]

        # Pad to next multiple of 32 so all PvT stride-2 stages divide evenly.
        orig_h, orig_w = inputs.shape[2:]
        pad_h = (32 - orig_h % 32) % 32
        pad_w = (32 - orig_w % 32) % 32
        if pad_h or pad_w:
            inputs = F.pad(inputs, (0, pad_w, 0, pad_h))
        padded_h, padded_w = inputs.shape[2:]

        label_feats, l_cls = self.forward_features(inputs)
        out_L, out_cls_l = self.regression(label_feats, l_cls)
        density_map = out_L[0]  # (B, 1, H', W') at padded resolution

        B = density_map.size(0)
        label_sum = density_map.view(B, -1).sum(1).view(B, 1, 1, 1)
        label_normed = density_map / (label_sum + 1e-6)

        if self.training:
            if points is None:
                raise ValueError("points must be provided in training mode")
            # We compute loss on padded images to avoid complications around
            # rescaling when passing to the loss.
            return self.compute_loss(
                out_L,
                label_normed,
                out_cls_l,
                points,
                image_h=padded_h,
                image_w=padded_w,
                gt_discrete=gt_discrete,
            )

        # Crop output back to match original (unpadded) input resolution.
        out_h = orig_h // self.downsample_ratio
        out_w = orig_w // self.downsample_ratio
        density_map = density_map[:, :, :out_h, :out_w].contiguous()

        label_sum = density_map.view(B, -1).sum(1).view(B, 1, 1, 1)
        label_normed = density_map / (label_sum + 1e-6)

        return density_map, label_normed
