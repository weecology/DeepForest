"""TreeFormer with PvT-V2 backbone."""

import warnings

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
    """PvT-V2 backbone + Regression head for density estimation."""

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

    # Fixed dims Regression expects
    REG_DIMS = [128, 256, 512, 1024]

    def __init__(
        self,
        backbone: str = "OpenGVLab/pvt_v2_b3",
        pretrained: bool = True,
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
        log_count_loss: bool = False,
        use_uncertainty_head: bool = False,
        uncertainty_delta: float = 0.2,
        uncertainty_mse_weight: float = 1.0,
        count_prediction_mode: str = "absolute",
        **kwargs,
    ):
        """Initialize TreeFormerModel."""
        super().__init__()
        self.backbone_name = backbone
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

        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

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
        self.ot_iter = num_of_iter_in_ot

        # This is the output stride of the model and is
        # fixed for PvtV2. We store it for convenience
        # since it's used by various loss functions as a
        # scaling factor.
        self.downsample_ratio = 4

        self.sinkhorn_reg = sinkhorn_reg
        self.density_sigma = density_sigma
        self.mae_weight = mae_weight
        self.ot_weight = ot_weight
        self.density_l1_weight = density_l1_weight
        self.count_cls_weight = count_cls_weight
        self.enforce_count = enforce_count
        self.norm_cood = norm_cood
        self.log_count_loss = log_count_loss
        if count_prediction_mode not in {"absolute", "density"}:
            raise ValueError(
                "count_prediction_mode must be one of {'absolute', 'density'}"
            )
        self.count_prediction_mode = count_prediction_mode
        self.uncertainty_delta = uncertainty_delta
        self.uncertainty_mse_weight = uncertainty_mse_weight
        self.use_uncertainty_head = use_uncertainty_head

        if losses is None:
            losses = ["count", "ot", "density_l1", "count_cls"]
        self.losses = list(losses)
        if "count_cls" not in self.losses and enforce_count:
            warnings.warn(
                "enforce_count uses the CLS branch to rescale the density map, but "
                "count_cls is not active in losses. This preserves the requested "
                "legacy behavior, but it is unsafe and can degrade spatial quality.",
                UserWarning,
                stacklevel=2,
            )
        self.active_losses = set(self.losses)

        if use_uncertainty_head:
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
            )
            # Init final conv to near-zero so uncertainty
            # is approx 0 at the start,
            # matching Gominski et al's initialisation
            # (softplus(-5) ≈ 0.007 pixels).
            final_uncertainty_layer = self.uncertainty_head[-1]
            assert isinstance(final_uncertainty_layer, nn.Conv2d)
            assert final_uncertainty_layer.bias is not None
            nn.init.zeros_(final_uncertainty_layer.weight)
            nn.init.constant_(final_uncertainty_layer.bias, -5.0)

        # Losses that don't require a device are set up eagerly.
        self.density_l1 = nn.L1Loss(reduction="none")
        self.cls_l1 = nn.L1Loss()

        # OT_Loss is set up once the model device is known
        self._ot_loss: OT_Loss | None = None

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
            "log_count_loss": self.log_count_loss,
            "count_prediction_mode": self.count_prediction_mode,
            "use_uncertainty_head": self.use_uncertainty_head,
            "uncertainty_delta": self.uncertainty_delta,
            "uncertainty_mse_weight": self.uncertainty_mse_weight,
            **self.kwargs,
        }

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

        * ``False`` (default): density_map = score_map (unconstrained
          scale). count_loss trains the spatial head directly.
        * ``True`` (physically-consistent): density_map =
          score_normed * cls_count, so density_map.sum() == cls_count by
          construction. OT/TV losses train the spatial distribution; count_loss
          trains the CLS branch.

        Args:
            score_map:  (B, 1, H, W) raw non-negative output of the density head.
            cls_count:  (B,) absolute-count prediction from the CLS branch.

        Returns:
            density_map:    (B, 1, H, W) density used for count_loss and output.
            normed_density: (B, 1, H, W) spatially normalized map (sums to 1),
                            used for OT and TV losses.
        """
        B = score_map.size(0)
        score_sum = score_map.view(B, -1).sum(1).view(B, 1, 1, 1)
        normed = score_map / (score_sum + 1e-4)
        if self.enforce_count:
            count = cls_count.view(B, 1, 1, 1).clamp(min=1e-4)
            return normed * count, normed
        return score_map, normed

    def _count_areas(self, image_shapes: list[tuple[int, int]]) -> torch.Tensor:
        """Return per-image areas in input-pixel space."""
        return torch.tensor(
            [image_h * image_w for image_h, image_w in image_shapes],
            dtype=torch.float32,
            device=self.device,
        )

    def _output_shapes(
        self, image_shapes: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Return the valid output-map extent for each input image.

        This matches the existing eval-time crop behaviour, which keeps
        the stride-4 region corresponding to the unpadded image and
        ignores any batch-padding beyond it.
        """
        return [
            (
                max(image_h // self.downsample_ratio, 1),
                max(image_w // self.downsample_ratio, 1),
            )
            for image_h, image_w in image_shapes
        ]

    def _build_output_mask(
        self,
        image_shapes: list[tuple[int, int]],
        out_h: int,
        out_w: int,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Mask padded decoder outputs so losses ignore batch padding."""
        mask = torch.zeros(
            len(image_shapes),
            1,
            out_h,
            out_w,
            device=self.device,
            dtype=dtype,
        )
        for index, (valid_h, valid_w) in enumerate(self._output_shapes(image_shapes)):
            mask[index, :, :valid_h, :valid_w] = 1.0
        return mask

    def _cls_outputs_to_count(
        self,
        cls_output: torch.Tensor,
        image_shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Convert CLS-head outputs into absolute counts.

        In ``absolute`` mode, the head predicts counts directly. In ``density``
        mode, the head predicts count density and is converted to absolute count
        using each image's true area.
        """
        absolute_count = cls_output.reshape(len(image_shapes))
        if self.count_prediction_mode == "density":
            absolute_count = absolute_count * self._count_areas(image_shapes)
        return absolute_count

    def _scale_points_to_output(
        self,
        points: list,
        image_shapes: list[tuple[int, int]],
        output_shapes: list[tuple[int, int]],
    ) -> list:
        """Scale image-space point coordinates into output-map coordinates."""
        scaled_points = []
        for p, (image_h, image_w), (out_h, out_w) in zip(
            points, image_shapes, output_shapes, strict=True
        ):
            scale = torch.tensor(
                [out_w / image_w, out_h / image_h],
                dtype=torch.float32,
                device=self.device,
            )
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
        uncertainty: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build a batched Gaussian density map (B, 1, out_h, out_w) from point
        lists.

        Points are expected in output-map coordinates. A Gaussian is
        applied with sigma rescaled from full-image pixels into output-
        map pixels, then count-normalized so each map sums to the number
        of ground-truth points.

        When ``uncertainty`` (B, 1, out_h, out_w) is provided, each object's
        Gaussian is drawn with ``sigma + uncertainty[b, 0, y, x]``, following
        the spatial-uncertainty modulation in Gominsky et al.
        """
        # sigma is applied directly in output-map space (matching the reference
        # which uses sigma=5.0 on the 64x64 output map, not image-space pixels).
        sigma = self.density_sigma
        maps = []
        for b_idx, p in enumerate(points):
            p_np = p.cpu().numpy()
            discrete = self.gen_discrete_map(out_h, out_w, p_np)
            if discrete.sum() > 0:
                if uncertainty is not None:
                    unc_np = uncertainty[b_idx, 0].detach().cpu().numpy()
                    smoothed = np.zeros((out_h, out_w), dtype=np.float32)
                    p_int = p_np.round().astype(int)
                    p_int[:, 0] = np.clip(p_int[:, 0], 0, out_w - 1)
                    p_int[:, 1] = np.clip(p_int[:, 1], 0, out_h - 1)
                    for pt in p_int:
                        local_sigma = sigma + float(unc_np[pt[1], pt[0]])
                        delta = np.zeros((out_h, out_w), dtype=np.float32)
                        delta[pt[1], pt[0]] = 1.0
                        smoothed += gaussian_filter(delta, sigma=max(local_sigma, 0.5))
                    smoothed *= discrete.sum() / (smoothed.sum() + 1e-6)
                else:
                    smoothed = gaussian_filter(discrete, sigma=sigma)
                    smoothed = smoothed * (
                        discrete.sum() / smoothed.sum()
                    )  # count-normalize
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
        image_shapes: list[tuple[int, int]],
        gt_discrete: torch.Tensor | None = None,
        uncertainty: torch.Tensor | None = None,
    ) -> dict:
        """Compute supervised losses.

        Args:
            density_maps:   [y0, y1, y2] multi-scale outputs from Regression head
            normed_density: density_maps[0] normalised to sum to 1, (B, 1, H', W')
            cls_outputs:    [yc0, yc1, yc2] count scalars from CLS-token pathway
            targets:        list of B target dicts or point tensors in image space
            image_shapes:   original (height, width) for each image in the batch
            gt_discrete:    optional pre-computed (B, 1, H', W') discrete map;
                            computed from points if None

        Returns:
            dict with 'loss' (total scalar) plus individual named terms
        """
        density_map = density_maps[0]  # primary output, (B, 1, H', W')
        B, _, H, W = density_map.shape
        points = self._coerce_points(targets)
        output_shapes = self._output_shapes(image_shapes)
        areas = self._count_areas(image_shapes)

        point_counts = torch.tensor(
            [len(p) for p in points], dtype=torch.float32, device=self.device
        )
        scaled_points = self._scale_points_to_output(points, image_shapes, output_shapes)

        if gt_discrete is None:
            gt_discrete = self._make_gt_density(scaled_points, H, W)

        active = self.active_losses
        zero = density_map.new_zeros(1)

        # ---- MAE count loss -----------------------------------------------
        # In density mode with enforce_count, density_map.sum() equals
        # raw_cls * area (absolute count). We must compare in density space
        # (divide by area) to avoid area-amplified gradients on the CLS head.
        if "count" in active:
            pred_sum = density_map.view(B, -1).sum(1)
            if self.count_prediction_mode == "density" and not self.log_count_loss:
                pred_count = pred_sum / areas
                gt_count = point_counts / areas
            else:
                pred_count = pred_sum
                gt_count = point_counts
            if self.log_count_loss:
                count_loss = (
                    self.cls_l1(torch.log1p(pred_count), torch.log1p(gt_count))
                    * self.mae_weight
                )
            else:
                count_loss = self.cls_l1(pred_count, gt_count) * self.mae_weight
        else:
            count_loss = zero

        # ---- Optimal transport loss ----------------------------------------
        if "ot" in active:
            ot_raw, ot_wd_val, _, ot_avg_its = self._get_ot_loss()(
                normed_density, density_map, scaled_points
            )
            ot_loss = ot_raw * self.ot_weight
            # Wasserstein distance diagnostic, not used for backprop.
            ot_wd = torch.tensor(
                ot_wd_val, device=density_map.device, dtype=torch.float32
            )
            sinkhorn_its = torch.tensor(
                ot_avg_its, device=density_map.device, dtype=torch.float32
            )
        else:
            ot_loss = zero
            ot_wd = zero
            sinkhorn_its = zero

        # ---- Density L1 loss (pixel-wise L1 between normalized density maps) ----
        if "density_l1" in active:
            gt_discrete_normed = gt_discrete / (point_counts.view(B, 1, 1, 1) + 1e-4)
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
        # In density mode, keep the loss in density space: compare raw CLS
        # output (count density) to gt_count/area directly. This avoids
        # multiplying by area and then dividing it back out, which would
        # amplify gradients by ~area^2 before Adam can adapt.
        if "count_cls" in active:
            cls_preds = torch.stack(
                [c.reshape(B) for c in cls_outputs]
            )  # raw CLS outputs, (3, B)
            gt_counts = point_counts.unsqueeze(0).expand(3, -1)  # (3, B)
            if self.count_prediction_mode == "density":
                # Raw CLS predicts count density; GT must match.
                gt_counts = gt_counts / areas.unsqueeze(0)
            if self.log_count_loss:
                count_cls_loss = (
                    self.cls_l1(
                        torch.log1p(cls_preds.clamp(min=0)), torch.log1p(gt_counts)
                    )
                    * self.count_cls_weight
                )
            else:
                count_cls_loss = self.cls_l1(cls_preds, gt_counts) * self.count_cls_weight
        else:
            count_cls_loss = zero

        # ---- Uncertainty MSE + regularisation (Gominsky et al., arXiv 2508.21437) ----------
        if uncertainty is not None and "uncertainty_mse" in active:
            gt_uncertainty = self._make_gt_density(
                scaled_points, H, W, uncertainty=uncertainty
            )
            uncertainty_mse_loss = (
                F.mse_loss(density_map, gt_uncertainty) * self.uncertainty_mse_weight
            )
        else:
            uncertainty_mse_loss = zero

        if uncertainty is not None and "uncertainty_reg" in active:
            uncertainty_reg_loss = uncertainty.pow(2).mean() * self.uncertainty_delta
        else:
            uncertainty_reg_loss = zero

        total = (
            count_loss
            + ot_loss
            + density_l1_loss
            + count_cls_loss
            + uncertainty_mse_loss
            + uncertainty_reg_loss
        )
        result = {
            "loss": total,
            "count_loss": count_loss,
            "ot_loss": ot_loss,
            "ot_wd": ot_wd,
            "sinkhorn_its": sinkhorn_its,
            "density_l1_loss": density_l1_loss,
            "count_cls_loss": count_cls_loss,
        }
        if uncertainty is not None:
            result["uncertainty_mse_loss"] = uncertainty_mse_loss
            result["uncertainty_reg_loss"] = uncertainty_reg_loss
        return result

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
            output_mask = self._build_output_mask(
                shapes,
                out_L[0].shape[-2],
                out_L[0].shape[-1],
                dtype=out_L[0].dtype,
            )
            primary_counts = self._cls_outputs_to_count(out_cls_l[0], shapes)
            density_map, label_normed = self._normalize_density(
                out_L[0] * output_mask,
                primary_counts,
            )
            uncertainty = (
                F.softplus(self.uncertainty_head(label_feats[0])) * output_mask
                if self.use_uncertainty_head
                else None
            )
            return self.compute_loss(
                [density_map] + out_L[1:],
                label_normed,
                out_cls_l,
                targets,
                image_shapes=shapes,
                gt_discrete=gt_discrete,
                uncertainty=uncertainty,
            )

        # Eval: crop each output back to its original spatial extent.
        density_list, normed_list = [], []
        for i, (valid_h, valid_w) in enumerate(self._output_shapes(shapes)):
            crop = out_L[0][i : i + 1, :, :valid_h, :valid_w].contiguous()
            cls_count = self._cls_outputs_to_count(out_cls_l[0][i : i + 1], [shapes[i]])
            dm, nd = self._normalize_density(crop, cls_count)
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
        num_classes = (
            len(label_dict) if label_dict is not None else self.config.num_classes or 1
        )

        if pretrained:
            model = TreeFormerModel.from_pretrained(
                pretrained, revision=revision, **hf_args
            )
        else:
            model = TreeFormerModel(
                num_classes=num_classes,
                label_dict=label_dict,
                num_of_iter_in_ot=cfg.num_of_iter_in_ot,
                sinkhorn_reg=cfg.sinkhorn_reg,
                density_sigma=cfg.density_sigma,
                mae_weight=cfg.mae_weight,
                ot_weight=cfg.ot_weight,
                density_l1_weight=cfg.density_l1_weight,
                count_cls_weight=cfg.count_cls_weight,
                losses=list(cfg.losses) if cfg.losses is not None else None,
                norm_cood=cfg.norm_cood,
                enforce_count=cfg.enforce_count,
                log_count_loss=cfg.log_count_loss,
                count_prediction_mode=cfg.count_prediction_mode,
                use_uncertainty_head=cfg.use_uncertainty_head,
                uncertainty_delta=cfg.uncertainty_delta,
                uncertainty_mse_weight=cfg.uncertainty_mse_weight,
                **hf_args,
            )

        return model.to(map_location)
