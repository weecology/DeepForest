"""Point density prediction model based on TreeFormer.

PvT-V2 backbone + multi-scale Regression head producing a density map at
1/4 input resolution.

See: 10.1109/TGRS.2023.3295802
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from scipy.ndimage import gaussian_filter
from transformers import AutoConfig, AutoImageProcessor, PvtV2Model

from deepforest.losses.ot_loss import OT_Loss
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

        # Instantiate architecture but don't pull weights
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
        self.active_losses = set(self.losses)

        # OT_Loss is created lazily on first use, once the device is known.
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
            "score_thresh": self.score_thresh,
            "score_integration_radius": self.score_integration_radius,
            **self.kwargs,
        }

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _get_ot_loss(self) -> OT_Loss:
        """Return the optimal-transport loss, created on first call with the
        current device."""
        if self._ot_loss is None:
            self._ot_loss = OT_Loss(
                self.norm_cood,
                self.device,
                self.ot_iter,
                self.sinkhorn_reg,
            )
        return self._ot_loss

    def _normalize_density(
        self,
        score_map: torch.Tensor,
        cls_count: torch.Tensor | None,
        gt_count: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (density_map, normed_density) from a raw score map and count.

        With ``enforce_count`` the density map is rescaled so its sum matches a
        count: during training the ground-truth ``gt_count`` is used when
        provided, otherwise the CLS prediction ``cls_count``. ``cls_count`` may
        be ``None`` when ``gt_count`` is always supplied (the training path).
        Without ``enforce_count`` the raw ``score_map`` is returned unscaled.
        ``normed_density`` is the spatially normalised map (sums to 1), used by
        the OT and density-L1 losses.
        """
        B = score_map.size(0)
        score_sum = score_map.view(B, -1).sum(1).view(B, 1, 1, 1)
        normed = score_map / (score_sum + 1e-4)
        if self.enforce_count:
            if self.training and gt_count is not None:
                count = gt_count.view(B, 1, 1, 1).clamp(min=1e-4)
            else:
                count = cls_count.view(B, 1, 1, 1).abs().clamp(min=1e-4)
            return normed * count, normed
        return score_map, normed

    def _count_areas(self, image_shapes: list[tuple[int, int]]) -> torch.Tensor:
        """Per-image pixel areas, used to convert between density and count."""
        return torch.tensor(
            [image_h * image_w for image_h, image_w in image_shapes],
            dtype=torch.float32,
            device=self.device,
        )

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

    def _build_output_mask(
        self,
        image_shapes: list[tuple[int, int]],
        out_h: int,
        out_w: int,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Mask padded decoder outputs so losses ignore batch padding."""
        mask = torch.zeros(
            len(image_shapes), 1, out_h, out_w, device=self.device, dtype=dtype
        )
        for index, (valid_h, valid_w) in enumerate(self._output_shapes(image_shapes)):
            mask[index, :, :valid_h, :valid_w] = 1.0
        return mask

    def _cls_outputs_to_count(
        self,
        cls_output: torch.Tensor,
        image_shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Convert CLS-head count-density prediction to absolute count by
        multiplying by the area of each image."""
        count_density = cls_output.reshape(len(image_shapes))
        return count_density * self._count_areas(image_shapes)

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

    def _cast_points(self, targets: list) -> list[torch.Tensor]:
        """Cast training targets to a list of (N, 2) point tensors."""
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
            density_map: list of per-image density tensors (each
                ``(1, 1, H, W)``) or a ``(B, 1, H, W)`` batch tensor.
            images: batch tensor ``(B, C, H_img, W_img)`` or list of
                ``(C, H_img, W_img)`` tensors, used to rescale points to
                image-pixel coordinates.

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

    def rasterize_points(
        self, im_height: int, im_width: int, points: np.ndarray
    ) -> np.ndarray:
        """Rasterize (N, 2) (x, y) points (in output-map space) into an impulse
        map of per-pixel point counts."""
        discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
        points_np = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        points_np = np.rint(points_np).astype(int)
        p_h = np.clip(points_np[:, 1], 0, im_height - 1)
        p_w = np.clip(points_np[:, 0], 0, im_width - 1)
        np.add.at(discrete_map, (p_h, p_w), 1)
        return discrete_map

    def _make_gt_density(self, points: list, out_h: int, out_w: int) -> torch.Tensor:
        """Build a batched Gaussian density map (B, 1, H, W) from point lists.

        Rasterizes points to impulses, applies a Gaussian blur with
        ``density_sigma``, and rescales so the smoothed map preserves
        the discrete point count.
        """
        sigma = self.density_sigma
        maps = []
        for p in points:
            p_np = p.cpu().numpy()
            discrete = self.rasterize_points(out_h, out_w, p_np)
            if discrete.sum() > 0:
                smoothed = gaussian_filter(discrete, sigma=sigma)
                smoothed = smoothed * (discrete.sum() / smoothed.sum())
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
        gt_density: torch.Tensor | None = None,
        points: list | None = None,
    ) -> dict:
        """Compute the supervised TreeFormer/DM-Count losses.

        Spatial structure is trained by optimal transport (``ot``) and a
        pixel-wise L1 against a Gaussian GT density (``density_l1``). The total
        count is trained by MAE on the density-map sum (``count``, in log
        space) and optionally an auxiliary CLS count head (``count_cls``). The
        CLS head predicts count density (count / area) so it transfers across
        image sizes. Individual terms are enabled via ``self.active_losses``.

        Args:
            density_maps: ``[y0, y1, y2]`` multi-scale outputs; index 0 primary.
            normed_density: ``density_maps[0]`` normalised to sum to 1, (B,1,H,W).
            cls_outputs: ``[yc0, yc1, yc2]`` count-density scalars from CLS head.
            targets: list of B target dicts (or point tensors) in image space.
            image_shapes: original ``(H, W)`` for each image in the batch.
            gt_density: optional precomputed ``(B, 1, H, W)`` Gaussian density;
                computed from points when ``None``.
            points: optional pre-cast list of per-image point tensors; cast from
                ``targets`` when ``None``.

        Returns:
            dict with ``"loss"`` (total) plus individual named terms/diagnostics.
        """
        density_map = density_maps[0]  # (B, 1, H', W')
        B, _, H, W = density_map.shape
        if points is None:
            points = self._cast_points(targets)
        output_shapes = self._output_shapes(image_shapes)
        areas = self._count_areas(image_shapes)

        point_counts = torch.tensor(
            [len(p) for p in points], dtype=torch.float32, device=self.device
        )
        scaled_points = self._scale_points_to_output(points, image_shapes, output_shapes)

        if gt_density is None:
            gt_density = self._make_gt_density(scaled_points, H, W)

        active = self.active_losses
        zero = density_map.new_zeros(())
        pred_sum = density_map.view(B, -1).sum(1)

        # Unweighted count diagnostics, so calibration is visible in the logs.
        count_mae = self.cls_l1(pred_sum, point_counts)
        cls_preds = torch.stack([c.reshape(B) for c in cls_outputs])  # (3, B)
        gt_counts = point_counts.unsqueeze(0).expand(3, -1)  # (3, B)

        # ---- MAE count loss -------------------------------------------------
        if "count" in active:
            count_loss = (
                self.cls_l1(torch.log1p(pred_sum), torch.log1p(point_counts))
                * self.mae_weight
            )
        else:
            count_loss = zero

        # ---- Optimal transport loss -----------------------------------------
        if "ot" in active:
            (
                ot_raw,
                ot_wd_val,
                _,
                ot_avg_its,
                ot_K_min,
                ot_beta_abs_max,
                ot_sinkhorn_err,
            ) = self._get_ot_loss()(normed_density, density_map, scaled_points)
            ot_loss = ot_raw * self.ot_weight
            # Wasserstein distance + Sinkhorn diagnostics (not backpropagated).
            ot_wd = torch.tensor(
                ot_wd_val, device=density_map.device, dtype=torch.float32
            )
            sinkhorn_its = torch.tensor(
                ot_avg_its, device=density_map.device, dtype=torch.float32
            )
            sinkhorn_K_min = torch.tensor(
                ot_K_min, device=density_map.device, dtype=torch.float32
            )
            sinkhorn_beta_abs_max = torch.tensor(
                ot_beta_abs_max, device=density_map.device, dtype=torch.float32
            )
            sinkhorn_err = torch.tensor(
                ot_sinkhorn_err, device=density_map.device, dtype=torch.float32
            )
        else:
            ot_loss = zero
            ot_wd = zero
            sinkhorn_its = zero
            sinkhorn_K_min = zero
            sinkhorn_beta_abs_max = zero
            sinkhorn_err = zero

        # ---- Density pixel-wise L1 loss between normalised density maps -
        if "density_l1" in active:
            gt_density_normed = gt_density / (point_counts.view(B, 1, 1, 1) + 1e-4)
            per_pixel = self.density_l1(normed_density, gt_density_normed)
            density_l1_loss = (
                per_pixel.sum(dim=[1, 2, 3]).mul(point_counts).mean()
                * self.density_l1_weight
            )
        else:
            density_l1_loss = zero

        # ---- Auxiliary CLS count regression in density space ---------
        if "count_cls" in active:
            gt_counts_normed = gt_counts / areas.unsqueeze(0)
            count_cls_loss = (
                self.cls_l1(cls_preds, gt_counts_normed) * self.count_cls_weight
            )
            cls_pred_counts = cls_preds * areas.unsqueeze(0)
            count_cls_mae = self.cls_l1(cls_pred_counts, gt_counts)
        else:
            count_cls_loss = zero

        total = count_loss + ot_loss + density_l1_loss + count_cls_loss
        result = {
            "loss": total,
            "count_mae": count_mae,
            "count_loss": count_loss,
            "ot_loss": ot_loss,
            "ot_wd": ot_wd,
            "sinkhorn_its": sinkhorn_its,
            "sinkhorn_K_min": sinkhorn_K_min,
            "sinkhorn_beta_abs_max": sinkhorn_beta_abs_max,
            "sinkhorn_err": sinkhorn_err,
            "density_l1_loss": density_l1_loss,
        }
        if "count_cls" in active:
            result["count_cls_mae"] = count_cls_mae
            result["count_cls_loss"] = count_cls_loss

        return result

    def forward(
        self,
        inputs: torch.Tensor | list[torch.Tensor],
        targets: list | None = None,
        gt_density: torch.Tensor | None = None,
    ):
        """Forward pass.

        In training mode returns a loss dict; in eval mode returns a list of
        per-image prediction dicts (see ``postprocess_density``).

        Args:
            inputs: ``(B, C, H, W)`` tensor or list of ``(C, H, W)`` tensors.
            targets: list of target dicts (required during training).
            gt_density: optional precomputed ``(B, 1, H', W')`` Gaussian density
                target for the density-L1 loss; computed from points if omitted.
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

        if self.training:
            if targets is None:
                raise ValueError("targets must be provided in training mode")
            # Train on the full batched output, masking the padded region.
            output_mask = self._build_output_mask(
                shapes, out_L[0].shape[-2], out_L[0].shape[-1], dtype=out_L[0].dtype
            )
            points = self._cast_points(targets)
            gt_counts = torch.tensor(
                [len(p) for p in points], dtype=torch.float32, device=self.device
            )
            # In training the map is scaled by the ground-truth counts when
            # enforce_count is set, and left raw otherwise. The CLS count
            # prediction is only used at inference, so pass None here.
            density_map, label_normed = self._normalize_density(
                out_L[0] * output_mask, None, gt_count=gt_counts
            )
            return self.compute_loss(
                [density_map] + out_L[1:],
                label_normed,
                out_cls_l,
                targets,
                image_shapes=shapes,
                gt_density=gt_density,
                points=points,
            )

        # Eval: crop each output to its valid spatial extent and normalise.
        # Cropping naturally excludes batch padding, which is required for
        # inference peak-finding.
        density_list = []
        for i, (valid_h, valid_w) in enumerate(self._output_shapes(shapes)):
            crop = out_L[0][i : i + 1, :, :valid_h, :valid_w].contiguous()
            cls_count = self._cls_outputs_to_count(out_cls_l[0][i : i + 1], [shapes[i]])
            dm, _ = self._normalize_density(crop, cls_count)
            density_list.append(dm)

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

        # Loss/normalisation hyperparameters from config, passed only when
        # training from scratch. A loaded checkpoint keeps its own saved config.
        point_cfg = self.config.point
        loss_kwargs = {
            "density_sigma": point_cfg.density_sigma,
            "mae_weight": point_cfg.mae_weight,
            "ot_weight": point_cfg.ot_weight,
            "density_l1_weight": point_cfg.density_l1_weight,
            "count_cls_weight": point_cfg.count_cls_weight,
            "sinkhorn_reg": point_cfg.sinkhorn_reg,
            "num_of_iter_in_ot": point_cfg.num_of_iter_in_ot,
            "losses": list(point_cfg.losses) if point_cfg.losses is not None else None,
            "norm_cood": point_cfg.norm_cood,
            "enforce_count": point_cfg.enforce_count,
        }
        scratch_args = {**loss_kwargs, **hf_args}

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
                **scratch_args,
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
                **scratch_args,
            )

        if map_location is not None:
            model = model.to(map_location)
        return model
