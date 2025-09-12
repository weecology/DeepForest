from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoImageProcessor, AutoModel


class Dinov3Model(nn.Module):
    def __init__(
        self,
        repo_id="facebook/dinov3-vitl16-pretrain-sat493m",
        frozen=True,
        use_conv_pyramid=True,
        fpn_out_channels=256,
    ):
        super().__init__()

        self.model = AutoModel.from_pretrained(repo_id)
        self.processor = AutoImageProcessor.from_pretrained(repo_id)
        self.image_mean = torch.Tensor(self.processor.image_mean)
        self.image_std = torch.Tensor(self.processor.image_std)
        self.frozen = frozen
        self.use_conv_pyramid = use_conv_pyramid
        self.fpn_out_channels = fpn_out_channels

        if self.frozen:
            # Freeze DINO model parameters
            for param in self.model.parameters():
                param.requires_grad = False

        # Use final layer
        self.feature_layer = -1

        # Infer hidden size from model config
        self.hidden_size = self.model.config.hidden_size

        if self.use_conv_pyramid:
            # ViTDet-style simple feature pyramid with conv/deconv layers
            # Following Detectron2 scale_factors=(4.0, 2.0, 1.0, 0.5)
            # All operations applied in parallel to base features
            self.fpn_4x = nn.Conv2d(
                self.hidden_size,
                self.fpn_out_channels,
                kernel_size=3,
                stride=4,
                padding=1,
            )  # 1/64 scale (4x downsample)
            self.fpn_2x = nn.Conv2d(
                self.hidden_size,
                self.fpn_out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )  # 1/32 scale (2x downsample)
            self.fpn_1x = nn.Conv2d(
                self.hidden_size,
                self.fpn_out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )  # 1/16 scale (base)
            self.fpn_0_5x = nn.ConvTranspose2d(
                self.hidden_size,
                self.fpn_out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )  # 1/8 scale (2x upsample)

            # Initialize conv layers
            for module in [self.fpn_4x, self.fpn_2x, self.fpn_1x, self.fpn_0_5x]:
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            self.out_channels = self.fpn_out_channels
        else:
            # Multi-scale pooling to create pyramid,
            # uses average pooling in the forward pass.
            self.scales = [1, 2, 4, 8]
            self.out_channels = self.hidden_size

    def _reshape_to_spatial(self, x, h, w):
        """Reshape token sequence back to spatial format."""
        batch_size = x.shape[0]

        # Skip CLS token (first) and register tokens, keep only patch tokens
        num_register_tokens = self.model.config.num_register_tokens
        patch_tokens = x[:, 1 + num_register_tokens :, :]

        patch_tokens = patch_tokens.contiguous().view(batch_size, h, w, self.hidden_size)
        patch_tokens = patch_tokens.permute(0, 3, 1, 2)  # BHWC -> BCHW
        return patch_tokens

    def forward(self, x: torch.Tensor) -> OrderedDict:
        # Note that 'x' is normalized by RetinaNet, so be careful.
        batch_size, _, height, width = x.shape

        # Calculate patch grid size (assuming patch size 16)
        patch_size = self.model.config.patch_size
        h_patches = height // patch_size
        w_patches = width // patch_size

        # Here, we expect x to be normalized.
        encoded_inputs = {"pixel_values": x}

        # This is the actual "forward pass", we extract the hidden states.
        hidden_states = self.model(
            **encoded_inputs, output_hidden_states=True
        ).hidden_states

        # Get features from final layer
        layer_features = hidden_states[self.feature_layer]

        # Reshape to spatial format
        base_features = self._reshape_to_spatial(layer_features, h_patches, w_patches)

        if self.use_conv_pyramid:
            # Create multi-scale features using ViTDet-style simple feature pyramid
            # Apply all conv/deconv operations in parallel to the base features
            feat_4x = self.fpn_4x(base_features)  # 4x downsample for 1/64 scale
            feat_2x = self.fpn_2x(base_features)  # 2x downsample for 1/32 scale
            feat_1x = self.fpn_1x(base_features)  # Base resolution 1/16 scale
            feat_0_5x = self.fpn_0_5x(base_features)  # 2x upsample for 1/8 scale

            features = {
                "feat_0": feat_0_5x,  # 1/8 scale (highest resolution - for small objects)
                "feat_1": feat_1x,  # 1/16 scale (medium-high resolution)
                "feat_2": feat_2x,  # 1/32 scale (medium-low resolution)
                "feat_3": feat_4x,  # 1/64 scale (lowest resolution - for large objects)
            }
        else:
            # Create multi-scale features using pooling (original approach)
            features = {}
            for i, scale in enumerate(self.scales):
                if scale == 1:
                    # Original resolution
                    features[f"feat_{i}"] = base_features
                else:
                    # Downsample using average pooling
                    pooled_features = F.avg_pool2d(
                        base_features, kernel_size=scale, stride=scale
                    )
                    features[f"feat_{i}"] = pooled_features

        # Return as OrderedDict to match torchvision backbone interface

        return OrderedDict(features)

    def normalize(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor X [B, C, H, W] using per-channel mean and
        std.

        Args:
            X: input tensor [B, C, H, W]
        Returns:
            Normalized tensor [B, C, H, W]
        """
        # reshape mean and std to [C, 1, 1] so they broadcast across H, W
        mean = self.image_mean.view(-1, 1, 1)
        std = self.image_std.view(-1, 1, 1)
        return (X - mean) / std
