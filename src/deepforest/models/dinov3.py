from collections import OrderedDict

import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModel


class Dinov3Model(nn.Module):
    """DINOv3 vision transformer backbone for torchvision detectors.

    The final layer patch tokens are turned into a four level feature
    pyramid (1/8 to 1/64 scale) with conv/deconv layers, following the
    ViTDet simple feature pyramid. The output matches the torchvision
    backbone interface for retinanet: an OrderedDict of feature maps
    and an ``out_channels`` attribute.

    Args:
        repo_id: Hugging Face repository ID of the DINOv3 model.
    """

    out_channels: int = 256

    def __init__(self, repo_id: str):
        super().__init__()

        self.repo_id = repo_id
        self.model = AutoModel.from_pretrained(repo_id)
        # The mask token is only used in masked pretraining. Freeze to 
        # avoid warnings about unused parameters during training.
        self.model.embeddings.mask_token.requires_grad_(False)
        processor = AutoImageProcessor.from_pretrained(repo_id)
        self.image_mean = processor.image_mean
        self.image_std = processor.image_std

        hidden_size = self.model.config.hidden_size

        # ViTDet scale_factors=(4.0, 2.0, 1.0, 0.5), applied in
        # parallel to the 1/16 scale patch features.
        self.fpn_0_5x = nn.ConvTranspose2d(
            hidden_size, self.out_channels, kernel_size=4, stride=2, padding=1
        )  # 1/8 scale
        self.fpn_1x = nn.Conv2d(
            hidden_size, self.out_channels, kernel_size=3, stride=1, padding=1
        )  # 1/16 scale
        self.fpn_2x = nn.Conv2d(
            hidden_size, self.out_channels, kernel_size=3, stride=2, padding=1
        )  # 1/32 scale
        self.fpn_4x = nn.Conv2d(
            hidden_size, self.out_channels, kernel_size=3, stride=4, padding=1
        )  # 1/64 scale

        for module in [self.fpn_0_5x, self.fpn_1x, self.fpn_2x, self.fpn_4x]:
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(module.bias, 0)

    def _reshape_to_spatial(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Reshape the token sequence to a [B, C, H, W] feature map."""
        # Skip the CLS token and register tokens, keep only patch tokens
        patch_tokens = x[:, 1 + self.model.config.num_register_tokens :, :]
        patch_tokens = patch_tokens.contiguous().view(x.shape[0], h, w, -1)
        return patch_tokens.permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> OrderedDict:
        """Compute the feature pyramid.

        Args:
            x: Image batch [B, C, H, W], already normalized by RetinaNet.

        Returns:
            OrderedDict of four feature maps, highest resolution first.
        """
        patch_size = self.model.config.patch_size
        h_patches = x.shape[2] // patch_size
        w_patches = x.shape[3] // patch_size

        # Use the normalized output. hidden_states[-1] skips the final norm
        # layer, which leaves its parameters unused and breaks DDP training.
        tokens = self.model(pixel_values=x).last_hidden_state
        base_features = self._reshape_to_spatial(tokens, h_patches, w_patches)

        return OrderedDict(
            feat_0=self.fpn_0_5x(base_features),
            feat_1=self.fpn_1x(base_features),
            feat_2=self.fpn_2x(base_features),
            feat_3=self.fpn_4x(base_features),
        )
