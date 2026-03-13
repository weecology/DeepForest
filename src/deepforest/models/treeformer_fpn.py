"""TreeFormer FPN: 2-scale FPN neck + single density head.

Uses stride-4 and stride-8 backbone features only, fused by a standard
torchvision FeaturePyramidNetwork. The finest FPN level feeds a lightweight
DensityHead. The CLS count branch (nn.Linear + noise) is identical to the
base TreeFormer decoder.

Forward signatures (unchanged from base):
  training:  model(inputs, points)  -> loss_dict
  eval:      model(inputs)          -> (density_map, normed_density)
"""

from collections import OrderedDict

import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork

from deepforest.models.treeformer import TreeFormerModel
from deepforest.models.treeformer_decoder import FeatureDropDecoder, FeatureNoiseDecoder


class DensityHead(nn.Module):
    """Lightweight conv stack: density map from 128-channel FPN output."""

    def __init__(self, in_channels: int = 128, num_classes: int = 1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.head(x)


class RegressionFPN(nn.Module):
    """2-scale FPN neck (stride-4 + stride-8) + DensityHead + CLS count branch.

    Using only the two finest backbone scales keeps the receptive field
    equivalent to the base TreeFormer's single-step neck fusion (x0 +
    upsample(x1)), preventing coarse-scale diffusion.
    """

    FPN_IN_CHANNELS = [128, 256]
    FPN_OUT_CHANNELS = 128

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.FPN_IN_CHANNELS,
            out_channels=self.FPN_OUT_CHANNELS,
        )
        self.density_head = DensityHead(self.FPN_OUT_CHANNELS, num_classes)

        # CLS count branch
        self.noise1 = FeatureDropDecoder(1, 256, 256)
        self.noise0 = FeatureNoiseDecoder(1, 128, 128)
        self.noise_cls = nn.Dropout(p=0.3)

        self.cls_lin1 = nn.Linear(1024, 512, bias=False)
        self.cls_lin2 = nn.Linear(512, 256, bias=False)
        self.cls_lin3 = nn.Linear(256, 128, bias=False)
        self.cls_lin4 = nn.Linear(128, num_classes, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, cls):
        """Forward pass.

        Args:
            x:   list of 4 spatial tensors [(B,128,H/4,W/4) … (B,1024,H/32,W/32)]
            cls: list of 3 GAP vectors [(B,256), (B,512), (B,1024)]

        Returns:
            ([y0], [yc0, yc1, yc2])  — y0 shape: (B, 1, H/4, W/4)
        """
        fpn_in = OrderedDict([("0", x[0]), ("1", x[1])])
        fpn_out = self.fpn(fpn_in)
        y0 = self.density_head(fpn_out["0"])

        if self.training:
            lin1_out = self.cls_lin1(cls[2])
            yc2 = self.cls_lin4(
                self.cls_lin3(self.cls_lin2(self.noise_cls(lin1_out)))
            ).squeeze()

            lin2_out = self.cls_lin2(cls[1])
            lin2_noisy = self.noise1(lin2_out[:, :, None, None]).squeeze(-1).squeeze(-1)
            yc1 = self.cls_lin4(self.cls_lin3(lin2_noisy)).squeeze()

            lin3_out = self.cls_lin3(cls[0])
            lin3_noisy = self.noise0(lin3_out[:, :, None, None]).squeeze(-1).squeeze(-1)
            yc0 = self.cls_lin4(lin3_noisy).squeeze()

        else:
            yc2 = self.cls_lin4(
                self.cls_lin3(self.cls_lin2(self.cls_lin1(cls[2])))
            ).squeeze()
            yc1 = self.cls_lin4(self.cls_lin3(self.cls_lin2(cls[1]))).squeeze()
            yc0 = self.cls_lin4(self.cls_lin3(cls[0])).squeeze()

        return [y0], [yc0, yc1, yc2]


class TreeFormerFPN(TreeFormerModel):
    """TreeFormer with 2-scale FPN neck and single density head.

    Replaces the multi-scale decoder in TreeFormer with a standard FPN
    fusing only the stride-4 and stride-8 backbone features. The CLS
    count branch and enforce_count behaviour are inherited from the
    base.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regression = RegressionFPN(num_classes=self.num_classes)
