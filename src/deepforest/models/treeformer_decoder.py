"""Decoder modules for the TreeFormer density-map prediction head.

Contains the multi-scale Regression head and its helper blocks. This
code is a reimplementation of the code found in the TreeFormer
repository, but updated to follow more modern PyTorch practices.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // ratio, in_planes, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, in_feature):
        x = in_feature
        b, c, _, _ = in_feature.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        out = avg_out
        return self.sigmoid(out).expand_as(in_feature) * in_feature


class FeatureDropDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super().__init__()

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, x):
        x = self.feature_dropout(x)
        return x


class FeatureNoiseDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, uniform_range=0.3):
        super().__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x


class DropOutDecoder(nn.Module):
    def __init__(
        self, upscale, conv_in_ch, num_classes, drop_rate=0.3, spatial_dropout=True
    ):
        super().__init__()
        self.dropout = (
            nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)
        )

    def forward(self, x):
        x = self.dropout(x)
        return x


class Regression(nn.Module):
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes

        self.v1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1, dilation=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.v2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.v3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(1024, 512, 3, padding=1, dilation=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.ca2 = nn.Sequential(
            ChannelAttention(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.ca1 = nn.Sequential(
            ChannelAttention(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.ca0 = nn.Sequential(
            ChannelAttention(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, dilation=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, dilation=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
        )

        self.res0 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
        )

        self.noise2 = nn.Dropout2d(p=0.3)
        self.noise1 = FeatureDropDecoder(1, 256, 256)
        self.noise0 = FeatureNoiseDecoder(1, 128, 128)
        self.noise_cls = nn.Dropout(p=0.3)

        self.upsam2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsam4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        self.cls_lin1 = nn.Linear(1024, 512, bias=False)
        self.cls_lin2 = nn.Linear(512, 256, bias=False)
        self.cls_lin3 = nn.Linear(256, 128, bias=False)
        self.cls_lin4 = nn.Linear(128, num_classes, bias=True)

        self.init_param()
        # Start the count-density head with a small positive prior
        # (~100 trees per 1024x1024 image). count_cls loss refines from here.
        nn.init.constant_(self.cls_lin4.bias, 1e-4)

    def forward(self, x, cls):
        x0, x1, x2, x3 = x[0], x[1], x[2], x[3]

        x2_1 = self.ca2(x2) + self.v3(x3)
        x1_1 = self.ca1(x1) + self.v2(x2_1)
        x0_1 = self.ca0(x0) + self.v1(x1_1)

        if self.training:
            lin1_out = self.cls_lin1(cls[2])
            yc2 = self.cls_lin4(
                self.cls_lin3(self.cls_lin2(self.noise_cls(lin1_out)))
            ).squeeze(-1)

            lin2_out = self.cls_lin2(cls[1])
            lin2_noisy = self.noise1(lin2_out[:, :, None, None]).squeeze(-1).squeeze(-1)
            yc1 = self.cls_lin4(self.cls_lin3(lin2_noisy)).squeeze(-1)

            lin3_out = self.cls_lin3(cls[0])
            lin3_noisy = self.noise0(lin3_out[:, :, None, None]).squeeze(-1).squeeze(-1)
            yc0 = self.cls_lin4(lin3_noisy).squeeze(-1)

            y2 = self.res2(self.upsam4(self.noise2(x2_1)))
            y1 = self.res1(self.upsam2(self.noise1(x1_1)))
            y0 = self.res0(self.noise0(x0_1))

        else:
            yc2 = self.cls_lin4(
                self.cls_lin3(self.cls_lin2(self.cls_lin1(cls[2])))
            ).squeeze(-1)
            yc1 = self.cls_lin4(self.cls_lin3(self.cls_lin2(cls[1]))).squeeze(-1)
            yc0 = self.cls_lin4(self.cls_lin3(cls[0])).squeeze(-1)

            y2 = self.res2(self.upsam4(x2_1))
            y1 = self.res1(self.upsam2(x1_1))
            y0 = self.res0(x0_1)

        return [y0, y1, y2], [yc0, yc1, yc2]

    def init_param(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
