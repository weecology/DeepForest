import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models import register_model
from timm.models.vision_transformer import _cfg
from torch.distributions.uniform import Uniform

__all__ = ["pvt_tiny", "pvt_small", "pvt_medium", "pvt_large"]


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Regression(nn.Module):
    def __init__(self):
        super().__init__()

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
            nn.Conv2d(128, 1, 3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, dilation=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
        )

        self.res0 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
        )

        self.noise2 = DropOutDecoder(1, 512, 512)
        self.noise1 = FeatureDropDecoder(1, 256, 256)
        self.noise0 = FeatureNoiseDecoder(1, 128, 128)

        self.upsam2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsam4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        self.conv1 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 1, kernel_size=1, bias=False)

        # cls2.view(8, 1024, 1, 1))

        self.init_param()

    def forward(self, x, cls):
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]
        x3 = x[3]
        cls0 = cls[0].view(cls[0].shape[0], cls[0].shape[1], 1, 1)
        cls1 = cls[1].view(cls[1].shape[0], cls[1].shape[1], 1, 1)
        cls2 = cls[2].view(cls[2].shape[0], cls[2].shape[1], 1, 1)

        x2_1 = self.ca2(x2) + self.v3(x3)
        x1_1 = self.ca1(x1) + self.v2(x2_1)
        x0_1 = self.ca0(x0) + self.v1(x1_1)

        if self.training:
            yc2 = self.conv4(
                self.conv3(self.conv2(self.noise2(self.conv1(cls2))))
            ).squeeze()
            yc1 = self.conv4(self.conv3(self.noise1(self.conv2(cls1)))).squeeze()
            yc0 = self.conv4(self.noise0(self.conv3(cls0))).squeeze()

            y2 = self.res2(self.upsam4(self.noise2(x2_1)))
            y1 = self.res1(self.upsam2(self.noise1(x1_1)))
            y0 = self.res0(self.noise0(x0_1))

        else:
            yc2 = self.conv4(self.conv3(self.conv2(self.conv1(cls2)))).squeeze()
            yc1 = self.conv4(self.conv3(self.conv2(cls1))).squeeze()
            yc0 = self.conv4(self.conv3(cls0)).squeeze()

            y2 = self.res2(self.upsam4(x2_1))
            y1 = self.res1(self.upsam2(x1_1))
            y0 = self.res0(x0_1)

        return [y0, y1, y2], [yc0, yc1, yc2]

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


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


## ChannelAttetion
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


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim {dim} should be divided by num_heads {num_heads}."
        )

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 4:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PyramidVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dims=None,
        num_heads=None,
        mlp_ratios=None,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=None,
        sr_ratios=None,
        num_stages=4,
    ):
        if sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]
        if depths is None:
            depths = [3, 4, 6, 3]
        if mlp_ratios is None:
            mlp_ratios = [4, 4, 4, 4]
        if num_heads is None:
            num_heads = [1, 2, 4, 8]
        if embed_dims is None:
            embed_dims = [64, 128, 256, 512]
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = PatchEmbed(
                img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                patch_size=patch_size if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
            )
            num_patches = (
                patch_embed.num_patches if i == 0 else patch_embed.num_patches + 1
            )
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            block = nn.ModuleList(
                [
                    Block(
                        dim=embed_dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratios[i],
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[cur + j],
                        norm_layer=norm_layer,
                        sr_ratio=sr_ratios[i],
                    )
                    for j in range(depths[i])
                ]
            )
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

        self.norm = norm_layer(embed_dims[3])

        # cls_token
        self.cls_token_1 = nn.Parameter(torch.zeros(1, 1, embed_dims[1]))
        self.cls_token_2 = nn.Parameter(torch.zeros(1, 1, embed_dims[2]))
        self.cls_token_3 = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        # classification head
        self.head = (
            nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        )

        self.regression = Regression()

        # init weights
        for i in range(num_stages):
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            trunc_normal_(pos_embed, std=0.02)
        trunc_normal_(self.cls_token_1, std=0.02)
        trunc_normal_(self.cls_token_2, std=0.02)
        trunc_normal_(self.cls_token_3, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {"cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return (
                F.interpolate(
                    pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(
                        0, 3, 1, 2
                    ),
                    size=(H, W),
                    mode="bilinear",
                )
                .reshape(1, -1, H * W)
                .permute(0, 2, 1)
            )

    def forward_features(self, x):
        B = x.shape[0]
        outputs = []
        cls_output = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, (H, W) = patch_embed(x)

            if i == 0:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)
            elif i == 1:
                cls_tokens = self.cls_token_1.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                pos_embed_ = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
                pos_embed = torch.cat((pos_embed[:, 0:1], pos_embed_), dim=1)
            elif i == 2:
                cls_tokens = self.cls_token_2.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                pos_embed_ = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
                pos_embed = torch.cat((pos_embed[:, 0:1], pos_embed_), dim=1)

            elif i == 3:
                cls_tokens = self.cls_token_3.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                pos_embed_ = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
                pos_embed = torch.cat((pos_embed[:, 0:1], pos_embed_), dim=1)

            x = pos_drop(x + pos_embed)
            for blk in block:
                x = blk(x, H, W)

            if i == 0:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            else:
                x_cls = x[:, 1, :]
                x = x[:, 1:, :].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                cls_output.append(x_cls)

            outputs.append(x)
        return outputs, cls_output

    def forward(self, label_x, unlabel_x=None):
        if self.training:
            # labeled image processing
            label_x, l_cls = self.forward_features(label_x)
            out_label_x, out_cls_l = self.regression(label_x, l_cls)
            label_x_1, label_x_2, label_x_3 = out_label_x

            B, C, H, W = label_x_1.size()
            label_sum = (
                label_x_1.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            )
            label_normed = label_x_1 / (label_sum + 1e-6)

            # unlabeled image processing
            B, C, H, W = unlabel_x.shape
            unlabel_x, ul_cls = self.forward_features(unlabel_x)
            out_unlabel_x, out_cls_ul = self.regression(unlabel_x, ul_cls)
            y0, y1, y2 = out_unlabel_x

            unlabel_x_1 = self.generate_feature_patches(y0)
            unlabel_x_2 = self.generate_feature_patches(y1)
            unlabel_x_3 = self.generate_feature_patches(y2)

            assert unlabel_x_1.shape[0] == B * 5
            assert unlabel_x_2.shape[0] == B * 5
            assert unlabel_x_3.shape[0] == B * 5

            unlabel_x_1 = torch.split(unlabel_x_1, split_size_or_sections=B, dim=0)
            unlabel_x_2 = torch.split(unlabel_x_2, split_size_or_sections=B, dim=0)
            unlabel_x_3 = torch.split(unlabel_x_3, split_size_or_sections=B, dim=0)

            return (
                [label_x_1, label_x_2, label_x_3],
                [unlabel_x_1, unlabel_x_2, unlabel_x_3],
                label_normed,
                out_cls_l,
                out_cls_ul,
            )

        else:
            label_x, l_cls = self.forward_features(label_x)
            out_label_x, out_cls_l = self.regression(label_x, l_cls)
            label_x_1, label_x_2, label_x_3 = out_label_x
            B, C, H, W = label_x_1.size()
            label_sum = (
                label_x_1.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            )
            label_normed = label_x_1 / (label_sum + 1e-6)

            return [label_x_1, label_x_2, label_x_3], label_normed

    def generate_feature_patches(self, unlabel_x, ratio=0.75):
        # unlabeled image processing

        unlabel_x_1 = unlabel_x
        b, c, h, w = unlabel_x.shape

        new_h2 = int(h * ratio)
        new_w2 = int(w * ratio)  # 48*48

        x_margin = max((h - new_h2) // 2, 0)
        y_margin = max((w - new_w2) // 2, 0)
        center_x = random.randint(h // 2 - x_margin, h // 2 + x_margin)
        center_y = random.randint(w // 2 - y_margin, w // 2 + y_margin)

        unlabel_x_2 = unlabel_x[
            :,
            :,
            center_x - new_h2 // 2 : center_x + new_h2 // 2,
            center_y - new_w2 // 2 : center_y + new_w2 // 2,
        ]

        new_h3 = int(new_h2 * ratio)
        new_w3 = int(new_w2 * ratio)
        unlabel_x_3 = unlabel_x[
            :,
            :,
            center_x - new_h3 // 2 : center_x + new_h3 // 2,
            center_y - new_w3 // 2 : center_y + new_w3 // 2,
        ]

        new_h4 = int(new_h3 * ratio)
        new_w4 = int(new_w3 * ratio)
        unlabel_x_4 = unlabel_x[
            :,
            :,
            center_x - new_h4 // 2 : center_x + new_h4 // 2,
            center_y - new_w4 // 2 : center_y + new_w4 // 2,
        ]

        new_h5 = int(new_h4 * ratio)
        new_w5 = int(new_w4 * ratio)
        unlabel_x_5 = unlabel_x[
            :,
            :,
            center_x - new_h5 // 2 : center_x + new_h5 // 2,
            center_y - new_w5 // 2 : center_y + new_w5 // 2,
        ]

        unlabel_x_2 = nn.functional.interpolate(unlabel_x_2, size=(h, w), mode="bilinear")
        unlabel_x_3 = nn.functional.interpolate(unlabel_x_3, size=(h, w), mode="bilinear")
        unlabel_x_4 = nn.functional.interpolate(unlabel_x_4, size=(h, w), mode="bilinear")
        unlabel_x_5 = nn.functional.interpolate(unlabel_x_5, size=(h, w), mode="bilinear")

        unlabel_x = torch.cat(
            [unlabel_x_1, unlabel_x_2, unlabel_x_3, unlabel_x_4, unlabel_x_5], dim=0
        )

        return unlabel_x


def _conv_filter(state_dict, patch_size=16):
    """Convert patch embedding weight from manual patchify + linear proj to
    conv."""
    out_dict = {}
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@register_model
def pvt_tiny(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    )
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_small(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    )
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_medium(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 18, 3],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    )
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_large(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 8, 27, 3],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    )
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_treeformer(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4,
        embed_dims=[128, 256, 512, 1024],
        num_heads=[4, 8, 16, 32],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 18, 3],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    )
    model.default_cfg = _cfg()

    return model
