# code from https://github.com/facebookresearch/ConvNeXt
# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from siclib.models.base_model import BaseModel
from siclib.models.utils.modules import DropPath

logger = logging.getLogger(__name__)

# flake8: noqa
# mypy: ignore-errors

model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

depths_dict = {
    "convnext_tiny": [3, 3, 9, 3],
    "convnext_small": [3, 3, 27, 3],
    "convnext_base": [3, 3, 27, 3],
    "convnext_large": [3, 3, 27, 3],
    "convnext_xlarge": [3, 3, 27, 3],
}

dims_dict = {
    "convnext_tiny": [96, 192, 384, 768],
    "convnext_small": [96, 192, 384, 768],
    "convnext_base": [128, 256, 512, 1024],
    "convnext_large": [192, 384, 768, 1536],
    "convnext_xlarge": [256, 512, 1024, 2048],
}


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = residual + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXt(BaseModel):
    default_conf = {
        "model_size": "tiny",
        "pretrained": False,
        "num_classes": 5,
        "in_chans": 3,
        "drop_path_rate": 0.0,
        "layer_scale_init_value": 1e-6,
        "head_init_scale": 1.0,
        "return_intermediate": False,
    }

    required_data_keys = ["image"]

    def _init(self, conf):  # sourcery skip: extract-method
        logger.debug(f"Initializing ConvNeXt with {conf}")

        assert conf.model_size in ["tiny", "small", "base", "large", "xlarge"], conf.model_size

        in_chans = conf.in_chans
        num_classes = conf.num_classes
        depths = depths_dict[f"convnext_{conf.model_size}"]
        dims = dims_dict[f"convnext_{conf.model_size}"]
        drop_path_rate = conf.drop_path_rate
        layer_scale_init_value = conf.layer_scale_init_value
        head_init_scale = conf.head_init_scale
        self.return_intermediate = conf.return_intermediate

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        if num_classes != 0:
            self.head = nn.Linear(dims[-1], num_classes)
        else:
            self.output_dim = dims[-1]

        self.apply(self._init_weights)
        if num_classes != 0:
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
        self.num_classes = num_classes

        if self.conf.pretrained:
            url = model_urls[f"convnext_{self.conf.model_size}_1k"]
            logger.info(f"Loading ConvNeXt weights from {url}")
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")

            keys = list(checkpoint["model"].keys())
            for k in keys:
                if "head" in k:
                    logger.warning(f"Drop {k} from pretrained weights")
                    checkpoint["model"].pop(k)

            self.load_state_dict(checkpoint["model"], strict=False)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def get_intermediate(self, x):
        intermediate_features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            intermediate_features.append(x)
        return intermediate_features

    def _forward(self, data):
        x = data["image"]

        if self.return_intermediate:
            return {"features": self.get_intermediate(x)}

        x = self.forward_features(x)
        if self.num_classes != 0:
            x = self.head(x)
        return {"features": x}

    def loss(self, pred, data):
        raise NotImplementedError
