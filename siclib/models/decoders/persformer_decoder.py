"""Base decoder for Persformer.

Adapted from https://github.com/jinlinyi/PerspectiveFields
"""

import logging

import torch
from torch import nn
from torch.nn import functional as F

from siclib.models.base_model import BaseModel
from siclib.models.utils.modules import MLP, ConvModule, FeatureFusionBlock

logger = logging.getLogger(__name__)


# flake8: noqa
# mypy: ignore-errors
def resize(input, size=None, scale_factor=None, mode="nearest", align_corners=None, warning=True):
    if warning and (size is not None and align_corners):
        input_h, input_w = tuple(int(x) for x in input.shape[2:])
        output_h, output_w = tuple(int(x) for x in size)
        if (output_h > input_h or output_w > output_h) and (
            (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
            and (output_h - 1) % (input_h - 1)
            and (output_w - 1) % (input_w - 1)
        ):
            logger.warn(
                f"When align_corners={align_corners}, "
                "the output would more aligned if "
                f"input size {(input_h, input_w)} is `x+1` and "
                f"out size {(output_h, output_w)} is `nx+1`"
            )
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class PersformerDecoder(BaseModel):
    """Persformer decoder head."""

    default_conf = {
        "predict_uncertainty": False,
        "use_original_architecture": True,
        "out_channels": 32,
        "in_channels": [64, 128, 320, 512],
        "in_index": [0, 1, 2, 3],
        "input_transform": "multiple_select",
        "align_corners": False,
        "embed_dim": 768,
    }

    required_data_keys = ["hl", "ll"]

    def _init(self, conf):
        logger.debug(f"Persformer decoder with {conf}")
        self._init_inputs(conf.in_channels, conf.in_index, conf.input_transform)

        self.in_index = conf.in_index
        self.align_corners = conf.align_corners
        (
            c1_in_channels,
            c2_in_channels,
            c3_in_channels,
            c4_in_channels,
        ) = self.in_channels

        self.out_channels = conf.out_channels

        embedding_dim = conf.embed_dim
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_c4_proc = torch.nn.Conv2d(
            embedding_dim, 256, kernel_size=3, stride=1, padding=1
        )
        self.linear_c3_proc = torch.nn.Conv2d(
            embedding_dim, 256, kernel_size=3, stride=1, padding=1
        )
        self.linear_c2_proc = torch.nn.Conv2d(
            embedding_dim, 256, kernel_size=3, stride=1, padding=1
        )
        self.linear_c1_proc = torch.nn.Conv2d(
            embedding_dim, 256, kernel_size=3, stride=1, padding=1
        )

        self.fusion1 = FeatureFusionBlock(256)
        self.fusion2 = FeatureFusionBlock(256)
        self.fusion3 = FeatureFusionBlock(256)
        self.fusion4 = FeatureFusionBlock(256, unit2only=True)

        self.conv_fuse_conv0 = (
            ConvModule(in_channels=256 + 64, out_channels=64, kernel_size=3, padding=1)
            if conf.use_original_architecture
            else ConvModule(in_channels=256, out_channels=64, kernel_size=3, padding=1)
        )

        self.conv_fuse_conv1 = ConvModule(
            in_channels=64, out_channels=self.out_channels, kernel_size=3, padding=1
        )

        if not conf.use_original_architecture:
            self.ll_fusion = FeatureFusionBlock(64, upsample=False)

        self.predict_uncertainty = conf.predict_uncertainty
        if self.predict_uncertainty:
            self.linear_pred_uncertainty = nn.Sequential(
                ConvModule(
                    in_channels=64, out_channels=self.out_channels, kernel_size=3, padding=1
                ),
                nn.Conv2d(in_channels=self.out_channels, out_channels=1, kernel_size=1),
            )

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.
        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform
        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ["resize_concat", "multiple_select"]
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            # assert isinstance(in_channels, (list, tuple))
            # assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == "resize_concat":
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == "resize_concat":
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def _forward(self, features):
        x = self._transform_inputs(features["hl"])  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = self.linear_c4_proc(_c4)
        _c4 = self.fusion4(_c4)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = self.linear_c3_proc(_c3)
        _c3 = self.fusion3(_c4, _c3)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = self.linear_c2_proc(_c2)
        _c2 = self.fusion2(_c3, _c2)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c1 = self.linear_c1_proc(_c1)
        _c1 = self.fusion1(_c2, _c1)

        if self.conf.use_original_architecture:
            feats_ll = features["ll"]
            x = torch.cat([_c1, feats_ll], dim=1)
        else:
            x = _c1

        x = self.conv_fuse_conv0(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        if not self.conf.use_original_architecture:
            # clone to avoid in-place problems
            feats_ll = features["ll"].clone()
            x = self.ll_fusion(x, feats_ll)

        u = self.linear_pred_uncertainty(x).squeeze(1) if self.predict_uncertainty else None
        x = self.conv_fuse_conv1(x)
        return x, u

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError
