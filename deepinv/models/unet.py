from __future__ import annotations
import warnings
import torch
import torch.nn as nn
from .drunet import test_pad
from .base import Denoiser
from .utils import fix_dim, conv_nd, batchnorm_nd, maxpool_nd


class BFBatchNorm2d(nn.BatchNorm2d):
    r"""
    From :footcite:t:`mohan2020robust`.
    """

    def __init__(
        self, num_features, eps=1e-5, momentum=0.1, use_bias=False, affine=True
    ):
        super(BFBatchNorm2d, self).__init__(num_features, eps, momentum)
        self.use_bias = use_bias
        self.affine = affine

    def forward(self, x):
        self._check_input_dim(x)
        y = x.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        if self.use_bias:
            mu = y.mean(dim=1)
        sigma2 = y.var(dim=1)
        if self.training is not True:
            if self.use_bias:
                y = y - self.running_mean.view(-1, 1)
            y = y / (self.running_var.view(-1, 1) ** 0.5 + self.eps)
        else:
            if self.track_running_stats is True:
                with torch.no_grad():
                    if self.use_bias:
                        self.running_mean = (
                            1 - self.momentum
                        ) * self.running_mean + self.momentum * mu
                    self.running_var = (
                        1 - self.momentum
                    ) * self.running_var + self.momentum * sigma2
            if self.use_bias:
                y = y - mu.view(-1, 1)
            y = y / (sigma2.view(-1, 1) ** 0.5 + self.eps)
        if self.affine:
            y = self.weight.view(-1, 1) * y
            if self.use_bias:
                y += self.bias.view(-1, 1)

        return y.view(return_shape).transpose(0, 1)


class UNet(Denoiser):
    r"""
    U-Net convolutional denoiser.

    This network is a fully convolutional denoiser based on the U-Net architecture. The number of downsample steps
    can be controlled with the ``scales`` parameter. The number of trainable parameters increases with the number of
    scales.

    .. warning::
        When using the bias-free batch norm ``BFBatchNorm2d`` via ``batch_norm="biasfree"``, NaNs may be encountered
        during training, causing the whole training procedure to fail.

    :param int in_channels: input image channels
    :param int out_channels: output image channels
    :param bool residual: use a skip-connection between output and output.
    :param bool circular_padding: circular padding for the convolutional layers.
    :param bool cat: use skip-connections between intermediate levels.
    :param bool bias: use learnable biases.
    :param bool, str batch_norm: if False, no batchnorm applied, if ``True``, use :class:`torch.nn.BatchNorm2d`,
        if ``batch_norm="biasfree"``, use ``BFBatchNorm2d`` from :footcite:t:`mohan2020robust`.
    :param int scales: Number of downsampling steps used in the U-Net. The options are 2,3,4 and 5.
        The number of trainable parameters increases with the scale.
    :param torch.device, str device: Device to put the model on.
    :param str, int dim: Whether to build 2D or 3D network (if str, can be "2", "2d", "3D", etc.)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        residual: bool = True,
        circular_padding: bool = False,
        cat: bool = True,
        bias: bool = True,
        batch_norm: bool | str = True,
        scales: int = 4,
        device: torch.device | str = None,
        dim: str | int = 2,
    ):
        super(UNet, self).__init__()
        self.name = "unet"

        if residual and in_channels != out_channels:  # pragma: no cover
            raise warnings.warn(
                "residual is True, but in_channels != out_channels: Falling back to non residual denoiser."
            )

        dim = fix_dim(dim)

        conv = conv_nd(dim)
        batchnorm = batchnorm_nd(dim)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.residual = residual
        self.cat = cat
        self.compact = scales
        self.Maxpool = maxpool_nd(dim)(kernel_size=2, stride=2)

        biasfree = batch_norm == "biasfree"
        if biasfree and dim == 3:  # pragma: no cover
            raise NotImplementedError("Bias-free batchnorm is not implemented for 3D")

        def conv_block(ch_in, ch_out):
            if batch_norm:
                return nn.Sequential(
                    conv(
                        ch_in,
                        ch_out,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias,
                        padding_mode="circular" if circular_padding else "zeros",
                    ),
                    (
                        BFBatchNorm2d(ch_out, use_bias=bias)
                        if biasfree
                        else batchnorm(ch_out)
                    ),
                    nn.ReLU(inplace=True),
                    conv(
                        ch_out,
                        ch_out,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias,
                        padding_mode="circular" if circular_padding else "zeros",
                    ),
                    (
                        BFBatchNorm2d(ch_out, use_bias=bias)
                        if biasfree
                        else batchnorm(ch_out)
                    ),
                    nn.ReLU(inplace=True),
                )
            else:
                return nn.Sequential(
                    conv(
                        ch_in,
                        ch_out,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias,
                        padding_mode="circular" if circular_padding else "zeros",
                    ),
                    nn.ReLU(inplace=True),
                    conv(
                        ch_out,
                        ch_out,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias,
                        padding_mode="circular" if circular_padding else "zeros",
                    ),
                    nn.ReLU(inplace=True),
                )

        def up_conv(ch_in, ch_out):
            if batch_norm:
                return nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    conv(
                        ch_in,
                        ch_out,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias,
                        padding_mode="circular" if circular_padding else "zeros",
                    ),
                    (
                        BFBatchNorm2d(ch_out, use_bias=bias)
                        if biasfree
                        else batchnorm(ch_out)
                    ),
                    nn.ReLU(inplace=True),
                )
            else:
                return nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    conv(
                        ch_in,
                        ch_out,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias,
                        padding_mode="circular" if circular_padding else "zeros",
                    ),
                    nn.ReLU(inplace=True),
                )

        self.Conv1 = conv_block(ch_in=in_channels, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = (
            conv_block(ch_in=128, ch_out=256) if self.compact in [3, 4, 5] else None
        )
        self.Conv4 = (
            conv_block(ch_in=256, ch_out=512) if self.compact in [4, 5] else None
        )
        self.Conv5 = conv_block(ch_in=512, ch_out=1024) if self.compact in [5] else None

        self.Up5 = up_conv(ch_in=1024, ch_out=512) if self.compact in [5] else None
        self.Up_conv5 = (
            conv_block(ch_in=1024, ch_out=512) if self.compact in [5] else None
        )

        self.Up4 = up_conv(ch_in=512, ch_out=256) if self.compact in [4, 5] else None
        self.Up_conv4 = (
            conv_block(ch_in=512, ch_out=256) if self.compact in [4, 5] else None
        )

        self.Up3 = up_conv(ch_in=256, ch_out=128) if self.compact in [3, 4, 5] else None
        self.Up_conv3 = (
            conv_block(ch_in=256, ch_out=128) if self.compact in [3, 4, 5] else None
        )

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = conv(
            in_channels=64,
            out_channels=out_channels,
            bias=bias,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if self.compact == 5:
            self._forward = self.forward_standard
        if self.compact == 4:
            self._forward = self.forward_compact4
        if self.compact == 3:
            self._forward = self.forward_compact3
        if self.compact == 2:
            self._forward = self.forward_compact2

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor, sigma=None, **kwargs) -> torch.Tensor:
        r"""
        Run the denoiser on noisy image. The noise level is not used in this denoiser.

        :param torch.Tensor x: noisy image.
        :param float sigma: noise level (not used).
        """

        factor = 2 ** (self.compact - 1)
        if x.size(2) % factor == 0 and x.size(3) % factor == 0:
            return self._forward(x)
        else:
            return test_pad(self._forward, x, modulo=factor)

    def forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        # encoding path
        cat_dim = 1
        input = x
        x1 = self.Conv1(input)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        if self.cat:
            d5 = torch.cat((x4, d5), dim=cat_dim)
            d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        if self.cat:
            d4 = torch.cat((x3, d4), dim=cat_dim)
            d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        if self.cat:
            d3 = torch.cat((x2, d3), dim=cat_dim)
            d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        if self.cat:
            d2 = torch.cat((x1, d2), dim=cat_dim)
            d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        out = d1 + x if self.residual and self.in_channels == self.out_channels else d1
        return out

    def forward_compact4(self, x: torch.Tensor) -> torch.Tensor:
        # def forward_compact4(self, x):
        # encoding path
        cat_dim = 1
        input = x

        x1 = self.Conv1(input)  # 1->64

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)  # 64->128

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)  # 128->256

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)  # 256->512

        d4 = self.Up4(x4)  # 512->256
        if self.cat:
            d4 = torch.cat((x3, d4), dim=cat_dim)
            d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)  # 256->128
        if self.cat:
            d3 = torch.cat((x2, d3), dim=cat_dim)
            d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)  # 128->64
        if self.cat:
            d2 = torch.cat((x1, d2), dim=cat_dim)
            d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        out = d1 + x if self.residual and self.in_channels == self.out_channels else d1
        return out

    def forward_compact3(self, x: torch.Tensor) -> torch.Tensor:
        # encoding path
        cat_dim = 1
        input = x
        x1 = self.Conv1(input)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        d3 = self.Up3(x3)
        if self.cat:
            d3 = torch.cat((x2, d3), dim=cat_dim)
            d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        if self.cat:
            d2 = torch.cat((x1, d2), dim=cat_dim)
            d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        out = d1 + x if self.residual and self.in_channels == self.out_channels else d1
        return out

    def forward_compact2(self, x: torch.Tensor) -> torch.Tensor:
        # encoding path
        cat_dim = 1
        input = x
        x1 = self.Conv1(input)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        d2 = self.Up2(x2)
        if self.cat:
            d2 = torch.cat((x1, d2), dim=cat_dim)
            d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        out = d1 + x if self.residual and self.in_channels == self.out_channels else d1
        return out
