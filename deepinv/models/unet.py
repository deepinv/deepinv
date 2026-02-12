from __future__ import annotations
from typing import Any, Sequence
from dataclasses import dataclass
import warnings

import torch
import torch.nn as nn
from .drunet import test_pad
from .base import Denoiser
from .utils import fix_dim, conv_nd, batchnorm_nd, maxpool_nd


class UNet(Denoiser):
    r"""
    U-Net convolutional denoiser.

    The architecture follows the design described in :footcite:t:`jin2017deep`, which is adapted for
    inverse problems by using zero padding and a global residual connection.
    This differs from the original U-Net :footcite:p:`ronneberger2015u` commonly used for medical segmentation.

    The number of stages in the network is controlled by ``scales``. The width of each stage is
    controlled by ``channels_per_scale``, which gives the number of feature maps at each stage,
    from shallow to deeper stages.


    .. warning::
        When using the bias-free batch norm via ``batch_norm="biasfree"``, NaNs may be encountered
        during training, causing the whole training procedure to fail.

    :param int in_channels: input image channels
    :param int out_channels: output image channels
    :param bool residual: use a skip-connection between input and output.
    :param bool circular_padding: circular padding for the convolutional layers.
    :param bool cat: use skip-connections between intermediate levels.
    :param bool bias: use learnable biases in conv and norm layers.
    :param bool, str batch_norm: if False, disable normalization entirely, if ``True``, use batch normalization,
        if ``batch_norm="biasfree"``, use the bias-free batch norm from :footcite:t:`mohan2020robust`.
    :param int scales: Number of stages.
    :param Sequence[int] channels_per_scale: Number of feature maps at each stage (from shallow to deep).
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
        scales: int | None = None,
        channels_per_scale: Sequence[int] | None = None,
        device: torch.device | str = "cpu",
        dim: str | int = 2,
    ):
        super(UNet, self).__init__()
        self.name = "unet"

        if residual and in_channels != out_channels:  # pragma: no cover
            warnings.warn(
                "residual is True, but in_channels != out_channels: Falling back to non residual denoiser."
            )

        if (scales is not None and channels_per_scale is not None) and (
            len(channels_per_scale) != scales
        ):  # pragma: no cover
            raise RuntimeError(
                f"Both scales and channels_per_scale was passed, but scales ({scales}) does not match the length of channels_per_scale ({len(channels_per_scale)})"
            )

        if scales is None:
            if channels_per_scale is not None:
                scales = len(channels_per_scale)
            else:
                scales = 4  # legacy default

        if scales < 2:  # pragma: no cover
            raise ValueError(f"UNet requires at least 2 scales. Got {scales} scales")

        if channels_per_scale is None:
            channels_per_scale = [64 * (2**k) for k in range(scales)]

        dim = fix_dim(dim)

        conv = conv_nd(dim)
        self.Maxpool = maxpool_nd(dim)(kernel_size=2, stride=2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual
        self.cat = cat

        b = _Blocks(
            dim=dim,
            circular_padding=circular_padding,
            biasfree_norm=batch_norm == "biasfree",
            use_bias=bias,
            norm=batch_norm,
        )

        cps = channels_per_scale  # shorthand

        for i in range(scales):
            ch_in = in_channels if i == 0 else cps[i - 1]
            ch_out = cps[i]
            setattr(self, f"Conv{i+1}", b.conv_block(ch_in=ch_in, ch_out=ch_out))

        for i in range(scales - 1):
            ch_in = cps[-i - 1]
            ch_out = cps[-i - 2]
            setattr(self, f"Up{scales - i}", b.up_conv(ch_in=ch_in, ch_out=ch_out))
            setattr(
                self,
                f"Up_conv{scales - i}",
                b.conv_block(ch_in=ch_out * 2, ch_out=ch_out),
            )

        self.Conv_1x1 = conv(
            in_channels=cps[0],
            out_channels=out_channels,
            bias=bias,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self._enc_names = tuple(f"Conv{i+1}" for i in range(scales))
        self._up_names = tuple(f"Up{i+2}" for i in range(scales - 1))[::-1]
        self._upc_names = tuple(f"Up_conv{i + 2}" for i in range(scales - 1))[::-1]

        if device is not None:
            self.to(device)

    @property
    def compact(self):
        warnings.warn("UNet.compact is deprecated and read-only.")
        return len(self._enc_names)

    def forward(self, x: torch.Tensor, sigma: Any = None, **kwargs) -> torch.Tensor:
        r"""
        Run the denoiser on noisy image. The noise level is not used in this denoiser.

        :param torch.Tensor x: noisy image.
        :param float sigma: noise level (not used).
        """

        factor = 2 ** (len(self._upc_names))
        if x.size(2) % factor == 0 and x.size(3) % factor == 0:
            return self._forward(x)
        else:
            return test_pad(self._forward, x, modulo=factor)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        network_input = x

        enc_feats = []
        for i, name in enumerate(self._enc_names):
            block = getattr(self, name)
            x = block(x) if i == 0 else block(self.Maxpool(x))
            enc_feats.append(x)

        for i, (up_name, upc_name) in enumerate(zip(self._up_names, self._upc_names)):
            x = getattr(self, up_name)(x)
            if self.cat:
                skip = enc_feats[-2 - i]
                x = torch.cat((skip, x), dim=1)
                x = getattr(self, upc_name)(x)

        x = self.Conv_1x1(x)

        return (
            x + network_input
            if self.residual and self.in_channels == self.out_channels
            else x
        )

    def forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        warnings.warn("forward_standard is deprecated, please use unet._forward")
        return self._forward(x)

    def forward_compact4(self, x: torch.Tensor) -> torch.Tensor:
        warnings.warn("forward_compact4 is deprecated, please use unet._forward")
        return self._forward(x)

    def forward_compact3(self, x: torch.Tensor) -> torch.Tensor:
        warnings.warn("forward_compact3 is deprecated, please use unet._forward")
        return self._forward(x)

    def forward_compact2(self, x: torch.Tensor) -> torch.Tensor:
        warnings.warn("forward_compact2 is deprecated, please use unet._forward")
        return self._forward(x)


@dataclass(frozen=True)
class _Blocks:
    dim: int
    circular_padding: bool
    biasfree_norm: bool
    use_bias: bool
    norm: bool

    @property
    def padding_mode(self) -> str:
        return "circular" if self.circular_padding else "zeros"

    def make_norm(self, ch_out: int) -> nn.Module:
        if not self.norm:
            return None
        return (
            bfbatchnorm_nd(self.dim)(ch_out, use_bias=self.use_bias)
            if self.biasfree_norm
            else batchnorm_nd(self.dim)(ch_out)
        )

    def conv_block(self, ch_in: int, ch_out: int) -> nn.Module:
        conv = conv_nd(self.dim)
        layers = []

        layers.append(
            conv(
                ch_in,
                ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.use_bias,
                padding_mode=self.padding_mode,
            ),
        )
        norm = self.make_norm(ch_out)
        if norm is not None:
            layers.append(norm)
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            conv(
                ch_out,
                ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.use_bias,
                padding_mode=self.padding_mode,
            )
        )
        norm = self.make_norm(ch_out)
        if norm is not None:
            layers.append(norm)
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def up_conv(self, ch_in: int, ch_out: int) -> nn.Module:
        conv = conv_nd(self.dim)
        layers = []

        layers.append(nn.Upsample(scale_factor=2))
        layers.append(
            conv(
                ch_in,
                ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.use_bias,
                padding_mode=self.padding_mode,
            ),
        )
        norm = self.make_norm(ch_out)
        if norm is not None:
            layers.append(norm)
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)


def _bf_forward(
    bn: nn.BatchNorm2d | nn.BatchNorm3d, x: torch.Tensor, use_bias: bool
) -> torch.Tensor:
    bn._check_input_dim(x)
    y = x.transpose(0, 1)
    return_shape = y.shape
    y = y.contiguous().view(x.size(1), -1)

    if use_bias:
        mu = y.mean(dim=1)
    sigma2 = y.var(dim=1)

    if not bn.training:
        if use_bias:
            y = y - bn.running_mean.view(-1, 1)
        y = y / (bn.running_var.view(-1, 1).sqrt() + bn.eps)
    else:
        if bn.track_running_stats:
            with torch.no_grad():
                if use_bias:
                    bn.running_mean.mul_(1 - bn.momentum).add_(bn.momentum * mu)
                bn.running_var.mul_(1 - bn.momentum).add_(bn.momentum * sigma2)

        if use_bias:
            y = y - mu.view(-1, 1)
        y = y / (sigma2.view(-1, 1).sqrt() + bn.eps)

    if bn.affine:
        y = bn.weight.view(-1, 1) * y
        if use_bias:
            y = y + bn.bias.view(-1, 1)

    return y.view(return_shape).transpose(0, 1)


class BFBatchNorm2d(nn.BatchNorm2d):
    r"""
    From :footcite:t:`mohan2020robust`.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        use_bias: bool = False,
        affine: bool = True,
    ):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine)
        self.use_bias = use_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _bf_forward(self, x, self.use_bias)


class BFBatchNorm3d(nn.BatchNorm3d):
    r"""
    From :footcite:t:`mohan2020robust`.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        use_bias: bool = False,
        affine: bool = True,
    ):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine)
        self.use_bias = use_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _bf_forward(self, x, self.use_bias)


def bfbatchnorm_nd(dim: int) -> nn.Module:
    return {2: BFBatchNorm2d, 3: BFBatchNorm3d}[dim]
