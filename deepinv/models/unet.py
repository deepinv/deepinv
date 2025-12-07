from __future__ import annotations
from typing import Mapping, Any, Sequence
import warnings
import torch
import torch.nn as nn
from torch.nn.modules.module import _IncompatibleKeys
from .drunet import test_pad
from .base import Denoiser
from .utils import fix_dim, conv_nd, batchnorm_nd, maxpool_nd


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
    :param bool, str batch_norm: if False, no batchnorm applied, if ``True``, use :class:`torch.nn.BatchNorm2d` (or 3d variant, when building a 3D network),
        if ``batch_norm="biasfree"``, use ``BFBatchNorm2d`` (not yet implemented for 3D) from :footcite:t:`mohan2020robust`.
    :param int scales: Number of downsampling steps stages (network depth). Must be one of ``{2, 3, 4, 5}``.
        The number of trainable parameters increases with the scale.
    :param Sequence[int] channels_per_scale: Number of feature maps at each encoder stage (from shallow to deep). If None, defaults to ``[64, 128, 256, 512, 1024]``.
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
        scales: int = None,
        channels_per_scale: Sequence[int] = None,
        device: torch.device | str = None,
        dim: str | int = 2,
    ):
        super(UNet, self).__init__()
        self.name = "unet"

        if residual and in_channels != out_channels:  # pragma: no cover
            raise warnings.warn(
                "residual is True, but in_channels != out_channels: Falling back to non residual denoiser."
            )

        if scales is None:
            if channels_per_scale is not None:
                scales = len(channels_per_scale)
            else:
                scales = 4  # legacy default

        if scales not in (2, 3, 4, 5):  # pragma: no cover
            raise ValueError("`scales` must be one of {2, 3, 4, 5}.")

        if channels_per_scale is None:
            channels_per_scale = [64, 128, 256, 512, 1024]

        if len(channels_per_scale) < scales:  # pragma: no cover
            raise ValueError(
                f"`channels_per_scale` must have length at least `scales` "
                f"(got len={len(channels_per_scale)}, scales={scales})."
            )

        cps = channels_per_scale

        dim = fix_dim(dim)

        conv = conv_nd(dim)
        batchnorm = batchnorm_nd(dim)
        self.Maxpool = maxpool_nd(dim)(kernel_size=2, stride=2)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.residual = residual
        self.cat = cat
        self.depth = scales
        self.compact = scales  # backward compatibility, old attribute name

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

        conv_block = ConvBlock

        self.Conv1 = conv_block(ch_in=in_channels, ch_out=cps[0])
        self.Conv2 = conv_block(ch_in=cps[0], ch_out=cps[1])
        self.Conv3 = conv_block(ch_in=cps[1], ch_out=cps[2]) if self.depth > 2 else None
        self.Conv4 = conv_block(ch_in=cps[2], ch_out=cps[3]) if self.depth > 3 else None
        self.Conv5 = conv_block(ch_in=cps[3], ch_out=cps[4]) if self.depth > 4 else None

        self.Up5 = up_conv(ch_in=cps[4], ch_out=cps[3]) if self.depth > 4 else None
        self.Up_conv5 = (
            conv_block(ch_in=cps[3] * 2, ch_out=cps[3])
            if (self.depth > 4 and self.cat)
            else None
        )

        self.Up4 = up_conv(ch_in=cps[3], ch_out=cps[2]) if self.depth > 3 else None
        self.Up_conv4 = (
            conv_block(ch_in=cps[2] * 2, ch_out=cps[2])
            if (self.depth > 3 and self.cat)
            else None
        )

        self.Up3 = up_conv(ch_in=cps[2], ch_out=cps[1]) if self.depth > 2 else None
        self.Up_conv3 = (
            conv_block(ch_in=cps[1] * 2, ch_out=cps[1])
            if (self.depth > 2 and self.cat)
            else None
        )

        self.Up2 = up_conv(ch_in=cps[1], ch_out=cps[0])
        self.Up_conv2 = (
            conv_block(ch_in=cps[0] * 2, ch_out=cps[0]) if self.cat else None
        )

        self.Conv_1x1 = conv(
            in_channels=cps[0],
            out_channels=out_channels,
            bias=bias,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self._forward = self._forward_unet  # avoid breaking changes

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor, sigma=None, **kwargs) -> torch.Tensor:
        r"""
        Run the denoiser on noisy image. The noise level is not used in this denoiser.

        :param torch.Tensor x: noisy image.
        :param float sigma: noise level (not used).
        """

        factor = 2 ** (self.depth - 1)
        if x.size(2) % factor == 0 and x.size(3) % factor == 0:
            return self._forward_unet(x)
        else:
            return test_pad(self._forward_unet, x, modulo=factor)

    def _forward_unet(self, x: torch.Tensor) -> torch.Tensor:

        network_input = x

        enc_blocks = [self.Conv1, self.Conv2, self.Conv3, self.Conv4, self.Conv5]
        enc_feats = []

        for i, block in enumerate(enc_blocks):
            if block is None:
                break

            if i == 0:
                x = block(x)
            else:
                x = block(self.Maxpool(x))

            enc_feats.append(x)

        n_levels = len(enc_feats)

        up_blocks = [self.Up2, self.Up3, self.Up4, self.Up5]
        dec_blocks = [self.Up_conv2, self.Up_conv3, self.Up_conv4, self.Up_conv5]

        for level in range(n_levels - 1):
            id_decoder = (n_levels - 2) - level

            x = up_blocks[id_decoder](x)

            if self.cat:
                skip = enc_feats[-2 - level]
                x = torch.cat((skip, x), dim=1)
                x = dec_blocks[id_decoder](x)

        x = self.Conv_1x1(x)

        return (
            x + network_input
            if self.residual and self.in_channels == self.out_channels
            else x
        )

    def forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        # These are kept to avoid breaking changes
        return self._forward_unet(x)

    def forward_compact4(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_unet(x)

    def forward_compact3(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_unet(x)

    def forward_compact2(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_unet(x)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> _IncompatibleKeys:
        if not self.cat:
            # Filter out legacy Up_conv* params from old checkpoints
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if not any(
                    k.startswith(prefix)
                    for prefix in ("Up_conv2", "Up_conv3", "Up_conv4", "Up_conv5")
                )
            }

        return super().load_state_dict(state_dict, strict=strict, assign=assign)


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

class ConvBlock:
    def __init__(self, dim : int, batchnorm : bool | str, circular_padding : bool, bias: bool, ):
        (pass)