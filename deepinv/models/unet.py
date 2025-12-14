from __future__ import annotations
from typing import Any, Sequence
from dataclasses import dataclass
import re, warnings

import torch
import torch.nn as nn
from .drunet import test_pad
from .base import Denoiser
from .utils import fix_dim, conv_nd, batchnorm_nd, maxpool_nd


class UNet(Denoiser):
    r"""
    U-Net convolutional denoiser.

    This network is a fully convolutional denoiser based on the U-Net architecture. The depth of the network is
    controlled by ``scales``, which sets the number of encoder downsampling stages (and corresponding decoder
    upsampling stages). The width of each stage is controlled by ``channels_per_scale``, which gives the number of
    feature maps from shallow to deep.

    If ``scales`` is not given, it is inferred from ``channels_per_scale`` (its length). If both are omitted, defaults to
    configuration with ``scales=4`` and ``channels_per_scale=[64, 128, 256, 512, 1024]``. When
    ``scales`` is specified explicitly together with ``channels_per_scale``, only the first ``scales`` entries
    of ``channels_per_scale`` are used; its length must be at least ``scales``. The number of trainable parameters
    increases with both ``scales`` and the values in ``channels_per_scale``.

    .. warning::
        When using the bias-free batch norm ``BFBatchNorm2d`` via ``batch_norm="biasfree"``, NaNs may be encountered
        during training, causing the whole training procedure to fail.

    :param int in_channels: input image channels
    :param int out_channels: output image channels
    :param bool residual: use a skip-connection between output and output.
    :param bool circular_padding: circular padding for the convolutional layers.
    :param bool cat: use skip-connections between intermediate levels.
    :param bool bias: use learnable biases.
    :param bool, str batch_norm: if False, no batchnorm applied, if ``True``, use batch normalization,
        if ``batch_norm="biasfree"``, use the biasfree batchnorm from :footcite:t:`mohan2020robust`.
    :param int scales: Number of downsampling stages.
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
        ):
            raise RuntimeError(
                f"Both scales and channels_per_scale was passed, but scales ({scales}) does not match the length of channels_per_scale ({len(channels_per_scale)})"
            )

        if scales is None:
            if channels_per_scale is not None:
                scales = len(channels_per_scale)
            else:
                scales = 4  # legacy default

        if channels_per_scale is None:
            channels_per_scale = [64 * (2**k) for k in range(scales)]

        dim = fix_dim(dim)

        conv = conv_nd(dim)
        self.Maxpool = maxpool_nd(dim)(kernel_size=2, stride=2)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.residual = residual
        self.cat = cat
        self.compact = scales  # backward compatibility, old attribute name

        biasfree = batch_norm == "biasfree"
        if biasfree and dim == 3:  # pragma: no cover
            raise NotImplementedError("Bias-free batchnorm is not implemented for 3D")

        b = _Blocks(
            dim=dim,
            circular_padding=circular_padding,
            biasfree_norm=biasfree,
            use_bias=bias,
            norm=batch_norm,
        )

        cps = channels_per_scale  # shorthand

        self.enc_blocks = nn.ModuleList()

        for i in range(scales):
            ch_in = in_channels if i == 0 else cps[i - 1]
            ch_out = cps[i]
            self.enc_blocks.append(b.conv_block(ch_in=ch_in, ch_out=ch_out))

        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(scales - 1):
            ch_in = cps[-1 - i]
            ch_out = cps[-2 - i]
            self.up_blocks.append(b.up_conv(ch_in=ch_in, ch_out=ch_out))
            if self.cat:
                self.dec_blocks.append(b.conv_block(ch_in=ch_out * 2, ch_out=ch_out))

        self.Conv_1x1 = conv(
            in_channels=cps[0],
            out_channels=out_channels,
            bias=bias,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor, sigma: Any = None, **kwargs) -> torch.Tensor:
        r"""
        Run the denoiser on noisy image. The noise level is not used in this denoiser.

        :param torch.Tensor x: noisy image.
        :param float sigma: noise level (not used).
        """

        factor = 2 ** (len(self.up_blocks))
        if x.size(2) % factor == 0 and x.size(3) % factor == 0:
            return self._forward(x)
        else:
            return test_pad(self._forward, x, modulo=factor)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        network_input = x

        enc_feats = []
        for i, block in enumerate(self.enc_blocks):
            x = block(x) if i == 0 else block(self.Maxpool(x))
            enc_feats.append(x)

        for i in range(len(self.up_blocks)):
            x = self.up_blocks[i](x)
            if self.cat:
                skip = enc_feats[-2 - i]
                x = torch.cat((skip, x), dim=1)
                x = self.dec_blocks[i](x)

        x = self.Conv_1x1(x)

        return (
            x + network_input
            if self.residual and self.in_channels == self.out_channels
            else x
        )

    def forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        # These are kept to avoid breaking changes
        return self._forward(x)

    def forward_compact4(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)

    def forward_compact3(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)

    def forward_compact2(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Backwards compatibility: translate legacy checkpoints that used individual attributes into current ModuleList scheme.
        to_add = {}
        to_del = []
        for k, v in list(state_dict.items()):
            if not k.startswith(prefix):
                continue
            local_k = k[len(prefix) :]

            m = re.match(r"^Conv(\d+)\.(.*)$", local_k)
            if m:
                idx = int(m.group(1)) - 1
                to_add[prefix + f"enc_blocks.{idx}.{m.group(2)}"] = v
                to_del.append(k)
                continue

            m = re.match(r"^Up(\d+)\.(.*)$", local_k)
            if m:
                idx = list(range(len(self.up_blocks)))[-int(m.group(1)) + 1]
                to_add[prefix + f"up_blocks.{idx}.{m.group(2)}"] = v
                to_del.append(k)
                continue

            m = re.match(r"^Up_conv(\d+)\.(.*)$", local_k)
            if m:
                if getattr(self, "cat", False):
                    idx = list(range(len(self.up_blocks)))[-int(m.group(1)) + 1]
                    to_add[prefix + f"dec_blocks.{idx}.{m.group(2)}"] = v
                    to_del.append(k)
                else:
                    to_del.append(k)
        state_dict.update(to_add)
        for k in to_del:
            state_dict.pop(k, None)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


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
            BFBatchNorm2d(ch_out, use_bias=self.use_bias)
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
        super(BFBatchNorm2d, self).__init__(num_features, eps, momentum)
        self.use_bias = use_bias
        self.affine = affine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        y = x.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        if self.use_bias:
            mu = y.mean(dim=1)
        sigma2 = y.var(dim=1)
        if not self.training:
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
