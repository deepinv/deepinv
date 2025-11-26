# Code borrowed from Kai Zhang https://github.com/cszn/DPIR/tree/master/models
from __future__ import annotations
import torch
import torch.nn as nn
from .utils import (
    get_weights_url,
    test_onesplit,
    test_pad,
    fix_dim,
    conv_nd,
    conv_transpose_nd,
    batchnorm_nd,
    instancenorm_nd,
    maxpool_nd,
    avgpool_nd,
)
from .base import Denoiser
from typing import Sequence  # noqa: F401
from collections import OrderedDict

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class DRUNet(Denoiser):
    r"""
    DRUNet denoiser network.

    The network architecture is based on the paper :footcite:t:`zhang2021plug`.
    and has a U-Net like structure, with convolutional blocks in the encoder and decoder parts.

    The network takes into account the noise level of the input image, which is encoded as an additional input channel.

    A pretrained network for (in_channels=out_channels=1 or in_channels=out_channels=3)
    can be downloaded via setting ``pretrained='download'``.

    :param int in_channels: number of channels of the input.
    :param int out_channels: number of channels of the output.
    :param Sequence[int,int,int,int] nc: number of channels per convolutional layer, the network has a fixed number of 4 scales with ``nb`` blocks per scale (default: ``[64,128,256,512]``).
    :param int nb: number of convolutional blocks per layer.
    :param str act_mode: activation mode, "R" for ReLU, "L" for LeakyReLU "E" for ELU and "s" for Softplus.
    :param str downsample_mode: Downsampling mode, "avgpool" for average pooling, "maxpool" for max pooling, and
        "strideconv" for convolution with stride 2.
    :param str upsample_mode: Upsampling mode, "convtranspose" for convolution transpose, "pixelshuffle" for pixel
        shuffling, and "upconv" for nearest neighbour upsampling with additional convolution. "pixelshuffle" is not implemented for 3D.
    :param str, None pretrained: use a pretrained network. If ``pretrained=None``, the weights will be initialized at random
        using Pytorch's default initialization. If ``pretrained='download'``, the weights will be downloaded from an
        online repository (only available for the default architecture with 3 or 1 input/output channels).
        Finally, ``pretrained`` can also be set as a path to the user's own pretrained weights.
        See :ref:`pretrained-weights <pretrained-weights>` for more details.
    :param torch.device, str device: Device to put the model on.
    :param str, int dim: Whether to build 2D or 3D network (if str, can be "2", "2d", "3D", etc.)

    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        nc: tuple[int, int, int, int] = (64, 128, 256, 512),
        nb: int = 4,
        act_mode: str = "R",
        downsample_mode: str = "strideconv",
        upsample_mode: str = "convtranspose",
        pretrained: str | None = "download",
        device: torch.device | str = None,
        dim: str | int = 2,
    ):
        super(DRUNet, self).__init__()

        dim = fix_dim(dim)

        in_channels = in_channels + 1  # accounts for the input noise channel
        self.m_head = conv(in_channels, nc[0], bias=False, mode="C", dim=dim)

        # downsample
        if downsample_mode == "avgpool":
            downsample_block = downsample_avgpool
        elif downsample_mode == "maxpool":
            downsample_block = downsample_maxpool
        elif downsample_mode == "strideconv":
            downsample_block = downsample_strideconv
        else:  # pragma: no cover
            raise NotImplementedError(
                "downsample mode [{:s}] is not found".format(downsample_mode)
            )

        self.m_down1 = sequential(
            *[
                ResBlock(nc[0], nc[0], bias=False, mode="C" + act_mode + "C", dim=dim)
                for _ in range(nb)
            ],
            downsample_block(nc[0], nc[1], bias=False, mode="2", dim=dim),
        )
        self.m_down2 = sequential(
            *[
                ResBlock(nc[1], nc[1], bias=False, mode="C" + act_mode + "C", dim=dim)
                for _ in range(nb)
            ],
            downsample_block(nc[1], nc[2], bias=False, mode="2", dim=dim),
        )
        self.m_down3 = sequential(
            *[
                ResBlock(nc[2], nc[2], bias=False, mode="C" + act_mode + "C", dim=dim)
                for _ in range(nb)
            ],
            downsample_block(nc[2], nc[3], bias=False, mode="2", dim=dim),
        )

        self.m_body = sequential(
            *[
                ResBlock(nc[3], nc[3], bias=False, mode="C" + act_mode + "C", dim=dim)
                for _ in range(nb)
            ]
        )

        # upsample
        if upsample_mode == "upconv":
            upsample_block = upsample_upconv
        elif upsample_mode == "pixelshuffle":
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == "convtranspose":
            upsample_block = upsample_convtranspose
        else:  # pragma: no cover
            raise NotImplementedError(
                "upsample mode [{:s}] is not found".format(upsample_mode)
            )

        self.m_up3 = sequential(
            upsample_block(nc[3], nc[2], bias=False, mode="2", dim=dim),
            *[
                ResBlock(nc[2], nc[2], bias=False, mode="C" + act_mode + "C", dim=dim)
                for _ in range(nb)
            ],
        )
        self.m_up2 = sequential(
            upsample_block(nc[2], nc[1], bias=False, mode="2", dim=dim),
            *[
                ResBlock(nc[1], nc[1], bias=False, mode="C" + act_mode + "C", dim=dim)
                for _ in range(nb)
            ],
        )
        self.m_up1 = sequential(
            upsample_block(nc[1], nc[0], bias=False, mode="2", dim=dim),
            *[
                ResBlock(nc[0], nc[0], bias=False, mode="C" + act_mode + "C", dim=dim)
                for _ in range(nb)
            ],
        )

        self.m_tail = conv(nc[0], out_channels, bias=False, mode="C", dim=dim)
        if pretrained is not None:
            if pretrained == "download":
                if dim == 3:  # pragma: no cover
                    raise ValueError(
                        "No 3D weights for DRUNet are available for download. Please set pretrained to None or path to your own pretrained weights."
                    )
                if in_channels == 4:
                    name = "drunet_deepinv_color_finetune_22k.pth"
                elif in_channels == 2:
                    name = "drunet_deepinv_gray_finetune_26k.pth"
                url = get_weights_url(model_name="drunet", file_name=name)
                ckpt_drunet = torch.hub.load_state_dict_from_url(
                    url, map_location=lambda storage, loc: storage, file_name=name
                )
            else:
                ckpt_drunet = torch.load(
                    pretrained, map_location=lambda storage, loc: storage
                )

            self.load_state_dict(ckpt_drunet, strict=True)
            self.eval()
        else:
            self.apply(weights_init_drunet)

        self.dim = dim

        if device is not None:
            self.to(device)

    def forward_unet(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        return x

    def forward(self, x: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
        r"""
        Run the denoiser on image with noise level :math:`\sigma`.

        :param torch.Tensor x: noisy image
        :param float, torch.Tensor sigma: noise level. If ``sigma`` is a float, it is used for all images in the batch.
            If ``sigma`` is a tensor, it must be of shape ``(batch_size,)``.
        """
        if isinstance(sigma, torch.Tensor):
            if sigma.ndim > 0:
                noise_level_map = sigma.view(x.size(0), 1, 1, 1)
                noise_level_map = noise_level_map.expand(-1, 1, x.size(2), x.size(3))
            else:
                noise_level_map = torch.ones(
                    (x.size(0), 1, *x.shape[2:]), device=x.device
                ) * sigma[None, None, None, None].to(x.device)
        else:
            noise_level_map = (
                torch.ones((x.size(0), 1, *x.shape[2:]), device=x.device) * sigma
            )
        x = torch.cat((x, noise_level_map), 1)
        shape_is_safe = all((s % 8 == 0 and s > 31) for s in x.shape[2:])
        if shape_is_safe:
            x = self.forward_unet(x)
        elif self.training or (x.size(2) < 32 or x.size(3) < 32):
            x = test_pad(self.forward_unet, x, modulo=16)
        else:
            if self.dim == 3:  # pragma: no cover
                raise NotImplementedError(
                    f"test_onesplit is not implemented yet for 3D. Please pass images with spatial shape smaller than 32, or multiple of 8 and larger than 32 to DRUNet."
                )
            x = test_onesplit(self.forward_unet, x, refield=64)
        return x


"""
Functional blocks below
"""


"""
# --------------------------------------------
# Advanced nn.Sequential
# https://github.com/xinntao/BasicSR
# --------------------------------------------
"""


def sequential(*args):
    """Advanced nn.Sequential.
    Args:
        nn.Sequential, nn.Module
    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):  # pragma: no cover
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


"""
# --------------------------------------------
# Useful blocks
# https://github.com/xinntao/BasicSR
# --------------------------------
# conv + normaliation + relu (conv)
# (PixelUnShuffle)
# (ConditionalBatchNorm2d)
# concat (ConcatBlock)
# sum (ShortcutBlock)
# resblock (ResBlock)
# Channel Attention (CA) Layer (CALayer)
# Residual Channel Attention Block (RCABlock)
# Residual Channel Attention Group (RCAGroup)
# Residual Dense Block (ResidualDenseBlock_5C)
# Residual in Residual Dense Block (RRDB)
# --------------------------------------------
"""


# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def conv(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="CBR",
    negative_slope=0.2,
    dim=2,
):
    L = []
    for t in mode:
        if t == "C":
            L.append(
                conv_nd(dim)(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
        elif t == "T":
            L.append(
                conv_transpose_nd(dim)(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
        elif t == "B":
            L.append(
                batchnorm_nd(dim)(out_channels, momentum=0.9, eps=1e-04, affine=True)
            )
        elif t == "I":
            L.append(instancenorm_nd(dim)(out_channels, affine=True))
        elif t == "R":
            L.append(nn.ReLU(inplace=True))
        elif t == "r":
            L.append(nn.ReLU(inplace=False))
        elif t == "L":
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == "l":
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == "E":
            L.append(nn.ELU(inplace=False))
        elif t == "s":
            L.append(nn.Softplus())
        elif t.isdigit():
            if dim == 3:  # pragma: no cover
                raise ValueError(
                    "PixelShuffle is not implemented for 3D models. Please use upconv or convtranspose"
                )
            L.append(nn.PixelShuffle(upscale_factor=int(t)))
        elif t == "U":
            L.append(nn.Upsample(scale_factor=2, mode="nearest"))
        elif t == "u":
            L.append(nn.Upsample(scale_factor=3, mode="nearest"))
        elif t == "v":
            L.append(nn.Upsample(scale_factor=4, mode="nearest"))
        elif t == "M":
            L.append(maxpool_nd(dim)(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == "A":
            L.append(avgpool_nd(dim)(kernel_size=kernel_size, stride=stride, padding=0))
        else:  # pragma: no cover
            raise NotImplementedError("Undefined type: ".format(t))
    return sequential(*L)


# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        mode="CRC",
        negative_slope=0.2,
        dim=2,
    ):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, "Only support in_channels==out_channels."
        if mode[0] in ["R", "L"]:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias,
            mode,
            negative_slope,
            dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.res(x)
        return x + res


"""
# --------------------------------------------
# Upsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# upsample_pixelshuffle
# upsample_upconv
# upsample_convtranspose
# --------------------------------------------
"""


# --------------------------------------------
# conv + subp (+ relu)
# --------------------------------------------
def upsample_pixelshuffle(
    in_channels=64,
    out_channels=3,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="2R",
    negative_slope=0.2,
    dim=2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    up1 = conv(
        in_channels,
        out_channels * (int(mode[0]) ** 2),
        kernel_size,
        stride,
        padding,
        bias,
        mode="C" + mode,
        negative_slope=negative_slope,
        dim=dim,
    )
    return up1


# --------------------------------------------
# nearest_upsample + conv (+ R)
# --------------------------------------------
def upsample_upconv(
    in_channels=64,
    out_channels=3,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="2R",
    negative_slope=0.2,
    dim=2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR"
    if mode[0] == "2":
        uc = "UC"
    elif mode[0] == "3":
        uc = "uC"
    elif mode[0] == "4":
        uc = "vC"
    mode = mode.replace(mode[0], uc)
    up1 = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode=mode,
        negative_slope=negative_slope,
        dim=dim,
    )
    return up1


# --------------------------------------------
# convTranspose (+ relu)
# --------------------------------------------
def upsample_convtranspose(
    in_channels=64,
    out_channels=3,
    kernel_size=2,
    stride=2,
    padding=0,
    bias=True,
    mode="2R",
    negative_slope=0.2,
    dim=2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "T")
    up1 = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode,
        negative_slope,
        dim,
    )
    return up1


"""
# --------------------------------------------
# Downsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# downsample_strideconv
# downsample_maxpool
# downsample_avgpool
# --------------------------------------------
"""


# --------------------------------------------
# strideconv (+ relu)
# --------------------------------------------
def downsample_strideconv(
    in_channels=64,
    out_channels=64,
    kernel_size=2,
    stride=2,
    padding=0,
    bias=True,
    mode="2R",
    negative_slope=0.2,
    dim=2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "C")
    down1 = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode,
        negative_slope,
        dim,
    )
    return down1


# --------------------------------------------
# maxpooling + conv (+ relu)
# --------------------------------------------
def downsample_maxpool(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="2R",
    negative_slope=0.2,
    dim=2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 3BR."
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], "MC")
    pool = conv(
        kernel_size=kernel_size_pool,
        stride=stride_pool,
        mode=mode[0],
        negative_slope=negative_slope,
        dim=dim,
    )
    pool_tail = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode=mode[1:],
        negative_slope=negative_slope,
        dim=dim,
    )
    return sequential(pool, pool_tail)


# --------------------------------------------
# averagepooling + conv (+ relu)
# --------------------------------------------
def downsample_avgpool(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="2R",
    negative_slope=0.2,
    dim=2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 3BR."
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], "AC")
    pool = conv(
        kernel_size=kernel_size_pool,
        stride=stride_pool,
        mode=mode[0],
        negative_slope=negative_slope,
        dim=dim,
    )
    pool_tail = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode=mode[1:],
        negative_slope=negative_slope,
        dim=dim,
    )
    return sequential(pool, pool_tail)


def weights_init_drunet(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.orthogonal_(m.weight.data, gain=0.2)
