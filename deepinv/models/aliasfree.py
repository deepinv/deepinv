# https://github.com/hmichaeli/alias_free_convnets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F

import math
import warnings
from typing import Tuple


from deepinv.models.drunet import test_pad
from deepinv.models.unet import BFBatchNorm2d
from deepinv.physics.blur import gaussian_blur
from deepinv.physics.functional import conv2d


class ConvNextBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mode="up_poly_per_channel",
        bias=False,
        ksize=7,
        padding_mode="circular",
        batch_norm=False,
        rotation_equivariant=False,
    ):
        super().__init__()

        if rotation_equivariant:
            ksize = 1

        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=ksize,
            groups=in_channels,
            stride=1,
            padding=ksize // 2,
            bias=bias,
            padding_mode=padding_mode,
        )
        if batch_norm:
            self.BatchNorm = (
                BFBatchNorm2d(in_channels, use_bias=bias)
                if bias
                else nn.BatchNorm2d(in_channels)
            )
        else:
            self.BatchNorm = nn.Identity()

        self.conv2 = nn.Conv2d(
            in_channels,
            4 * in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            padding_mode=padding_mode,
        )

        if mode == "up_poly_per_channel":
            self.nonlin = UpPolyActPerChannel(
                4 * in_channels,
                rotation_equivariant=rotation_equivariant,
            )
        else:
            self.nonlin = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
            4 * in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            padding_mode=padding_mode,
        )
        if in_channels != out_channels:
            self.convout = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
        else:
            self.convout = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.BatchNorm(out)
        out = self.nonlin(out)
        out = self.conv3(out)
        out = out + x
        out = self.convout(out)
        return out


def ConvBlock(
    in_channels, out_channels, mode="relu", bias=False, rotation_equivariant=False
):
    if rotation_equivariant:
        ksize = 1
    else:
        ksize = 3

    seq = nn.Sequential()
    seq.append(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=1,
            padding=ksize // 2,
            bias=bias,
            padding_mode="circular",
            groups=1,
        )
    )

    if mode == "up_poly_per_channel":
        seq.append(
            UpPolyActPerChannel(
                out_channels,
                rotation_equivariant=rotation_equivariant,
            )
        )
    elif mode == "relu":
        seq.append(nn.ReLU(inplace=True))
    else:
        raise ValueError(f"Mode {mode} not supported")

    seq.append(
        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=ksize,
            stride=1,
            padding=ksize // 2,
            bias=bias,
            padding_mode="circular",
        )
    )

    if mode == "up_poly_per_channel":
        seq.append(
            UpPolyActPerChannel(
                out_channels,
                data_format="channels_first",
                rotation_equivariant=rotation_equivariant,
            )
        )
    else:
        seq.append(nn.ReLU(inplace=True))

    return seq


# https://github.com/huggingface/pytorch-image-models/blob/f689c850b90b16a45cc119a7bc3b24375636fc63/timm/layers/weight_init.py


def create_lpf_rect(shape, cutoff=0.5):
    assert len(shape) == 2, "Only 2D low-pass filters are supported"
    lpfs = []
    for N in shape:
        cutoff_low = int((N * cutoff) // 2)
        cutoff_high = int(N - cutoff_low)
        lpf = torch.ones(N)
        lpf[cutoff_low + 1 : cutoff_high] = 0
        # if N is divisible by 4, the Nyquist frequency should be 0
        # N % 4 = 0 means the downsampeled signal is even
        if N % 4 == 0:
            lpf[cutoff_low] = 0
            lpf[cutoff_high] = 0
        lpfs.append(lpf)
    return lpfs[0][:, None] * lpfs[1][None, :]


def create_lpf_disk(shape, cutoff=0.5):
    assert len(shape) == 2, "Only 2D low-pass filters are supported"
    N, M = shape
    u = torch.linspace(-1, 1, N)
    v = torch.linspace(-1, 1, M)
    U, V = torch.meshgrid(u, v, indexing="ij")
    mask = (U**2 + V**2) < cutoff**2
    mask = mask.to(torch.float32)
    mask = torch.fft.ifftshift(mask)
    return mask


# upsample using FFT
def create_recon_rect(shape, cutoff=0.5):
    assert len(shape) == 2, "Only 2D low-pass filters are supported"
    lpfs = []
    for N in shape:
        cutoff_low = int((N * cutoff) // 2)
        cutoff_high = int(N - cutoff_low)
        lpf = torch.ones(N)
        lpf[cutoff_low + 1 : cutoff_high] = 0
        # if N is divisible by 4, the Nyquist frequency should be 0.5
        # N % 4 = 0 means the downsampeled signal is even
        # NOTE: This is the only difference with create_lpf_rect.
        if N % 4 == 0:
            lpf[cutoff_low] = 0.5
            lpf[cutoff_high] = 0.5
        lpfs.append(lpf)
    return lpfs[0][:, None] * lpfs[1][None, :]


class LPF_RFFT(nn.Module):
    """
    saves rect in first use
    """

    def __init__(
        self,
        cutoff=0.5,
        transform_mode="rfft",
        rotation_equivariant=False,
    ):
        super(LPF_RFFT, self).__init__()
        self.cutoff = cutoff
        assert transform_mode in [
            "fft",
            "rfft",
        ], f"transform_mode={transform_mode} is not supported"
        self.transform_mode = transform_mode
        self.transform = torch.fft.fft2 if transform_mode == "fft" else torch.fft.rfft2
        self.itransform = (
            (lambda x, **kwargs: torch.real(torch.fft.ifft2(x)))
            if transform_mode == "fft"
            else torch.fft.irfft2
        )
        self.rotation_equivariant = rotation_equivariant
        self.masks = {}

    def forward(self, x):
        # A tuple containing the shape of x used as a key
        # for caching the masks
        shape = x.shape[-2:]
        x_fft = self.transform(x)
        if shape not in self.masks:
            if not self.rotation_equivariant:
                mask = create_lpf_rect(shape, self.cutoff)
            else:
                mask = create_lpf_disk(shape, self.cutoff)
            N = x.shape[-1]
            mask = mask[:, : int(N / 2 + 1)] if self.transform_mode == "rfft" else mask
            self.masks[shape] = mask
        mask = self.masks[shape]
        mask = mask.to(x.device)
        x_fft *= mask
        out = self.itransform(x_fft, s=(x.shape[-2], x.shape[-1]))

        return out


class LPF_RECON_RFFT(nn.Module):
    """
    saves rect in first use
    """

    def __init__(self, cutoff=0.5, transform_mode="rfft"):
        super(LPF_RECON_RFFT, self).__init__()
        self.cutoff = cutoff
        assert transform_mode in [
            "fft",
            "rfft",
        ], f"mode={transform_mode} is not supported"
        self.transform_mode = transform_mode
        self.transform = torch.fft.fft2 if transform_mode == "fft" else torch.fft.rfft2
        self.itransform = (
            (lambda x: torch.real(torch.fft.ifft2(x)))
            if transform_mode == "fft"
            else torch.fft.irfft2
        )
        self.rect = {}

    def forward(self, x):
        # A tuple containing the shape of x used as a key
        # for caching the masks
        shape = x.shape[-2:]
        x_fft = self.transform(x)
        if shape not in self.rect:
            rect = create_recon_rect(shape, self.cutoff)
            N = x.shape[-1]
            rect = rect[:, : int(N / 2 + 1)] if self.transform_mode == "rfft" else rect
            self.rect[shape] = rect
        rect = self.rect[shape]
        rect = rect.to(x.device)
        x_fft *= rect
        out = self.itransform(x_fft)
        return out


class UpsampleRFFT(nn.Module):
    """
    input shape is unknown
    """

    def __init__(self, up=2, transform_mode="rfft"):
        super(UpsampleRFFT, self).__init__()
        self.up = up
        self.recon_filter = LPF_RECON_RFFT(cutoff=1 / up, transform_mode=transform_mode)

    def forward(self, x):
        # pad zeros
        batch_size, num_channels, in_height, in_width = x.shape
        x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
        x = torch.nn.functional.pad(x, [0, self.up - 1, 0, 0, 0, self.up - 1])
        x = x.reshape(
            [batch_size, num_channels, in_height * self.up, in_width * self.up]
        )
        x = self.recon_filter(x) * (self.up**2)
        return x


class PolyActPerChannel(nn.Module):
    def __init__(
        self,
        channels,
        init_coef=None,
        data_format="channels_first",
        in_scale=1,
        out_scale=1,
        train_scale=False,
    ):
        super(PolyActPerChannel, self).__init__()
        self.channels = channels
        if init_coef is None:
            init_coef = [0.0169394634313126, 0.5, 0.3078363963999393]
        self.deg = len(init_coef) - 1
        coef = torch.Tensor(init_coef)
        coef = coef.repeat([channels, 1])
        coef = torch.unsqueeze(torch.unsqueeze(coef, -1), -1)
        self.coef = nn.Parameter(coef, requires_grad=True)

        if train_scale:
            self.in_scale = nn.Parameter(
                torch.tensor([in_scale * 1.0]), requires_grad=True
            )
            self.out_scale = nn.Parameter(
                torch.tensor([out_scale * 1.0]), requires_grad=True
            )

        else:
            if in_scale != 1:
                self.register_buffer("in_scale", torch.tensor([in_scale * 1.0]))
            else:
                self.in_scale = None

            if out_scale != 1:
                self.register_buffer("out_scale", torch.tensor([out_scale * 1.0]))
            else:
                self.out_scale = None

        self.data_format = data_format

    def forward(self, x):
        if self.data_format == "channels_last":
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        if self.in_scale is not None:
            x = self.in_scale * x

        x = self.calc_polynomial(x)

        if self.out_scale is not None:
            x = self.out_scale * x

        if self.data_format == "channels_last":
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        return x

    def __repr__(self):
        # print_coef = self.coef.cpu().detach().numpy()
        print_in_scale = self.in_scale.cpu().detach().numpy() if self.in_scale else None
        print_out_scale = (
            self.out_scale.cpu().detach().numpy() if self.out_scale else None
        )

        return "PolyActPerChannel(channels={}, in_scale={}, out_scale={})".format(
            self.channels, print_in_scale, print_out_scale
        )

    def calc_polynomial(self, x):
        if self.deg == 2:
            # maybe this is faster?
            res = self.coef[:, 0] + self.coef[:, 1] * x + self.coef[:, 2] * (x**2)
        else:
            res = self.coef[:, 0] + self.coef[:, 1] * x
            for i in range(2, self.deg):
                res = res + self.coef[:, i] * (x**i)

        return res


class UpPolyActPerChannel(nn.Module):
    def __init__(
        self,
        channels,
        up=2,
        transform_mode="rfft",
        rotation_equivariant=False,
        **kwargs,
    ):
        super(UpPolyActPerChannel, self).__init__()
        self.up = up
        self.lpf = LPF_RFFT(
            cutoff=1 / up,
            transform_mode=transform_mode,
            rotation_equivariant=rotation_equivariant,
        )
        self.upsample = UpsampleRFFT(up, transform_mode=transform_mode)

        self.pact = PolyActPerChannel(channels, **kwargs)

    def forward(self, x):
        out = self.upsample(x)
        out = self.pact(out)
        out = self.lpf(out)
        out = out[:, :, :: self.up, :: self.up]
        return out


class circular_pad(nn.Module):
    def __init__(self, padding=(1, 1, 1, 1)):
        super(circular_pad, self).__init__()
        self.pad_sizes = padding

    def forward(self, x):
        return F.pad(x, pad=self.pad_sizes, mode="circular")


# Code originally from https://github.com/adobe/antialiased-cnns
# Copyright 2019 Adobe. All rights reserved.
## Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International


class BlurPool(nn.Module):
    def __init__(
        self,
        channels,
        pad_type="circular",
        filt_size=1,
        stride=2,
        pad_off=0,
        filter_type="ideal",
        cutoff=0.5,
        scale_l2=False,
        eps=1e-6,
        transform_mode="rfft",
        rotation_equivariant=False,
    ):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels
        self.pad_type = pad_type
        self.filter_type = filter_type
        self.scale_l2 = scale_l2
        self.eps = eps

        if filter_type == "ideal":
            self.filt = LPF_RFFT(
                cutoff=cutoff,
                transform_mode=transform_mode,
                rotation_equivariant=rotation_equivariant,
            )

        elif filter_type == "basic":
            a = self.get_rect(self.filt_size)

        if filter_type == "basic":
            filt = torch.Tensor(a[:, None] * a[None, :])
            filt = filt / torch.sum(filt)
            self.filt = Filter(filt, channels, pad_type, self.pad_sizes, scale_l2)
            if self.filt_size == 1 and self.pad_off == 0:
                self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filter_type == "ideal":
            if self.scale_l2:
                inp_norm = torch.norm(inp, p=2, dim=(-1, -2), keepdim=True)
            out = self.filt(inp)
            if self.scale_l2:
                out_norm = torch.norm(out, p=2, dim=(-1, -2), keepdim=True)
                out = out * (inp_norm / (out_norm + self.eps))
            return out[:, :, :: self.stride, :: self.stride]

        elif self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, :: self.stride, :: self.stride]
            else:
                return self.pad(inp)[:, :, :: self.stride, :: self.stride]

        else:
            return self.filt(inp)[:, :, :: self.stride, :: self.stride]

    @staticmethod
    def get_rect(filt_size):
        if filt_size == 1:
            a = np.array(
                [
                    1.0,
                ]
            )
        elif filt_size == 2:
            a = np.array([1.0, 1.0])
        elif filt_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif filt_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif filt_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif filt_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif filt_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])

        return a

    def __repr__(self):
        return (
            f"BlurPool(channels={self.channels}, pad_type={self.pad_type}, "
            f" stride={self.stride}, filter_type={self.filter_type},  filt_size={self.filt_size}, "
            f"scale_l2={self.scale_l2})"
        )


class Filter(nn.Module):
    def __init__(
        self, filt, channels, pad_type=None, pad_sizes=None, scale_l2=False, eps=1e-6
    ):
        super(Filter, self).__init__()
        self.register_buffer("filt", filt[None, None, :, :].repeat((channels, 1, 1, 1)))
        if pad_sizes is not None:
            self.pad = get_pad_layer(pad_type)(pad_sizes)
        else:
            self.pad = None
        self.scale_l2 = scale_l2
        self.eps = eps

    def forward(self, x):
        if self.scale_l2:
            inp_norm = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        if self.pad is not None:
            x = self.pad(x)
        out = F.conv2d(x, self.filt, groups=x.shape[1])
        if self.scale_l2:
            out_norm = torch.norm(out, p=2, dim=(-1, -2), keepdim=True)
            out = out * (inp_norm / (out_norm + self.eps))
        return out


def get_pad_layer(pad_type):
    if pad_type in ["refl", "reflect"]:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ["repl", "replicate"]:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == "zero":
        PadLayer = nn.ZeroPad2d
    elif pad_type == "circular":
        PadLayer = circular_pad
    else:
        print("Pad type [%s] not recognized" % pad_type)
    return PadLayer


class AliasFreeUNet(nn.Module):
    r"""
    An efficient translation-equivariant UNet denoiser

    The network is implemented using circular convolutions, filtered polynomial nonlinearities and ideal up and downsampling operators as suggested by `Karras et al. (2021) <https://doi.org/10.48550/arXiv.2106.12423>`_ and `Michaeli et al. (2023) <https://doi.org/10.1109/CVPR52729.2023.01567>`_.

    A rotation-equivariant version is also available where all convolutions are
    made with 1x1 filters, and where square-shaped masks used for anti-aliasing
    are replaced by disk-shaped masks.

    :param int in_channels: number of input channels.
    :param int out_channels: number of output channels.
    :param bool residual: if True, the output is the sum of the input and the denoised image.
    :param bool cat: if True, the network uses skip connections.
    :param int scales: number of scales in the network.
    :param str block_kind: type of block to use in the network. Options are ``ConvBlock`` and ``ConvNextBlock``.
    :param bool rotation_equivariant: if True, the network is rotation-equivariant.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        residual=True,
        cat=True,
        bias=False,
        scales=4,
        block_kind="ConvNextBlock",
        rotation_equivariant=False,
    ):
        super().__init__()
        mode = "up_poly_per_channel"

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.residual = residual
        self.cat = cat
        self.scales = scales

        self.hidden_channels = 64

        if block_kind == "ConvBlock":
            main_block = ConvBlock
        elif block_kind == "ConvNextBlock":
            main_block = ConvNextBlock
        else:
            raise NotImplementedError()

        out_ch = self.hidden_channels

        self.conv_in = ConvBlock(
            in_channels=in_channels,
            out_channels=out_ch,
            bias=bias,
            mode="up_poly_per_channel",
            rotation_equivariant=rotation_equivariant,
        )

        for i in range(1, scales):
            in_ch = out_ch
            out_ch = in_ch * 2
            setattr(
                self,
                f"Downsample{i}",
                BlurPool(channels=in_ch, rotation_equivariant=rotation_equivariant),
            )
            setattr(
                self,
                f"DownBlock{i}",
                main_block(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    bias=bias,
                    mode="up_poly_per_channel",
                    rotation_equivariant=rotation_equivariant,
                ),
            )

        for i in range(scales - 1, 0, -1):
            in_ch = out_ch
            out_ch = in_ch // 2
            setattr(
                self,
                f"Upsample{i}",
                UpsampleRFFT(),
            )
            setattr(
                self,
                f"UpBlock{i}",
                main_block(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    bias=bias,
                    mode="up_poly_per_channel",
                    rotation_equivariant=rotation_equivariant,
                ),
            )
            setattr(
                self,
                f"CatBlock{i}",
                ConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    bias=bias,
                    mode="up_poly_per_channel",
                    rotation_equivariant=rotation_equivariant,
                ),
            )

        self.conv_out = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            bias=bias,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x, sigma=None):
        r"""
        Run the denoiser on noisy image. The noise level is not used in this denoiser.

        :param torch.Tensor x: noisy image.
        :param float sigma: noise level (not used).
        """
        factor = 2 ** (self.scales - 1)
        if x.size(2) % factor == 0 and x.size(3) % factor == 0:
            return self._forward(x)
        else:
            return test_pad(self._forward, x, modulo=factor)

    def _forward(self, x):
        cat_list = []

        m = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        x = x - m

        out = self.conv_in(x)

        cat_list.append(out)
        for i in range(1, self.scales):
            out = getattr(self, f"Downsample{i}")(out)
            out = getattr(self, f"DownBlock{i}")(out)
            if self.cat and i < self.scales - 1:
                cat_list.append(out)

        for i in range(self.scales - 1, 0, -1):
            out = getattr(self, f"Upsample{i}")(out)
            out = getattr(self, f"UpBlock{i}")(out)
            if self.cat:
                out = torch.cat((cat_list.pop(), out), dim=1)
                out = getattr(self, f"CatBlock{i}")(out)

        out = self.conv_out(out)

        if self.residual and self.in_channels == self.out_channels:
            out = out + x

        out = out + m

        return out
