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


# Adapted from
# https://github.com/hmichaeli/alias_free_convnets/blob/9018d9858b2db44cac329c7844cbd0d873519952/models/layer_norm.py#L50
class LayerNorm_AF(nn.Module):
    """
    Alias-Free Layer Normalization

    From `"Alias-Free Convnets: Fractional Shift Invariance via Polynomial Activations" by Michaeli et al. <https://doi.org/10.48550/arXiv.2303.08085>`_

    :param int in_channels: number of input channels.
    :param int out_channels: number of output channels.
    :param bool residual: if True, the output is the sum of the input and the denoised image.
    :param bool cat: if True, the network uses skip connections.
    :param int scales: number of scales in the network.
    :param bool rotation_equivariant: if True, the network is rotation-equivariant.

    :param Union[int, list, torch.Size] normalized_shape: Input shape from an expected input of size.
    :param float eps: A value added to the denominator for numerical stability. Default: 1e-6.
    :param bool bias: If set to False, the layer will not learn an additive bias. Default: True.
    """

    def __init__(self, normalized_shape, eps=1e-6, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.bias = None
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
        self.u_dims = (1, 2, 3)
        self.s_dims = (1, 2, 3)

    def forward(self, x):
        """
        Forward pass for layer normalization.

        :param torch.Tensor x: Input tensor
        """
        u = x.mean(self.u_dims, keepdim=True)
        s = (x - u).pow(2).mean(self.s_dims, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x
        if self.bias is not None:
            x = x + self.bias[:, None, None]
        return x


# Adapted from
# https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py#L15
class ConvNextBlock(nn.Module):
    """
    ConvNextBlock implements a block of the ConvNeXt architecture with optional normalization and non-linearity.

    From `"A ConvNet for the 2020s" by Liu et al. <https://arxiv.org/abs/2201.03545>`_

    :param int in_channels: Number of input channels.
    :param int out_channels: Number of output channels.
    :param str mode: Type of non-linearity to use. Default is "up_poly_per_channel".
    :param bool bias: If True, adds a learnable bias to the convolutional layers. Default is False.
    :param int ksize: Kernel size for the depthwise convolution. Default is 7.
    :param str padding_mode: Padding mode for the convolutional layers. Default is "circular".
    :param str norm: Type of normalization to use. Options are "BatchNorm2d", "LayerNorm", "LayerNorm_AF", or "Identity". Default is "LayerNorm_AF".
    :param bool rotation_equivariant: If True, makes the block rotation equivariant by setting kernel size to 1. Default is False. For more details, see `"Alias-Free Generative Adversarial Networks" by Karras et al. <https://doi.org/10.48550/arXiv.2106.12423>`_.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        mode="up_poly_per_channel",
        bias=False,
        ksize=7,
        padding_mode="circular",
        norm="LayerNorm_AF",
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

        self.conv2 = nn.Conv2d(
            in_channels,
            4 * in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            padding_mode=padding_mode,
        )

        if norm == "BatchNorm2d":
            self.norm = (
                BFBatchNorm2d(4 * in_channels, use_bias=bias)
                if bias
                else nn.BatchNorm2d(4 * in_channels)
            )
            self.norm_order = "channels_first"
        elif norm == "LayerNorm":
            self.norm = nn.LayerNorm(4 * in_channels, bias=bias)
            self.norm_order = "channels_last"
        elif norm == "LayerNorm_AF":
            self.norm = LayerNorm_AF(4 * in_channels, bias=bias)
            self.norm_order = "channels_first"
        else:
            self.norm = nn.Identity()
            self.norm_order = "channels_first"

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
        """
        Forward pass for the ConvNextBlock.

        :param torch.Tensor x: Input tensor
        """
        out = self.conv1(x)
        out = self.conv2(out)
        if self.norm_order == "channels_first":
            out = self.norm(out)
        elif self.norm_order == "channels_last":
            out = out.permute(0, 2, 3, 1)
            out = self.norm(out)
            out = out.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"norm_order={self.norm_order} is not supported")
        out = self.nonlin(out)
        out = self.conv3(out)
        out = out + x
        out = self.convout(out)
        return out


# Adapted from
# https://github.com/hmichaeli/alias_free_convnets/blob/9018d9858b2db44cac329c7844cbd0d873519952/models/ideal_lpf.py#L5
def create_lpf_rect(shape, cutoff=0.5):
    """
    Creates a rectangular low-pass filter (LPF) for 2D signals.

    :param Tuple[int] shape: The shape of the filter, should be a tuple of two integers.
    :param float cutoff: The cutoff frequency as a fraction of the Nyquist frequency. Default is 0.5.
    :return: A 2D tensor representing the low-pass filter.
    """
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


# TODO: merge with create_lpf_rect
def create_lpf_disk(shape, cutoff=0.5):
    """
    Create a 2D low-pass filter with a disk-shaped cutoff.

    :param shape: Tuple[int, int], the shape of the filter (N, M).
    :param cutoff: float, the cutoff frequency for the low-pass filter. Default is 0.5.
    :return: torch.Tensor, a 2D tensor representing the low-pass filter.
    """
    assert len(shape) == 2, "Only 2D low-pass filters are supported"
    N, M = shape
    u = torch.linspace(-1, 1, N)
    v = torch.linspace(-1, 1, M)
    U, V = torch.meshgrid(u, v, indexing="ij")
    mask = (U**2 + V**2) < cutoff**2
    mask = mask.to(torch.float32)
    mask = torch.fft.ifftshift(mask)
    return mask


# Adapted from
# https://github.com/hmichaeli/alias_free_convnets/blob/9018d9858b2db44cac329c7844cbd0d873519952/models/ideal_lpf.py#L31
def create_recon_rect(shape, cutoff=0.5):
    """
    Create a 2D rectangular reconstruction filter.

    :param shape: Tuple[int, int], the shape of the filter (N, M).
    :param cutoff: float, the cutoff frequency for the reconstruction filter. Default is 0.5.
    :return: torch.Tensor, a 2D tensor representing the reconstruction filter.
    """
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


# https://github.com/hmichaeli/alias_free_convnets/blob/9018d9858b2db44cac329c7844cbd0d873519952/models/ideal_lpf.py#L45
class LPF_RFFT(nn.Module):
    """
    A module that applies a low-pass filter in the frequency domain using fast fourier transforms (FFTs).

    :param cutoff: float, the cutoff frequency for the low-pass filter. Default is 0.5.
    :param transform_mode: str, the transform mode to use ('fft' or 'rfft'). Default is 'rfft'.
    :param rotation_equivariant: bool, whether to use a rotation-equivariant filter. Default is False. For more details, see `"Alias-Free Generative Adversarial Networks" by Karras et al. <https://doi.org/10.48550/arXiv.2106.12423>`_.
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
        """
        Apply the low-pass filter to the input tensor.

        :param x: torch.Tensor, the input tensor to be filtered.
        :return: torch.Tensor, the filtered output tensor.
        """
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


# Adapted from
# https://github.com/hmichaeli/alias_free_convnets/blob/9018d9858b2db44cac329c7844cbd0d873519952/models/ideal_lpf.py#L73
class LPF_RECON_RFFT(nn.Module):
    """
    A module that applies a rectangular reconstruction filter in the frequency domain using FFT or RFFT.

    :param cutoff: float, the cutoff frequency for the reconstruction filter. Default is 0.5.
    :param transform_mode: str, the transform mode to use ('fft' or 'rfft'). Default is 'rfft'.
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
        """
        Apply the rectangular reconstruction filter to the input tensor.

        :param x: torch.Tensor, the input tensor to be filtered.
        :return: torch.Tensor, the filtered output tensor.
        """
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


# Adapted from
# https://github.com/hmichaeli/alias_free_convnets/blob/9018d9858b2db44cac329c7844cbd0d873519952/models/ideal_lpf.py#L99
class UpsampleRFFT(nn.Module):
    """
    A module that upsamples an input tensor in the frequency domain using FFT or RFFT.

    :param up: int, the upsampling factor. Default is 2.
    :param transform_mode: str, the transform mode to use ('fft' or 'rfft'). Default is 'rfft'.
    """

    def __init__(self, up=2, transform_mode="rfft"):
        super(UpsampleRFFT, self).__init__()
        self.up = up
        self.recon_filter = LPF_RECON_RFFT(cutoff=1 / up, transform_mode=transform_mode)

    def forward(self, x):
        """
        Upsample the input tensor.

        :param x: torch.Tensor, the input tensor to be upsampled.
        :return: torch.Tensor, the upsampled output tensor.
        """
        # pad zeros
        batch_size, num_channels, in_height, in_width = x.shape
        x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
        x = torch.nn.functional.pad(x, [0, self.up - 1, 0, 0, 0, self.up - 1])
        x = x.reshape(
            [batch_size, num_channels, in_height * self.up, in_width * self.up]
        )
        x = self.recon_filter(x) * (self.up**2)
        return x


# Adapted from
# https://github.com/hmichaeli/alias_free_convnets/blob/9018d9858b2db44cac329c7844cbd0d873519952/models/activation.py#L117
class PolyActPerChannel(nn.Module):
    """
    A module that applies a polynomial activation function per channel.

    From `"Alias-Free Convnets: Fractional Shift Invariance via Polynomial Activations" by Michaeli et al. <https://doi.org/10.48550/arXiv.2303.08085>`_

    :param channels: int, the number of channels.
    """

    def __init__(self, channels):
        super(PolyActPerChannel, self).__init__()
        self.channels = channels
        init_coef = [0.0169394634313126, 0.5, 0.3078363963999393]
        self.deg = len(init_coef) - 1
        coef = torch.Tensor(init_coef)
        coef = coef.repeat([channels, 1])
        coef = torch.unsqueeze(torch.unsqueeze(coef, -1), -1)
        self.coef = nn.Parameter(coef, requires_grad=True)

    def forward(self, x):
        """
        Apply the polynomial activation function to the input tensor.

        :param x: torch.Tensor, the input tensor.
        :return: torch.Tensor, the output tensor after applying the activation function.
        """
        if self.deg == 2:
            # maybe this is faster?
            res = self.coef[:, 0] + self.coef[:, 1] * x + self.coef[:, 2] * (x**2)
        else:
            res = self.coef[:, 0] + self.coef[:, 1] * x
            for i in range(2, self.deg):
                res = res + self.coef[:, i] * (x**i)
        return res

    def __repr__(self):
        return "PolyActPerChannel(channels={})".format(self.channels)

# Adapted from
# https://github.com/hmichaeli/alias_free_convnets/blob/9018d9858b2db44cac329c7844cbd0d873519952/models/activation.py#L187
class UpPolyActPerChannel(nn.Module):
    """
    A module that applies an upsampled polynomial activation function per channel.

    From `"Alias-Free Convnets: Fractional Shift Invariance via Polynomial Activations" by Michaeli et al. <https://doi.org/10.48550/arXiv.2303.08085>`_

    :param channels: int, the number of channels.
    :param up: int, the upsampling factor. Default is 2.
    :param transform_mode: str, the transform mode to use ('fft' or 'rfft'). Default is 'rfft'.
    :param rotation_equivariant: bool, whether to use a rotation-equivariant filter. Default is False.
    :param kwargs: additional keyword arguments for the polynomial activation function.
    """
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
        """
        Apply the upsampled polynomial activation function per channel to the input tensor.

        :param x: torch.Tensor, the input tensor.
        :return: torch.Tensor, the output tensor after applying the activation function.
        """
        out = self.upsample(x)
        out = self.pact(out)
        out = self.lpf(out)
        out = out[:, :, :: self.up, :: self.up]
        return out


# Adapted from
# https://github.com/adobe/antialiased-cnns/blob/b27a34a26f3ab039113d44d83c54d0428598ac9c/antialiased_cnns/blurpool.py#L13
class BlurPool(nn.Module):
    """
    A module that applies a blur pooling operation.

    From `"Alias-Free Generative Adversarial Networks" by Karras et al. <https://doi.org/10.48550/arXiv.2106.12423>`_

    :param channels: int, the number of channels.
    :param pad_type: str, the type of padding to use. Default is "circular".
    :param filt_size: int, the size of the filter. Default is 1.
    :param stride: int, the stride of the pooling operation. Default is 2.
    :param pad_off: int, the padding offset. Default is 0.
    :param cutoff: float, the cutoff frequency for the low-pass filter. Default is 0.5.
    :param scale_l2: bool, whether to scale the output by the L2 norm of the input. Default is False.
    :param eps: float, a small value to avoid division by zero. Default is 1e-6.
    :param transform_mode: str, the transform mode to use ('fft' or 'rfft'). Default is 'rfft'.
    :param rotation_equivariant: bool, whether to use a rotation-equivariant filter. Default is False.
    """

    def __init__(
        self,
        channels,
        pad_type="circular",
        filt_size=1,
        stride=2,
        pad_off=0,
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
        self.scale_l2 = scale_l2
        self.eps = eps

        self.filt = LPF_RFFT(
            cutoff=cutoff,
            transform_mode=transform_mode,
            rotation_equivariant=rotation_equivariant,
        )

    def forward(self, inp):
        """
        Apply the blur pooling operation to the input tensor.

        :param inp: torch.Tensor, the input tensor.
        :return: torch.Tensor, the output tensor after applying the blur pooling operation.
        """
        if self.scale_l2:
            inp_norm = torch.norm(inp, p=2, dim=(-1, -2), keepdim=True)
        out = self.filt(inp)
        if self.scale_l2:
            out_norm = torch.norm(out, p=2, dim=(-1, -2), keepdim=True)
            out = out * (inp_norm / (out_norm + self.eps))
        return out[:, :, :: self.stride, :: self.stride]

    def __repr__(self):
        return (
            f"BlurPool(channels={self.channels}, pad_type={self.pad_type}, "
            f" stride={self.stride}, filt_size={self.filt_size}, "
            f"scale_l2={self.scale_l2})"
        )


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
        rotation_equivariant=False,
        hidden_channels=64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.residual = residual
        self.cat = cat
        self.scales = scales

        out_ch = self.hidden_channels

        self.conv_in = ConvNextBlock(
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
                ConvNextBlock(
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
                ConvNextBlock(
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
                ConvNextBlock(
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
