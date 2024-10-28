# https://github.com/hmichaeli/alias_free_convnets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out

from doctest import UnexpectedException
import math
import warnings
from typing import Tuple

# https://github.com/huggingface/pytorch-image-models/blob/f689c850b90b16a45cc119a7bc3b24375636fc63/timm/layers/grid.py


def ndgrid(*tensors) -> Tuple[torch.Tensor, ...]:
    """generate N-D grid in dimension order.

    The ndgrid function is like meshgrid except that the order of the first two input arguments are switched.

    That is, the statement
    [X1,X2,X3] = ndgrid(x1,x2,x3)

    produces the same result as

    [X2,X1,X3] = meshgrid(x2,x1,x3)

    This naming is based on MATLAB, the purpose is to avoid confusion due to torch's change to make
    torch.meshgrid behaviour move from matching ndgrid ('ij') indexing to numpy meshgrid defaults of ('xy').

    """
    try:
        return torch.meshgrid(*tensors, indexing="ij")
    except TypeError:
        # old PyTorch < 1.10 will follow this path as it does not have indexing arg,
        # the old behaviour of meshgrid was 'ij'
        return torch.meshgrid(*tensors)


def meshgrid(*tensors) -> Tuple[torch.Tensor, ...]:
    """generate N-D grid in spatial dim order.

    The meshgrid function is similar to ndgrid except that the order of the
    first two input and output arguments is switched.

    That is, the statement

    [X,Y,Z] = meshgrid(x,y,z)
    produces the same result as

    [Y,X,Z] = ndgrid(y,x,z)
    Because of this, meshgrid is better suited to problems in two- or three-dimensional Cartesian space,
    while ndgrid is better suited to multidimensional problems that aren't spatially based.
    """

    # NOTE: this will throw in PyTorch < 1.10 as meshgrid did not support indexing arg or have
    # capability of generating grid in xy order before then.
    return torch.meshgrid(*tensors, indexing="xy")


# https://github.com/huggingface/pytorch-image-models/blob/f689c850b90b16a45cc119a7bc3b24375636fc63/timm/layers/drop.py

""" DropBlock, DropPath

PyTorch implementations of DropBlock and DropPath (Stochastic Depth) regularization layers.

Papers:
DropBlock: A regularization method for convolutional networks (https://arxiv.org/abs/1810.12890)

Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)

Code:
DropBlock impl inspired by two Tensorflow impl that I liked:
 - https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_model.py#L74
 - https://github.com/clovaai/assembled-cnn/blob/master/nets/blocks.py

Hacked together by / Copyright 2020 Ross Wightman
"""


def drop_block_2d(
    x,
    drop_prob: float = 0.1,
    block_size: int = 7,
    gamma_scale: float = 1.0,
    with_noise: bool = False,
    inplace: bool = False,
    batchwise: bool = False,
):
    """DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. This layer has been tested on a few training
    runs with success, but needs further validation and possibly optimization for lower runtime impact.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    # seed_drop_rate, the gamma parameter
    gamma = (
        gamma_scale
        * drop_prob
        * total_size
        / clipped_block_size**2
        / ((W - block_size + 1) * (H - block_size + 1))
    )

    # Forces the block to be inside the feature map.
    w_i, h_i = ndgrid(
        torch.arange(W, device=x.device), torch.arange(H, device=x.device)
    )
    valid_block = (
        (w_i >= clipped_block_size // 2) & (w_i < W - (clipped_block_size - 1) // 2)
    ) & ((h_i >= clipped_block_size // 2) & (h_i < H - (clipped_block_size - 1) // 2))
    valid_block = torch.reshape(valid_block, (1, 1, H, W)).to(dtype=x.dtype)

    if batchwise:
        # one mask for whole batch, quite a bit faster
        uniform_noise = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device)
    else:
        uniform_noise = torch.rand_like(x)
    block_mask = ((2 - gamma - valid_block + uniform_noise) >= 1).to(dtype=x.dtype)
    block_mask = -F.max_pool2d(
        -block_mask,
        kernel_size=clipped_block_size,  # block_size,
        stride=1,
        padding=clipped_block_size // 2,
    )

    if with_noise:
        normal_noise = (
            torch.randn((1, C, H, W), dtype=x.dtype, device=x.device)
            if batchwise
            else torch.randn_like(x)
        )
        if inplace:
            x.mul_(block_mask).add_(normal_noise * (1 - block_mask))
        else:
            x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = (
            block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-7)
        ).to(x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


def drop_block_fast_2d(
    x: torch.Tensor,
    drop_prob: float = 0.1,
    block_size: int = 7,
    gamma_scale: float = 1.0,
    with_noise: bool = False,
    inplace: bool = False,
):
    """DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. Simplied from above without concern for valid
    block mask at edges.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = (
        gamma_scale
        * drop_prob
        * total_size
        / clipped_block_size**2
        / ((W - block_size + 1) * (H - block_size + 1))
    )

    block_mask = torch.empty_like(x).bernoulli_(gamma)
    block_mask = F.max_pool2d(
        block_mask.to(x.dtype),
        kernel_size=clipped_block_size,
        stride=1,
        padding=clipped_block_size // 2,
    )

    if with_noise:
        normal_noise = torch.empty_like(x).normal_()
        if inplace:
            x.mul_(1.0 - block_mask).add_(normal_noise * block_mask)
        else:
            x = x * (1.0 - block_mask) + normal_noise * block_mask
    else:
        block_mask = 1 - block_mask
        normalize_scale = (
            block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-6)
        ).to(dtype=x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


class DropBlock2d(nn.Module):
    """DropBlock. See https://arxiv.org/pdf/1810.12890.pdf"""

    def __init__(
        self,
        drop_prob: float = 0.1,
        block_size: int = 7,
        gamma_scale: float = 1.0,
        with_noise: bool = False,
        inplace: bool = False,
        batchwise: bool = False,
        fast: bool = True,
    ):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.batchwise = batchwise
        self.fast = fast  # FIXME finish comparisons of fast vs not

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        if self.fast:
            return drop_block_fast_2d(
                x,
                self.drop_prob,
                self.block_size,
                self.gamma_scale,
                self.with_noise,
                self.inplace,
            )
        else:
            return drop_block_2d(
                x,
                self.drop_prob,
                self.block_size,
                self.gamma_scale,
                self.with_noise,
                self.inplace,
                self.batchwise,
            )


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


# https://github.com/huggingface/pytorch-image-models/blob/f689c850b90b16a45cc119a7bc3b24375636fc63/timm/layers/weight_init.py


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
"""
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


def create_lpf_rect(N, cutoff=0.5):
    cutoff_low = int((N * cutoff) // 2)
    cutoff_high = int(N - cutoff_low)
    rect_1d = torch.ones(N)
    rect_1d[cutoff_low + 1 : cutoff_high] = 0
    if N % 4 == 0:
        # if N is divides by 4, nyquist freq should be 0
        # N % 4 =0 means the downsampeled signal is even
        rect_1d[cutoff_low] = 0
        rect_1d[cutoff_high] = 0

    rect_2d = rect_1d[:, None] * rect_1d[None, :]
    return rect_2d


def create_fixed_lpf_rect(N, size):
    rect_1d = torch.ones(N)
    if size < N:
        cutoff_low = size // 2
        cutoff_high = int(N - cutoff_low)
        rect_1d[cutoff_low + 1 : cutoff_high] = 0
    rect_2d = rect_1d[:, None] * rect_1d[None, :]
    return rect_2d


# upsample using FFT
def create_recon_rect(N, cutoff=0.5):
    cutoff_low = int((N * cutoff) // 2)
    cutoff_high = int(N - cutoff_low)
    rect_1d = torch.ones(N)
    rect_1d[cutoff_low + 1 : cutoff_high] = 0
    if N % 4 == 0:
        # if N is divides by 4, nyquist freq should be 0.5
        # N % 4 =0 means the downsampeled signal is even
        rect_1d[cutoff_low] = 0.5
        rect_1d[cutoff_high] = 0.5
    rect_2d = rect_1d[:, None] * rect_1d[None, :]
    return rect_2d


class LPF_RFFT(nn.Module):
    """
    saves rect in first use
    """

    def __init__(self, cutoff=0.5, transform_mode="rfft", fixed_size=None):
        super(LPF_RFFT, self).__init__()
        self.cutoff = cutoff
        self.fixed_size = fixed_size
        assert transform_mode in [
            "fft",
            "rfft",
        ], f"transform_mode={transform_mode} is not supported"
        self.transform_mode = transform_mode
        self.transform = torch.fft.fft2 if transform_mode == "fft" else torch.fft.rfft2
        self.itransform = (
            (lambda x: torch.real(torch.fft.ifft2(x)))
            if transform_mode == "fft"
            else torch.fft.irfft2
        )

    def forward(self, x):
        x_fft = self.transform(x)
        if not hasattr(self, "rect"):
            N = x.shape[-1]
            rect = (
                create_lpf_rect(N, self.cutoff)
                if not self.fixed_size
                else create_fixed_lpf_rect(N, self.fixed_size)
            )
            rect = rect[:, : int(N / 2 + 1)] if self.transform_mode == "rfft" else rect
            self.register_buffer("rect", rect)
            self.to(x.device)
        x_fft *= self.rect
        # out = self.itransform(x_fft) # support odd inputs - need to specify signal size (irfft default is even)
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

    def forward(self, x):
        x_fft = self.transform(x)
        if not hasattr(self, "rect"):
            N = x.shape[-1]
            rect = create_recon_rect(N, self.cutoff)
            rect = rect[:, : int(N / 2 + 1)] if self.transform_mode == "rfft" else rect
            self.register_buffer("rect", rect)
            self.to(x.device)
        x_fft *= self.rect
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


def subpixel_shift(images, up=2, shift_x=1, shift_y=1, up_method="ideal"):
    """
    effective fractional shift is (shift_x / up, shift_y / up)
    """

    assert up_method == "ideal", 'Only "ideal" interpolation kenrel is supported'
    up_layer = UpsampleRFFT(up=up).to(images.device)
    up_img_batch = up_layer(images)
    # img_batch_1 = up_img_batch[:, :, 1::2, 1::2]
    img_batch_1 = torch.roll(up_img_batch, shifts=(-shift_x, -shift_y), dims=(2, 3))[
        :, :, ::up, ::up
    ]
    return img_batch_1


def get_activation(activation, channels=None, data_format="channels_first", **kwargs):
    if data_format not in ["channels_last", "channels_first"]:
        raise NotImplementedError

    if activation == "relu":
        return nn.ReLU()

    elif activation == "gelu":
        return nn.GELU()

    elif activation == "poly":
        return PolyAct(**kwargs)

    elif activation == "up_poly":
        return UpPolyAct(data_format=data_format, **kwargs)

    elif activation == "poly_per_channel":
        return PolyActPerChannel(channels, data_format=data_format, **kwargs)

    elif activation == "up_poly_per_channel":
        return UpPolyActPerChannel(channels, data_format=data_format, **kwargs)

    elif activation == "lpf_poly_per_channel":
        return LPFPolyActPerChannel(
            channels=channels, data_format=data_format, **kwargs
        )

    else:
        assert False, "Un implemented activation {}".format(activation)


class UpPolyAct(nn.Module):
    def __init__(self, up=2, data_format="channels_first", **kwargs):
        super(UpPolyAct, self).__init__()
        self.up = up
        # self.lpf = IdealLPF(cutoff=1/up)
        # self.upsample = IdealUpsample(up)
        self.lpf = LPF_RFFT(cutoff=1 / up)
        self.upsample = UpsampleRFFT(up)
        self.data_format = data_format

        self.pact = PolyAct(**kwargs)

    def forward(self, x):
        if self.data_format == "channels_last":
            inp = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        out = self.upsample(inp)
        out = self.pact(out)
        out = self.lpf(out)
        out = out[:, :, :: self.up, :: self.up]

        if self.data_format == "channels_last":
            out = out.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        return out


class PolyAct(nn.Module):
    def __init__(
        self,
        trainable=False,
        init_coef=None,
        in_scale=1,
        out_scale=1,
        train_scale=False,
    ):
        super(PolyAct, self).__init__()
        if init_coef is None:
            init_coef = [0.0169394634313126, 0.5, 0.3078363963999393, 0.0]
        self.deg = len(init_coef) - 1
        self.trainable = trainable
        coef = torch.Tensor(init_coef)

        if trainable:
            self.coef = nn.Parameter(coef, requires_grad=True)
        else:
            self.register_buffer("coef", coef)

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

    def forward(self, x):
        if self.in_scale is not None:
            x = self.in_scale * x

        x = self.calc_polynomial(x)

        if self.out_scale is not None:
            x = self.out_scale * x

        return x

    def __repr__(self):
        print_coef = self.coef.cpu().detach().numpy()
        return "PolyAct(trainable={}, coef={})".format(self.trainable, print_coef)

    def calc_polynomial(self, x):
        res = self.coef[0] + self.coef[1] * x
        for i in range(2, self.deg):
            res = res + self.coef[i] * (x**i)

        return res


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
        data_format="channels_first",
        transform_mode="rfft",
        **kwargs,
    ):
        super(UpPolyActPerChannel, self).__init__()
        self.up = up
        self.lpf = LPF_RFFT(cutoff=1 / up, transform_mode=transform_mode)
        self.upsample = UpsampleRFFT(up, transform_mode=transform_mode)
        self.data_format = data_format

        self.pact = PolyActPerChannel(channels, **kwargs)

    def forward(self, x):
        if self.data_format == "channels_last":
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        out = self.upsample(x)
        # print("[UpPolyActPerChannel] up: ", out)
        out = self.pact(out)
        # print("[UpPolyActPerChannel] pact: ", out)
        out = self.lpf(out)
        out = out[:, :, :: self.up, :: self.up]

        if self.data_format == "channels_last":
            out = out.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        return out


class LPFPolyActPerChannel(nn.Module):
    def __init__(
        self,
        channels,
        init_coef=None,
        data_format="channels_first",
        in_scale=1,
        out_scale=1,
        train_scale=False,
        cutoff=0.5,
        fixed_lpf_size=None,
    ):
        super(LPFPolyActPerChannel, self).__init__()
        self.fixed_lpf_size = fixed_lpf_size
        self.lpf = LPF_RFFT(cutoff=cutoff, fixed_size=fixed_lpf_size)

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

        x_lpf = self.lpf(x)

        x = self.coef[:, 0] + self.coef[:, 1] * x + self.coef[:, 2] * (x * x_lpf)

        if self.out_scale is not None:
            x = self.out_scale * x

        if self.data_format == "channels_last":
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        return x


class circular_pad(nn.Module):
    def __init__(self, padding=(1, 1, 1, 1)):
        super(circular_pad, self).__init__()
        self.pad_sizes = padding

    def forward(self, x):

        return F.pad(x, pad=self.pad_sizes, mode="circular")


# Change layer norm implementation
# originaly they normalized only on channels. now normaliazing on all layer [C,H,W] -
# solving shit invariance issue
def LayerNorm(
    normalized_shape, eps=1e-6, data_format="channels_last", normalization_type="C"
):
    # normalizing on channels
    if normalization_type == "C":
        return LayerNorm_C(normalized_shape, eps, data_format)

    # normalize with mean on channels, std on layer
    elif normalization_type == "CHW2":
        return LayerNorm_AF(
            normalized_shape, eps, data_format, u_dims=1, s_dims=(1, 2, 3)
        )

    else:
        raise NotImplementedError


class LayerNorm_C(nn.Module):
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
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class LayerNorm_AF(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-6,
        data_format="channels_last",
        u_dims=(1, 2, 3),
        s_dims=(1, 2, 3),
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
        self.u_dims = u_dims
        self.s_dims = s_dims

    def forward(self, x):
        if self.data_format == "channels_last":
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        u = x.mean(self.u_dims, keepdim=True)
        s = (x - u).pow(2).mean(self.s_dims, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]

        if self.data_format == "channels_last":
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        return x


def get_norm_layer(normalization_type, dim, data_format="channels_first", **kwargs):
    if normalization_type == "batch":
        assert (
            data_format == "channels_first"
        ), "BatchNorm2d doesn't support channels last"
        return nn.BatchNorm2d(dim)

    elif normalization_type == "instance":
        assert (
            data_format == "channels_first"
        ), "InstanceNorm2d doesn't support channels last"
        return nn.InstanceNorm2d(dim)

    if "num_groups" in kwargs:
        num_groups = kwargs["num_groups"]
    elif "channels_per_group" in kwargs:
        num_groups = int(dim / kwargs["channels_per_group"])
    else:
        num_groups = None

    if normalization_type == "group":
        assert (
            data_format == "channels_first"
        ), "GroupNorm doesn't support channels last"
        assert (
            num_groups
        ), "missing key word argument for GroupNorm / LayerNormSTDGroups num_groups"
        return nn.GroupNorm(num_groups=num_groups, num_channels=dim)

    else:
        return LayerNorm(
            dim,
            eps=1e-6,
            normalization_type=normalization_type,
            data_format=data_format,
        )


# support other pad types - copied from cifar model
def pad_layer(pad_type, padding):
    if pad_type == "zero":
        padding = nn.ZeroPad2d(padding)
    elif pad_type == "reflect":
        padding = nn.ReflectionPad2d(padding)
    elif pad_type == "replicate_pad":
        padding = nn.ReplicationPad2d(padding)
    elif pad_type == "circular":
        padding = circular_pad(padding)
    else:
        assert False, "pad type {} not supported".format(pad_type)
    return padding


def conv3x3(
    in_planes, out_planes, stride=1, groups=1, bias=True, conv_pad_type="zeros"
):
    """3x3 convolution with padding"""

    padding = pad_layer(conv_pad_type, [1, 1, 1, 1])

    return nn.Sequential(
        padding,
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
        ),
    )


def conv2x2(
    in_planes,
    out_planes,
    padding=0,
    stride=1,
    groups=1,
    bias=True,
    conv_pad_type="zeros",
):
    """3x3 convolution with padding"""
    if padding != 0:
        if type(padding) == int:
            padding = pad_layer(conv_pad_type, [padding, padding, padding, padding])
        elif type(padding) == list and len(padding) == 4:
            padding = pad_layer(conv_pad_type, padding)
        else:
            raise UnexpectedException
    else:
        padding = nn.Identity()

    return nn.Sequential(
        padding,
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=2,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
        ),
    )


def conv4x4(
    in_planes,
    out_planes,
    padding=0,
    stride=1,
    groups=1,
    bias=True,
    conv_pad_type="zeros",
):
    """3x3 convolution with padding"""
    if padding != 0:
        if type(padding) == int:
            padding = pad_layer(conv_pad_type, [padding, padding, padding, padding])
        elif type(padding) == list and len(padding) == 4:
            padding = pad_layer(conv_pad_type, padding)
        else:
            raise UnexpectedException
    else:
        padding = nn.Identity()

    return nn.Sequential(
        padding,
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=4,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
        ),
    )


def conv7x7(
    in_planes,
    out_planes,
    padding=0,
    stride=1,
    groups=1,
    bias=True,
    conv_pad_type="zeros",
):
    """3x3 convolution with padding"""
    if padding > 0:
        padding = pad_layer(conv_pad_type, [padding, padding, padding, padding])
    else:
        padding = nn.Identity()

    return nn.Sequential(
        padding,
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=7,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
        ),
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class MLP(nn.Module):
    def __init__(self, dim, expand_ratio, activation, activation_kwargs={}):
        super(MLP, self).__init__()
        self.pwconv1 = nn.Linear(
            dim, expand_ratio * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = get_activation(
            activation,
            channels=expand_ratio * dim,
            data_format="channels_last",
            **activation_kwargs,
        )
        self.pwconv2 = nn.Linear(expand_ratio * dim, dim)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x


class AALMLP(nn.Module):
    def __init__(self, dim, expand_ratio, activation_kwargs={}):
        """channels last"""
        super(AALMLP, self).__init__()
        transform_mode = activation_kwargs.pop("transform_mode", "rfft")
        self.upsample = UpsampleRFFT(2, transform_mode=transform_mode)
        self.pwconv1 = nn.Linear(
            dim, expand_ratio * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = PolyActPerChannel(
            expand_ratio * dim, data_format="channels_last", **activation_kwargs
        )
        self.pwconv2 = nn.Linear(expand_ratio * dim, dim)
        self.lpf = LPF_RFFT(cutoff=0.5, transform_mode=transform_mode)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.lpf(x)
        x = x[:, :, ::2, ::2]
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        return x


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
        filter_type="basic",
        cutoff=0.5,
        scale_l2=False,
        eps=1e-6,
        transform_mode="rfft",
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
            self.filt = LPF_RFFT(cutoff=cutoff, transform_mode=transform_mode)

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


class BlurPool1D(nn.Module):
    def __init__(self, channels, pad_type="reflect", filt_size=3, stride=2, pad_off=0):
        super(BlurPool1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels

        # print('Filter size [%i]' % filt_size)
        if self.filt_size == 1:
            a = np.array(
                [
                    1.0,
                ]
            )
        elif self.filt_size == 2:
            a = np.array([1.0, 1.0])
        elif self.filt_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.filt_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.filt_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.filt_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.filt_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])

        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer("filt", filt[None, None, :].repeat((self.channels, 1, 1)))

        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, :: self.stride]
            else:
                return self.pad(inp)[:, :, :: self.stride]
        else:
            return F.conv1d(
                self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1]
            )


def get_pad_layer_1d(pad_type):
    if pad_type in ["refl", "reflect"]:
        PadLayer = nn.ReflectionPad1d
    elif pad_type in ["repl", "replicate"]:
        PadLayer = nn.ReplicationPad1d
    elif pad_type == "zero":
        PadLayer = nn.ZeroPad1d
    else:
        print("Pad type [%s] not recognized" % pad_type)
    return PadLayer


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

    def __init__(
        self,
        dim,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        conv_pad_type="circular",
        activation="gelu",
        activation_kwargs={},
        normalization_type="C",
        normalization_kwargs={},
    ):
        super().__init__()
        self.dwconv = conv7x7(
            dim, dim, padding=3, groups=dim, conv_pad_type=conv_pad_type
        )

        self.norm = get_norm_layer(
            normalization_type,
            dim,
            data_format="channels_first",
            **normalization_kwargs,
        )
        if activation == "up_poly_per_channel":
            # use AALMLP - faster implementation
            self.mlp = AALMLP(
                dim=dim, expand_ratio=4, activation_kwargs=activation_kwargs
            )
        else:
            self.mlp = MLP(
                dim=dim,
                expand_ratio=4,
                activation=activation,
                activation_kwargs=activation_kwargs,
            )

        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path_layer = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        x = self.mlp(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path_layer(x)
        return x


class ConvNeXtAFC(nn.Module):
    r"""ConvNeXtAFC

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        activation (str): Model activation function. Default: 'gelu'.
        For Aliasing free implementation use up_poly_per_channel
        activation_kwargs (dict): keyword arguments for activation function
        normalization_type (str): Default: C - original Layernorm. for Alias-free LayerNorm use CHW2
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        conv_pad_type="circular",
        blurpool_kwargs={},
        activation="gelu",
        activation_kwargs={},
        normalization_type="C",
        init_weight_std=0.02,
        stem_mode=None,
        stem_activation=None,
        stem_activation_kwargs={},
        normalization_kwargs={},
    ):
        super().__init__()
        layer_block = Block
        self.init_weight_std = init_weight_std
        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        if not stem_mode:

            stem = nn.Sequential(
                conv4x4(
                    in_chans,
                    dims[0],
                    stride=1,
                    conv_pad_type=conv_pad_type,
                    padding=[1, 2, 1, 2],
                ),
                BlurPool(
                    dims[0], conv_pad_type, stride=4, cutoff=0.25, **blurpool_kwargs
                ),
                get_norm_layer(
                    normalization_type,
                    dims[0],
                    data_format="channels_first",
                    **normalization_kwargs,
                ),
            )

        elif stem_mode == "activation":
            if not stem_activation:
                stem_activation = activation

            stem = nn.Sequential(
                conv4x4(
                    in_chans,
                    dims[0],
                    stride=1,
                    conv_pad_type=conv_pad_type,
                    padding=[1, 2, 1, 2],
                ),
                get_activation(
                    stem_activation, dims[0], "channels_first", **stem_activation_kwargs
                ),
                BlurPool(
                    dims[0], conv_pad_type, stride=4, cutoff=0.25, **blurpool_kwargs
                ),
                get_norm_layer(
                    normalization_type,
                    dims[0],
                    data_format="channels_first",
                    **normalization_kwargs,
                ),
            )

        elif stem_mode == "activation_residual":
            if not stem_activation:
                stem_activation = activation

            stem = nn.Sequential(
                conv4x4(
                    in_chans,
                    dims[0],
                    stride=1,
                    conv_pad_type=conv_pad_type,
                    padding=[1, 2, 1, 2],
                ),
                Residual(
                    get_activation(
                        stem_activation,
                        dims[0],
                        "channels_first",
                        **stem_activation_kwargs,
                    )
                ),
                BlurPool(
                    dims[0], conv_pad_type, stride=4, cutoff=0.25, **blurpool_kwargs
                ),
                get_norm_layer(
                    normalization_type,
                    dims[0],
                    data_format="channels_first",
                    **normalization_kwargs,
                ),
            )

        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_norm_layer(
                    normalization_type,
                    dims[i],
                    data_format="channels_first",
                    **normalization_kwargs,
                ),
                conv2x2(
                    dims[i],
                    dims[i + 1],
                    stride=1,
                    conv_pad_type=conv_pad_type,
                    padding=[0, 1, 0, 1],
                ),
                BlurPool(dims[i + 1], conv_pad_type, stride=2, **blurpool_kwargs),
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
                    layer_block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        conv_pad_type=conv_pad_type,
                        activation=activation,
                        activation_kwargs=activation_kwargs,
                        normalization_type=normalization_type,
                        normalization_kwargs=normalization_kwargs,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=self.init_weight_std)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(
            x.mean([-2, -1])
        )  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class Residual(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.residual = nn.Sequential(*layers)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.gamma * self.residual(x)


class ResidualPerChannel(nn.Module):
    def __init__(self, channels, *layers):
        super().__init__()
        self.residual = nn.Sequential(*layers)
        self.gamma = nn.Parameter(1e-6 * torch.ones((channels, 1, 1)))

    def forward(self, x):
        return x + self.gamma * self.residual(x)


class AliasFreeDenoiser(nn.Module):
    def __init__(self, size, **kwargs):
        super().__init__()
        if size == "tiny":
            self.model = ConvNeXtAFC(
                depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs
            )
        elif size == "small":
            self.model = ConvNeXtAFC(
                depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs
            )
        elif size == "base":
            self.model = ConvNeXtAFC(
                depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs
            )
        elif size == "large":
            self.model = ConvNeXtAFC(
                depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs
            )
        else:
            raise ValueError(f"Unknown size {size}")

    def forward(self, x, sigma=None):
        return x
