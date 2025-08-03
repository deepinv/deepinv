"""
Code reproduced with modifications from https://github.com/sanghyun-son/bicubic_pytorch

Author:      Sanghyun Son
Email:       sonsang35@gmail.com (primary), thstkdgus35@snu.ac.kr (secondary)
Version:     1.2.0
Last update: July 9th, 2020 (KST)
"""

import math
from typing import Optional

import torch
from torch.nn import functional as F

_I = Optional[int]
_D = Optional[torch.dtype]


def cubic_contribution(x: torch.Tensor, a: float = -0.5) -> torch.Tensor:
    ax = x.abs()
    ax2 = ax * ax
    ax3 = ax * ax2

    range_01 = ax.le(1)
    range_12 = torch.logical_and(ax.gt(1), ax.le(2))

    cont_01 = (a + 2) * ax3 - (a + 3) * ax2 + 1
    cont_01 = cont_01 * range_01.to(dtype=x.dtype)

    cont_12 = (a * ax3) - (5 * a * ax2) + (8 * a * ax) - (4 * a)
    cont_12 = cont_12 * range_12.to(dtype=x.dtype)

    cont = cont_01 + cont_12
    return cont


def gaussian_contribution(x: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    range_3sigma = x.abs() <= 3 * sigma + 1
    # Normalization will be done after
    cont = torch.exp(-x.pow(2) / (2 * sigma**2))
    cont = cont * range_3sigma.to(dtype=x.dtype)
    return cont


def reflect_padding(
    x: torch.Tensor, dim: int, pad_pre: int, pad_post: int
) -> torch.Tensor:
    """
    Apply reflect padding to the given Tensor.
    Note that it is slightly different from the PyTorch functional.pad,
    where boundary elements are used only once.
    Instead, we follow the MATLAB implementation
    which uses boundary elements twice.

    For example,
    [a, b, c, d] would become [b, a, b, c, d, c] with the PyTorch implementation,
    while our implementation yields [a, a, b, c, d, d].
    """
    b, c, h, w = x.size()
    if dim == 2 or dim == -2:
        padding_buffer = x.new_zeros(b, c, h + pad_pre + pad_post, w)
        padding_buffer[..., pad_pre : (h + pad_pre), :].copy_(x)
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1, :].copy_(x[..., p, :])
        for p in range(pad_post):
            padding_buffer[..., h + pad_pre + p, :].copy_(x[..., -(p + 1), :])
    else:
        padding_buffer = x.new_zeros(b, c, h, w + pad_pre + pad_post)
        padding_buffer[..., pad_pre : (w + pad_pre)].copy_(x)
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1].copy_(x[..., p])
        for p in range(pad_post):
            padding_buffer[..., w + pad_pre + p].copy_(x[..., -(p + 1)])

    return padding_buffer


def padding(
    x: torch.Tensor,
    dim: int,
    pad_pre: int,
    pad_post: int,
    padding_type: Optional[str] = "reflect",
) -> torch.Tensor:

    if padding_type is None:
        return x
    elif padding_type == "reflect":
        x_pad = reflect_padding(x, dim, pad_pre, pad_post)
    else:
        raise ValueError("{} padding is not supported!".format(padding_type))

    return x_pad


def get_padding(
    base: torch.Tensor, kernel_size: int, x_size: int
) -> tuple[int, int, torch.Tensor]:

    base = base.long()
    r_min = base.min()
    r_max = base.max() + kernel_size - 1

    if r_min <= 0:
        pad_pre = -r_min
        pad_pre = pad_pre.item()
        base += pad_pre
    else:
        pad_pre = 0

    if r_max >= x_size:
        pad_post = r_max - x_size + 1
        pad_post = pad_post.item()
    else:
        pad_post = 0

    return pad_pre, pad_post, base


def get_weight(
    dist: torch.Tensor,
    kernel_size: int,
    kernel: str = "cubic",
    sigma: float = 2.0,
    antialiasing_factor: float = 1,
) -> torch.Tensor:

    buffer_pos = dist.new_zeros(kernel_size, len(dist))
    for idx, buffer_sub in enumerate(buffer_pos):
        buffer_sub.copy_(dist - idx)

    # Expand (downsampling) / Shrink (upsampling) the receptive field.
    buffer_pos *= antialiasing_factor
    if kernel == "cubic":
        weight = cubic_contribution(buffer_pos)
    elif kernel == "gaussian":
        weight = gaussian_contribution(buffer_pos, sigma=sigma)
    else:
        raise ValueError("{} kernel is not supported!".format(kernel))

    weight /= weight.sum(dim=0, keepdim=True)
    return weight


def reshape_tensor(x: torch.Tensor, dim: int, kernel_size: int) -> torch.Tensor:
    # Resize height
    if dim == 2 or dim == -2:
        k = (kernel_size, 1)
        h_out = x.size(-2) - kernel_size + 1
        w_out = x.size(-1)
    # Resize width
    else:
        k = (1, kernel_size)
        h_out = x.size(-2)
        w_out = x.size(-1) - kernel_size + 1

    unfold = F.unfold(x, k)
    unfold = unfold.view(unfold.size(0), -1, h_out, w_out)
    return unfold


def reshape_input(x: torch.Tensor) -> tuple[torch.Tensor, _I, _I, _I, _I]:

    if x.dim() == 4:
        b, c, h, w = x.size()
    elif x.dim() == 3:
        c, h, w = x.size()
        b = None
    elif x.dim() == 2:
        h, w = x.size()
        b = c = None
    else:
        raise ValueError("{}-dim Tensor is not supported!".format(x.dim()))

    x = x.view(-1, 1, h, w)
    return x, b, c, h, w


def reshape_output(x: torch.Tensor, b: _I, c: _I) -> torch.Tensor:

    rh = x.size(-2)
    rw = x.size(-1)
    # Back to the original dimension
    if b is not None:
        x = x.view(b, c, rh, rw)  # 4-dim
    else:
        if c is not None:
            x = x.view(c, rh, rw)  # 3-dim
        else:
            x = x.view(rh, rw)  # 2-dim

    return x


def cast_input(x: torch.Tensor) -> tuple[torch.Tensor, _D]:
    if x.dtype not in (torch.float64, torch.float32):
        dtype = x.dtype
        x = x.float()
    else:
        dtype = None

    return x, dtype


def cast_output(x: torch.Tensor, dtype: _D) -> torch.Tensor:
    if dtype is not None:
        if not dtype.is_floating_point:
            x = x.round()
        # To prevent over/underflow when converting types
        if dtype is torch.uint8:
            x = x.clamp(0, 255)

        x = x.to(dtype=dtype)

    return x


def resize_1d(
    x: torch.Tensor,
    dim: int,
    size: Optional[int],
    scale: Optional[float],
    kernel: str = "cubic",
    sigma: float = 2.0,
    padding_type: str = "reflect",
    antialiasing: bool = True,
) -> torch.Tensor:
    """
    Args:
        x (torch.Tensor): A torch.Tensor of dimension (B x C, 1, H, W).
        dim (int):
        scale (float):
        size (int):

    Return:
    """
    # Identity case
    if scale == 1:
        return x

    # Default bicubic kernel with antialiasing (only when downsampling)
    if kernel == "cubic":
        kernel_size = 4
    else:
        kernel_size = math.floor(6 * sigma)

    if antialiasing and (scale < 1):
        antialiasing_factor = scale
        kernel_size = math.ceil(kernel_size / antialiasing_factor)
    else:
        antialiasing_factor = 1

    # We allow margin to both sizes
    kernel_size += 2

    # Weights only depend on the shape of input and output,
    # so we do not calculate gradients here.
    with torch.no_grad():
        pos = torch.linspace(
            0,
            size - 1,
            steps=size,
            dtype=x.dtype,
            device=x.device,
        )
        pos = (pos + 0.5) / scale - 0.5
        base = pos.floor() - (kernel_size // 2) + 1
        dist = pos - base
        weight = get_weight(
            dist,
            kernel_size,
            kernel=kernel,
            sigma=sigma,
            antialiasing_factor=antialiasing_factor,
        )
        pad_pre, pad_post, base = get_padding(base, kernel_size, x.size(dim))

    # To backpropagate through x
    x_pad = padding(x, dim, pad_pre, pad_post, padding_type=padding_type)
    unfold = reshape_tensor(x_pad, dim, kernel_size)
    # Subsampling first
    if dim == 2 or dim == -2:
        sample = unfold[..., base, :]
        weight = weight.view(1, kernel_size, sample.size(2), 1)
    else:
        sample = unfold[..., base]
        weight = weight.view(1, kernel_size, 1, sample.size(3))

    # Apply the kernel
    x = sample * weight
    x = x.sum(dim=1, keepdim=True)
    return x


def imresize_matlab(
    x: torch.Tensor,
    scale: Optional[float] = None,
    sizes: Optional[tuple[int, int]] = None,
    kernel: str = "cubic",
    sigma: float = 2,
    padding_type: str = "reflect",
    antialiasing: bool = True,
) -> torch.Tensor:
    """MATLAB imresize reimplementation.

    A standalone PyTorch implementation for fast and efficient bicubic resampling.
    The resulting values are the same to MATLAB function imresize('bicubic') with `reflect` padding.

    Code reproduced with modifications from https://github.com/sanghyun-son/bicubic_pytorch

    :param torch.Tensor x: input tensor of shape (H,W), (C,H,W) or (B,C,H,W)
    :param float scale: imresize scale factor. > 1 = upsample.
    :param tuple sizes: optional output image size following MATLAB convention.
    :param str kernel: downsampling kernel, choose between 'cubic' (for MATLAB bicubic) or 'gaussian'.
    :param float sigma: Gaussian kernel size. Ignored if kernel is not gaussian.
    :param str padding_type: reflect padding only.
    :param bool antialiasing: whether to perform antialiasing.
    :return: torch.Tensor: output resized image.

    :Example:

        >>> import torch
        >>> from deepinv.physics.functional import imresize_matlab
        >>> x = torch.randn(1, 1, 8, 8)
        >>> y = imresize_matlab(x, scale=2)
        >>> y.shape
        torch.Size([1, 1, 16, 16])

    """

    if scale is None and sizes is None:
        raise ValueError("One of scale or sizes must be specified!")
    if scale is not None and sizes is not None:
        raise ValueError("Please specify scale or sizes to avoid conflict!")

    x, b, c, h, w = reshape_input(x)

    if sizes is None:
        # Determine output size
        sizes = (math.ceil(h * scale), math.ceil(w * scale))
        scales = (scale, scale)

    if scale is None:
        scales = (sizes[0] / h, sizes[1] / w)

    x, dtype = cast_input(x)

    kwargs = {
        "kernel": kernel,
        "sigma": sigma,
        "padding_type": padding_type,
        "antialiasing": antialiasing,
    }
    # Core resizing module
    x = resize_1d(x, -2, size=sizes[0], scale=scales[0], **kwargs)
    x = resize_1d(x, -1, size=sizes[1], scale=scales[1], **kwargs)

    x = reshape_output(x, b, c)
    x = cast_output(x, dtype)
    return x
