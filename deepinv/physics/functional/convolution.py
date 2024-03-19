# %%
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Optional, Tuple


def conv2d(x: Tensor, filter: Tensor, padding: str = "valid") -> Tensor:
    r"""
    A helper function performing the 2d convolution of images :math:`x` and `filter`.  The transposed of this operation is :meth:`deepinv.physics.functional.conv_transposed2d()`

    :param torch.Tensor x: Image of size `(B, C, W, H)`.
    :param torch.Tensor filter: Filter of size `(b, c, w, h)` where `b` can be either `1` or `B` and `c` can be either `1` or `C`.

    If `b = 1` or `c = 1`, then this function supports broadcasting as the same as `numpy <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_. Otherwise, each channel of each image is convolved with the corresponding kernel.

    :param padding: ( options = `valid`, `circular`, `replicate`, `reflect`. If `padding = 'valid'` the blurred output is smaller than the image (no padding), otherwise the blurred output has the same size as the image.

    :return torch.Tensor : the output
    """
    assert x.dim() == filter.dim() == 4, "Input and filter must be 4D tensors"

    # Get dimensions of the input and the filter
    B, C, H, W = x.size()
    b, c, h, w = filter.size()

    if c != C:
        assert c == 1
        print(
            "Warning: the number of channels of the input is different than the one of the filter. The filter is expanded in the channel dimension"
        )
        filter = filter.expand(-1, C, -1, -1)

    if b != B:
        assert b == 1
        filter = filter.expand(B, -1, -1, -1)

    if padding != "valid":
        ph = int((h - 1) / 2)
        pw = int((w - 1) / 2)
        x = F.pad(x, (pw, pw, ph, ph), mode=padding, value=0)
        B, C, H, W = x.size()

    # Move batch dim of the input into channels
    x = x.reshape(1, -1, H, W)
    # Expand the channel dim of the filter and move it into batch dimension
    filter = filter.reshape(B * C, -1, h, w)
    # Perform the convolution, using the groups parameter
    output = F.conv2d(x, filter, padding="valid", groups=B * C)
    # Make it in the good shape
    output = output.view(B, C, output.size(-2), -1)

    return output


def conv_transpose2d(y: Tensor, filter: Tensor, padding: str = "valid") -> Tensor:
    r"""
    A helper function performing the 2d transposed convolution 2d of x and filter. The transposed of this operation is :meth:`deepinv.physics.functional.conv2d()`

    :param torch.Tensor x: Image of size `(B, C, W, H)`.
    :param torch.Tensor filter: Filter of size `(b, c, w, h)` ) where `b` can be either `1` or `B` and `c` can be either `1` or `C`.

    If `b = 1` or `c = 1`, then this function supports broadcasting as the same as `numpy <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_. Otherwise, each channel of each image is convolved with the corresponding kernel.

    :param str padding: options are ``'valid'``, ``'circular'``, ``'replicate'`` and ``'reflect'``.
        If ``padding='valid'`` the blurred output is smaller than the image (no padding)
        otherwise the blurred output has the same size as the image.
    """

    assert y.dim() == filter.dim() == 4, "Input and filter must be 4D tensors"

    # Get dimensions of the input and the filter
    B, C, H, W = y.size()
    b, c, h, w = filter.size()

    ph = int((h - 1) / 2)
    pw = int((w - 1) / 2)

    if c != C:
        assert c == 1
        filter = filter.expand(-1, C, -1, -1)

    if b != B:
        assert b == 1
        filter = filter.expand(B, -1, -1, -1)

    # Move batch dim of the input into channels
    y = y.reshape(1, -1, H, W)
    # Expand the channel dim of the filter and move it into batch dimension
    filter = filter.reshape(B * C, -1, h, w)
    # Perform the convolution, using the groups parameter
    x = F.conv_transpose2d(y, filter, groups=B * C)
    # Make it in the good shape
    x = x.view(B, C, x.size(-2), -1)

    if padding == "valid":
        out = x
    elif padding == "zero":
        out = x[:, :, ph:-ph, pw:-pw]
    elif padding == "circular":
        out = x[:, :, ph:-ph, pw:-pw]
        # sides
        out[:, :, :ph, :] += x[:, :, -ph:, pw:-pw]
        out[:, :, -ph:, :] += x[:, :, :ph, pw:-pw]
        out[:, :, :, :pw] += x[:, :, ph:-ph, -pw:]
        out[:, :, :, -pw:] += x[:, :, ph:-ph, :pw]
        # corners
        out[:, :, :ph, :pw] += x[:, :, -ph:, -pw:]
        out[:, :, -ph:, -pw:] += x[:, :, :ph, :pw]
        out[:, :, :ph, -pw:] += x[:, :, -ph:, :pw]
        out[:, :, -ph:, :pw] += x[:, :, :ph, -pw:]

    elif padding == "reflect":
        out = x[:, :, ph:-ph, pw:-pw]
        # sides
        out[:, :, 1 : 1 + ph, :] += x[:, :, :ph, pw:-pw].flip(dims=(2,))
        out[:, :, -ph - 1 : -1, :] += x[:, :, -ph:, pw:-pw].flip(dims=(2,))
        out[:, :, :, 1 : 1 + pw] += x[:, :, ph:-ph, :pw].flip(dims=(3,))
        out[:, :, :, -pw - 1 : -1] += x[:, :, ph:-ph, -pw:].flip(dims=(3,))
        # corners
        out[:, :, 1 : 1 + ph, 1 : 1 + pw] += x[:, :, :ph, :pw].flip(dims=(2, 3))
        out[:, :, -ph - 1 : -1, -pw - 1 : -1] += x[:, :, -ph:, -pw:].flip(dims=(2, 3))
        out[:, :, -ph - 1 : -1, 1 : 1 + pw] += x[:, :, -ph:, :pw].flip(dims=(2, 3))
        out[:, :, 1 : 1 + ph, -pw - 1 : -1] += x[:, :, :ph, -pw:].flip(dims=(2, 3))

    elif padding == "replicate":
        out = x[:, :, ph:-ph, pw:-pw]
        # sides
        out[:, :, 0, :] += x[:, :, :ph, pw:-pw].sum(2)
        out[:, :, -1, :] += x[:, :, -ph:, pw:-pw].sum(2)
        out[:, :, :, 0] += x[:, :, ph:-ph, :pw].sum(3)
        out[:, :, :, -1] += x[:, :, ph:-ph, -pw:].sum(3)
        # corners
        out[:, :, 0, 0] += x[:, :, :ph, :pw].sum(3).sum(2)
        out[:, :, -1, -1] += x[:, :, -ph:, -pw:].sum(3).sum(2)
        out[:, :, -1, 0] += x[:, :, -ph:, :pw].sum(3).sum(2)
        out[:, :, 0, -1] += x[:, :, :ph, -pw:].sum(3).sum(2)

    return out


# %%
if __name__ == "__main__":
    B = 4
    C = 3
    H = 256
    W = 256
    from skimage.data import astronaut
    from skimage.transform import resize
    import matplotlib.pyplot as plt

    img = resize(astronaut(), (H, W))

    device = "cuda"
    dtype = torch.float64

    x = torch.from_numpy(img).permute(2, 0, 1)[None].to(device=device, dtype=dtype)
    x = x.expand(B, -1, -1, -1)

    # filter = torch.randn((B, C, H // 2, W // 2), device=device, dtype=dtype)
    # filter = gaussian_blur(3.).expand(
    #     B, C, -1, -1).to(device=device, dtype=dtype)

    filter = torch.randn((B, C, H // 2 + 1, W // 2 + 1), device=device, dtype=dtype)
    # 'valid', 'circular', 'replicate', 'reflect'
    padding = "reflect"
    # output_v1 = conv2d_v1(x, filter[:, 0:1, ...])
    # print(output_v1.shape)
    # plt.imshow(output_v1[0].permute(1, 2, 0).cpu().numpy())
    # plt.show()

    # output_v2 = conv2d_v2(x, filter)
    # print(output_v2.shape)
    # plt.imshow(output_v2[0].permute(1, 2, 0).cpu().numpy())
    # plt.show()

    filter = filter[:, 0:1, ...]

    Ax = conv2d(x, filter, padding)
    print(Ax.shape)
    plt.imshow(Ax[0].permute(1, 2, 0).cpu().numpy())
    plt.show()

    y = torch.randn_like(Ax)

    z = conv_transpose2d(y, filter, padding)

    print((Ax * y).sum(dim=(1, 2, 3)))
    print((x * z).sum(dim=(1, 2, 3)))

    print((Ax * y).sum(dim=(1, 2, 3)) - (x * z).sum(dim=(1, 2, 3)))
# %%
