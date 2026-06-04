"""
This module implements the Discrete Cosine Transform (DCT) and its inverse (IDCT) for PyTorch tensors.

Parts of this code are adapted from the 'torch-dct' repository by zh217:
https://github.com/zh217/torch-dct

Original code is licensed under the MIT License.
Modifications have been made for integration with the DeepInverse library.
"""

import torch

pi = torch.pi


def dct(x, norm=None):
    r"""
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    Parts of this code are adapted from the `torch-dct` repository by zh217: https://github.com/zh217/torch-dct

    :param torch.Tensor x: the input signal
    :param None, str norm: the normalization, `None` or `'ortho'`
    :return: (:class:`torch.Tensor`) the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = torch.tensor(x_shape[-1], device=x.device)
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= torch.sqrt(N) * 2
        V[:, 1:] /= torch.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(x, norm=None):
    r"""
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    Parts of this code are adapted from the `torch-dct` repository by zh217: https://github.com/zh217/torch-dct

    :param torch.Tensor x: the input signal
    :param None, str norm: the normalization, `None` or `'ortho'`
    :return: (:class:`torch.Tensor`) the inverse DCT-II of the signal over the last dimension
    """

    x_shape = x.shape
    N = torch.tensor(x_shape[-1], device=x.device)

    X_v = x.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= torch.sqrt(N) * 2
        X_v[:, 1:] *= torch.sqrt(N / 2) * 2

    k = (
        torch.arange(x_shape[-1], dtype=x.dtype, device=x.device)[None, :]
        * pi
        / (2 * N)
    )
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    r"""
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    Parts of this code are adapted from the `torch-dct` repository by zh217: https://github.com/zh217/torch-dct

    :param torch.Tensor x: the input signal
    :param None, str norm: the normalization, `None` or `'ortho'`
    :return: (:class:`torch.Tensor`) the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    r"""
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    Parts of this code are adapted from the `torch-dct` repository by zh217: https://github.com/zh217/torch-dct

    :param torch.Tensor x: the input signal
    :param None, str norm: the normalization, `None` or `'ortho'`
    :return: (:class:`torch.Tensor`) the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)
