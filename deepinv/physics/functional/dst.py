import numpy as np
import torch


def dst1(x):
    r"""
    Orthogonal Discrete Sine Transform, Type I
    The transform is performed across the last dimension of the input signal
    Due to orthogonality we have ``dst1(dst1(x)) = x``.

    :param torch.Tensor x: the input signal
    :return: (torch.tensor) the DST-I of the signal over the last dimension

    """
    x_shape = x.shape

    b = int(np.prod(x_shape[:-1]))
    n = x_shape[-1]
    x = x.view(-1, n)

    z = torch.zeros(b, 1, device=x.device)
    x = torch.cat([z, x, z, -x.flip([1])], dim=1)
    x = torch.view_as_real(torch.fft.rfft(x, norm="ortho"))
    x = x[:, 1:-1, 1]
    return x.view(*x_shape)
