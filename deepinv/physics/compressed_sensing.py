from __future__ import annotations
from deepinv.physics.forward import LinearPhysics
import torch
import numpy as np
from torch import Tensor


def dst1(x: Tensor) -> Tensor:
    r"""
    Orthogonal Discrete Sine Transform, Type I
    The transform is performed across the last dimension of the input signal
    Due to orthogonality we have ``dst1(dst1(x)) = x``.

    :param torch.Tensor x: the input signal
    :return: (torch.Tensor) the DST-I of the signal over the last dimension

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


class CompressedSensing(LinearPhysics):
    r"""
    Compressed Sensing forward operator. Creates a random sampling :math:`m \times n` matrix where :math:`n` is the
    number of elements of the signal, i.e., ``np.prod(img_size)`` and ``m`` is the number of measurements.

    This class generates a random iid Gaussian matrix

    .. math::

        A_{i,j} \sim \mathcal{N}(0,\frac{1}{m})

    For image sizes bigger than 32 x 32, the forward computation can be prohibitively expensive due to its :math:`O(mn)` complexity.
    In this case, we recommend using :class:`deepinv.physics.StructuredRandom` instead.

    An existing operator can be loaded from a saved .pth file via ``self.load_state_dict(save_path)``,
    in a similar fashion to :class:`torch.nn.Module`.

    .. note::

        The forward operator has a norm which tends to :math:`(1+\sqrt{n/m})^2` for large :math:`n`
        and :math:`m` due to the `Marcenko-Pastur law
        <https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution>`_.

    If ``dtype=torch.cfloat``, the forward operator will be generated as a random i.i.d. complex Gaussian matrix

    .. math::

        A_{i,j} \sim \mathcal{N} \left( 0, \frac{1}{2m} \right) + \mathrm{i} \mathcal{N} \left( 0, \frac{1}{2m} \right).

    :param int m: number of measurements.
    :param tuple img_size: shape (C, H, W) of inputs.
    :param bool channelwise: Channels are processed independently using the same random forward operator.
    :param torch.dtype dtype: Forward matrix is stored as a dtype. For complex matrices, use torch.cfloat. Default is torch.float.
    :param str device: Device to store the forward matrix.
    :param torch.Generator rng: (optional) a pseudorandom random number generator for the parameter generation.
        If ``None``, the default Generator of PyTorch will be used.

    |sep|

    :Examples:

        Compressed sensing operator with 100 measurements for a 3x3 image:

        >>> from deepinv.physics import CompressedSensing
        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn(1, 1, 3, 3) # Define random 3x3 image
        >>> physics = CompressedSensing(m=10, img_size=(1, 3, 3), rng=torch.Generator('cpu'))
        >>> physics(x)
        tensor([[-1.7769,  0.6160, -0.8181, -0.5282, -1.2197,  0.9332, -0.1668,  1.5779,
                  0.6752, -1.5684]])

    """

    def __init__(
        self,
        m: int,
        img_size: tuple[int],
        channelwise: bool = False,
        dtype: torch.dtype = torch.float,
        device: torch.device | str = "cpu",
        rng: torch.Generator = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.name = f"CS_m{m}"
        self.img_size = img_size
        self.channelwise = channelwise
        self.dtype = dtype

        if rng is None:
            self.rng = torch.Generator(device=device)
        else:
            # Make sure that the random generator is on the same device as the physic generator
            assert rng.device == torch.device(
                device
            ), f"The random generator is not on the same device as the Physics Generator. Got random generator on {rng.device} and the Physics Generator on {device}."
            self.rng = rng
        self.register_buffer("initial_random_state", self.rng.get_state())

        if channelwise:
            n = int(np.prod(img_size[1:]))
        else:
            n = int(np.prod(img_size))

        _A = torch.randn(
            (m, n), device=device, dtype=dtype, generator=self.rng
        ) / np.sqrt(m)
        _A_dagger = torch.linalg.pinv(_A)

        self.register_buffer("_A", _A)
        self.register_buffer("_A_dagger", _A_dagger)
        self.register_buffer("_A_adjoint", self._A.conj().T)
        self.to(device=device, dtype=dtype)

    def A(self, x: Tensor, **kwargs) -> Tensor:
        N, C = x.shape[:2]
        if self.channelwise:
            x = x.reshape(N * C, -1)
        else:
            x = x.reshape(N, -1)

        y = torch.einsum("in, mn->im", x, self._A)

        if self.channelwise:
            y = y.view(N, C, -1)

        return y

    def A_adjoint(self, y: Tensor, **kwargs) -> Tensor:
        y = y.type(self.dtype)
        N = y.shape[0]
        C, H, W = self.img_size[0], self.img_size[1], self.img_size[2]

        if self.channelwise:
            N2 = N * C
            y = y.view(N2, -1)
        else:
            N2 = N

        x = torch.einsum("im, nm->in", y, self._A_adjoint)  # x:(N, n, 1)
        x = x.view(N, C, H, W)
        return x

    def A_dagger(self, y: Tensor, **kwargs) -> Tensor:
        y = y.type(self.dtype)

        N = y.shape[0]
        C, H, W = self.img_size[0], self.img_size[1], self.img_size[2]

        if self.channelwise:
            y = y.reshape(N * C, -1)

        x = torch.einsum("im, nm->in", y, self._A_dagger)
        x = x.reshape(N, C, H, W)
        return x
