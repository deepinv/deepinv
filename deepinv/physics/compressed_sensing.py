from deepinv.physics.forward import LinearPhysics
import torch
import numpy as np
from deepinv.physics.functional import random_choice
from torch import Tensor
from deepinv.utils.decorators import _deprecated_alias


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


class CompressedSensing(LinearPhysics):
    r"""
    Compressed Sensing forward operator. Creates a random sampling :math:`m \times n` matrix where :math:`n` is the
    number of elements of the signal, i.e., ``np.prod(img_size)`` and ``m`` is the number of measurements.

    This class generates a random iid Gaussian matrix if ``fast=False``

    .. math::

        A_{i,j} \sim \mathcal{N}(0,\frac{1}{m})

    or a Subsampled Orthogonal with Random Signs matrix (SORS) if ``fast=True`` (see :footcite:t:`oymak2018isometric`)

    .. math::

        A = \text{diag}(m)D\text{diag}(s)

    where :math:`s\in\{-1,1\}^{n}` is a random sign flip with probability 0.5,
    :math:`D\in\mathbb{R}^{n\times n}` is a fast orthogonal transform (DST-1) and
    :math:`\text{diag}(m)\in\mathbb{R}^{m\times n}` is random subsampling matrix, which keeps :math:`m` out of :math:`n` entries.

    For image sizes bigger than 32 x 32, the forward computation can be prohibitively expensive due to its :math:`O(mn)` complexity.
    In this case, we recommend using :class:`deepinv.physics.StructuredRandom` instead.

    .. deprecated:: 0.2.2
       The ``fast`` option is deprecated and might be removed in future versions. Use :class:`deepinv.physics.StructuredRandom` instead.

    An existing operator can be loaded from a saved .pth file via ``self.load_state_dict(save_path)``,
    in a similar fashion to :class:`torch.nn.Module`.

    .. note::

        If ``fast=False``, the forward operator has a norm which tends to :math:`(1+\sqrt{n/m})^2` for large :math:`n`
        and :math:`m` due to the `Marcenko-Pastur law
        <https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution>`_.
        If ``fast=True``, the forward operator has a unit norm.

    If ``dtype=torch.cfloat``, the forward operator will be generated as a random i.i.d. complex Gaussian matrix to be used with ``fast=False``

    .. math::

        A_{i,j} \sim \mathcal{N} \left( 0, \frac{1}{2m} \right) + \mathrm{i} \mathcal{N} \left( 0, \frac{1}{2m} \right).

    :param int m: number of measurements.
    :param tuple img_size: shape (C, H, W) of inputs.
    :param bool fast: The operator is iid Gaussian if false, otherwise A is a SORS matrix with the Discrete Sine Transform (type I).
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

    @_deprecated_alias(img_shape="img_size")
    def __init__(
        self,
        m,
        img_size,
        fast=False,
        channelwise=False,
        dtype=torch.float,
        device="cpu",
        rng: torch.Generator = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = f"CS_m{m}"
        self.img_size = img_size
        self.fast = fast
        self.channelwise = channelwise
        self.dtype = dtype
        self.device = device

        if rng is None:
            self.rng = torch.Generator(device=device)
        else:
            # Make sure that the random generator is on the same device as the physic generator
            assert rng.device == torch.device(
                device
            ), f"The random generator is not on the same device as the Physics Generator. Got random generator on {rng.device} and the Physics Generator on {self.device}."
            self.rng = rng
        self.register_buffer("initial_random_state", self.rng.get_state())

        if channelwise:
            n = int(np.prod(img_size[1:]))
        else:
            n = int(np.prod(img_size))

        if self.fast:
            self.n = n
            D = torch.where(
                torch.rand(self.n, device=device, generator=self.rng) > 0.5, -1.0, 1.0
            )

            mask = torch.zeros(self.n, device=device)
            idx = torch.sort(
                random_choice(self.n, size=m, replace=False, rng=self.rng)
            ).values
            mask[idx] = 1
            mask = mask.type(torch.bool)

            self.register_buffer("D", D)
            self.register_buffer("mask", mask)
        else:
            _A = torch.randn(
                (m, n), device=device, dtype=dtype, generator=self.rng
            ) / np.sqrt(m)
            _A_dagger = torch.linalg.pinv(_A)

            self.register_buffer("_A", _A)
            self.register_buffer("_A_dagger", _A_dagger)
            self.register_buffer("_A_adjoint", self._A.conj().T.type(dtype).to(device))
        self.to(device)

    def A(self, x: Tensor, **kwargs) -> Tensor:
        N, C = x.shape[:2]
        if self.channelwise:
            x = x.reshape(N * C, -1)
        else:
            x = x.reshape(N, -1)

        if self.fast:
            y = dst1(x * self.D)[:, self.mask]
        else:
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

        if self.fast:
            y2 = torch.zeros((N2, self.n), device=y.device)
            y2[:, self.mask] = y.type(y2.dtype)
            x = dst1(y2) * self.D
        else:
            x = torch.einsum("im, nm->in", y, self._A_adjoint)  # x:(N, n, 1)

        x = x.view(N, C, H, W)
        return x

    def A_dagger(self, y: Tensor, **kwargs) -> Tensor:
        y = y.type(self.dtype)
        if self.fast:
            return self.A_adjoint(y)
        else:
            N = y.shape[0]
            C, H, W = self.img_size[0], self.img_size[1], self.img_size[2]

            if self.channelwise:
                y = y.reshape(N * C, -1)

            x = torch.einsum("im, nm->in", y, self._A_dagger)
            x = x.reshape(N, C, H, W)
        return x
