from deepinv.physics.forward import LinearPhysics
import torch
import numpy as np
from deepinv.physics.functional import random_choice


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
    number of elements of the signal, i.e., ``np.prod(img_shape)`` and ``m`` is the number of measurements.

    This class generates a random iid Gaussian matrix if ``fast=False``

    .. math::

        A_{i,j} \sim \mathcal{N}(0,\frac{1}{m})

    or a Subsampled Orthogonal with Random Signs matrix (SORS) if ``fast=True`` (see https://arxiv.org/abs/1506.03521)

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
    :param tuple img_shape: shape (C, H, W) of inputs.
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
        >>> physics = CompressedSensing(m=10, img_shape=(1, 3, 3), rng=torch.Generator('cpu'))
        >>> physics(x)
        tensor([[-1.7769,  0.6160, -0.8181, -0.5282, -1.2197,  0.9332, -0.1668,  1.5779,
                  0.6752, -1.5684]])

    """

    def __init__(
        self,
        m,
        img_shape,
        fast=False,
        channelwise=False,
        dtype=torch.float,
        device="cpu",
        rng: torch.Generator = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = f"CS_m{m}"
        self.img_shape = img_shape
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
        self.initial_random_state = self.rng.get_state()

        if channelwise:
            n = int(np.prod(img_shape[1:]))
        else:
            n = int(np.prod(img_shape))

        if self.fast:
            self.n = n
            self.D = torch.where(
                torch.rand(self.n, device=device, generator=self.rng) > 0.5, -1.0, 1.0
            )

            self.mask = torch.zeros(self.n, device=device)
            idx = torch.sort(
                random_choice(self.n, size=m, replace=False, rng=self.rng)
            ).values
            self.mask[idx] = 1
            self.mask = self.mask.type(torch.bool)

            self.D = torch.nn.Parameter(self.D, requires_grad=False)
            self.mask = torch.nn.Parameter(self.mask, requires_grad=False)
        else:
            self._A = torch.randn(
                (m, n), device=device, dtype=dtype, generator=self.rng
            ) / np.sqrt(m)
            self._A_dagger = torch.linalg.pinv(self._A)
            self._A = torch.nn.Parameter(self._A, requires_grad=False)
            self._A_dagger = torch.nn.Parameter(self._A_dagger, requires_grad=False)
            self._A_adjoint = (
                torch.nn.Parameter(self._A.conj().T, requires_grad=False)
                .type(dtype)
                .to(device)
            )

    def A(self, x, **kwargs):
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

    def A_adjoint(self, y, **kwargs):
        y = y.type(self.dtype)
        N = y.shape[0]
        C, H, W = self.img_shape[0], self.img_shape[1], self.img_shape[2]

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

    def A_dagger(self, y, **kwargs):
        y = y.type(self.dtype)
        if self.fast:
            return self.A_adjoint(y)
        else:
            N = y.shape[0]
            C, H, W = self.img_shape[0], self.img_shape[1], self.img_shape[2]

            if self.channelwise:
                y = y.reshape(N * C, -1)

            x = torch.einsum("im, nm->in", y, self._A_dagger)
            x = x.reshape(N, C, H, W)
        return x


# if __name__ == "__main__":
#     device = "cuda:0"
#
#     # for comparing fast=True and fast=False forward matrices.
#     for i in range(1):
#         n = 2 ** (i + 4)
#         im_size = (1, n, n)
#         m = int(np.prod(im_size))
#         x = torch.randn((1,) + im_size, device=device)
#
#         print((dst1(dst1(x)) - x).flatten().abs().sum())
#
#         physics = CompressedSensing(img_shape=im_size, m=m, fast=True, device=device)
#
#         print((physics.A_adjoint(physics.A(x)) - x).flatten().abs().sum())
#         print(f"adjointness: {physics.adjointness_test(x)}")
#         print(f"norm: {physics.power_method(x, verbose=False)}")
#         start = torch.cuda.Event(enable_timing=True)
#         end = torch.cuda.Event(enable_timing=True)
#         start.record()
#         for j in range(100):
#             y = physics.A(x)
#             xhat = physics.A_dagger(y)
#         end.record()
#
#         # print((xhat-x).pow(2).flatten().mean())
#
#         # Waits for everything to finish running
#         torch.cuda.synchronize()
#         print(start.elapsed_time(end))
