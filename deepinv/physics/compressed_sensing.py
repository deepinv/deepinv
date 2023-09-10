from deepinv.physics.forward import LinearPhysics
import torch
import numpy as np


def dst1(x):
    r"""
    Orthogonal Discrete Sine Transform, Type I
    The transform is performed across the last dimension of the input signal
    Due to orthogonality we have ``dst1(dst1(x)) = x``.

    :param torch.tensor x: the input signal
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

    It is recommended to use ``fast=True`` for image sizes bigger than 32 x 32, since the forward computation with
    ``fast=False`` has an :math:`O(mn)` complexity, whereas with ``fast=True`` it has an :math:`O(n \log n)` complexity.

    An existing operator can be loaded from a saved .pth file via ``self.load_state_dict(save_path)``,
    in a similar fashion to :class:`torch.nn.Module`.

    .. note::

        If ``fast=False``, the forward operator has a norm which tends to :math:`(1+\sqrt{n/m})^2` for large :math:`n`
        and :math:`m` due to the `Marcenko-Pastur law
        <https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution>`_.
        If ``fast=True``, the forward operator has a unit norm.

    :param int m: number of measurements.
    :param tuple img_shape: shape (C, H, W) of inputs.
    :param bool fast: The operator is iid Gaussian if false, otherwise A is a SORS matrix with the Discrete Sine Transform (type I).
    :param bool channelwise: Channels are processed independently using the same random forward operator.
    :param torch.type dtype: Forward matrix is stored as a dtype.
    :param str device: Device to store the forward matrix.

    """

    def __init__(
        self,
        m,
        img_shape,
        fast=False,
        channelwise=False,
        dtype=torch.float,
        device="cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = f"CS_m{m}"
        self.img_shape = img_shape
        self.fast = fast
        self.channelwise = channelwise
        self.dtype = dtype

        if channelwise:
            n = int(np.prod(img_shape[1:]))
        else:
            n = int(np.prod(img_shape))

        if self.fast:
            self.n = n
            self.D = torch.ones(self.n, device=device)
            self.D[torch.rand_like(self.D) > 0.5] = -1.0
            self.mask = torch.zeros(self.n, device=device)
            idx = np.sort(np.random.choice(self.n, size=m, replace=False))
            self.mask[torch.from_numpy(idx)] = 1
            self.mask = self.mask.type(torch.bool)

            self.D = torch.nn.Parameter(self.D, requires_grad=False)
            self.mask = torch.nn.Parameter(self.mask, requires_grad=False)
        else:
            self._A = torch.randn((m, n), device=device) / np.sqrt(m)
            self._A_dagger = torch.linalg.pinv(self._A)
            self._A = torch.nn.Parameter(self._A, requires_grad=False)
            self._A_dagger = torch.nn.Parameter(self._A_dagger, requires_grad=False)
            self._A_adjoint = (
                torch.nn.Parameter(self._A.t(), requires_grad=False)
                .type(dtype)
                .to(device)
            )

    def A(self, x):
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

    def A_adjoint(self, y):
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

    def A_dagger(self, y):
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
