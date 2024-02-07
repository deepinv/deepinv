from deepinv.physics.forward import LinearPhysics
import torch
import numpy as np


class BasePhaseRetrieval(LinearPhysics):
    r"""
    Base Phase Retrieval forward operator. The matrices A, A_adjoint, and A_dagger are all initialized as zeros of proper dimensions.

    :param int m: number of measurements.
    :param tuple img_shape: shape (C, H, W) of inputs.
    :param bool channelwise: Channels are processed independently using the same random forward operator.
    :param torch.type dtype: Forward matrix is stored as a dtype.
    :param str device: Device to store the forward matrix.
    """

    def __init__(
        self,
        m,
        img_shape,
        channelwise=False,
        dtype=torch.cfloat,
        device="cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = f"PR_m{m}"
        self.img_shape = img_shape
        self.channelwise = channelwise
        self.dtype = dtype

        if channelwise:
            n = int(np.prod(img_shape[1:]))
        else:
            n = int(np.prod(img_shape))

        self._A = torch.zeros((m, n), device=device, dtype=self.dtype)
        self._A_adjoint = torch.zeros((n, m), device=device, dtype=self.dtype)
        self._A_dagger = torch.zeros((n, m), device=device, dtype=self.dtype)

    def A(self, x: torch.Tensor) -> torch.Tensor:
        N, C = x.shape[:2]
        if self.channelwise:
            x = x.reshape(N * C, -1)
        else:
            x = x.reshape(N, -1)

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

        x = torch.einsum("im, nm->in", y, self._A_adjoint)  # x:(N, n, 1)

        x = x.view(N, C, H, W)
        return x

    def A_dagger(self, y):
        N = y.shape[0]
        C, H, W = self.img_shape[0], self.img_shape[1], self.img_shape[2]

        if self.channelwise:
            y = y.reshape(N * C, -1)

        x = torch.einsum("im, nm->in", y, self._A_dagger)
        x = x.reshape(N, C, H, W)
        return x

    def forward(self, x):
        return self.sensor(self.noise(self.A(x).abs().square()))


class RandomPhaseRetrieval(BasePhaseRetrieval):
    r"""
    Random Phase Retrieval forward operator. Creates a random sampling :math:`m \times n` matrix where :math:`n` is the
    number of elements of the signal, i.e., ``np.prod(img_shape)`` and ``m`` is the number of measurements.

    This class generates a random i.i.d. Gaussian matrix

    .. math::

        (\text{Re}(A_{i,j}), \text{Im}(A_{i,j})) \sim \mathcal{N} \left( (0,0),\begin{pmatrix} \frac{1}{2m} & 0\\ 0 &  \frac{1}{2m} \end{pmatrix} \right).

    An existing operator can be loaded from a saved .pth file via ``self.load_state_dict(save_path)``,
    in a similar fashion to :class:`torch.nn.Module`.

    :param int m: number of measurements.
    :param tuple img_shape: shape (C, H, W) of inputs.
    :param bool channelwise: Channels are processed independently using the same random forward operator.
    :param torch.type dtype: Forward matrix is stored as a dtype.
    :param str device: Device to store the forward matrix.

    |sep|

    :Examples:

        Random phase retrieval operator with 10 measurements for a 3x3 image:

        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn((1, 1, 3, 3),dtype=torch.cfloat) # Define random 3x3 image
        >>> physics = RandomPhaseRetrieval(m=10,img_shape=(1, 3, 3))
        >>> physics(x)
        tensor([[0.9987, 2.1279, 0.7651, 3.1675, 0.5760, 0.2864, 0.0099, 0.4901, 0.6011, 1.4841]])

    """

    def __init__(
        self,
        m,
        img_shape,
        channelwise=False,
        dtype=torch.cfloat,
        device="cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = f"RPR_m{m}"
        self.img_shape = img_shape
        self.channelwise = channelwise
        self.dtype = dtype

        # elementwise modulus then square
        self.sensor_model = lambda x: torch.square(x.abs())

        if channelwise:
            n = int(np.prod(img_shape[1:]))
        else:
            n = int(np.prod(img_shape))

        A_real = torch.randn((m, n), device=device) / np.sqrt(2 * m)
        A_imag = torch.randn((m, n), device=device) / np.sqrt(2 * m)
        self._A = torch.view_as_complex(torch.stack((A_real, A_imag), dim=-1))
        # dagger is inverse, adjoint is conjugate transpose
        self._A_dagger = torch.linalg.pinv(self._A)
        self._A = torch.nn.Parameter(self._A, requires_grad=False)
        self._A_dagger = torch.nn.Parameter(self._A_dagger, requires_grad=False)
        self._A_adjoint = (
            torch.nn.Parameter(self._A.t().conj(), requires_grad=False)
            .type(dtype)
            .to(device)
        )
