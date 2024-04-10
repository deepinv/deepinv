from deepinv.physics.forward import Physics, LinearPhysics
from deepinv.physics.compressed_sensing import CompressedSensing
from deepinv.optim.phase_retrieval import spectral_methods
import torch
import numpy as np


class PhaseRetrieval(Physics):
    r"""
    Phase Retrieval base class corresponding to the operator

    .. math::

        A(x) = |Bx|^2.

    The linear operator :math:`B` is defined by a :meth:`deepinv.physics.LinearPhysics` object.

    An existing operator can be loaded from a saved .pth file via ``self.load_state_dict(save_path)``, in a similar fashion to :class:`torch.nn.Module`.

    :param deepinv.physics.forward.LinearPhysics B: the linear forward operator.
    """

    def __init__(
        self,
        B: LinearPhysics,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = f"PR_m{self.m}"

        self.B = B

    def A(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Applies the forward operator to the input x.

        Note here the operation includes the modulus operation.

        :param torch.Tensor x: signal/image.
        """
        return self.B(x).abs().square()

    def A_dagger(self, y: torch.Tensor) -> torch.Tensor:
        r"""
        Computes a initial reconstruction for the image :math:`x` from the measurements :math:`y`.

        :param torch.Tensor y: measurements.
        :return: (torch.Tensor) an initial reconstruction for image :math:`x`.
        """
        return spectral_methods(50, torch.rand((y.shape[0],) + self.img_shape), y, self)

    def B_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return self.B.A_adjoint(y)

    def B_dagger(self, y):
        r"""
        Computes the linear pseudo-inverse of :math:`B`.

        :param torch.Tensor y: measurements.
        :return: (torch.Tensor) the reconstruction image :math:`x`.
        """
        return self.B.A_dagger(y)

    def forward(self, x):
        r"""
        Applies the phase retrieval measurement operator, i.e. :math:`y = N(|Bx|^2)` (with noise :math:`N` and/or sensor non-linearities).

        :param torch.Tensor,list[torch.Tensor] x: signal/image
        :return: (torch.Tensor) noisy measurements
        """
        return self.sensor(self.noise(self.A(x)))

    def A_vjp(self, x, v):
        r"""
        Computes the product between a vector :math:`v` and the Jacobian of the forward operator :math:`A` at the input x, defined as:

        .. math::

            A_{vjp}(x, v) = 2 \overline{B}^{\top} diag(Bx) v.

        :param torch.Tensor x: signal/image.
        :param torch.Tensor v: vector.
        :return: (torch.Tensor) the VJP product between :math:`v` and the Jacobian.
        """
        return 2 * self.B_adjoint(self.B(x) * v)


class RandomPhaseRetrieval(PhaseRetrieval):
    r"""
    Random Phase Retrieval forward operator. Creates a random :math:`m \times n` sampling matrix :math:`B` where :math:`n` is the number of elements of the signal and :math:`m` is the number of measurements.

    This class generates a random i.i.d. Gaussian matrix

    .. math::

        B_{i,j} \sim \mathcal{N} \left( 0, \frac{1}{2m}) \right) + \mathrm{i} \mathcal{N} \left( 0, \frac{1}{2m} \right).

    An existing operator can be loaded from a saved .pth file via ``self.load_state_dict(save_path)``, in a similar fashion to :class:`torch.nn.Module`.

    :param int m: number of measurements.
    :param tuple img_shape: shape (C, H, W) of inputs.
    :param bool channelwise: Channels are processed independently using the same random forward operator.
    :param torch.type dtype: Forward matrix is stored as a dtype. Default is torch.cfloat.
    :param str device: Device to store the forward matrix.

    |sep|

    :Examples:

        Random phase retrieval operator with 10 measurements for a 3x3 image:

        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn((1, 1, 3, 3),dtype=torch.cfloat) # Define random 3x3 image
        >>> physics = RandomPhaseRetrieval(m=10,img_shape=(1, 3, 3))
        >>> physics(x)
        tensor([[1.1901, 4.0743, 0.1858, 2.3197, 0.0734, 0.4557, 0.1231, 0.6597, 1.7768,
                 0.3864]])
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
        self.m = m
        self.img_shape = img_shape
        self.channelwise = channelwise
        self.dtype = dtype
        self.device = device
        B = CompressedSensing(
            m=m,
            img_shape=img_shape,
            fast=False,
            channelwise=channelwise,
            dtype=dtype,
            device=device,
        )
        super().__init__(B)
        self.name = f"RPR_m{self.m}"


class PseudoRandomPhaseRetrieval(PhaseRetrieval):
    r"""
    Pseudo-random Phase Retrieval class corresponding to the operator

    .. math::

        A(x) = |F \prod_{i=1}^N (D_i F) x|^2,

    where :math:`F` is the Discrete Fourier Transform (DFT) matrix, and :math:`D_i` are diagonal matrices with elements of unit norm and random phases, and :math:`N` is the number of layers.

    The phase of the diagonal elements of the matrices :math:`D_i` are drawn from a uniform distribution in the interval :math:`[0, 2\pi]`.

    :param int n_layers: number of layers.
    :param tuple img_shape: shape (C, H, W) of inputs.
    :param torch.type dtype: Signals are processed in dtype. Default is torch.cfloat.
    :param str device: Device for computation.
    """

    def __init__(
        self,
        n_layers,
        img_shape,
        zero_padding=0,
        dtype=torch.cfloat,
        device="cpu",
        **kwargs,
    ):
        self.n_layers = n_layers
        self.zero_padding = zero_padding

        self.img_shape = img_shape
        self.img_shape_padding = (
            img_shape[0],
            img_shape[1] + 2 * zero_padding,
            img_shape[2] + 2 * zero_padding,
        )

        self.n = torch.prod(torch.tensor(self.img_shape))
        self.m = torch.prod(torch.tensor(self.img_shape_padding))

        self.dtype = dtype
        self.device = device
        self.diagonals = []
        for _ in range(self.n_layers):
            diagonal = torch.rand(self.img_shape_padding)
            diagonal = 2 * torch.pi * diagonal
            diagonal = torch.exp(1j * diagonal)
            self.diagonals.append(diagonal)

        def A(x):
            x = torch.nn.ZeroPad2d(self.zero_padding)(x)
            for i in range(self.n_layers):
                diagonal = self.diagonals[i]
                x = torch.fft.fft2(x, norm="ortho")
                x = diagonal * x
            x = torch.fft.fft2(x, norm="ortho")
            return x

        def A_adjoint(y):
            for i in range(self.n_layers):
                diagonal = self.diagonals[-i - 1]
                y = torch.fft.ifft2(y, norm="ortho")
                y = torch.conj(diagonal) * y
            y = torch.fft.ifft2(y, norm="ortho")
            return y[
                :,
                :,
                self.zero_padding : -self.zero_padding,
                self.zero_padding : -self.zero_padding,
            ]

        super().__init__(LinearPhysics(A=A, A_adjoint=A_adjoint))
        self.name = f"PRPR_m{self.m}"

    def B_dagger(self, y):
        return self.B.A_adjoint(y)
