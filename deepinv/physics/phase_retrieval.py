from functools import partial
import math

from dotmap import DotMap
import numpy as np
from scipy.fft import dct, idct
import torch

from deepinv.physics.compressed_sensing import CompressedSensing
from deepinv.physics.forward import Physics, LinearPhysics
from deepinv.optim.phase_retrieval import compare, merge_order, spectral_methods


def generate_diagonal(
    shape,
    mode,
    dtype=torch.complex64,
    device="cpu",
    config: DotMap = None,
):
    r"""
    Generate a random tensor as the diagonal matrix.
    """

    #! all distributions should be normalized to have E[|x|^2] = 1
    if mode == "uniform_phase":
        # Generate REAL-VALUED random numbers in the interval [0, 1)
        diag = torch.rand(shape)
        diag = 2 * np.pi * diag
        diag = torch.exp(1j * diag)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return diag.to(device)


class PhaseRetrieval(Physics):
    r"""
    Phase Retrieval base class corresponding to the operator

    .. math::

        \forw{x} = |Bx|^2.

    The linear operator :math:`B` is defined by a :class:`deepinv.physics.LinearPhysics` object.

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

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Applies the forward operator to the input x.

        Note here the operation includes the modulus operation.

        :param torch.Tensor x: signal/image.
        """
        return self.B(x, **kwargs).abs().square()

    def A_dagger(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Computes an initial reconstruction for the image :math:`x` from the measurements :math:`y`.

        We use the spectral methods defined in :class:`deepinv.optim.phase_retrieval.spectral_methods` to obtain an initial inverse.

        :param torch.Tensor y: measurements.
        :return: (torch.Tensor) an initial reconstruction for image :math:`x`.
        """
        return spectral_methods(y, self, **kwargs)

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.A_dagger(y, **kwargs)

    def B_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.B.A_adjoint(y, **kwargs)

    def B_dagger(self, y):
        r"""
        Computes the linear pseudo-inverse of :math:`B`.

        :param torch.Tensor y: measurements.
        :return: (torch.Tensor) the reconstruction image :math:`x`.
        """
        return self.B.A_dagger(y)

    def forward(self, x, **kwargs):
        r"""
        Applies the phase retrieval measurement operator, i.e. :math:`y = \noise{|Bx|^2}` (with noise :math:`N` and/or sensor non-linearities).

        :param torch.Tensor,list[torch.Tensor] x: signal/image
        :return: (torch.Tensor) noisy measurements
        """
        return self.sensor(self.noise(self.A(x, **kwargs)))

    def A_vjp(self, x, v):
        r"""
        Computes the product between a vector :math:`v` and the Jacobian of the forward operator :math:`A` at the input x, defined as:

        .. math::

            A_{vjp}(x, v) = 2 \overline{B}^{\top} \text{diag}(Bx) v.

        :param torch.Tensor x: signal/image.
        :param torch.Tensor v: vector.
        :return: (torch.Tensor) the VJP product between :math:`v` and the Jacobian.
        """
        return 2 * self.B_adjoint(self.B(x) * v)

    def release_memory(self):
        del self.B
        torch.cuda.empty_cache()
        return


class RandomPhaseRetrieval(PhaseRetrieval):
    r"""
    Random Phase Retrieval forward operator. Creates a random :math:`m \times n` sampling matrix :math:`B` where :math:`n` is the number of elements of the signal and :math:`m` is the number of measurements.

    This class generates a random i.i.d. Gaussian matrix

    .. math::

        B_{i,j} \sim \mathcal{N} \left( 0, \frac{1}{2m} \right) + \mathrm{i} \mathcal{N} \left( 0, \frac{1}{2m} \right).

    An existing operator can be loaded from a saved .pth file via ``self.load_state_dict(save_path)``, in a similar fashion to :class:`torch.nn.Module`.

    :param int m: number of measurements.
    :param tuple img_shape: shape (C, H, W) of inputs.
    :param bool channelwise: Channels are processed independently using the same random forward operator.
    :param torch.type dtype: Forward matrix is stored as a dtype. Default is torch.complex64.
    :param str device: Device to store the forward matrix.
    :param torch.Generator (Optional) rng: a pseudorandom random number generator for the parameter generation.
        If ``None``, the default Generator of PyTorch will be used.

    |sep|

    :Examples:

        Random phase retrieval operator with 10 measurements for a 3x3 image:

        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn((1, 1, 3, 3),dtype=torch.complex64) # Define random 3x3 image
        >>> physics = RandomPhaseRetrieval(m=10,img_shape=(1, 3, 3), rng=torch.Generator('cpu'))
        >>> physics(x)
        tensor([[1.1901, 4.0743, 0.1858, 2.3197, 0.0734, 0.4557, 0.1231, 0.6597, 1.7768,
                    0.3864]])
    """

    def __init__(
        self,
        m,
        img_shape,
        channelwise=False,
        dtype=torch.complex64,
        device="cpu",
        config: DotMap = DotMap(),
        rng: torch.Generator = None,
        **kwargs,
    ):
        self.m = m
        self.input_shape = img_shape
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

        B = CompressedSensing(
            m=m,
            img_shape=img_shape,
            fast=False,
            channelwise=channelwise,
            dtype=dtype,
            device=device,
            config=config,
            rng=self.rng,
        )
        super().__init__(B, **kwargs)
        self.name = f"RPR_m{self.m}"

    def get_A_squared_mean(self):
        return self.B._A.var() + self.B._A.mean() ** 2


class StructuredRandomPhaseRetrieval(PhaseRetrieval):
    r"""
    Structured Random Phase Retrieval class corresponding to the operator

    .. math::

        A(x) = |F \prod_{i=1}^N (D_i F) x|^2,

    where :math:`F` is the Discrete Fourier Transform (DFT) matrix, and :math:`D_i` are diagonal matrices with elements of unit norm and random phases, and :math:`N` is the number of layers.

    The phase of the diagonal elements of the matrices :math:`D_i` are drawn from a uniform distribution in the interval :math:`[0, 2\pi]`.

    :param tuple input_shape: shape (C, H, W) of inputs.
    :param tuple output_shape: shape (C, H, W) of outputs.
    :param int n_layers: number of layers. an extra F is at the end if there is a 0.5
    :param str transform: structured transform to use. Default is 'fft'.
    :param str diagonal_mode: sampling distribution for the diagonal elements. Default is 'uniform_phase'.
    :param DotMap distri_config: configuration for the diagonal distribution.
    :param bool shared_weights: if True, the same diagonal matrix is used for all layers. Default is False.
    :param torch.type dtype: Signals are processed in dtype. Default is torch.complex64.
    :param str device: Device for computation.
    """

    def __init__(
        self,
        input_shape: tuple,
        output_shape: tuple,
        n_layers: int,
        transform="fft",
        diagonal_mode="uniform_phase",
        distri_config: DotMap = DotMap(),
        shared_weights=False,
        dtype=torch.complex64,
        device="cpu",
        **kwargs,
    ):
        if output_shape is None:
            output_shape = input_shape

        height_order = compare(input_shape[1], output_shape[1])
        width_order = compare(input_shape[2], output_shape[2])

        order = merge_order(height_order, width_order)

        if order == "<":
            self.mode = "oversampling"
        elif order == ">":
            self.mode = "undersampling"
        elif order == "=":
            self.mode = "equisampling"
        else:
            raise ValueError(
                "Does not support different sampling schemes on height and width."
            )

        change_top = math.ceil(abs(input_shape[1] - output_shape[1]) / 2)
        change_bottom = math.floor(abs(input_shape[1] - output_shape[1]) / 2)
        change_left = math.ceil(abs(input_shape[2] - output_shape[2]) / 2)
        change_right = math.floor(abs(input_shape[2] - output_shape[2]) / 2)
        assert change_top + change_bottom == abs(input_shape[1] - output_shape[1])
        assert change_left + change_right == abs(input_shape[2] - output_shape[2])

        def padding(tensor: torch.Tensor):
            return torch.nn.ZeroPad2d(
                (change_left, change_right, change_top, change_bottom)
            )(tensor)

        self.padding = padding

        def trimming(tensor: torch.Tensor):
            if change_bottom == 0:
                tensor = tensor[..., change_top:, :]
            else:
                tensor = tensor[..., change_top:-change_bottom, :]
            if change_right == 0:
                tensor = tensor[..., change_left:]
            else:
                tensor = tensor[..., change_left:-change_right]
            return tensor

        self.trimming = trimming

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n = torch.prod(torch.tensor(self.input_shape))
        self.m = torch.prod(torch.tensor(self.output_shape))
        self.oversampling_ratio = self.m / self.n
        assert (
            n_layers % 1 == 0.5 or n_layers % 1 == 0
        ), "n_layers must be an integer or an integer plus 0.5"
        self.n_layers = n_layers
        self.structure = self.get_structure(self.n_layers)
        self.shared_weights = shared_weights
        self.distri_config = distri_config
        self.distri_config.m = self.m
        self.distri_config.n = self.n

        self.dtype = dtype
        self.device = device

        self.diagonals = []

        if not shared_weights:
            for _ in range(math.floor(self.n_layers)):
                if self.mode == "oversampling":
                    diagonal = generate_diagonal(
                        self.output_shape,
                        mode=diagonal_mode,
                        dtype=self.dtype,
                        device=self.device,
                        config=self.distri_config,
                    )
                else:
                    diagonal = generate_diagonal(
                        self.input_shape,
                        mode=diagonal_mode,
                        dtype=self.dtype,
                        device=self.device,
                        config=self.distri_config,
                    )
                self.diagonals.append(diagonal)
        else:
            if self.mode == "oversampling":
                diagonal = generate_diagonal(
                    self.output_shape,
                    mode=diagonal_mode,
                    dtype=self.dtype,
                    device=self.device,
                    config=self.distri_config,
                )
            else:
                diagonal = generate_diagonal(
                    self.input_shape,
                    mode=diagonal_mode,
                    dtype=self.dtype,
                    device=self.device,
                    config=self.distri_config,
                )
            self.diagonals = self.diagonals + [diagonal] * math.floor(self.n_layers)

        if transform == "fft":
            transform_func = partial(torch.fft.fft2, norm="ortho")
            transform_func_inv = partial(torch.fft.ifft2, norm="ortho")
        else:
            raise ValueError(f"Unimplemented transform: {transform}")

        def A(x):
            assert (
                x.shape[1:] == self.input_shape
            ), f"x doesn't have the correct shape {x.shape[1:]} != {self.input_shape}"

            if self.mode == "oversampling":
                x = self.padding(x)

            if self.n_layers - math.floor(self.n_layers) == 0.5:
                x = transform_func(x)
            for i in range(math.floor(self.n_layers)):
                diagonal = self.diagonals[i]
                x = diagonal * x
                x = transform_func(x)

            if self.mode == "undersampling":
                x = self.trimming(x)

            return x

        def A_adjoint(y):
            assert (
                y.shape[1:] == self.output_shape
            ), f"y doesn't have the correct shape {y.shape[1:]} != {self.output_shape}"

            if self.mode == "undersampling":
                y = self.padding(y)

            for i in range(math.floor(self.n_layers)):
                diagonal = self.diagonals[-i - 1]
                y = transform_func_inv(y)
                y = torch.conj(diagonal) * y
            if self.n_layers - math.floor(self.n_layers) == 0.5:
                y = transform_func_inv(y)

            if self.mode == "oversampling":
                y = self.trimming(y)

            return y

        super().__init__(LinearPhysics(A=A, A_adjoint=A_adjoint), **kwargs)
        self.name = f"PRPR_m{self.m}"

    def B_dagger(self, y):
        return self.B.A_adjoint(y)

    def get_A_squared_mean(self):
        if self.n_layers == 0.5:
            print(
                "warning: computing the mean of the squared operator for a single Fourier transform."
            )
            return None
        return self.diagonals[0].var() + self.diagonals[0].mean() ** 2

    @staticmethod
    def get_structure(n_layers) -> str:
        r"""Returns the structure of the operator as a string.

        :param float n_layers: number of layers.

        :return: (str) the structure of the operator, e.g., "FDFD".
        """
        return "FD" * math.floor(n_layers) + "F" * (n_layers % 1 == 0.5)
