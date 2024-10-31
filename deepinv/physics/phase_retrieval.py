from functools import partial
import math

import numpy as np
import torch

from deepinv.physics.compressed_sensing import CompressedSensing
from deepinv.physics.forward import Physics, LinearPhysics
from deepinv.optim.phase_retrieval import spectral_methods


def compare(a: int, b: int):
    r"""
    Compare two numbers.

    :param int a: First number.
    :param int b: Second number.

    :return: The comparison result (>, < or =).
    """
    if a > b:
        return ">"
    elif a < b:
        return "<"
    else:
        return "="


def merge_order(a: str, b: str):
    r"""
    Merge two orders.

    If a and b are the same, return the same order.
    If a and b are one ">" and one "<", return "!".
    If a or b is "=", return the other order.

    :param str a: First order.
    :param str b: Second order.

    :return: The merged order.
    """
    if a == ">" and b == "<":
        return "!"
    elif a == "<" and b == ">":
        return "!"
    elif a == ">" or b == ">":
        return ">"
    elif a == "<" or b == "<":
        return "<"
    else:
        return "="


def padding(tensor: torch.Tensor, input_shape: tuple, output_shape: tuple):
    r"""
    Zero padding function for oversampling in structured random phase retrieval.

    :param torch.Tensor tensor: input tensor.
    :param tuple input_shape: shape of the input tensor.
    :param tuple output_shape: shape of the output tensor.

    :return: (torch.Tensor) the zero-padded tensor.
    """
    change_top = math.ceil(abs(input_shape[1] - output_shape[1]) / 2)
    change_bottom = math.floor(abs(input_shape[1] - output_shape[1]) / 2)
    change_left = math.ceil(abs(input_shape[2] - output_shape[2]) / 2)
    change_right = math.floor(abs(input_shape[2] - output_shape[2]) / 2)
    assert change_top + change_bottom == abs(input_shape[1] - output_shape[1])
    assert change_left + change_right == abs(input_shape[2] - output_shape[2])
    return torch.nn.ZeroPad2d((change_left, change_right, change_top, change_bottom))(
        tensor
    )


def trimming(tensor: torch.Tensor, input_shape: tuple, output_shape: tuple):
    r"""
    Trimming function for undersampling in structured random phase retrieval.

    :param torch.Tensor tensor: input tensor.
    :param tuple input_shape: shape of the input tensor.
    :param tuple output_shape: shape of the output tensor.

    :return: (torch.Tensor) the trimmed tensor.
    """
    change_top = math.ceil(abs(input_shape[1] - output_shape[1]) / 2)
    change_bottom = math.floor(abs(input_shape[1] - output_shape[1]) / 2)
    change_left = math.ceil(abs(input_shape[2] - output_shape[2]) / 2)
    change_right = math.floor(abs(input_shape[2] - output_shape[2]) / 2)
    assert change_top + change_bottom == abs(input_shape[1] - output_shape[1])
    assert change_left + change_right == abs(input_shape[2] - output_shape[2])
    if change_bottom == 0:
        tensor = tensor[..., change_top:, :]
    else:
        tensor = tensor[..., change_top:-change_bottom, :]
    if change_right == 0:
        tensor = tensor[..., change_left:]
    else:
        tensor = tensor[..., change_left:-change_right]
    return tensor


def generate_diagonal(
    shape,
    mode,
    dtype=torch.cfloat,
    device="cpu",
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
        self.name = "Phase Retrieval"

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
    :param bool use_haar: Use Haar matrix instead of Gaussian matrix. Default is False.
    :param bool compute_inverse: Compute the pseudo-inverse of the forward matrix. Default is False.
    :param torch.type dtype: Forward matrix is stored as a dtype. Default is torch.cfloat.
    :param str device: Device to store the forward matrix.
    :param torch.Generator (Optional) rng: a pseudorandom random number generator for the parameter generation.
        If ``None``, the default Generator of PyTorch will be used.

    |sep|

    :Examples:

        Random phase retrieval operator with 10 measurements for a 3x3 image:

        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn((1, 1, 3, 3),dtype=torch.cfloat) # Define random 3x3 image
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
        dtype=torch.cfloat,
        device="cpu",
        use_haar=False,
        compute_inverse=False,
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
            use_haar=use_haar,
            compute_inverse=compute_inverse,
            dtype=dtype,
            device=device,
            rng=self.rng,
        )
        super().__init__(B, **kwargs)
        self.name = "Random Phase Retrieval"

    def get_A_squared_mean(self):
        return self.B._A.var() + self.B._A.mean() ** 2


class StructuredRandomLinearOperator(LinearPhysics):
    r"""
    Linear operator for structured random phase retrieval.

    :param tuple input_shape: shape (C, H, W) of inputs.
    :param tuple output_shape: shape (C, H, W) of outputs.
    :param str mode: sampling mode.
    :param float n_layers: number of layers.
    :param function transform_func: structured transform function.
    :param function transform_func_inv: structured inverse transform function.
    :param list diagonals: list of diagonal matrices.
    """

    def __init__(
        self,
        input_shape,
        output_shape,
        mode,
        n_layers,
        transform_func,
        transform_func_inv,
        diagonals,
        **kwargs,
    ):

        def A(x):
            assert (
                x.shape[1:] == input_shape
            ), f"x doesn't have the correct shape {x.shape[1:]} != {input_shape}"

            if mode == "oversampling":
                x = padding(x, input_shape, output_shape)

            if n_layers - math.floor(n_layers) == 0.5:
                x = transform_func(x)
            for i in range(math.floor(n_layers)):
                diagonal = diagonals[i]
                x = diagonal * x
                x = transform_func(x)

            if mode == "undersampling":
                x = trimming(x, input_shape, output_shape)

            return x

        def A_adjoint(y):
            assert (
                y.shape[1:] == output_shape
            ), f"y doesn't have the correct shape {y.shape[1:]} != {self.output_shape}"

            if mode == "undersampling":
                y = padding(y, input_shape, output_shape)

            for i in range(math.floor(n_layers)):
                diagonal = diagonals[-i - 1]
                y = transform_func_inv(y)
                y = torch.conj(diagonal) * y
            if n_layers - math.floor(n_layers) == 0.5:
                y = transform_func_inv(y)

            if mode == "oversampling":
                y = trimming(y, input_shape, output_shape)

            return y

        super().__init__(A=A, A_adjoint=A_adjoint, **kwargs)


class StructuredRandomPhaseRetrieval(PhaseRetrieval):
    r"""
    Structured random phase retrieval model corresponding to the operator

    .. math::

        A(x) = |\prod_{i=1}^N (F D_i) x|^2,

    where :math:`F` is the Discrete Fourier Transform (DFT) matrix, and :math:`D_i` are diagonal matrices with elements of unit norm and random phases, and :math:`N` refers to the number of layers. It is also possible to replace :math:`x` with :math:`Fx` as an additional 0.5 layer.

    The phase of the diagonal elements of the matrices :math:`D_i` are drawn from a uniform distribution in the interval :math:`[0, 2\pi]`.

    :param tuple input_shape: shape (C, H, W) of inputs.
    :param tuple output_shape: shape (C, H, W) of outputs.
    :param float n_layers: number of layers :math:`N`. If ``layers=N + 0.5``, a first :math`F` transform is included, ie :math:`A(x)=|\prod_{i=1}^N (F D_i) F x|^2`
    :param str transform: structured transform to use. Default is 'fft'.
    :param str diagonal_mode: sampling distribution for the diagonal elements. Default is 'uniform_phase'.
    :param bool shared_weights: if True, the same diagonal matrix is used for all layers. Default is False.
    :param torch.type dtype: Signals are processed in dtype. Default is torch.cfloat.
    :param str device: Device for computation. Default is 'cpu'.
    """

    def __init__(
        self,
        input_shape: tuple,
        output_shape: tuple,
        n_layers: int,
        transform="fft",
        diagonal_mode="uniform_phase",
        shared_weights=False,
        dtype=torch.cfloat,
        device="cpu",
        **kwargs,
    ):
        if output_shape is None:
            output_shape = input_shape

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

        self.dtype = dtype
        self.device = device

        # determine the sampling mode
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

        # generate diagonal matrices
        self.diagonals = []

        if not shared_weights:
            for _ in range(math.floor(self.n_layers)):
                if self.mode == "oversampling":
                    diagonal = generate_diagonal(
                        shape=self.output_shape,
                        mode=diagonal_mode,
                        dtype=self.dtype,
                        device=self.device,
                    )
                else:
                    diagonal = generate_diagonal(
                        shape=self.input_shape,
                        mode=diagonal_mode,
                        dtype=self.dtype,
                        device=self.device,
                    )
                self.diagonals.append(diagonal)
        else:
            if self.mode == "oversampling":
                diagonal = generate_diagonal(
                    shape=self.output_shape,
                    mode=diagonal_mode,
                    dtype=self.dtype,
                    device=self.device,
                )
            else:
                diagonal = generate_diagonal(
                    shape=self.input_shape,
                    mode=diagonal_mode,
                    dtype=self.dtype,
                    device=self.device,
                )
            self.diagonals = self.diagonals + [diagonal] * math.floor(self.n_layers)

        # determine transform functions
        if transform == "fft":
            transform_func = partial(torch.fft.fft2, norm="ortho")
            transform_func_inv = partial(torch.fft.ifft2, norm="ortho")
        else:
            raise ValueError(f"Unimplemented transform: {transform}")

        B = StructuredRandomLinearOperator(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            mode=self.mode,
            n_layers=self.n_layers,
            transform_func=transform_func,
            transform_func_inv=transform_func_inv,
            diagonals=self.diagonals,
            **kwargs,
        )

        super().__init__(B, **kwargs)
        self.name = "Structured Random Phase Retrieval"

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
