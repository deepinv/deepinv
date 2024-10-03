from functools import partial
import math

import numpy as np
from scipy.fft import dct, idct
import torch

from deepinv.physics.compressed_sensing import CompressedSensing
from deepinv.physics.forward import Physics, LinearPhysics
from deepinv.optim.phase_retrieval import compare,merge_order,spectral_methods

def dct2(x:torch.Tensor,device):
    r""" 2D DCT

    DCT is performed along the last two dimensions of the input tensor.
    """
    return torch.from_numpy(dct(dct(x.cpu().numpy(), axis=-1, norm='ortho'), axis=-2, norm='ortho')).to(device)

def idct2(x:torch.Tensor,device):
    r""" 2D IDCT

    IDCT is performed along the last two dimensions of the input tensor.
    """
    return torch.from_numpy(idct(idct(x.cpu().numpy(), axis=-2, norm='ortho'), axis=-1, norm='ortho')).to(device)

def generate_diagonal(
    tensor_shape,
    mode,
    dtype=torch.complex64,
    device="cpu",
    df=3
):
    r"""
    Generate a random tensor as the diagonal matrix.
    """

    #! all distributions should be normalized to have unit variance
    if mode == "uniform_phase":
        # Generate REAL-VALUED random numbers in the interval [0, 1)
        diagonal = torch.rand(tensor_shape, device=device)
        diagonal = 2 * np.pi * diagonal
        diagonal = torch.exp(1j * diagonal)
    elif mode == "gaussian":
        diagonal = torch.randn(tensor_shape, dtype=dtype, device=device)
    elif mode == "laplace":
        #! variance = 2*scale^2
        #! variance of complex numbers is doubled
        laplace_dist = torch.distributions.laplace.Laplace(0,0.5)
        diagonal = (laplace_dist.sample(tensor_shape) + 1j*laplace_dist.sample(tensor_shape)).to(device)
    elif mode == "student-t":
        #! variance = df/(df-2) if df > 2
        #! variance of complex numbers is doubled
        student_t_dist = torch.distributions.studentT.StudentT(df,0,1)
        scale = torch.sqrt((torch.tensor(df)-2)/torch.tensor(df)/2)
        diagonal = (scale*(student_t_dist.sample(tensor_shape) + 1j*student_t_dist.sample(tensor_shape))).to(device)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return diagonal

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

    def A_dagger(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Computes a initial reconstruction for the image :math:`x` from the measurements :math:`y`.

        :param torch.Tensor y: measurements.
        :return: (torch.Tensor) an initial reconstruction for image :math:`x`.
        """
        return spectral_methods(y, self, **kwargs)

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.A_dagger(y, **kwargs)

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

    |sep|

    :Examples:

        Random phase retrieval operator with 10 measurements for a 3x3 image:

        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn((1, 1, 3, 3),dtype=torch.complex64) # Define random 3x3 image
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
        dtype=torch.complex64,
        device="cpu",
        use_haar=False,
        test=False,
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
            use_haar=use_haar,
            test=test,
        )
        super().__init__(B, **kwargs)
        self.name = f"RPR_m{self.m}"
    
    def get_A_var(self):
        return self.B._A.var()


class StructuredRandomPhaseRetrieval(PhaseRetrieval):
    r"""
    Pseudo-random Phase Retrieval class corresponding to the operator

    .. math::

        A(x) = |F \prod_{i=1}^N (D_i F) x|^2,

    where :math:`F` is the Discrete Fourier Transform (DFT) matrix, and :math:`D_i` are diagonal matrices with elements of unit norm and random phases, and :math:`N` is the number of layers.

    The phase of the diagonal elements of the matrices :math:`D_i` are drawn from a uniform distribution in the interval :math:`[0, 2\pi]`.

    :param int n_layers: number of layers.
    :param tuple img_shape: shape (C, H, W) of inputs.
    :param torch.type dtype: Signals are processed in dtype. Default is torch.complex64.
    :param str device: Device for computation.
    """

    def __init__(
        self,
        n_layers:int,
        input_shape:tuple,
        output_shape:tuple,
        diagonal_mode="uniform_phase",
        transform="fft",
        shared_weights=False,
        drop_tail=True,
        dtype=torch.complex64,
        device="cpu",
        df=3,
        **kwargs,
    ):
        if output_shape is None:
            output_shape = input_shape

        self.n_layers = n_layers
        self.shared_weights = shared_weights
        self.drop_tail = drop_tail

        height_order = compare(input_shape[1:], output_shape[1:])
        width_order = compare(input_shape[2:], output_shape[2:])

        order = merge_order(height_order, width_order)

        if order == "<":
            self.mode = "oversampling"
        elif order == ">":
            self.mode = "undersampling"
        elif order == "=":
            self.mode = "equisampling"
        else:
            raise ValueError(f"Does not support different sampling schemes on height and width.")
        
        change_top = math.ceil(abs(input_shape[1] - output_shape[1])/2)
        change_bottom = math.floor(abs(input_shape[1] - output_shape[1])/2)
        change_left = math.ceil(abs(input_shape[2] - output_shape[2])/2)
        change_right = math.floor(abs(input_shape[2] - output_shape[2])/2)
        assert change_top + change_bottom == abs(input_shape[1] - output_shape[1])
        assert change_left + change_right == abs(input_shape[2] - output_shape[2])

        def padding(tensor: torch.Tensor):
            return torch.nn.ZeroPad2d((change_left,change_right,change_top,change_bottom))(tensor)
        self.padding = padding

        def trimming(tensor: torch.Tensor):
            if change_bottom == 0:
                tensor = tensor[...,change_top:,:]
            else:
                tensor = tensor[...,change_top:-change_bottom,:]
            if change_right == 0:
                tensor = tensor[...,change_left:]
            else:
                tensor = tensor[...,change_left:-change_right]
            return tensor
        self.trimming = trimming

        self.img_shape = input_shape
        self.output_shape = output_shape
        self.n = torch.prod(torch.tensor(self.img_shape))
        self.m = torch.prod(torch.tensor(self.output_shape))
        self.oversampling_ratio = self.m / self.n

        self.dtype = dtype
        self.device = device

        self.diagonals = []

        if not shared_weights:
            for _ in range(self.n_layers):
                if self.mode == "oversampling":
                    diagonal = generate_diagonal(self.output_shape, mode=diagonal_mode, dtype=self.dtype, device=self.device, df=df)
                else:
                    diagonal = generate_diagonal(self.img_shape, mode=diagonal_mode, dtype=self.dtype, device=self.device, df=df)
                self.diagonals.append(diagonal)
        else:
            if self.mode == "oversampling":
                diagonal = generate_diagonal(self.output_shape, mode=diagonal_mode, dtype=self.dtype, device=self.device, df=df)
            else:
                diagonal = generate_diagonal(self.img_shape, mode=diagonal_mode, dtype=self.dtype, device=self.device, df=df)
            self.diagonals = self.diagonals + [diagonal] * self.n_layers

        if transform == "fft":
            transform_func = partial(torch.fft.fft2, norm="ortho")
            transform_func_inv = partial(torch.fft.ifft2, norm="ortho")
        elif transform == "dct":
            transform_func = partial(dct2, device=self.device)
            transform_func_inv = partial(idct2, device=self.device)
        else:
            raise ValueError(f"Unimplemented transform: {transform}")
        
        def A(x):
            assert x.shape[1:] == self.img_shape, f"x doesn't have the correct shape {x.shape[1:]} != {self.img_shape}"

            if self.mode == "oversampling":
                x = self.padding(x)

            if not drop_tail:
                x = transform_func(x)
            for i in range(self.n_layers):
                diagonal = self.diagonals[i]
                x = diagonal * x
                x = transform_func(x)

            if self.mode == "undersampling":
                x = self.trimming(x)

            return x

        def A_adjoint(y):
            assert y.shape[1:] == self.output_shape, f"y doesn't have the correct shape {y.shape[1:]} != {self.output_shape}"

            if self.mode == "undersampling":
                y = self.padding(y)

            for i in range(self.n_layers):
                diagonal = self.diagonals[-i - 1]
                y = transform_func_inv(y)
                y = torch.conj(diagonal) * y
            if not drop_tail:
                y = transform_func_inv(y)

            if self.mode == "oversampling":
                y = self.trimming(y)

            return y

        super().__init__(LinearPhysics(A=A, A_adjoint=A_adjoint), **kwargs)
        self.name = f"PRPR_m{self.m}"

    def B_dagger(self, y):
        return self.B.A_adjoint(y)
    
    def get_A_var(self):
        return self.diagonals[0].var()