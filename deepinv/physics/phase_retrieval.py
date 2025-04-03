from functools import partial
import math
import torch
import numpy as np
from deepinv.optim.phase_retrieval import spectral_methods
from deepinv.physics.compressed_sensing import CompressedSensing
from deepinv.physics.forward import Physics, LinearPhysics
from deepinv.physics.structured_random import (
    compare,
    generate_diagonal,
    StructuredRandom,
)


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
        :return: (:class:`torch.Tensor`) an initial reconstruction for image :math:`x`.
        """
        return spectral_methods(y, self, **kwargs)

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.B_adjoint(y, **kwargs)

    def B_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.B.A_adjoint(y, **kwargs)

    def B_dagger(self, y):
        r"""
        Computes the linear pseudo-inverse of :math:`B`.

        :param torch.Tensor y: measurements.
        :return: (:class:`torch.Tensor`) the reconstruction image :math:`x`.
        """
        return self.B.A_dagger(y)

    def forward(self, x, **kwargs):
        r"""
        Applies the phase retrieval measurement operator, i.e. :math:`y = \noise{|Bx|^2}` (with noise :math:`N` and/or sensor non-linearities).

        :param torch.Tensor,list[torch.Tensor] x: signal/image
        :return: (:class:`torch.Tensor`) noisy measurements
        """
        return self.sensor(self.noise(self.A(x, **kwargs)))

    def A_vjp(self, x, v):
        r"""
        Computes the product between a vector :math:`v` and the Jacobian of the forward operator :math:`A` at the input x, defined as:

        .. math::

            A_{vjp}(x, v) = 2 \overline{B}^{\top} \text{diag}(Bx) v.

        :param torch.Tensor x: signal/image.
        :param torch.Tensor v: vector.
        :return: (:class:`torch.Tensor`) the VJP product between :math:`v` and the Jacobian.
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
    :param bool unitary: Use a random unitary matrix instead of Gaussian matrix. Default is False.
    :param bool compute_inverse: Compute the pseudo-inverse of the forward matrix. Default is False.
    :param torch.dtype dtype: Forward matrix is stored as a dtype. Default is torch.cfloat.
    :param str device: Device to store the forward matrix.
    :param torch.Generator rng: (optional) a pseudorandom random number generator for the parameter generation.
        If ``None``, the default Generator of PyTorch will be used.

    |sep|

    :Examples:

        Random phase retrieval operator with 10 measurements for a 3x3 image:

        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn((1, 1, 3, 3),dtype=torch.cfloat) # Define random 3x3 image
        >>> physics = RandomPhaseRetrieval(m=6, img_shape=(1, 3, 3), rng=torch.Generator('cpu'))
        >>> physics(x)
        tensor([[3.8405, 2.2588, 0.0146, 3.0864, 1.8075, 0.1518]])

    """

    def __init__(
        self,
        m,
        img_shape,
        channelwise=False,
        dtype=torch.cfloat,
        device="cpu",
        unitary=False,
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
            unitary=unitary,
            compute_inverse=compute_inverse,
            dtype=dtype,
            device=device,
            rng=self.rng,
        )
        super().__init__(B, **kwargs)
        self.name = "Random Phase Retrieval"

    def get_A_squared_mean(self):
        return self.B._A.var() + self.B._A.mean() ** 2


class StructuredRandomPhaseRetrieval(PhaseRetrieval):
    r"""
    Structured random phase retrieval model corresponding to the operator

    .. math::

        A(x) = |\prod_{i=1}^N (F D_i) x|^2,

    where :math:`F` is the Discrete Fourier Transform (DFT) matrix, and :math:`D_i` are diagonal matrices with elements of unit norm and random phases, and :math:`N` refers to the number of layers. It is also possible to replace :math:`x` with :math:`Fx` as an additional 0.5 layer.

    For oversampling, we first pad the input signal with zeros to match the output shape and pass it to :math:`A(x)`. For undersampling, we first pass the signal in its original shape to :math:`A(x)` and trim the output signal to match the output shape.

    The phase of the diagonal elements of the matrices :math:`D_i` are drawn from a uniform distribution in the interval :math:`[0, 2\pi]`.

    :param tuple input_shape: shape (C, H, W) of inputs.
    :param tuple output_shape: shape (C, H, W) of outputs.
    :param float n_layers: number of layers :math:`N`. If ``layers=N + 0.5``, a first :math:`F` transform is included, i.e., :math:`A(x)=|\prod_{i=1}^N (F D_i) F x|^2`.
    :param str transform: structured transform to use. Default is 'fft'.
    :param str diagonal_mode: sampling distribution for the diagonal elements. Default is 'uniform_phase'.
    :param bool shared_weights: if True, the same diagonal matrix is used for all layers. Default is False.
    :param torch.dtype dtype: Signals are processed in dtype. Default is torch.cfloat.
    :param str device: Device for computation. Default is `cpu`.
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

        self.mode = compare(input_shape, output_shape)

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

        B = StructuredRandom(
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


class PtychographyLinearOperator(LinearPhysics):
    r"""
    Forward linear operator for phase retrieval in ptychography.

    Models multiple applications of the shifted probe and Fourier transform on an input image.

    This operator performs multiple 2D Fourier transforms on the probe function applied to the shifted input image according to specific offsets, and concatenates them.
    The probe function is applied element by element to the input image.

    .. math::

        B = \left[ \begin{array}{c} B_1 \\ B_2 \\ \vdots \\ B_{n_{\text{img}}} \end{array} \right],
        B_l = F \text{diag}(p) T_l, \quad l = 1, \dots, n_{\text{img}},

    where :math:`F` is the 2D Fourier transform, :math:`\text{diag}(p)` is associated with the probe :math:`p` and :math:`T_l` is a 2D shift.

    :param tuple img_size: Shape of the input image (height, width).
    :param None, torch.Tensor probe: A tensor of shape ``img_size`` representing the probe function. If ``None``, a disk probe is generated with :func:`deepinv.physics.phase_retrieval.build_probe` with disk shape and radius 10.
    :param None, torch.Tensor shifts: A 2D array of shape ``(N, 2)`` corresponding to the ``N`` shift positions for the probe. If ``None``, shifts are generated with :func:`deepinv.physics.phase_retrieval.generate_shifts` with ``N=25``.
    :param torch.device, str device: Device "cpu" or "gpu".
    """

    def __init__(
        self,
        img_size,
        probe=None,
        shifts=None,
        device="cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.device = device
        self.img_size = img_size

        if probe is not None:
            self.probe = probe
        else:
            probe = build_probe(
                img_size=img_size, type="disk", probe_radius=10, device=device
            )

        self.init_probe = probe.clone()

        if shifts is not None:
            self.shifts = shifts
            self.n_img = len(shifts)
        else:
            self.n_img = 25
            self.shifts = generate_shifts(img_size=img_size, n_img=self.n_img)

        self.probe = probe / self.get_overlap_img(self.shifts).mean().sqrt()
        self.probe = torch.cat(
            [
                self.shift(self.probe, x_shift, y_shift)
                for x_shift, y_shift in self.shifts
            ],
            dim=0,
        ).unsqueeze(0)

    def A(self, x, **kwargs):
        """
        Applies the forward operator to the input image ``x`` by shifting the probe,
        multiplying element-wise, and performing a 2D Fourier transform.

        :param torch.Tensor x: Input image tensor.
        :return: Concatenated Fourier transformed tensors after applying shifted probes.
        """
        op_fft2 = partial(torch.fft.fft2, norm="ortho")
        return op_fft2(self.probe * x)

    def A_adjoint(self, y, **kwargs):
        """
        Applies the adjoint operator to ``y``.

        :param torch.Tensor y: Transformed image data tensor of size (batch_size, n_img, height, width).
        :return: Reconstructed image tensor.
        """
        op_ifft2 = partial(torch.fft.ifft2, norm="ortho")
        return (self.probe * op_ifft2(y)).sum(dim=1).unsqueeze(1)

    def shift(self, x, x_shift, y_shift, pad_zeros=True):
        """
        Applies a shift to the tensor ``x`` by ``x_shift`` and ``y_shift``.

        :param torch.Tensor x: Input tensor.
        :param int x_shift: Shift in x-direction.
        :param int y_shift: Shift in y-direction.
        :param bool pad_zeros: If True, pads shifted regions with zeros.
        :return: Shifted tensor.
        """
        x = torch.roll(x, (x_shift, y_shift), dims=(-2, -1))

        if pad_zeros:
            if x_shift < 0:
                x[..., x_shift:, :] = 0
            elif x_shift > 0:
                x[..., 0:x_shift, :] = 0
            if y_shift < 0:
                x[..., :, y_shift:] = 0
            elif y_shift > 0:
                x[..., :, 0:y_shift] = 0
        return x

    def get_overlap_img(self, shifts):
        """
        Computes the overlapping image intensities from probe shifts, used for normalization.

        :param torch.Tensor shifts: Tensor of probe shifts.
        :return: Tensor representing the overlap image.
        """
        overlap_img = torch.zeros_like(self.init_probe, dtype=torch.float32)
        for x_shift, y_shift in shifts:
            overlap_img += torch.abs(self.shift(self.init_probe, x_shift, y_shift)) ** 2
        return overlap_img


class Ptychography(PhaseRetrieval):
    r"""
    Ptychography forward operator.

    Corresponding to the operator

    .. math::

         \forw{x} = \left| Bx \right|^2

    where :math:`B` is the linear forward operator defined by a :class:`deepinv.physics.PtychographyLinearOperator` object.

    :param tuple in_shape: Shape of the input image.
    :param None, torch.Tensor probe: A tensor of shape ``img_size`` representing the probe function.
        If None, a disk probe is generated with ``deepinv.physics.phase_retrieval.build_probe`` function.
    :param None, torch.Tensor shifts: A 2D array of shape (``n_img``, 2) corresponding to the shifts for the probe.
        If None, shifts are generated with ``deepinv.physics.phase_retrieval.generate_shifts`` function.
    :param torch.device, str device: Device "cpu" or "gpu".
    """

    def __init__(
        self,
        in_shape=None,
        probe=None,
        shifts=None,
        device="cpu",
        **kwargs,
    ):
        B = PtychographyLinearOperator(
            img_size=in_shape,
            probe=probe,
            shifts=shifts,
            device=device,
        )
        self.probe = B.probe
        self.shifts = B.shifts
        self.device = device

        super().__init__(B, **kwargs)
        self.name = f"Ptychography_PR"


def build_probe(img_size, type="disk", probe_radius=10, device="cpu"):
    """
    Builds a probe based on the specified type and radius.

    :param tuple img_size: Shape of the input image.
    :param str type: Type of probe shape, e.g., "disk".
    :param int probe_radius: Radius of the probe shape.
    :param torch.device device: Device "cpu" or "gpu".
    :return: Tensor representing the constructed probe.
    """
    if type == "disk" or type is None:
        x = torch.arange(img_size[1], dtype=torch.float64)
        y = torch.arange(img_size[2], dtype=torch.float64)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        probe = torch.zeros(img_size, device=device)
        probe[
            torch.sqrt(
                (X - img_size[1] // 2) ** 2 + (Y - img_size[2] // 2) ** 2
            ).unsqueeze(0)
            < probe_radius
        ] = 1
    else:
        raise NotImplementedError(f"Probe type {type} not implemented")
    return probe


def generate_shifts(img_size, n_img=25, fov=None):
    """
    Generates the array of probe shifts across the image.
    Based on probe radius and field of view.

    :param img_size: Size of the image.
    :param int n_img: Number of shifts (must be a perfect square).
    :param int fov: Field of view for shift computation.
    :return np.ndarray: Array of (x, y) shifts.
    """
    if fov is None:
        fov = img_size[-1]
    start_shift = -fov // 2
    end_shift = fov // 2

    if n_img != int(np.sqrt(n_img)) ** 2:
        raise ValueError("n_img needs to be a perfect square")

    side_n_img = int(np.sqrt(n_img))
    shifts = np.linspace(start_shift, end_shift, side_n_img).astype(int)
    y_shifts, x_shifts = np.meshgrid(shifts, shifts, indexing="ij")
    return np.concatenate(
        [x_shifts.reshape(n_img, 1), y_shifts.reshape(n_img, 1)], axis=1
    )
