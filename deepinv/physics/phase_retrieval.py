from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from deepinv.optim.phase_retrieval import spectral_methods
from deepinv.physics.compressed_sensing import CompressedSensing
from deepinv.physics.forward import LinearPhysics, Physics
from deepinv.physics.structured_random import (
    StructuredRandom,
    compare,
    generate_diagonal,
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
    :param tuple img_size: shape (C, H, W) of inputs.
    :param bool channelwise: Channels are processed independently using the same random forward operator.
    :param torch.dtype dtype: Forward matrix is stored as a dtype. Default is torch.cfloat.
    :param str device: Device to store the forward matrix.
    :param torch.Generator rng: (optional) a pseudorandom random number generator for the parameter generation.
        If ``None``, the default Generator of PyTorch will be used.

    |sep|

    :Examples:

        Random phase retrieval operator with 10 measurements for a 3x3 image:

        >>> from deepinv.physics import RandomPhaseRetrieval
        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn((1, 1, 3, 3),dtype=torch.cfloat) # Define random 3x3 image
        >>> physics = RandomPhaseRetrieval(m=6, img_size=(1, 3, 3), rng=torch.Generator('cpu'))
        >>> physics(x)
        tensor([[3.8405, 2.2588, 0.0146, 3.0864, 1.8075, 0.1518]])

    """

    def __init__(
        self,
        m,
        img_size,
        channelwise=False,
        dtype=torch.cfloat,
        device="cpu",
        rng: torch.Generator = None,
        **kwargs,
    ):
        self.m = m
        self.img_size = img_size
        self.channelwise = channelwise
        self.dtype = dtype
        if rng is None:
            self.rng = torch.Generator(device=device)
        else:
            # Make sure that the random generator is on the same device as the physic generator
            if rng.device != torch.device(device):  # pragma: no cover
                raise ValueError(
                    f"The random generator is not on the same device as the Physics Generator. Got random generator on {rng.device} and the Physics Generator on {device}."
                )
            self.rng = rng

        B = CompressedSensing(
            m=m,
            img_size=img_size,
            channelwise=channelwise,
            dtype=dtype,
            device=device,
            rng=self.rng,
        )
        super().__init__(B, **kwargs)
        self.register_buffer("initial_random_state", self.rng.get_state())
        self.name = "Random Phase Retrieval"
        self.to(device)

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

    :param tuple img_size: shape (C, H, W) of inputs.
    :param tuple output_size: shape (C, H, W) of outputs.
    :param float n_layers: number of layers :math:`N`. If ``layers=N + 0.5``, a first :math:`F` transform is included, i.e., :math:`A(x)=|\prod_{i=1}^N (F D_i) F x|^2`.
    :param str transform: structured transform to use. Default is 'fft'.
    :param str diagonal_mode: sampling distribution for the diagonal elements. Default is 'uniform_phase'.
    :param bool shared_weights: if True, the same diagonal matrix is used for all layers. Default is False.
    :param torch.dtype dtype: Signals are processed in dtype. Default is torch.cfloat.
    :param str device: Device for computation. Default is `cpu`.
    """

    def __init__(
        self,
        img_size: tuple,
        output_size: tuple,
        n_layers: int,
        transform="fft",
        diagonal_mode="uniform_phase",
        shared_weights=False,
        dtype=torch.cfloat,
        device="cpu",
        **kwargs,
    ):
        if output_size is None:
            output_size = img_size

        self.img_size = img_size
        self.output_size = output_size
        self.n = torch.prod(torch.tensor(self.img_size))
        self.m = torch.prod(torch.tensor(self.output_size))
        self.oversampling_ratio = self.m / self.n
        if not (n_layers % 1 == 0.5 or n_layers % 1 == 0):  # pragma: no cover
            raise ValueError("n_layers must be an integer or an integer plus 0.5")
        self.n_layers = n_layers
        self.structure = self.get_structure(self.n_layers)
        self.shared_weights = shared_weights

        self.dtype = dtype

        self.mode = compare(img_size, output_size)

        # generate diagonal matrices
        self.diagonals = []

        if not shared_weights:
            for _ in range(math.floor(self.n_layers)):
                if self.mode == "oversampling":
                    diagonal = generate_diagonal(
                        shape=self.output_size,
                        mode=diagonal_mode,
                        dtype=self.dtype,
                        device=device,
                    )
                else:
                    diagonal = generate_diagonal(
                        shape=self.img_size,
                        mode=diagonal_mode,
                        dtype=self.dtype,
                        device=device,
                    )
                self.diagonals.append(diagonal)
        else:
            if self.mode == "oversampling":
                diagonal = generate_diagonal(
                    shape=self.output_size,
                    mode=diagonal_mode,
                    dtype=self.dtype,
                    device=device,
                )
            else:
                diagonal = generate_diagonal(
                    shape=self.img_size,
                    mode=diagonal_mode,
                    dtype=self.dtype,
                    device=device,
                )
            self.diagonals = self.diagonals + [diagonal] * math.floor(self.n_layers)

        # determine transform functions
        if transform == "fft":
            transform_func = partial(torch.fft.fft2, norm="ortho")
            transform_func_inv = partial(torch.fft.ifft2, norm="ortho")
        else:
            raise ValueError(f"Unimplemented transform: {transform}")

        B = StructuredRandom(
            img_size=self.img_size,
            output_size=self.output_size,
            n_layers=self.n_layers,
            transform_func=transform_func,
            transform_func_inv=transform_func_inv,
            diagonals=self.diagonals,
            **kwargs,
        )

        super().__init__(B, **kwargs)
        self.name = "Structured Random Phase Retrieval"
        self.to(device)

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


@dataclass(frozen=True)
class PtychographyGeometry(ABC):
    """Base ptychography geometry that takes the experimental setup into account.
    All distances are in metres."""

    wavelength: float
    sample_detector_distance: float
    detector_shape: tuple[int, int]  # (height, width)
    detector_pixel_size: tuple[float, float]  # effective (dy, dx)

    @property
    @abstractmethod
    def object_pixel_size(self) -> tuple[float, float]:
        """Pixel size in the object plane."""

    @property
    def detector_extent(self) -> tuple[float, float]:
        height, width = self.detector_shape
        pixel_height, pixel_width = self.detector_pixel_size
        return height * pixel_height, width * pixel_width

    def object_extent(self, object_shape: tuple[int, int]) -> tuple[float, float]:
        height, width = object_shape
        pixel_height, pixel_width = self.object_pixel_size
        return height * pixel_height, width * pixel_width


@dataclass(frozen=True)
class FarFieldPtychographyGeometry(PtychographyGeometry):
    r"""
    Fraunhofer ptychography geometry.

    The object-plane pixel size is determined from the detector sampling by

    .. math::

        \Delta_o = \frac{\lambda z}{N \Delta_d},

    where :math:`\lambda` is the wavelength, :math:`z` is the
    sample-to-detector distance, :math:`N` is the number of detector
    pixels along a spatial dimension, :math:`\Delta_d` is the detector
    pixel size, and :math:`\Delta_o` is the resulting object-plane pixel
    size along that dimension. All distances are in metres.
    """

    @property
    def object_pixel_size(self) -> tuple[float, float]:
        height, width = self.detector_shape
        detector_dy, detector_dx = self.detector_pixel_size
        scale = self.wavelength * self.sample_detector_distance
        return (
            scale / (height * detector_dy),
            scale / (width * detector_dx),
        )


@dataclass(frozen=True)
class NearFieldPtychographyGeometry(PtychographyGeometry):
    r"""
    Near-field ptychography geometry using same-grid propagation.

    Same-grid Fresnel or angular-spectrum propagation preserves the transverse
    sampling grid, so

    .. math::

        \Delta_o = \Delta_d,

    where :math:`\Delta_d` is the detector pixel size and
    :math:`\Delta_o` is the object-plane pixel size along the same spatial
    dimension. All distances are in metres.
    """

    @property
    def object_pixel_size(self) -> tuple[float, float]:
        return self.detector_pixel_size


class PtychographyLinearOperator(LinearPhysics):
    r"""
    Forward linear operator for phase retrieval in ptychography.

    Models multiple applications of the shifted probe and Fourier transform on an input image.

    This operator extracts a probe-sized patch of the object at every scan
    position, multiplies it element-wise by the probe, and concatenates the 2D
    Fourier transforms of the resulting exit waves. The object can therefore
    be larger than the probe, as in a real ptychography experiment.

    .. math::

        B = \left[ \begin{array}{c} B_1 \\ B_2 \\ \vdots \\ B_{n_{\text{img}}} \end{array} \right],
        B_l = F \text{diag}(p) T_l, \quad l = 1, \dots, n_{\text{img}},

    where :math:`F` is the 2D Fourier transform, :math:`\text{diag}(p)` is associated with the probe :math:`p` and :math:`T_l` is a 2D shift.

    :param tuple img_size: Shape ``(C, H, W)`` of the input object.
    :param None, torch.Tensor probe: A tensor of shape ``(C, H_p, W_p)``
        representing the probe function, where ``H_p <= H`` and ``W_p <= W``.
        Each diffraction pattern has spatial shape ``(H_p, W_p)``. If ``None``,
        a disk probe is generated with :func:`deepinv.physics.phase_retrieval.build_probe`
        using the detector shape when ``geometry`` is provided, or ``img_size``
        otherwise.
    :param None, torch.Tensor shifts: A 2D array of shape ``(N, 2)`` corresponding to the ``N`` shift positions for the probe. If ``None``, shifts are generated with :func:`deepinv.physics.phase_retrieval.generate_shifts` with ``N=25``.
    :param torch.device, str device: Device "cpu" or "gpu".
    :param None, PtychographyGeometry geometry: Optional physical geometry
        associated with the dimensionless FFT operator. Currently only
        :class:`FarFieldPtychographyGeometry` is supported. Its detector shape
        must match the spatial shape of the probe and diffraction patterns.

    """

    def __init__(
        self,
        img_size,
        probe=None,
        shifts=None,
        device="cpu",
        geometry: PtychographyGeometry | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.img_size = img_size
        self.geometry = geometry

        # this would be removed if the near-field propagator
        if geometry is not None and not isinstance(
            geometry, FarFieldPtychographyGeometry
        ):
            raise NotImplementedError(
                "PtychographyLinearOperator currently supports only "
                "FarFieldPtychographyGeometry."
            )

        if shifts is None:
            self.n_img = 25
            shifts = generate_shifts(img_size=img_size, n_img=self.n_img)
        else:
            self.n_img = len(shifts)

        self.register_buffer("shifts", shifts)

        if probe is None:
            probe_size = (
                (img_size[0], *geometry.detector_shape)
                if geometry is not None
                else img_size
            )
            probe = build_probe(
                img_size=probe_size, type="disk", probe_radius=10, device=device
            )

        if probe.ndim != 3 or probe.shape[0] != img_size[0]:
            raise ValueError(
                "probe must have shape (C, H_p, W_p) with the same number of "
                f"channels as img_size; got probe.shape={tuple(probe.shape)} "
                f"and img_size={tuple(img_size)}."
            )
        if probe.shape[-2] > img_size[-2] or probe.shape[-1] > img_size[-1]:
            raise ValueError(
                "The probe spatial dimensions must not exceed the object spatial "
                f"dimensions; got probe.shape={tuple(probe.shape)} and "
                f"img_size={tuple(img_size)}."
            )
        if geometry is not None and geometry.detector_shape != tuple(probe.shape[-2:]):
            raise ValueError(
                f"geometry.detector_shape={geometry.detector_shape} must match "
                f"the probe and FFT output shape {tuple(probe.shape[-2:])}."
            )

        self.register_buffer("init_probe", probe.clone())
        self.probe_is_object_sized = tuple(probe.shape[-2:]) == tuple(img_size[-2:])

        probe = probe / self.get_overlap_img(self.shifts).mean().sqrt()
        if self.probe_is_object_sized:
            scan_probes = [
                self.shift(probe, x_shift, y_shift) for x_shift, y_shift in self.shifts
            ]
        else:
            scan_probes = [probe for _ in self.shifts]
        probe = torch.stack(scan_probes, dim=1)

        self.register_buffer("probe", probe)
        self.to(device)

    def A(self, x, **kwargs):
        """
        Applies the forward operator to the input image ``x`` by shifting the probe,
        multiplying element-wise, and performing a 2D Fourier transform.

        :param torch.Tensor x: Input image tensor.
        :return: Concatenated Fourier transformed tensors after applying shifted probes.
        """
        if x.ndim == len(self.img_size):
            x = x.unsqueeze(0)
        op_fft2 = partial(torch.fft.fft2, norm="ortho")
        if self.probe_is_object_sized:
            return op_fft2(self.probe * x)

        exit_waves = []
        for i, (x_shift, y_shift) in enumerate(self.shifts):
            object_patch = self.extract_patch(x, x_shift, y_shift)
            exit_waves.append(self.probe[:, i] * object_patch)
        return op_fft2(torch.cat(exit_waves, dim=1))

    def A_adjoint(self, y, **kwargs):
        """
        Applies the adjoint operator to ``y``.

        :param torch.Tensor y: Transformed image data tensor of size (batch_size, n_img, height, width).
        :return: Reconstructed image tensor.
        """
        op_ifft2 = partial(torch.fft.ifft2, norm="ortho")
        exit_waves = op_ifft2(y)
        if self.probe_is_object_sized:
            return (self.probe.conj() * exit_waves).sum(dim=1).unsqueeze(1)

        x = torch.zeros(
            (y.shape[0], *self.img_size), dtype=exit_waves.dtype, device=y.device
        )
        for i, (x_shift, y_shift) in enumerate(self.shifts):
            object_patch = self.probe[:, i].conj() * exit_waves[:, i].unsqueeze(1)
            x = x + self.place_patch(object_patch, x_shift, y_shift)
        return x

    def extract_patch(self, x, x_shift, y_shift):
        """Extract a probe-sized object patch, padding outside the object with zeros."""
        object_height, object_width = self.img_size[-2:]
        probe_height, probe_width = self.init_probe.shape[-2:]
        top = (object_height - probe_height) // 2
        left = (object_width - probe_width) // 2
        x = self.shift(x, -x_shift, -y_shift)
        return x[..., top : top + probe_height, left : left + probe_width]

    def place_patch(self, patch, x_shift, y_shift):
        """Apply the adjoint of :meth:`extract_patch` to a probe-sized patch."""
        object_height, object_width = self.img_size[-2:]
        probe_height, probe_width = patch.shape[-2:]
        top = (object_height - probe_height) // 2
        left = (object_width - probe_width) // 2
        patch = F.pad(
            patch,
            (
                left,
                object_width - probe_width - left,
                top,
                object_height - probe_height - top,
            ),
        )
        return self.shift(patch, x_shift, y_shift)

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
        overlap_img = torch.zeros(
            self.img_size,
            dtype=torch.float32,
            device=self.init_probe.device,
        )
        probe_intensity = torch.abs(self.init_probe) ** 2
        for x_shift, y_shift in shifts:
            overlap_img += self.place_patch(probe_intensity, x_shift, y_shift)
        return overlap_img


class Ptychography(PhaseRetrieval):
    r"""
    Ptychography forward operator.

    Corresponding to the operator

    .. math::

         \forw{x} = \left| Bx \right|^2

    where :math:`B` is the linear forward operator defined by a :class:`deepinv.physics.PtychographyLinearOperator` object.

    :param tuple img_size: Shape ``(C, H, W)`` of the input object.
    :param None, torch.Tensor probe: Probe of shape ``(C, H_p, W_p)``. Its
        spatial shape determines the diffraction-pattern shape and may be
        smaller than the object. If ``None``, a disk probe is generated.
    :param None, torch.Tensor shifts: A 2D array of shape (``n_img``, 2) corresponding to the shifts for the probe.
        If None, shifts are generated with ``deepinv.physics.phase_retrieval.generate_shifts`` function.
    :param torch.device, str device: Device "cpu" or "gpu".
    :param None, PtychographyGeometry geometry: Optional physical geometry
        associated with the dimensionless FFT operator. Currently only
        :class:`FarFieldPtychographyGeometry` is supported. If ``None``, the
        operator retains its existing pixel-based interpretation.

    |sep|

    :Examples:

    >>> from deepinv.physics import FarFieldPtychographyGeometry, Ptychography
    >>> import torch
    >>> img_size = (1, 64, 64)  # object shape
    >>> detector_shape = (32, 32)  # probe and diffraction-pattern shape
    >>> geometry = FarFieldPtychographyGeometry(
    ...     wavelength=632.8e-9,
    ...     sample_detector_distance=5e-2,
    ...     detector_shape=detector_shape,
    ...     detector_pixel_size=(36e-6, 36e-6),
    ... )
    >>> physics = Ptychography(img_size=img_size, geometry=geometry)
    >>> physics.geometry is geometry
    True
    >>> x = torch.randn(img_size, dtype=torch.cfloat)
    >>> y = physics(x)  # Apply the Ptychography forward operator
    >>> print(y.shape)  # 25 probe positions by default
    torch.Size([1, 25, 32, 32])
    """

    def __init__(
        self,
        img_size=None,
        probe=None,
        shifts=None,
        device="cpu",
        geometry: PtychographyGeometry | None = None,
        **kwargs,
    ):
        B = PtychographyLinearOperator(
            img_size=img_size,
            probe=probe,
            shifts=shifts,
            device=device,
            geometry=geometry,
        )
        self.probe = B.probe
        self.shifts = B.shifts
        self.img_size = img_size
        super().__init__(B, **kwargs)
        self.name = f"Ptychography_PR"
        self.to(device)

    @property
    def geometry(self) -> PtychographyGeometry | None:
        """Physical geometry associated with the linear ptychography operator."""
        return self.B.geometry


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
        probe = torch.zeros(img_size, device=device, dtype=torch.complex64)
        probe[
            torch.sqrt((X - img_size[1] // 2) ** 2 + (Y - img_size[2] // 2) ** 2)
            .unsqueeze(0)
            .expand(img_size[0], -1, -1)
            < probe_radius
        ] = 1
    else:
        raise NotImplementedError(f"Probe type {type} not implemented")
    return probe


def generate_shifts(
    img_size: Any, n_img: int = 25, fov: int | None = None
) -> torch.Tensor:
    """
    Generates the array of probe shifts across the image.
    Based on probe radius and field of view.

    :param img_size: Size of the image.
    :param int n_img: Number of shifts (must be a perfect square).
    :param int fov: Field of view for shift computation.
    :return: Array of (x, y) shifts.
    """
    if fov is None:
        fov = img_size[-1]
    start_shift = -fov // 2
    end_shift = fov // 2

    if n_img != int(np.sqrt(n_img)) ** 2:
        raise ValueError("n_img needs to be a perfect square")

    side_n_img = int(np.sqrt(n_img))
    shifts = torch.linspace(start_shift, end_shift, side_n_img).to(torch.int32)
    y_shifts, x_shifts = torch.meshgrid(shifts, shifts, indexing="ij")
    return torch.concatenate(
        [x_shifts.reshape(n_img, 1), y_shifts.reshape(n_img, 1)], dim=1
    )
