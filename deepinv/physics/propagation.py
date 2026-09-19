from __future__ import annotations

from collections.abc import Callable
import math

import torch
from torch import Tensor

from deepinv.physics.forward import LinearPhysics
from deepinv.utils._internal import _as_pair

TransferFunction = Tensor | Callable[[Tensor, Tensor], Tensor]


class AngularSpectrumPropagation(LinearPhysics):
    r"""
    Linear angular-spectrum propagation with a user-defined transfer function.

    This operator propagates a complex wave field :math:`u` according to

    .. math::

        P_H(u) = \mathcal{F}^{-1}\{H(f_x, f_y)\mathcal{F}\{u\}\}.

    The transfer function ``H`` can be supplied either as a tensor in the
    unshifted ordering used by :func:`torch.fft.fft2`, or as a callable with
    signature ``H(fx, fy)``. In the latter case, the callable is evaluated once
    at construction on frequency grids expressed in cycles per unit of
    ``pixel_size``. The resulting tensor is registered as a buffer, so it is
    included in the state dict and follows calls to :meth:`torch.nn.Module.to`.

    ``H`` must act pointwise without adding an output dimension. It may be a
    scalar, have shape ``(height, width)``, or contain leading dimensions that
    broadcast to ``img_size`` (for example, ``(channels, height, width)``).

    This class models propagation of an already formed complex field. Object
    transmission functions such as :math:`T(\delta, \beta)=\exp(g(\delta,
    \beta))` should be represented by a separate nonlinear
    :class:`deepinv.physics.Physics` operator and composed with this one.

    .. note::

        The FFT implementation imposes periodic boundary conditions. Pad the
        field and define ``H`` on the padded grid when circular wrap-around is
        not appropriate for the experiment.

    :param tuple[int, ...] img_size: Shape of an unbatched input. The final two
        entries are interpreted as ``(height, width)``. Both ``(H, W)`` and the
        DeepInv convention ``(C, H, W)`` are accepted.
    :param torch.Tensor, Callable H: Transfer-function tensor or callable
        ``H(fx, fy)``.
    :param float, tuple[float, float] pixel_size: Sampling pitch. A scalar uses
        the same pitch on both axes; a tuple is ordered as ``(dy, dx)``.
        Default: ``1.0``.
    :param torch.dtype dtype: Complex dtype used to evaluate and store ``H``.
        Default: ``torch.cfloat``.
    :param torch.device, str device: Device on which the frequency grid and
        transfer function are created. Default: ``"cpu"``.

    |sep|

    :Examples:

        Define a quadratic phase transfer function directly:

        >>> import torch
        >>> from deepinv.physics import AngularSpectrumPropagation
        >>> propagation = AngularSpectrumPropagation(
        ...     img_size=(1, 32, 48),
        ...     pixel_size=(2e-6, 3e-6),
        ...     H=lambda fx, fy: torch.exp(
        ...         -1j * 1e-10 * (fx.square() + fy.square())
        ...     ),
        ... )
        >>> x = torch.randn(2, 1, 32, 48, dtype=torch.cfloat)
        >>> propagation(x).shape
        torch.Size([2, 1, 32, 48])
    """

    def __init__(
        self,
        img_size: tuple[int, ...],
        H: TransferFunction,
        pixel_size: float | tuple[float, float] = 1.0,
        dtype: torch.dtype = torch.cfloat,
        device: torch.device | str = "cpu",
        **kwargs,
    ):
        img_size = tuple(img_size)
        if len(img_size) < 2 or any(size <= 0 for size in img_size):
            raise ValueError(
                f"img_size must contain at least two positive entries, got {img_size}."
            )
        if dtype not in (torch.cfloat, torch.cdouble):
            raise ValueError(
                f"dtype must be torch.cfloat or torch.cdouble, got {dtype}."
            )

        self.spatial_shape = img_size[-2:]
        self.pixel_size = _as_pair(pixel_size)
        super().__init__(img_size=img_size, device=device, **kwargs)

        transfer_function = self._make_transfer_function(H, dtype=dtype, device=device)
        self._validate_transfer_function(transfer_function)
        self.register_buffer("H", transfer_function)

    @staticmethod
    def frequency_grid(
        img_size: tuple[int, ...],
        pixel_size: float | tuple[float, float] = 1.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> tuple[Tensor, Tensor]:
        r"""Return unshifted spatial-frequency grids ``(fx, fy)``.

        Frequencies are in cycles per unit of ``pixel_size`` and follow the
        ordering expected by :func:`torch.fft.fft2`.

        :param tuple[int, ...] img_size: Shape whose final entries are
            ``(height, width)``.
        :param float, tuple[float, float] pixel_size: Scalar pitch or
            ``(dy, dx)``.
        :param torch.dtype dtype: Real floating-point dtype for the grids.
        :param torch.device, str device: Grid device.
        :return: The two grids ``(fx, fy)``, each of shape ``(height, width)``.
        """
        dy, dx = _as_pair(pixel_size)
        height, width = img_size[-2:]
        fy_values = torch.fft.fftfreq(height, d=dy, dtype=dtype, device=device)
        fx_values = torch.fft.fftfreq(width, d=dx, dtype=dtype, device=device)
        fy, fx = torch.meshgrid(fy_values, fx_values, indexing="ij")
        return fx, fy

    def _make_transfer_function(
        self,
        H: TransferFunction,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> Tensor:
        if callable(H):
            real_dtype = torch.float32 if dtype == torch.cfloat else torch.float64
            fx, fy = self.frequency_grid(
                self.img_size,
                pixel_size=self.pixel_size,
                dtype=real_dtype,
                device=device,
            )
            H = H(fx, fy)
        if not isinstance(H, Tensor):
            H = torch.as_tensor(H)
        return H.to(device=device, dtype=dtype)

    def _validate_transfer_function(self, H: Tensor) -> None:
        if H.ndim > len(self.img_size):
            raise ValueError(
                f"H with shape {tuple(H.shape)} has more dimensions than img_size "
                f"{self.img_size}. H must not add an output dimension."
            )
        input_shape = (1,) * (len(self.img_size) - H.ndim) + tuple(H.shape)
        if any(
            h_size not in (1, x_size)
            for h_size, x_size in zip(input_shape, self.img_size, strict=True)
        ):
            raise ValueError(
                f"H with shape {tuple(H.shape)} is not broadcastable to img_size "
                f"{self.img_size} without changing the output shape."
            )

    def _check_spatial_shape(self, x: Tensor) -> None:
        if x.ndim < 2 or tuple(x.shape[-2:]) != self.spatial_shape:
            raise ValueError(
                f"Expected the final two input dimensions to be {self.spatial_shape}, "
                f"got {tuple(x.shape)}."
            )

    @staticmethod
    def _apply_transfer_function(x: Tensor, H: Tensor) -> Tensor:
        spectrum = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
        return torch.fft.ifft2(spectrum * H, dim=(-2, -1), norm="ortho")

    def A(self, x: Tensor, **kwargs) -> Tensor:
        r"""Propagate a complex field without noise or sensor nonlinearities."""
        self._check_spatial_shape(x)
        return self._apply_transfer_function(x, self.H)

    def A_adjoint(self, y: Tensor, **kwargs) -> Tensor:
        r"""Apply the Hermitian adjoint of the propagation operator."""
        self._check_spatial_shape(y)
        return self._apply_transfer_function(y, self.H.conj())

    def A_dagger(self, y: Tensor, rcond: float | None = None, **kwargs) -> Tensor:
        r"""Apply the Fourier-domain Moore--Penrose pseudoinverse.

        Frequencies whose magnitude is no greater than ``rcond * max(abs(H))``
        are mapped to zero. If ``rcond`` is ``None``, machine precision for the
        real component dtype is used.
        """
        self._check_spatial_shape(y)
        if rcond is None:
            rcond = torch.finfo(self.H.real.dtype).eps
        if rcond < 0:
            raise ValueError(f"rcond must be non-negative, got {rcond}.")

        magnitude = self.H.abs()
        cutoff = rcond * magnitude.amax()
        invertible = magnitude > cutoff
        safe_H = torch.where(invertible, self.H, torch.ones_like(self.H))
        H_pinv = torch.where(invertible, safe_H.reciprocal(), torch.zeros_like(self.H))
        return self._apply_transfer_function(y, H_pinv)


class FresnelPropagation(AngularSpectrumPropagation):
    r"""
    Fresnel (paraxial angular-spectrum) propagation of a complex wave field.

    For wavelength :math:`\lambda`, propagation distance :math:`z`, and spatial
    frequencies :math:`(f_x, f_y)`, this class uses

    .. math::

        H(f_x, f_y) = \exp\left[-\mathrm{i}\pi\lambda z
        (f_x^2 + f_y^2)\right].

    The spatially constant phase :math:`\exp(\mathrm{i}2\pi z/\lambda)` is
    omitted by default because it has no effect on intensity measurements and
    can lose numerical precision for macroscopic propagation distances. Set
    ``include_global_phase=True`` when absolute field phase is required.

    :param tuple[int, ...] img_size: Shape of an unbatched input. The final two
        entries are ``(height, width)``.
    :param float wavelength: Wavelength in the same length unit as ``distance``
        and ``pixel_size``.
    :param float distance: Signed propagation distance.
    :param float, tuple[float, float] pixel_size: Scalar sampling pitch or
        ``(dy, dx)``. All physical length parameters must use the same unit.
    :param bool include_global_phase: Include
        :math:`\exp(\mathrm{i}2\pi z/\lambda)`. Default: ``False``.
    :param torch.dtype dtype: Complex dtype of the transfer function. Default:
        ``torch.cfloat``.
    :param torch.device, str device: Device on which the operator is created.

    |sep|

    :Examples:

        >>> import torch
        >>> from deepinv.physics import FresnelPropagation, PhaseRetrieval
        >>> B = FresnelPropagation(
        ...     img_size=(1, 64, 64),
        ...     wavelength=500e-9,
        ...     distance=0.1,
        ...     pixel_size=5e-6,
        ... )
        >>> physics = PhaseRetrieval(B=B)
        >>> transmission = torch.randn(2, 1, 64, 64, dtype=torch.cfloat)
        >>> physics(transmission).shape
        torch.Size([2, 1, 64, 64])

        If the reconstruction variable contains material parameters rather
        than the complex transmission itself, keep their nonlinear map outside
        ``B``:

        >>> from deepinv.physics import Physics
        >>> k = 2 * torch.pi / 500e-9
        >>> transmission_model = Physics(
        ...     A=lambda x: torch.exp(-1j * k * x[:, 0:1] - k * x[:, 1:2])
        ... )
        >>> material_physics = physics * transmission_model
    """

    def __init__(
        self,
        img_size: tuple[int, ...],
        wavelength: float,
        distance: float,
        pixel_size: float | tuple[float, float],
        include_global_phase: bool = False,
        dtype: torch.dtype = torch.cfloat,
        device: torch.device | str = "cpu",
        **kwargs,
    ):
        wavelength = float(wavelength)
        distance = float(distance)
        if not math.isfinite(wavelength) or wavelength <= 0:
            raise ValueError(
                f"wavelength must be positive and finite, got {wavelength}."
            )
        if not math.isfinite(distance):
            raise ValueError(f"distance must be finite, got {distance}.")

        def fresnel_transfer_function(fx: Tensor, fy: Tensor) -> Tensor:
            phase = -math.pi * wavelength * distance * (fx.square() + fy.square())
            if include_global_phase:
                phase = phase + 2 * math.pi * distance / wavelength
            return torch.exp(1j * phase)

        super().__init__(
            img_size=img_size,
            H=fresnel_transfer_function,
            pixel_size=pixel_size,
            dtype=dtype,
            device=device,
            **kwargs,
        )
        self.wavelength = wavelength
        self.distance = distance
        self.include_global_phase = include_global_phase
