from __future__ import annotations
from typing import Iterable
import math

import torch
from torch import Tensor
from torch.nn.functional import conv1d, pad as pad_fn

from deepinv.physics.forward import LinearPhysics


class UltrasoundPlaneWave(LinearPhysics):
    r"""
    2D plane-wave ultrafast ultrasound imaging operator.

    Models the linear operator :math:`A` mapping an image :math:`x` to the per-channel
    element raw data :math:`y` as a parabolic Radon transform :math:`G` followed by a
    convolution along the time axis with the pulse-echo impulse response :math:`h`:

    .. math::
        y = \forw{x} = \left( h \ast_t G \right) \left( x \right).

    For each transmit event :math:`k`, receive element :math:`i` and time sample
    :math:`t_n`, the sample is

    .. math::
        y_{k,i,n} = \left[h \ast_t G(x)\right]_{k,i,n}, \qquad
        \left[G(x)\right]_{k,i,n} = \sum_{j} a_{k,i}(\mathbf{r}_j)\, K\!\big(f_s\,(t_n - \tau_{k,i}(\mathbf{r}_j))\big)\, x_j,

    where :math:`\mathbf{r}_j = (x_j, z_j)` is the position of pixel :math:`j`,
    :math:`K` is the linear interpolation kernel, :math:`a_{k,i}` is the product of
    transmit and receive apodizations, and :math:`\tau_{k,i}` is the round-trip
    time-of-flight

    .. math::
        \tau_{k,i}(x, z) = \frac{x \sin\theta_k + z \cos\theta_k}{c}
                           + \frac{\|(x, z) - \mathbf{r}_i\|}{c}

    for the steering angle :math:`\theta_k`.

    The adjoint :meth:`A_adjoint` (also known as beamforming, or delay-and-sum in the special case of a Dirac pulse) follows the same formalism with a time-reversed pulse
    :math:`\tilde{h}(t) = h(-t)` and the transpose quadratic Radon transform:

    .. math::
        \left[A^\top y\right]_j = \left[G^\top \! \left(\tilde{h} \ast_t y\right)\right]_j, \qquad
        \left[G^\top y\right]_j = \sum_{k,i,n} a_{k,i}(\mathbf{r}_j)\, K\!\big(f_s\,(t_n - \tau_{k,i}(\mathbf{r}_j))\big)\, y_{k,i,n}.

    .. note::
        We treat signals as real RF tensors: :math:`x` has shape ``(B, 1, Z, X)`` and :math:`y` has shape
    ``(B, 1, n_angles, n_elements, n_samples)``. If you would like to treat signals instead as complex IQ data, please open a feature request issue on GitHub.

    .. note::
        We interpolate time linearly. If you would like to interpolate with more advanced kernels, please open a feature request issue on GitHub.


    :param tuple[int, int] img_size: spatial image size ``(Z, X)`` in pixels.
    :param Iterable[float], torch.Tensor angles: transmit steering angles in radians.
    :param torch.Tensor element_positions: receive element positions in meters, shape ``(n_elements, 2)`` with columns ``(x, z)``.
    :param int n_samples: number of time samples per channel.
    :param float sampling_frequency: sampling frequency in Hz.
    :param float sound_speed: speed of sound :math:`c` in m/s. (default: ``1540``)
    :param torch.Tensor pixel_grid: optional pixel positions in meters, shape
        ``(Z, X, 2)`` with columns ``(x, z)``. If ``None``, built from ``pixel_size`` and
        ``pixel_origin``. Stored flattened, as the ``(Z*X, 2)`` buffer ``pixel_grid``.
        (default: ``None``)
    :param tuple[float, float] pixel_size: pixel spacing ``(dz, dx)`` in meters.
        (default: :math:`c / (2 f_s)` along both axes)
    :param tuple[float, float] pixel_origin: grid origin ``(z0, x0)`` in meters.
        (default: ``(0, x_aperture_center)``)
    :param float, torch.Tensor t0: acquisition-start offset :math:`t_0` in seconds,
        scalar or per-angle tensor of shape ``(n_angles,)``. (default: ``0``)
    :param float f_number: receive f-number defining the aperture half-width
        :math:`|x_i - x_j| \le z_j / f_\#` at each pixel. ``None`` disables receive
        apodization. (default: ``None``)
    :param str receive_apod_window: receive apodization window inside the f-number
        aperture, one of ``"rect"`` or ``"hann"``. Ignored if ``f_number`` is ``None``.
        (default: ``"rect"``)
    :param str transmit_apod_window: transmit apodization window, one of ``"rect"`` or
        ``"hann"``. ``None`` disables transmit apodization. (default: ``None``)
    :param torch.Tensor pulse: optional real 1D pulse-echo impulse response :math:`h` (default: ``None``)
    :param bool normalize: if ``True``, :meth:`A` and :meth:`A_adjoint` are divided by
        the operator's spectral norm.
    :param torch.device, str device: device for buffers. (default: ``"cpu"``)


    |sep|

    :Examples:

        RF operator on a 32x32 image with 4 receive elements and 3 steering angles:

        .. doctest::

            >>> import torch
            >>> from deepinv.physics import UltrasoundPlaneWave
            >>> ele_pos = torch.stack(
            ...     [torch.linspace(-1e-3, 1e-3, 4), torch.zeros(4)], dim=-1
            ... )
            >>> physics = UltrasoundPlaneWave(
            ...     img_size=(32, 32),
            ...     angles=torch.linspace(-0.28, 0.28, 3),
            ...     element_positions=ele_pos,
            ...     n_samples=256,
            ...     sampling_frequency=40e6,
            ...     sound_speed=1540.0,
            ...     t0=0.0,
            ...     normalize=False,
            ... )
            >>> x = torch.randn(1, 1, 32, 32)
            >>> physics(x).shape # 1, 1, n_angles, n_elements, n_samples)
            torch.Size([1, 1, 3, 4, 256])
            >>> physics.A_adjoint_A(x).shape # (1, 1, Z, X)
            torch.Size([1, 1, 32, 32])

        The transmit sequence can be changed in place with
        :meth:`update_parameters`:

        .. doctest::

            >>> physics.update_parameters(angles=[0.0])  # keep a single transmit
            >>> physics(x).shape
            torch.Size([1, 1, 1, 4, 256])
    """

    def __init__(
        self,
        img_size: tuple[int, int],
        angles: Iterable[float] | Tensor,
        element_positions: Tensor,
        n_samples: int,
        sampling_frequency: float,
        sound_speed: float = 1540.0,
        t0: float | Tensor = 0.0,
        *,
        pixel_grid: Tensor | None = None,
        pixel_size: tuple[float, float] | None = None,
        pixel_origin: tuple[float, float] | None = None,
        f_number: float | None = None,
        receive_apod_window: str = "rect",
        transmit_apod_window: str | None = None,
        pulse: Tensor | None = None,
        normalize: bool = False,
        device: torch.device | str = "cpu",
    ):
        angles = torch.as_tensor(angles)
        element_positions = torch.as_tensor(element_positions)

        if pixel_grid is not None:
            pixel_grid = torch.as_tensor(pixel_grid)
        else:
            if pixel_size is None:
                pixel_size = (sound_speed / sampling_frequency / 2.0,) * 2
            if pixel_origin is None:
                pixel_origin = (
                    0.0,
                    0.5
                    * (
                        element_positions[:, 0].min() + element_positions[:, 0].max()
                    ).item()
                    - pixel_size[1] * (img_size[1] - 1) / 2.0,
                )
            pixel_grid = torch.stack(
                torch.meshgrid(
                    pixel_origin[1] + pixel_size[1] * torch.arange(img_size[1]),
                    pixel_origin[0] + pixel_size[0] * torch.arange(img_size[0]),
                    indexing="xy",
                ),
                dim=-1,
            )

        t0 = torch.as_tensor(t0)
        if t0.ndim == 0:
            t0 = t0.expand(len(angles))

        if pulse is not None:
            pulse_echo_ir = torch.as_tensor(pulse).reshape(-1)
            pulse_echo_ir = pulse_echo_ir / torch.linalg.norm(pulse_echo_ir)

        super().__init__(img_size=(1, img_size[0], img_size[1]), device=device)
        self.register_buffer("element_positions", element_positions.contiguous())
        self.register_buffer("pixel_grid", pixel_grid.reshape(-1, 2).contiguous())
        self.register_buffer("t0", t0.contiguous())
        self.register_buffer("angles", angles.contiguous())
        self.register_buffer(
            "pulse_echo_ir", pulse_echo_ir.contiguous() if pulse is not None else None
        )

        self.pixel_size = pixel_size
        self.pixel_origin = pixel_origin
        self.n_samples = n_samples
        self.fs = sampling_frequency
        self.c = sound_speed
        self.f_number = None if f_number is None else f_number
        self.receive_apod_window = receive_apod_window
        self.transmit_apod_window = transmit_apod_window
        self.to(device)

        self.register_buffer("receive_delays", self._receive_delays(), persistent=False)
        self.register_buffer(
            "receive_apodization", self._receive_apod(), persistent=False
        )

        self.normalize = False
        self.register_buffer("operator_norm", None)
        if normalize:
            x = torch.randn(
                (1, 1, img_size[0], img_size[1]),
                generator=torch.Generator(device).manual_seed(0),
                device=device,
            )
            self.register_buffer(
                "operator_norm", self.compute_norm(x, squared=False, verbose=False)
            )
            self.normalize = True

    def _receive_delays(self) -> Tensor:
        r"""Receive time-of-flight :math:`\tau_\mathrm{rx}(x, z; x_e, z_e) = \|(x, z) - (x_e, z_e)\|/c`,
        shape ``(n_elements, Z*X)``.
        """
        return (
            torch.hypot(
                self.pixel_grid[:, 0].unsqueeze(0)
                - self.element_positions[:, 0].unsqueeze(1),
                self.pixel_grid[:, 1].unsqueeze(0)
                - self.element_positions[:, 1].unsqueeze(1),
            )
            / self.c
        )

    def _transmit_delays(self, theta_k: Tensor) -> Tensor:
        r"""Transmit time-of-flight of the plane wave steered at :math:`\theta_k`,
        :math:`\tau_\mathrm{tx}(x, z) = (x \sin\theta_k + z \cos\theta_k)/c`, shape ``(Z*X,)``.
        """
        return (
            self.pixel_grid[:, 0] * torch.sin(theta_k)
            + self.pixel_grid[:, 1] * torch.cos(theta_k)
        ) / self.c

    def _receive_apod(self) -> Tensor:
        r"""Receive apodization, shape ``(n_elements, Z*X)``."""
        if self.f_number is None:
            return torch.ones(
                (self.element_positions.shape[0], math.prod(self.img_size)),
                dtype=torch.float32,
                device=self.pixel_grid.device,
            )
        lateral_distance = self.pixel_grid[:, 0].unsqueeze(0) - self.element_positions[
            :, 0
        ].unsqueeze(1)
        min_width = (
            max(
                0.5
                * torch.diff(torch.sort(self.element_positions[:, 0])[0]).mean().item(),
                1e-6,
            )
            if self.element_positions.shape[0] > 1
            else 1e-3
        )
        u = lateral_distance / torch.clamp(
            (
                self.pixel_grid[:, 1].unsqueeze(0)
                - self.element_positions[:, 1].unsqueeze(1)
            ).abs()
            / self.f_number,
            min=torch.finfo(self.pixel_grid.dtype).eps,
        )
        win = (
            0.5 * (1.0 + torch.cos(math.pi * u))
            if self.receive_apod_window == "hann"
            else torch.ones_like(u)
        )
        apod = torch.where(u.abs() <= 1.0, win, torch.zeros_like(u))
        apod = torch.where(
            lateral_distance.abs() <= min_width, torch.ones_like(apod), apod
        )
        return apod.to(torch.float32)

    def _transmit_apod(self, theta_k: Tensor) -> Tensor:
        r"""Transmit apodization for a given steering angle :math:`\theta_k`, shape ``(Z*X,)``."""
        if self.transmit_apod_window is None:
            return torch.ones(
                math.prod(self.img_size),
                dtype=torch.float32,
                device=self.pixel_grid.device,
            )
        x_min = self.element_positions[:, 0].min() * 1.2
        x_max = self.element_positions[:, 0].max() * 1.2
        u = (
            self.pixel_grid[:, 0]
            - self.pixel_grid[:, 1] * torch.tan(theta_k)
            - 0.5 * (x_min + x_max)
        ) / (0.5 * (x_max - x_min))
        win = (
            0.5 * (1.0 + torch.cos(math.pi * u))
            if self.transmit_apod_window == "hann"
            else torch.ones_like(u)
        )
        return torch.where(u.abs() <= 1.0, win, torch.zeros_like(u)).to(torch.float32)

    def _apply_pulse(self, sig: Tensor, adjoint: bool = False) -> Tensor:
        r"""Convolve along the time axis with the pulse-echo impulse response.

        In the adjoint the kernel is time-reversed as the adjoint of convolution is correlation.
        ``(N, 1, n_samples)``.
        """
        h = self.pulse_echo_ir.flip(-1) if adjoint else self.pulse_echo_ir
        L = h.numel()
        if L % 2:
            return conv1d(sig, h.reshape(1, 1, -1), padding="same")
        pad = (L // 2, L - 1 - L // 2)
        return conv1d(pad_fn(sig, pad[::-1] if adjoint else pad), h.reshape(1, 1, -1))

    def _interp1d(self, positions: Tensor, signal: Tensor, n_samples: int) -> Tensor:
        r"""Linear-interpolation gather along the time axis.

        Reads each element's signal at fractional sample positions, interpolating linearly
        between the two samples around each position. The time axis is zero-padded by
        one sample on each side, so that positions falling outside the record read a zero
        guard sample rather than a valid one.

        :param torch.Tensor positions: fractional sample positions at which to read, of
            shape ``(n_elements, Z*X)``.
        :param torch.Tensor signal: signal to read, of shape ``(B, n_elements, n_samples)``.
        :param int n_samples: length of the time axis of ``signal``.
        :return: (:class:`torch.Tensor`) the interpolated values, of shape
            ``(B, n_elements, Z*X)``.
        """
        floor_index = torch.floor(positions).to(torch.long)
        weight_after = (positions - floor_index.to(positions.dtype)).clamp(min=0.0)
        weight_before = (1.0 - weight_after).clamp(min=0.0).unsqueeze(0)
        weight_after = weight_after.unsqueeze(0)
        index_before = (
            (floor_index + 1)
            .clamp(0, n_samples + 1)
            .unsqueeze(0)
            .expand(signal.shape[0], *floor_index.shape)
        )
        index_after = (
            (floor_index + 2)
            .clamp(0, n_samples + 1)
            .unsqueeze(0)
            .expand(signal.shape[0], *floor_index.shape)
        )
        padded = pad_fn(signal, (1, 1))
        return (
            torch.gather(padded, 2, index_before) * weight_before
            + torch.gather(padded, 2, index_after) * weight_after
        )

    def _interp1d_adjoint(
        self, positions: Tensor, values: Tensor, n_samples: int
    ) -> Tensor:
        r"""Adjoint of :meth:`_interp1d`: scatter-add along the time axis.

        Scatters each value into the two samples around its position, with the same
        weights as :meth:`_interp1d`. The two zero guard samples of the padded axis collect
        the contributions falling outside the record, and are dropped on return.

        :param torch.Tensor positions: fractional sample positions at which to accumulate,
            of shape ``(n_elements, Z*X)``.
        :param torch.Tensor values: values to accumulate, of shape ``(B, n_elements, Z*X)``.
        :param int n_samples: length of the time axis to accumulate into.
        :return: (:class:`torch.Tensor`) the accumulated signal, of shape
            ``(B, n_elements, n_samples)``.
        """
        floor_index = torch.floor(positions).to(torch.long)
        weight_after = (positions - floor_index.to(positions.dtype)).clamp(min=0.0)
        weight_before = (1.0 - weight_after).clamp(min=0.0).unsqueeze(0)
        weight_after = weight_after.unsqueeze(0)
        index_before = (
            (floor_index + 1)
            .clamp(0, n_samples + 1)
            .unsqueeze(0)
            .expand(values.shape[0], *floor_index.shape)
        )
        index_after = (
            (floor_index + 2)
            .clamp(0, n_samples + 1)
            .unsqueeze(0)
            .expand(values.shape[0], *floor_index.shape)
        )
        padded = torch.zeros(
            (values.shape[0], *positions.shape[:-1], n_samples + 2),
            dtype=values.dtype,
            device=values.device,
        )
        padded.scatter_add_(2, index_before, values * weight_before)
        padded.scatter_add_(2, index_after, values * weight_after)
        return padded[..., 1:-1]

    def A(self, x: Tensor, **kwargs) -> Tensor:
        r"""Forward operator :math:`y = \forw{x} = \left(h \ast_t G\right)(x)`.

        :param torch.Tensor x: image of shape ``(B, 1, Z, X)``.
        :return: RF per-channel raw data of shape ``(B, 1, n_angles, n_elements, n_samples)``.
        """
        if x.ndim != 4 or tuple(x.shape[1:]) != tuple(self.img_size):
            raise ValueError(
                f"Expected image of shape (B, *{tuple(self.img_size)}), got {tuple(x.shape)}."
            )
        reflectivity = x.reshape(x.shape[0], 1, -1)

        y = torch.zeros(
            (
                x.shape[0],
                len(self.angles),
                self.element_positions.shape[0],
                self.n_samples,
            ),
            dtype=torch.float32,
            device=x.device,
        )
        for transmit, angle in enumerate(self.angles):
            y[:, transmit] = self._interp1d_adjoint(
                (
                    self._transmit_delays(angle).unsqueeze(0)
                    + self.receive_delays
                    + self.t0[transmit]
                )
                * self.fs,
                reflectivity
                * self.receive_apodization
                * self._transmit_apod(angle).unsqueeze(0),
                self.n_samples,
            )

        if self.pulse_echo_ir is not None:
            y = self._apply_pulse(y.reshape(-1, 1, self.n_samples)).reshape(y.shape)
        y = y.unsqueeze(1)
        return y / self.operator_norm if self.normalize else y

    def A_adjoint(self, y: Tensor, **kwargs) -> Tensor:
        r"""Adjoint (beamforming) operator :math:`x = A^\top y = G^\top(\tilde{h} \ast_t y)`.

        :param torch.Tensor y: raw RF data of shape ``(B, 1, n_transmits, n_elements, n_samples)``.
        :return: beamformed image of shape ``(B, 1, Z, X)``.
        """
        expected_shape = (
            1,
            len(self.angles),
            self.element_positions.shape[0],
            self.n_samples,
        )
        if y.ndim != 5 or tuple(y.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected measurement of shape (B, *{expected_shape}), got {tuple(y.shape)}."
            )
        channels = y[:, 0]
        if self.pulse_echo_ir is not None:
            channels = self._apply_pulse(
                channels.reshape(-1, 1, self.n_samples), adjoint=True
            ).reshape(channels.shape)

        x = torch.zeros(
            (y.shape[0], math.prod(self.img_size)), dtype=torch.float32, device=y.device
        )
        for transmit, angle in enumerate(self.angles):
            x = x + (
                self._interp1d(
                    (
                        self._transmit_delays(angle).unsqueeze(0)
                        + self.receive_delays
                        + self.t0[transmit]
                    )
                    * self.fs,
                    channels[:, transmit],
                    self.n_samples,
                )
                * self.receive_apodization
                * self._transmit_apod(angle).unsqueeze(0)
            ).sum(dim=1)

        x = x.reshape(y.shape[0], *self.img_size)
        return x / self.operator_norm if self.normalize else x

    def update_parameters(
        self, angles: Iterable[float] | Tensor | None = None, **kwargs
    ):
        r"""Update the transmit steering angles in place.

        .. note::
            Changing ``angles`` changes the number of transmits, hence the expected shape
            of :math:`y`. The per-angle :math:`t_0` follows automatically when it is the
            same for all angles, otherwise the operator must be rebuilt.

        :param Iterable[float], torch.Tensor angles: new transmit steering angles in radians.
        """
        fixed = {
            "t0",
            "pixel_grid",
            "element_positions",
            "pulse_echo_ir",
        } & kwargs.keys()
        if fixed:
            raise NotImplementedError(
                f"Only 'angles' can be updated, got {sorted(fixed)}. "
                "Rebuild the operator to change any other setting."
            )

        if angles is not None:
            angles = torch.as_tensor(angles, device=self.angles.device).reshape(-1)
            if angles.numel() == 0:
                raise ValueError("angles must contain at least one steering angle.")
            self.angles = angles.contiguous()

            if self.t0.numel() != len(self.angles):
                if torch.unique(self.t0).numel() != 1:
                    raise ValueError(
                        f"angles now has {len(self.angles)} entries but t0 has "
                        f"{self.t0.numel()} angle-dependent entries; rebuild the operator."
                    )
                self.t0 = self.t0[:1].expand(len(self.angles)).contiguous()

            if self.normalize:
                self.normalize = False
                self.operator_norm = self.compute_norm(
                    torch.randn(
                        (1, *self.img_size),
                        generator=torch.Generator(self.angles.device).manual_seed(0),
                        device=self.angles.device,
                    ),
                    squared=False,
                    verbose=False,
                )
                self.normalize = True

        return super().update_parameters(**kwargs)
