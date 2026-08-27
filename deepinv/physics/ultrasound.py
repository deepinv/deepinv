from __future__ import annotations
from typing import Iterable
from warnings import warn

import math

from numpy import ndarray
import torch
from torch import Tensor

from deepinv.physics.forward import LinearPhysics


class UltrafastUltrasound(LinearPhysics):
    r"""
    2D Pulse-echo ultrafast ultrasound imaging operator (abstract base class).

    Models the linear operator mapping a tissue reflectivity map :math:`x` to the per-channel element raw data :math:`y`

    .. math::
        y = \forw{x} = \left( h \ast_t G \right) \left( x \right)

    as a quadratic Radon transform :math:`G` , i.e. a set of projections of :math:`x` along conics whose shape is set by the transmit/receive configuration,
    followed by a convolution with the pulse-echo impulse response :math:`h`.

    For each transmit event :math:`k`, receive element :math:`i` and time sample :math:`t_n`, the per-channel element raw data sample is

    .. math::
        y_{k,i,n} = \left[h \ast_t G(x)\right]_{k,i,n}, \qquad
        \left[G(x)\right]_{k,i,n} = \sum_{j} a_{k,i}(\mathbf{r}_j)\, K\!\big(f_s\,(t_n - \tau_{k,i}(\mathbf{r}_j))\big)\, \varphi_{k,i}(\mathbf{r}_j)\, x_j,

    where :math:`\mathbf{r}_j = (x_j, z_j)` is the position of pixel :math:`j`,
    :math:`\tau_{k,i}` is the round-trip time-of-flight,
    :math:`a_{k,i}` is the product of transmit and receive apodizations, :math:`K` is a 1D interpolation kernel, and
    :math:`\varphi_{k,i}` the IQ modulation phase.

    The adjoint :meth:`A_adjoint` follows the same formalism with a time-reversed pulse and the transpose quadratic Radon transform :math:`G^\top`

    .. math::
        \left[A^\top y\right]_j = \left[G^\top \! \left(\tilde{h} \ast_t y\right)\right]_j, \qquad
        \left[G^\top y\right]_j = \sum_{k,i,n} a_{k,i}(\mathbf{r}_j)\, K\!\big(f_s\,(t_n - \tau_{k,i}(\mathbf{r}_j))\big)\, \overline{\varphi_{k,i}(\mathbf{r}_j)}\, y_{k,i,n},

    where :math:`\tilde{h}(t) = h(-t)` is the time-reversed pulse and :math:`\overline{\varphi_{k,i}}` the conjugate modulation phase.

    Signals are batched real tensors whose layout depends on `signal_kind`: in `"iq"` mode :math:`x` has shape `(B, 2, Z, X)` with an `(I, Q)` channel pair on dim 1
    and :math:`y` has shape `(B, 2, n_transmits, n_elements, n_samples)`; in `"rf"` mode both have a single channel.

    .. note::
        This class is not meant to be instantiated directly. Subclass it to implement a new transmit scheme
        by overriding :meth:`transmit_delay` and :meth:`transmit_apod`, and handle operator normalization
        at the end of `__init__` (see :class:`UltrasoundPlaneWave` for a reference implementation).

    .. seealso::
        :class:`deepinv.physics.UltrasoundPlaneWave` for the ready-to-use plane-wave operator, and :class:`deepinv.physics.LinearPhysics` for the linear operator interface.

    :param tuple[int, int] img_size: spatial image size `(Z, X)` in pixels.
    :param torch.Tensor element_positions: receive element positions in meters, shape `(n_elements, 2)` with columns `(x, z)`.
    :param int n_transmits: number of transmit events.
    :param int n_samples: number of time samples per channel.
    :param float sampling_frequency: sampling frequency in Hz.
    :param float sound_speed: speed of sound in m/s. (default: `1540`)
    :param torch.Tensor pixel_grid: optional pixel positions in meters, shape `(Z, X, 2)` with columns `(x, z)`. If `None`, built from `pixel_size` and `pixel_origin`. (default: `None`)
    :param tuple[float, float] pixel_size: pixel spacing `(dz, dx)` in meters. (default: half a wavelength :math:`c / (2 f_{\mathrm{demod}})` along both axes)
    :param tuple[float, float] pixel_origin: grid origin `(z0, x0)` in meters. (default: `(0, x_aperture_center)`)
    :param float, torch.Tensor t0: acquisition-start offset :math:`t_0` in seconds, scalar or per-transmit tensor of shape `(n_transmits,)`. (default: `0.0`)
    :param float demodulation_frequency: demodulation frequency in Hz, required in `"iq"` mode and ignored in `"rf"` mode. (default: `None`)
    :param float f_number: receive f-number defining the aperture half-width :math:`|x_i - x_j| \le z_j / f_\#` at each pixel. `None` disables receive apodization. (default: `None`)
    :param str rx_apod_window: receive apodization window inside the f-number aperture, one of `"rect"`, `"hann"`, `"hamming"`, `"tukey0.25"`. Ignored if `f_number` is `None`. (default: `"rect"`)
    :param str tx_apod_window: transmit apodization window, one of `"rect"`, `"hann"`, `"hamming"`, `"tukey0.25"`. `None` disables transmit apodization. (default: `None`)
    :param str interp: interpolation kernel :math:`K`, one of `"nearest"`, `"linear"`, `"keys"`. (default: `"linear"`)
    :param str signal_kind: signal representation, `"iq"` or `"rf"`. (default: `"iq"`)
    :param torch.Tensor pulse: optional 1D pulse-echo impulse response :math:`h`, normalized to unit :math:`\ell_2` norm and convolved along the time axis in both :meth:`A` and :meth:`A_adjoint`. (default: `None`)
    :param torch.device, str device: device for buffers. (default: `"cpu"`)
    :param torch.dtype dtype: real dtype; `float32` uses internal `complex64`, `float64` uses `complex128`. (default: `torch.float32`)

    |sep|

    :Examples:

        Defining a new transmit scheme. Only the transmit delays and apodization need to be implemented:

        >>> import deepinv as dinv
        >>> class MyCustomUltrafastUltrasound(dinv.physics.UltrafastUltrasound):  # doctest: +SKIP
        ...     def __init__(self, *args, normalize=True, **kwargs):
        ...         super().__init__(*args, **kwargs)
        ...         # register any transmit-mode-specific buffers here, then normalize
        ...         self.normalize = False
        ...         if normalize:
        ...             x = torch.randn((1, self.n_channels, *self.img_size_spatial))
        ...             self.register_buffer(
        ...                 "operator_norm",
        ...                 self.compute_norm(x, squared=False, verbose=False),
        ...             )
        ...             self.normalize = True
        ...     def transmit_delay(self):
        ...         ...  # (n_transmits, Z*X) delays in seconds
        ...     def transmit_apod(self):
        ...         return None  # or (n_transmits, Z*X) weights

    """

    _VALID_WINDOWS = ("rect", "hann", "hamming", "tukey0.25")
    _VALID_SIGNAL_KINDS = ("iq", "rf")
    _VALID_INTERP = ("nearest", "linear", "keys")

    def __init__(
        self,
        img_size: tuple[int, int],
        element_positions: Tensor,
        n_transmits: int,
        n_samples: int,
        sampling_frequency: float,
        sound_speed: float = 1540.0,
        pixel_grid: Tensor | None = None,
        pixel_size: tuple[float, float] | None = None,
        pixel_origin: tuple[float, float] | None = None,
        t0: float | Tensor = 0.0,
        demodulation_frequency: float | None = None,
        f_number: float | None = None,
        rx_apod_window: str = "rect",
        tx_apod_window: str | None = None,
        interp: str = "linear",
        signal_kind: str = "iq",
        pulse: Tensor | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if signal_kind == "iq" and demodulation_frequency is None:
            raise ValueError("demodulation_frequency must be provided in 'iq' mode.")
        if sound_speed <= 0.0:
            raise ValueError(
                f"sound_speed must be strictly positive, got {sound_speed}."
            )
        if interp not in self._VALID_INTERP:
            raise ValueError(
                f"interp must be one of {self._VALID_INTERP}, got {interp!r}."
            )
        if signal_kind not in self._VALID_SIGNAL_KINDS:
            raise ValueError(
                f"signal_kind must be one of {self._VALID_SIGNAL_KINDS}, "
                f"got {signal_kind!r}."
            )
        if rx_apod_window not in self._VALID_WINDOWS:
            raise ValueError(
                f"rx_apod_window must be one of {self._VALID_WINDOWS}, "
                f"got {rx_apod_window!r}."
            )
        if tx_apod_window is not None and tx_apod_window not in self._VALID_WINDOWS:
            raise ValueError(
                f"tx_apod_window must be None or one of {self._VALID_WINDOWS}, "
                f"got {tx_apod_window!r}."
            )

        Z, X = int(img_size[0]), int(img_size[1])

        ele_pos = torch.as_tensor(element_positions, dtype=dtype)
        if ele_pos.ndim != 2 or ele_pos.shape[1] != 2:
            raise ValueError(
                "element_positions must have shape (n_elements, 2), got "
                f"{tuple(ele_pos.shape)}."
            )

        if pixel_grid is not None:
            grid = torch.as_tensor(pixel_grid, dtype=dtype)
            if grid.shape != (Z, X, 2):
                raise ValueError(
                    "pixel_grid must have shape (Z, X, 2), got "
                    f"{tuple(grid.shape)} for img_size=({Z}, {X})."
                )
        else:
            if pixel_size is None:
                ref_freq = (
                    demodulation_frequency
                    if demodulation_frequency is not None
                    and demodulation_frequency > 0.0
                    else sampling_frequency
                )
                lam = sound_speed / ref_freq
                pixel_size = (lam / 2.0, lam / 2.0)
            dz, dx = float(pixel_size[0]), float(pixel_size[1])
            if pixel_origin is None:
                x_center = 0.5 * (ele_pos[:, 0].min() + ele_pos[:, 0].max()).item()
                pixel_origin = (0.0, x_center - dx * (X - 1) / 2.0)
            z0, x0 = float(pixel_origin[0]), float(pixel_origin[1])
            z_axis = z0 + dz * torch.arange(Z, dtype=dtype)
            x_axis = x0 + dx * torch.arange(X, dtype=dtype)
            zz, xx = torch.meshgrid(z_axis, x_axis, indexing="ij")
            grid = torch.stack([xx, zz], dim=-1)

        t0_tensor = torch.as_tensor(t0, dtype=dtype)
        if t0_tensor.ndim == 0:
            t0_tensor = t0_tensor.expand(n_transmits).clone()
        elif t0_tensor.shape != (n_transmits,):
            raise ValueError(
                f"t0 must be scalar or shape ({n_transmits},), got "
                f"{tuple(t0_tensor.shape)}."
            )

        n_channels = 2 if signal_kind == "iq" else 1
        super().__init__(
            img_size=(n_channels, Z, X),
            device=device,
        )

        self.register_buffer("element_positions", ele_pos.contiguous())
        self.register_buffer("pixel_grid", grid.contiguous())
        self.register_buffer("t0", t0_tensor.contiguous())

        self.img_size_spatial = (Z, X)
        self.n_transmits = int(n_transmits)
        self.n_samples = int(n_samples)
        self.fs = float(sampling_frequency)
        self.c = float(sound_speed)
        self.demodulation_frequency = (
            None if demodulation_frequency is None else float(demodulation_frequency)
        )
        self.f_number = None if f_number is None else float(f_number)
        self.rx_apod_window = rx_apod_window
        self.tx_apod_window = tx_apod_window
        self.interp = interp
        self.signal_kind = signal_kind
        self.n_channels = n_channels
        self.dtype = dtype
        self._complex_dtype = (
            torch.complex128 if dtype == torch.float64 else torch.complex64
        )

        if pulse is not None:
            h = torch.as_tensor(pulse, dtype=self.dtype)
            h = h / torch.linalg.norm(h)
            self.register_buffer("pulse_echo_ir", h.contiguous())
        else:
            self.pulse_echo_ir = None

    @property
    def image_shape(self) -> tuple[int, int]:
        """Spatial image shape ``(Z, X)`` in pixels."""
        return self.img_size_spatial

    @property
    def measurement_shape(self) -> tuple[int, int, int, int]:
        """Per-sample measurement shape ``(n_channels, n_transmits, n_elements, n_samples)``."""
        return (self.n_channels, self.n_transmits, self.num_elements, self.n_samples)

    @property
    def num_elements(self) -> int:
        """Number of receive elements."""
        return int(self.element_positions.shape[0])

    @property
    def num_transmits(self) -> int:
        """Number of transmit events."""
        return self.n_transmits

    @property
    def num_samples(self) -> int:
        """Number of time samples per channel."""
        return self.n_samples

    @property
    def pixel_spacing(self) -> tuple[float, float]:
        """Pixel spacing ``(dz, dx)`` in meters, inferred from :attr:`pixel_grid`."""
        grid = self.pixel_grid
        dz = (grid[1, 0, 1] - grid[0, 0, 1]).item() if grid.shape[0] > 1 else 0.0
        dx = (grid[0, 1, 0] - grid[0, 0, 0]).item() if grid.shape[1] > 1 else 0.0
        return (dz, dx)

    @property
    def wavelength(self) -> float:
        r"""Nominal wavelength :math:`\lambda = c / f_{\mathrm{demod}}` in meters (falls back to :math:`c / f_s` if ``demodulation_frequency`` is ``None``)."""
        ref_freq = (
            self.demodulation_frequency
            if self.demodulation_frequency is not None
            else self.fs
        )
        return self.c / ref_freq

    def transmit_delay(self) -> Tensor:
        """Transmit delay ``(n_transmits, Z*X)`` in seconds. Subclass hook."""
        raise NotImplementedError(
            "Subclass must implement transmit_delay for its transmit mode."
        )

    def transmit_apod(self) -> Tensor | None:
        """Transmit apodization ``(n_transmits, Z*X)`` or ``None``. Subclass hook."""
        raise NotImplementedError(
            "Subclass must implement transmit_apod for its transmit mode."
        )

    def _receive_delay(self) -> Tensor:
        r"""Receive delay ``(n_elements, Z*X)`` in seconds.

        :math:`\tau_{\mathrm{rx}}(x, z; x_e, z_e) = \|(x,z) - (x_e, z_e)\|/c`.
        """
        grid = self.pixel_grid.reshape(-1, 2)
        dx = grid[:, 0].unsqueeze(0) - self.element_positions[:, 0].unsqueeze(1)
        dz = grid[:, 1].unsqueeze(0) - self.element_positions[:, 1].unsqueeze(1)
        return torch.hypot(dx, dz) / self.c

    @staticmethod
    def _window_shape(u: Tensor, kind: str) -> Tensor:
        """Compute a normalized apodization window at normalized positions ``u`` in [-1, 1]."""
        pi = math.pi
        if kind == "rect":
            return torch.ones_like(u)
        if kind == "hann":
            return 0.5 * (1.0 + torch.cos(pi * u))
        if kind == "hamming":
            return 0.54 + 0.46 * torch.cos(pi * u)
        # tukey0.25
        if kind == "tukey0.25":
            alpha = 0.25
            abs_u = torch.abs(u)
            flat = abs_u <= (1.0 - alpha)
            taper_arg = pi * (abs_u - (1.0 - alpha)) / alpha
            taper = 0.5 * (1.0 + torch.cos(taper_arg))
            return torch.where(flat, torch.ones_like(u), taper)
        raise ValueError(f"Unknown window kind: {kind}")

    def _receive_apod(self) -> Tensor | None:
        r"""Receive-side aperture apodization

        Combines the f-number cone (aperture half-width
        :math:`w = z_p / f_\#` at each pixel depth) with a tapered window
        applied to the normalized position :math:`u = (x_e - x_p) / w \in [-1, 1]`.

        Elements very close to the pixel in :math:`x` (``|dx| <= min_width``,
        a guard against divide-by-zero at boresight) always receive weight 1.
        """
        if self.f_number is None:
            return None
        grid = self.pixel_grid.reshape(-1, 2)

        if self.element_positions.shape[0] > 1:
            x_pos = torch.sort(self.element_positions[:, 0])[0]
            pitch = torch.diff(x_pos).mean().item()
            min_width = max(0.5 * pitch, 1e-6)
        else:
            min_width = 1e-3

        dx = grid[:, 0].unsqueeze(0) - self.element_positions[:, 0].unsqueeze(1)
        dz = grid[:, 1].unsqueeze(0) - self.element_positions[:, 1].unsqueeze(1)

        w_half = torch.abs(dz) / self.f_number
        eps = torch.full_like(dx, torch.finfo(self.pixel_grid.dtype).eps)
        u = dx / torch.maximum(w_half, eps)
        inside = torch.abs(u) <= 1.0

        window = self._window_shape(u, self.rx_apod_window)
        apod = torch.where(inside, window, torch.zeros_like(window))
        near_axis = torch.abs(dx) <= min_width
        apod = torch.where(near_axis, torch.ones_like(apod), apod)
        return apod.to(self.pixel_grid.dtype)

    def A(self, x: Tensor, **kwargs) -> Tensor:
        r"""Forward operator :math:`y = \forw{x} = \left(h \ast G\right) \left(x\right)`.

        :param torch.Tensor x: tissue reflectivity map of shape `(B, 2, Z, X)` in `"iq"` mode or `(B, 1, Z, X)` in `"rf"` mode.
        :return: channel data of shape `(B, C, n_transmits, n_elements, n_samples)`, divided by the operator norm if `normalize=True`.
        """
        B = x.shape[0]
        Z, X = self.img_size_spatial
        C = self.n_channels
        if x.ndim != 4 or x.shape[1] != C or x.shape[-2:] != (Z, X):
            raise ValueError(
                f"Expected image of shape (B, {C}, {Z}, {X}), got {tuple(x.shape)}."
            )
        n_t = self.n_transmits
        n_e = self.element_positions.shape[0]
        n_s = self.n_samples
        is_iq = self.signal_kind == "iq"

        if is_iq:
            x_flat = torch.view_as_complex(x.moveaxis(1, -1).contiguous()).reshape(
                B, Z * X
            )
            out_dtype = self._complex_dtype
        else:
            x_flat = x[:, 0].reshape(B, Z * X)
            out_dtype = self.dtype

        tau_rx = self._receive_delay()  # (n_e, Z*X)
        tau_tx_all = self.transmit_delay()  # (n_t, Z*X)
        apod_rx = self._receive_apod()  # (n_e, Z*X) or None
        apod_tx_all = self.transmit_apod()  # (n_t, Z*X) or None
        z_pix = self.pixel_grid.reshape(-1, 2)[:, 1] if is_iq else None

        y_out = torch.zeros((B, n_t, n_e, n_s), dtype=out_dtype, device=x.device)
        for k in range(n_t):
            tau_tx_k = tau_tx_all[k]  # (Z*X,)
            tau_full = tau_tx_k.unsqueeze(0) + tau_rx + self.t0[k]

            if is_iq:
                phase_arg = (
                    2.0
                    * math.pi
                    * self.demodulation_frequency
                    * (tau_full - 2.0 * z_pix / self.c)
                )
                weight = torch.exp(
                    torch.complex(torch.zeros_like(phase_arg), -phase_arg)
                )
                if apod_rx is not None:
                    weight = weight * apod_rx
            else:
                weight = apod_rx if apod_rx is not None else torch.ones_like(tau_full)

            if apod_tx_all is not None:
                weight = weight * apod_tx_all[k].unsqueeze(0)

            s_k = tau_full * self.fs
            contrib = x_flat.unsqueeze(1) * weight.unsqueeze(0)
            y_out[:, k] = self._interp1d_adjoint(s_k, contrib, n_s)

        if self.pulse_echo_ir is not None:
            y_flat = y_out.reshape(-1, 1, n_s)
            L_k = self.pulse_echo_ir.numel()
            pad_left = L_k // 2
            pad_right = L_k - 1 - pad_left
            if is_iq:
                y_real = y_flat.real
                y_imag = y_flat.imag
                y_real_padded = torch.nn.functional.pad(y_real, (pad_left, pad_right))
                y_imag_padded = torch.nn.functional.pad(y_imag, (pad_left, pad_right))
                conv_real = torch.nn.functional.conv1d(
                    y_real_padded, self.pulse_echo_ir.view(1, 1, -1)
                )
                conv_imag = torch.nn.functional.conv1d(
                    y_imag_padded, self.pulse_echo_ir.view(1, 1, -1)
                )
                y_out = torch.complex(conv_real, conv_imag).reshape(B, n_t, n_e, n_s)
            else:
                y_padded = torch.nn.functional.pad(y_flat, (pad_left, pad_right))
                conv = torch.nn.functional.conv1d(
                    y_padded, self.pulse_echo_ir.view(1, 1, -1)
                )
                y_out = conv.reshape(B, n_t, n_e, n_s)

        if is_iq:
            out = torch.view_as_real(y_out).moveaxis(-1, 1)
        else:
            out = y_out.unsqueeze(1)
        if self.normalize:
            out = out / self.operator_norm
        return out

    def A_adjoint(self, y: Tensor, **kwargs) -> Tensor:
        r"""Adjoint operator :math:`x = A^\top y = G^\top \! \left(\tilde{h} \ast_t y\right)`

        :param torch.Tensor y: channel data of shape `(B, 2, n_transmits, n_elements, n_samples)` in `"iq"` mode or `(B, 1, n_transmits, n_elements, n_samples)` in `"rf"` mode.
        :return: beamformed image of shape `(B, 2, Z, X)` or `(B, 1, Z, X)` respectively, divided by the operator norm if `normalize=True`.
        """
        B = y.shape[0]
        Z, X = self.img_size_spatial
        n_t = self.n_transmits
        n_e = self.element_positions.shape[0]
        expected = (self.n_channels, n_t, n_e, self.n_samples)
        if y.ndim != 5 or y.shape[1:] != expected:
            raise ValueError(
                f"Expected measurement of shape (B, "
                f"{', '.join(str(v) for v in expected)}), got {tuple(y.shape)}."
            )
        is_iq = self.signal_kind == "iq"

        if self.pulse_echo_ir is not None:
            n_s = y.shape[-1]
            y_flat_conv = y.reshape(-1, 1, n_s)
            L_k = self.pulse_echo_ir.numel()
            pad_left = L_k // 2
            pad_right = L_k - 1 - pad_left
            y_padded = torch.nn.functional.pad(y_flat_conv, (pad_right, pad_left))
            conv = torch.nn.functional.conv1d(
                y_padded, self.pulse_echo_ir.view(1, 1, -1).flip(-1)
            )
            y = conv.reshape(B, self.n_channels, n_t, n_e, n_s)

        if is_iq:
            y_flat = torch.view_as_complex(y.moveaxis(1, -1).contiguous())
            out_dtype = self._complex_dtype
        else:
            y_flat = y[:, 0]
            out_dtype = self.dtype

        tau_rx = self._receive_delay()
        tau_tx_all = self.transmit_delay()
        apod_rx = self._receive_apod()
        apod_tx_all = self.transmit_apod()
        z_pix = self.pixel_grid.reshape(-1, 2)[:, 1] if is_iq else None

        x_out = torch.zeros((B, Z * X), dtype=out_dtype, device=y.device)
        for k in range(n_t):
            tau_tx_k = tau_tx_all[k]
            tau_full = tau_tx_k.unsqueeze(0) + tau_rx + self.t0[k]

            if is_iq:
                phase_arg = (
                    2.0
                    * math.pi
                    * self.demodulation_frequency
                    * (tau_full - 2.0 * z_pix / self.c)
                )
                weight = torch.exp(
                    torch.complex(torch.zeros_like(phase_arg), phase_arg)
                )
                if apod_rx is not None:
                    weight = weight * apod_rx
            else:
                weight = apod_rx if apod_rx is not None else torch.ones_like(tau_full)

            if apod_tx_all is not None:
                weight = weight * apod_tx_all[k].unsqueeze(0)

            s_k = tau_full * self.fs
            gathered = self._interp1d(y_flat[:, k], s_k)
            x_out = x_out + (gathered * weight.unsqueeze(0)).sum(dim=1)

        if is_iq:
            out = torch.view_as_real(x_out.reshape(B, Z, X)).moveaxis(-1, 1)
        else:
            out = x_out.reshape(B, 1, Z, X)
        if self.normalize:
            out = out / self.operator_norm
        return out

    def _interp_taps(self, s: Tensor, n_samples: int) -> tuple[Tensor, Tensor]:
        r"""Returns interpolation weights and indices for the configured interpolation at a given sample position.

        :param torch.Tensor s: fractional sample positions, shape ``S``.
        :param int n_samples: length of the time axis.
        :return: ``indices`` of shape ``(K, *S)`` and
            ``weights`` of shape ``(K, *S)``,
            where ``K`` is 1 for ``nearest``, 2 for ``linear``, 4 for ``keys``.
        """
        if self.interp == "nearest":
            offsets = (0,)
            s_ref = torch.floor(s + 0.5)
        elif self.interp == "linear":
            offsets = (0, 1)
            s_ref = torch.floor(s)
        else:
            offsets = (-1, 0, 1, 2)
            s_ref = torch.floor(s)

        indices_list = []
        weights_list = []
        for off in offsets:
            idx = (s_ref + off).to(torch.long)
            valid = (idx >= 0) & (idx <= n_samples - 1)
            idx_clamped = idx.clamp(0, n_samples - 1)
            t = s - (s_ref + off)
            if self.interp == "nearest":
                w = torch.ones_like(t)
            else:
                abs_t = torch.abs(t)
                if self.interp == "linear":
                    w = (1.0 - abs_t).clamp(min=0.0)
                else:
                    t2 = abs_t * abs_t
                    t3 = t2 * abs_t
                    inner = 1.5 * t3 - 2.5 * t2 + 1.0
                    outer = -0.5 * t3 + 2.5 * t2 - 4.0 * abs_t + 2.0
                    w = torch.where(abs_t <= 1.0, inner, outer)
                    w = torch.where(abs_t <= 2.0, w, torch.zeros_like(w))
            w = torch.where(valid, w, torch.zeros_like(w))
            indices_list.append(idx_clamped)
            weights_list.append(w)
        return torch.stack(indices_list, dim=0), torch.stack(weights_list, dim=0)

    def _interp1d_adjoint(self, s: Tensor, contrib: Tensor, n_samples: int) -> Tensor:
        """Adjoint of :meth:`_interp1d`. Scatter-add the input ``contrib`` onto a
        length-``n_samples`` time axis at fractional positions ``s``.

        :param torch.Tensor s: fractional sample positions, shape ``(n_e, N)``.
        :param torch.Tensor contrib: contributions to accumulate, shape ``(B, n_e, N)``.
        :return: tensor of shape ``(B, n_e, n_samples)``, same dtype as ``contrib``.
        """
        B, n_e, N = contrib.shape
        indices, weights = self._interp_taps(s, n_samples)
        out = torch.zeros(
            (B, n_e, n_samples), dtype=contrib.dtype, device=contrib.device
        )
        for k in range(indices.shape[0]):
            idx_b = indices[k].unsqueeze(0).expand(B, n_e, N)
            w_c = weights[k].to(contrib.dtype)
            out = out.scatter_add(2, idx_b, contrib * w_c)
        return out

    def _interp1d(self, y_ke: Tensor, s: Tensor) -> Tensor:
        """Interpolate ``y_ke`` along the time axis at fractional positions ``s``.

        :param torch.Tensor y_ke: complex tensor of shape ``(B, n_e, n_samples)``.
        :param torch.Tensor s: fractional sample positions ``(n_e, N)``.
        :return: complex tensor of shape ``(B, n_e, N)``.
        """
        B, n_e, n_samples = y_ke.shape
        indices, weights = self._interp_taps(s, n_samples)
        out = None
        for k in range(indices.shape[0]):
            idx_b = indices[k].unsqueeze(0).expand(B, n_e, -1)
            gathered = torch.gather(y_ke, 2, idx_b)
            w_c = weights[k].to(y_ke.dtype)
            term = gathered * w_c
            out = term if out is None else out + term
        return out


class UltrasoundPlaneWave(UltrafastUltrasound):
    r"""
    2D plane-wave (PW) ultrafast ultrasound imaging operator.

    Specializes :class:`UltrafastUltrasound` to plane-wave transmits: for a steering angle :math:`\theta_k`, the quadratic Radon transform :math:`G` in

    .. math::
        y = \forw{x} = \left(h \ast_t G\right)(x)

    projects :math:`x` along parabolas with focus :math:`\mathbf{r}_i` (receive element position) and directrix perpendicular to :math:`(\sin\theta_k, \cos\theta_k)` (eccentricity 1), with round-trip time-of-flight

    .. math::
        \tau_{k,i}(\mathbf{r}) = \frac{x \sin\theta_k + z \cos\theta_k}{c} + \frac{\|\mathbf{r} - \mathbf{r}_i\|}{c}.

    The adjoint :math:`A^\top = G^\top (\tilde{h} \ast_t \cdot)` is delay-and-sum beamforming coherently compounded over angles and elements.
    See :class:`UltrafastUltrasound` for the full definition of :math:`G` (apodization, interpolation, modulation phase) and the adjoint.


    .. warning::
        By default, `normalize` is set to `True` if not specified; leaving it unset issues a warning. Normalization affects reconstruction dynamics, which may not be suitable for real-world applications.

    :param tuple[int, int] img_size: spatial image size `(Z, X)` in pixels.
    :param Iterable[float], torch.Tensor angles: transmit steering angles in radians.
    :param torch.Tensor element_positions: receive element positions in meters, shape `(n_elements, 2)` with columns `(x, z)`.
    :param int n_samples: number of time samples per channel.
    :param float sampling_frequency: sampling frequency in Hz.
    :param float sound_speed: speed of sound :math:`c` in m/s. (default: `1540`)
    :param torch.Tensor pixel_grid: optional pixel positions in meters, shape `(Z, X, 2)` with columns `(x, z)`. If `None`, built from `pixel_size` and `pixel_origin`. (default: `None`)
    :param tuple[float, float] pixel_size: pixel spacing `(dz, dx)` in meters. (default: half a wavelength :math:`c / (2 f_{\mathrm{demod}})` along both axes)
    :param tuple[float, float] pixel_origin: grid origin `(z0, x0)` in meters. (default: `(0, x_aperture_center)`)
    :param float, torch.Tensor t0: acquisition-start offset :math:`t_0` in seconds, scalar or per-angle tensor of shape `(n_angles,)`. (default: `0.0`)
    :param float demodulation_frequency: demodulation frequency in Hz, required in `"iq"` mode and ignored in `"rf"` mode. (default: `None`)
    :param float f_number: receive f-number defining the aperture half-width :math:`|x_i - x_j| \le z_j / f_\#` at each pixel. `None` disables receive apodization. (default: `None`)
    :param str rx_apod_window: receive apodization window inside the f-number aperture, one of `"rect"`, `"hann"`, `"hamming"`, `"tukey0.25"`. Ignored if `f_number` is `None`. (default: `"rect"`)
    :param str tx_apod_window: transmit apodization window, one of `"rect"`, `"hann"`, `"hamming"`, `"tukey0.25"`. `None` disables transmit apodization. (default: `None`)
    :param str interp: interpolation kernel. One of `"nearest"`, `"linear"`, `"keys"`. See :class:`UltrafastUltrasound` for kernel definitions. (default: `"linear"`)
    :param str signal_kind: signal representation, `"iq"` or `"rf"`. (default: `"iq"`)
    :param torch.Tensor pulse: optional 1D pulse-echo impulse response :math:`h`, normalized to unit :math:`\ell_2` norm and convolved along the time axis in both :meth:`A` and :meth:`A_adjoint`. (default: `None`)
    :param bool normalize: if `True`, :meth:`A` and :meth:`A_adjoint` are divided by the operator's spectral norm so it has unit norm. (default: `True`)
    :param torch.device, str device: device for buffers. (default: `"cpu"`)
    :param torch.dtype dtype: real dtype; `float32` uses internal `complex64`, `float64` uses `complex128`. (default: `torch.float32`)

    |sep|

    :Examples:

        IQ operator on a 32x32 image with 4 receive elements and 3 steering
        angles:

        .. doctest::

            >>> import torch
            >>> from deepinv.physics import UltrasoundPlaneWave
            >>> _ = torch.manual_seed(0)
            >>> ele_pos = torch.stack(
            ...     [torch.linspace(-1e-3, 1e-3, 4), torch.zeros(4)], dim=-1
            ... )
            >>> physics = UltrasoundPlaneWave(
            ...     img_size=(32, 32),
            ...     angles=torch.linspace(-0.28, 0.28, 3),
            ...     element_positions=ele_pos,
            ...     n_samples=256,
            ...     sampling_frequency=40e6,
            ...     demodulation_frequency=5e6,
            ...     normalize=False,
            ... )
            >>> x = torch.randn(1, 2, 32, 32)
            >>> y = physics(x)
            >>> print(y.shape)
            torch.Size([1, 2, 3, 4, 256])
            >>> print(physics.A_adjoint(y).shape)
            torch.Size([1, 2, 32, 32])

        Same operator on native RF signals (single channel):

        .. doctest::

            >>> physics_rf = UltrasoundPlaneWave(
            ...     img_size=(32, 32),
            ...     angles=torch.linspace(-0.28, 0.28, 3),
            ...     element_positions=ele_pos,
            ...     n_samples=256,
            ...     sampling_frequency=40e6,
            ...     signal_kind="rf",
            ...     normalize=False,
            ... )
            >>> x_rf = torch.randn(1, 1, 32, 32)
            >>> print(physics_rf(x_rf).shape)
            torch.Size([1, 1, 3, 4, 256])
    """

    def __init__(
        self,
        img_size: tuple[int, int],
        angles: Iterable[float] | Tensor,
        element_positions: Tensor,
        n_samples: int,
        sampling_frequency: float,
        sound_speed: float = 1540.0,
        pixel_grid: Tensor | None = None,
        pixel_size: tuple[float, float] | None = None,
        pixel_origin: tuple[float, float] | None = None,
        t0: float | Tensor = 0.0,
        demodulation_frequency: float | None = None,
        f_number: float | None = None,
        rx_apod_window: str = "rect",
        tx_apod_window: str | None = None,
        interp: str = "linear",
        signal_kind: str = "iq",
        pulse: Tensor | None = None,
        normalize: bool | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if isinstance(angles, (list, tuple, ndarray)):
            theta = torch.tensor(angles, dtype=dtype)
        elif isinstance(angles, torch.Tensor):
            theta = angles.to(dtype=dtype)
        else:
            raise ValueError(
                f"angles must be an iterable or Tensor, but got {type(angles)}"
            )
        if theta.ndim != 1:
            raise ValueError(f"angles must be 1-D, got shape {tuple(theta.shape)}.")
        n_transmits = theta.numel()

        super().__init__(
            img_size=img_size,
            element_positions=element_positions,
            n_transmits=n_transmits,
            n_samples=n_samples,
            sampling_frequency=sampling_frequency,
            sound_speed=sound_speed,
            pixel_grid=pixel_grid,
            pixel_size=pixel_size,
            pixel_origin=pixel_origin,
            t0=t0,
            demodulation_frequency=demodulation_frequency,
            f_number=f_number,
            rx_apod_window=rx_apod_window,
            tx_apod_window=tx_apod_window,
            interp=interp,
            signal_kind=signal_kind,
            pulse=pulse,
            device=device,
            dtype=dtype,
        )
        self.register_buffer("angles", theta.contiguous())
        self.to(device)

        if normalize is None:
            warn(
                "The default value of `normalize` is not specified and will "
                "be automatically set to `True`. Set `normalize` explicitly "
                "to `True` or `False` to avoid this warning."
            )
            normalize = True
        self.normalize = False
        if normalize:
            grid_device = self.pixel_grid.device
            x = torch.randn(
                (1, self.n_channels, *self.img_size_spatial),
                generator=torch.Generator(grid_device).manual_seed(0),
                device=grid_device,
                dtype=self.dtype,
            )
            self.register_buffer(
                "operator_norm",
                self.compute_norm(x, squared=False, verbose=False),
            )
            self.normalize = True

    @property
    def num_angles(self) -> int:
        """Number of transmit steering angles (alias of :attr:`num_transmits`)."""
        return self.n_transmits

    @property
    def angular_range(self) -> tuple[float, float]:
        """Min/max steering angle in degrees, as ``(theta_min, theta_max)``."""
        deg = torch.rad2deg(self.angles)
        return (deg.min().item(), deg.max().item())

    def select_transmits(
        self, indices: Iterable[int] | Tensor
    ) -> "UltrasoundPlaneWave":
        """Return a new operator restricted to a subset of transmit angles.

        Useful for angle-subset splitting in self-supervised losses such as
        :class:`deepinv.loss.Noise2InverseLoss`.

        :param indices: 1-D iterable of angle indices to keep.
        :return: a new :class:`UltrasoundPlaneWave` sharing element positions,
            pixel grid, and acquisition parameters, with sliced ``angles`` and
            ``t0``.
        """
        idx = torch.as_tensor(list(indices), dtype=torch.long)
        pulse = (
            self.pulse_echo_ir.detach().clone()
            if self.pulse_echo_ir is not None
            else None
        )
        return UltrasoundPlaneWave(
            img_size=self.img_size_spatial,
            angles=self.angles[idx].detach().clone(),
            element_positions=self.element_positions.detach().clone(),
            n_samples=self.n_samples,
            sampling_frequency=self.fs,
            sound_speed=self.c,
            pixel_grid=self.pixel_grid.detach().clone(),
            t0=self.t0[idx].detach().clone(),
            demodulation_frequency=self.demodulation_frequency,
            f_number=self.f_number,
            rx_apod_window=self.rx_apod_window,
            tx_apod_window=self.tx_apod_window,
            interp=self.interp,
            signal_kind=self.signal_kind,
            pulse=pulse,
            normalize=self.normalize,
            device=self.pixel_grid.device,
            dtype=self.dtype,
        )

    def select_angles(self, indices: Iterable[int] | Tensor) -> "UltrasoundPlaneWave":
        """Alias for :meth:`select_transmits`."""
        return self.select_transmits(indices)

    def transmit_delay(self) -> Tensor:
        r"""Plane-wave transmit delay ``(n_angles, Z*X)`` in seconds.

        :math:`\tau_{\mathrm{tx}}(x, z; \theta_k) = (x \sin\theta_k + z \cos\theta_k)/c`.
        """
        grid = self.pixel_grid.reshape(-1, 2)
        x = grid[:, 0]
        z = grid[:, 1]
        sin_t = torch.sin(self.angles).unsqueeze(-1)
        cos_t = torch.cos(self.angles).unsqueeze(-1)
        return (x * sin_t + z * cos_t) / self.c

    def transmit_apod(self) -> Tensor | None:
        r"""Plane-wave transmit apodization ``(n_angles, Z*X)``.

        For each transmit angle :math:`\theta_k`
        each pixel is projected back onto the aperture line,
        :math:`x_{\text{proj}} = x_p - z_p \tan\theta_k`. Pixels whose
        projection falls outside the aperture (with a 20% margin) are
        masked to zero. A tapered window is applied to the normalized position
        inside the aperture, matching :attr:`tx_apod_window`.
        """
        if self.tx_apod_window is None:
            return None
        grid = self.pixel_grid.reshape(-1, 2)
        x = grid[:, 0].unsqueeze(0)
        z = grid[:, 1].unsqueeze(0)
        tan_t = torch.tan(self.angles).unsqueeze(-1)
        x_proj = x - z * tan_t
        x_min = self.element_positions[:, 0].min() * 1.2
        x_max = self.element_positions[:, 0].max() * 1.2
        x_center = 0.5 * (x_min + x_max)
        x_half = 0.5 * (x_max - x_min)
        u = (x_proj - x_center) / x_half
        inside = torch.abs(u) <= 1.0
        window = self._window_shape(u, self.tx_apod_window)
        apod = torch.where(inside, window, torch.zeros_like(window))
        return apod.to(self.pixel_grid.dtype)
