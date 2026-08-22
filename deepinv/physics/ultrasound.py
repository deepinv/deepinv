from __future__ import annotations
from typing import Iterable, Sequence
from warnings import warn

import math

from numpy import ndarray
import torch
from torch import Tensor

from deepinv.physics.forward import LinearPhysics


class UltrafastUltrasound(LinearPhysics):
    r"""
    Base class for ultrafast ultrasound imaging physics.

    Implements the transmit-mode-independent core of a linearized ultrafast
    ultrasound forward operator and its adjoint (Delay-and-Sum algorithm).

    .. rubric:: Mathematical Formulation

    Following :footcite:t:`besson2018ultrafast`, the continuous forward model links the continuous
    transducer measurements :math:`s_{k,i}(t)` at receive element :math:`i` for transmit :math:`k`
    to the reflectivity function of the medium :math:`x(\mathbf{r})` over the spatial domain :math:`\Omega`:

    .. math::

        s_{k,i}(t) = \int_{\Omega} a_{k,i}(\mathbf{r}) \, x(\mathbf{r}) \, h\Big(t - \tau_{k,i}(\mathbf{r})\Big) \, d\mathbf{r},

    where:

    - :math:`x(\mathbf{r})` is the continuous tissue reflectivity function,
    - :math:`a_{k,i}(\mathbf{r})` is the combined transmit-receive apodization weight,
    - :math:`h(t)` is the pulse-echo impulse response of the system,
    - :math:`\tau_{k,i}(\mathbf{r}) = \tau_{\text{tx},k}(\mathbf{r}) + \tau_{\text{rx},i}(\mathbf{r})` is the round-trip propagation time of flight from the source to pixel :math:`\mathbf{r}` and back to element :math:`i`.

    In the discrete domain, this integral is discretized over a spatial grid of pixels :math:`\mathbf{r}_j`
    resulting in the standard linear inverse problem system:

    .. math::

        \mathbf{y} = \mathbf{A}\mathbf{x},

    where:

    - :math:`\mathbf{x} \in \mathbb{R}^{ZX}` (or :math:`\mathbb{C}^{ZX}`) is the vectorized reflectivity image consisting of :math:`Z \times X` spatial pixels,
    - :math:`\mathbf{y} \in \mathbb{R}^{M}` (or :math:`\mathbb{C}^{M}`) is the vectorized channel-data measurements of dimension :math:`M = n_{\text{transmits}} \times n_{\text{elements}} \times n_{\text{samples}}`,
    - :math:`\mathbf{A} \in \mathbb{R}^{M \times ZX}` (or :math:`\mathbb{C}^{M \times ZX}`) is the sparse linearized forward measurement matrix.

    For each individual measurement sample index :math:`n` corresponding to the :math:`i`-th receiver element and :math:`k`-th transmit event, the matrix-vector product is written element-wise as:

    .. math::

        y_{k,i,n} = \sum_{j} A_{(k,i,n), j} \, x_j,

    where :math:`y_{k,i,n}` is the recorded channel-data sample at time :math:`t_n = n / f_s`, and the
    forward matrix elements :math:`A_{(k,i,n), j}` are given by:

    .. math::

        A_{(k,i,n), j} = a_{k,i}(\mathbf{r}_j) \, K\Big(f_s (t_n - \tau_{k,i}(\mathbf{r}_j))\Big) \, \varphi_{k,i}(\mathbf{r}_j).

    Here :math:`K(\cdot)` is the interpolation kernel (nearest-neighbor, linear/triangular, or cubic), and
    :math:`\varphi_{k,i}(\mathbf{r}_j) = \exp(-i 2\pi f_{\text{demod}} (\tau_{k,i}(\mathbf{r}_j) - 2 z_j / c))`
    is the demodulation phase rotation term (active in ``"iq"`` mode, and :math:`1.0` in ``"rf"`` mode).

    Subclasses plug in a transmit mode by overriding two methods; everything else is
    shared: receive delay, receive apodization (f-number cone + tapered
    window), IQ demodulation phase compensation, the K-tap scatter (forward)
    and gather (adjoint) interpolation core, and the outer loops over
    transmits and receive elements.

    One concrete transmit mode ships with the library:
    :class:`UltrasoundPlaneWave` (steered plane waves). To add
    a new one (focused transmit, diverging wave, Hadamard-coded, phase-coded, …) subclass
    :class:`UltrafastUltrasound` and follow the extension contract below.

    .. rubric:: Extension contract

    A concrete subclass MUST:

    1. Accept its transmit parameterization in ``__init__`` and derive the
       integer ``n_transmits`` from it (e.g. ``len(angles)`` or
       ``virtual_sources.shape[0]``).
    2. Call ``super().__init__(..., n_transmits=n_transmits, ...)`` passing
       through all the shared kwargs.
    3. Register its transmit parameters as ``torch.nn.Module`` buffers *after*
       the ``super().__init__(...)`` call so they migrate with ``.to(device)``.
    4. Override :meth:`transmit_delay` to return a tensor of shape
       ``(n_transmits, Z*X)`` giving the transmit propagation delay in
       seconds for every pixel in the flat pixel grid.
    5. Override :meth:`transmit_apod` to return either ``None`` (no transmit
       apodization) or a tensor of shape ``(n_transmits, Z*X)`` used as a
       multiplicative per-pixel weight during forward and adjoint.
    6. Call ``self.to(device)`` at the end of ``__init__`` so the newly
       registered buffers land on the correct device.

    Everything else — the interp core, IQ phase, receive apodization, forward
    and adjoint drivers — is inherited and not intended to be overridden.
    Those internals are marked with a leading underscore.

    :param tuple img_size: spatial image size ``(Z, X)`` in pixels.
    :param torch.Tensor element_positions: receive element positions in
        meters, shape ``(n_elements, 2)`` with columns ``(x, z)``.
    :param int n_transmits: number of transmits in the acquisition.
    :param int n_samples: number of time samples per channel.
    :param float sampling_frequency: ADC sampling frequency :math:`f_s` in Hz.
    :param float sound_speed: assumed speed of sound in m/s. Default 1540.
    :param str signal_kind: signal representation. ``"iq"`` (default) treats
        images and measurements as demodulated IQ with an ``(I, Q)`` channel
        pair on dim 1 and applies IQ phase compensation. ``"rf"`` treats them
        as real single-channel radio-frequency signals with no phase term.
    :param bool normalize: If ``True``, :meth:`A` and :meth:`A_adjoint` are
        normalized so that the operator has unit norm. Mirrors the convention
        of :class:`deepinv.physics.Tomography`. (default: ``True``)

    See :class:`UltrasoundPlaneWave` for the full list of shared parameters
    (pixel grid, ``t0``, ``fdemod``, ``f_number``, apodization windows,
    interpolation kernel, device / dtype).
    """

    _VALID_INTERP = ("nearest", "linear", "keys")
    _VALID_WINDOWS = ("rect", "hann", "hamming", "tukey0.25")
    _VALID_SIGNAL_KINDS = ("iq", "rf")

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
        fdemod: float = 5e6,
        f_number: float | None = None,
        rx_apod_window: str = "rect",
        tx_apod_window: str | None = None,
        interp: str = "linear",
        signal_kind: str = "iq",
        pulse: Tensor | None = None,
        normalize: bool | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
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
                ref_freq = fdemod if fdemod > 0.0 else sampling_frequency
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
            grid = torch.stack([xx, zz], dim=-1)  # (Z, X, 2), columns (x, z)

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
            **kwargs,
        )

        self.register_buffer("element_positions", ele_pos.contiguous())
        self.register_buffer("pixel_grid", grid.contiguous())
        self.register_buffer("t0", t0_tensor.contiguous())

        self.img_size_spatial = (Z, X)
        self.n_transmits = int(n_transmits)
        self.n_samples = int(n_samples)
        self.fs = float(sampling_frequency)
        self.c = float(sound_speed)
        self.fdemod = float(fdemod)
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

        # Optional transducer pulse-echo impulse response modeling
        if pulse is not None:
            h = torch.as_tensor(pulse, dtype=self.dtype)
            h = h / torch.linalg.norm(h)
            self.register_buffer("pulse_echo_ir", h.contiguous())
        else:
            self.pulse_echo_ir = None

        # Normalize the operator so that ||A|| = 1, matching the Tomography
        # convention (see :class:`deepinv.physics.Tomography`). Subclasses must
        # have registered their transmit parameterization before calling
        # :meth:`_finalize_normalization` at the end of their own ``__init__``.
        self._normalize_flag = normalize
        self.normalize = False  # safe default; overwritten by _finalize_normalization

    # -------------------------------------------------------------------------
    # Geometry helpers.
    # -------------------------------------------------------------------------
    def _grid_flat(self) -> Tensor:
        """Flat pixel grid ``(Z*X, 2)`` in meters — columns ``(x, z)``."""
        return self.pixel_grid.reshape(-1, 2)

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
        grid = self._grid_flat()
        dx = grid[:, 0].unsqueeze(0) - self.element_positions[:, 0].unsqueeze(1)
        dz = grid[:, 1].unsqueeze(0) - self.element_positions[:, 1].unsqueeze(1)
        return torch.hypot(dx, dz) / self.c

    @staticmethod
    def _window_shape(u: Tensor, kind: str) -> Tensor:
        """Compute a normalized apodization window at positions ``u`` in [-1, 1].

        The mask enforcing ``|u| <= 1`` is applied by the caller; this helper
        just returns the analytical window shape.
        """
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
        r"""Receive-side aperture apodization ``(n_elements, Z*X)`` or ``None``.

        Combines the f-number cone (aperture half-width
        :math:`w = z_p / f_\#` at each pixel depth) with a tapered window
        applied to the normalized position :math:`u = (x_e - x_p) / w \in [-1, 1]`:

        - ``rect``: :math:`\mathbf{1}_{|u|\le 1}`.
        - ``hann``: :math:`0.5 (1 + \cos(\pi u))\,\mathbf{1}_{|u|\le 1}`.
        - ``hamming``: :math:`(0.54 + 0.46\cos(\pi u))\,\mathbf{1}_{|u|\le 1}`.
        - ``tukey`` with shape :math:`\alpha`: unit in the flat region
          :math:`|u|\le 1-\alpha` and a raised-cosine taper at the edges.

        Elements very close to the pixel in :math:`x` (``|dx| <= min_width``,
        a guard against divide-by-zero at boresight) always receive weight 1.
        """
        if self.f_number is None:
            return None
        grid = self._grid_flat()

        # Dynamically set min_width based on element pitch (0.5 * pitch) to prevent blind regions
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

    # -------------------------------------------------------------------------
    # Forward / adjoint (mode-independent).
    # -------------------------------------------------------------------------
    def _iq_phase_arg(self, tau_total: Tensor) -> Tensor:
        r"""Phase argument :math:`2\pi f_{\text{demod}} (\tau_{\text{full}} - 2z/c)`.

        ``tau_total`` is the total round-trip delay in seconds *including*
        the acquisition offset ``t0``. Shape ``(n_elements, Z*X)`` for a single
        transmit.
        """
        z_pix = self._grid_flat()[:, 1]  # (Z*X,)
        tshift = tau_total - 2.0 * z_pix / self.c
        return 2.0 * math.pi * self.fdemod * tshift

    def _finalize_normalization(self) -> None:
        r"""Finalize construction: warn on unspecified ``normalize`` and
        compute the operator norm if requested.

        Subclasses must call this at the end of their ``__init__``, after
        registering all transmit-mode-specific buffers, so that the power
        iteration inside :meth:`compute_norm` sees the full operator. Mirrors
        the pattern used by :class:`deepinv.physics.Tomography`.
        """
        normalize = self._normalize_flag
        if normalize is None:
            warn(
                "The default value of `normalize` is not specified and will "
                "be automatically set to `True`. Set `normalize` explicitly "
                "to `True` or `False` to avoid this warning.",
            )
            normalize = True
        self.normalize = False
        if normalize:
            device = self.pixel_grid.device
            x = torch.randn(
                (1, self.n_channels, *self.img_size_spatial),
                generator=torch.Generator(device).manual_seed(0),
                device=device,
                dtype=self.dtype,
            )
            operator_norm = self.compute_norm(x, squared=False, verbose=False)
            self.register_buffer("operator_norm", operator_norm)
            self.normalize = True

    def A(self, x: Tensor, **kwargs) -> Tensor:
        r"""Forward operator :math:`y = A x` (Besson kernel).

        :param torch.Tensor x: image ``(B, 2, Z, X)`` in IQ mode, or
            ``(B, 1, Z, X)`` in RF mode.
        :return: measurement ``(B, 2, n_transmits, n_elements, n_samples)`` or
            ``(B, 1, n_transmits, n_elements, n_samples)`` respectively.
        """
        self._check_image_shape(x)
        B = x.shape[0]
        Z, X = self.img_size_spatial
        n_t = self.n_transmits
        n_e = self.element_positions.shape[0]
        n_s = self.n_samples
        is_iq = self.signal_kind == "iq"

        if is_iq:
            x_flat = self._real_to_complex(x).reshape(B, Z * X)
            out_dtype = self._complex_dtype
        else:
            x_flat = x[:, 0].reshape(B, Z * X)
            out_dtype = self.dtype

        tau_rx = self._receive_delay()  # (n_e, Z*X)
        tau_tx_all = self.transmit_delay()  # (n_t, Z*X)
        apod_rx = self._receive_apod()  # (n_e, Z*X) or None
        apod_tx_all = self.transmit_apod()  # (n_t, Z*X) or None

        y_out = torch.zeros((B, n_t, n_e, n_s), dtype=out_dtype, device=x.device)
        for k in range(n_t):
            tau_tx_k = tau_tx_all[k]  # (Z*X,)
            tau_full = tau_tx_k.unsqueeze(0) + tau_rx + self.t0[k]

            if is_iq:
                phase_arg = self._iq_phase_arg(tau_full)
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
            y_out[:, k] = self._scatter_interp_1d(s_k, contrib, n_s)

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
            out = self._complex_to_real(y_out)
        else:
            out = y_out.unsqueeze(1)
        if self.normalize:
            out = out / self.operator_norm
        return out

    def A_adjoint(self, y: Tensor, **kwargs) -> Tensor:
        r"""Adjoint operator :math:`x = A^\top y` — Delay-and-Sum beamforming.

        :param torch.Tensor y: measurement of shape
            ``(B, 2, n_transmits, n_elements, n_samples)`` in IQ mode, or
            ``(B, 1, n_transmits, n_elements, n_samples)`` in RF mode.
        :return: image ``(B, 2, Z, X)`` or ``(B, 1, Z, X)`` respectively.
        """
        self._check_measurement_shape(y)
        B = y.shape[0]
        Z, X = self.img_size_spatial
        n_t = self.n_transmits
        n_e = self.element_positions.shape[0]
        is_iq = self.signal_kind == "iq"

        if self.pulse_echo_ir is not None:
            n_s = y.shape[-1]
            y_flat_conv = y.reshape(-1, 1, n_s)
            L_k = self.pulse_echo_ir.numel()
            pad_left = L_k // 2
            pad_right = L_k - 1 - pad_left
            # Adjoint padding is reversed: (pad_right, pad_left)
            y_padded = torch.nn.functional.pad(y_flat_conv, (pad_right, pad_left))
            conv = torch.nn.functional.conv1d(
                y_padded, self.pulse_echo_ir.view(1, 1, -1).flip(-1)
            )
            y = conv.reshape(B, self.n_channels, n_t, n_e, n_s)

        if is_iq:
            y_flat = self._real_to_complex(y)  # (B, n_t, n_e, n_s) complex
            out_dtype = self._complex_dtype
        else:
            y_flat = y[:, 0]  # (B, n_t, n_e, n_s) real
            out_dtype = self.dtype

        tau_rx = self._receive_delay()
        tau_tx_all = self.transmit_delay()
        apod_rx = self._receive_apod()
        apod_tx_all = self.transmit_apod()

        x_out = torch.zeros((B, Z * X), dtype=out_dtype, device=y.device)
        for k in range(n_t):
            tau_tx_k = tau_tx_all[k]
            tau_full = tau_tx_k.unsqueeze(0) + tau_rx + self.t0[k]

            if is_iq:
                phase_arg = self._iq_phase_arg(tau_full)
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
            gathered = self._gather_interp_1d(y_flat[:, k], s_k)
            x_out = x_out + (gathered * weight.unsqueeze(0)).sum(dim=1)

        if is_iq:
            out = self._complex_to_real(x_out.reshape(B, Z, X))
        else:
            out = x_out.reshape(B, 1, Z, X)
        if self.normalize:
            out = out / self.operator_norm
        return out

    # -------------------------------------------------------------------------
    # Complex <-> real channel-pair conversions.
    # -------------------------------------------------------------------------
    def _real_to_complex(self, x: Tensor) -> Tensor:
        """``(B, 2, ...)`` real → ``(B, ...)`` complex."""
        return torch.view_as_complex(x.moveaxis(1, -1).contiguous())

    def _complex_to_real(self, x: Tensor) -> Tensor:
        """``(B, ...)`` complex → ``(B, 2, ...)`` real."""
        return torch.view_as_real(x).moveaxis(-1, 1)

    # -------------------------------------------------------------------------
    # 1-D interpolation core (K-tap scatter / gather along the last axis).
    #
    # The forward is a scatter of pixel contributions into the time axis and
    # the adjoint is a gather; both share the same taps so the two are exact
    # transposes at the interpolation-kernel level. Out-of-range neighbors
    # are masked *independently* to match ``grid_sample(padding_mode='zeros')``.
    # -------------------------------------------------------------------------
    def _interp_taps(self, s: Tensor, n_samples: int) -> tuple[Tensor, Tensor]:
        """Return ``(indices, weights)`` for the configured interpolation.

        :param torch.Tensor s: fractional sample positions, shape ``S``.
        :param int n_samples: length of the time axis.
        :return: ``indices`` of shape ``(K, *S)`` (long, clamped) and
            ``weights`` of shape ``(K, *S)`` (float, OOB neighbors zeroed),
            where ``K`` is 1 for ``nearest``, 2 for ``linear``, 4 for ``keys``.
        """
        if self.interp == "nearest":
            offsets = (0,)
            s_ref = torch.floor(s + 0.5)  # round half up
        elif self.interp == "linear":
            offsets = (0, 1)
            s_ref = torch.floor(s)
        else:  # "keys"
            offsets = (-1, 0, 1, 2)
            s_ref = torch.floor(s)

        indices_list = []
        weights_list = []
        for off in offsets:
            idx = (s_ref + off).to(torch.long)
            valid = (idx >= 0) & (idx <= n_samples - 1)
            idx_clamped = idx.clamp(0, n_samples - 1)
            t = s - (s_ref + off)  # signed distance to this tap
            w = self._interp_kernel(t)
            w = torch.where(valid, w, torch.zeros_like(w))
            indices_list.append(idx_clamped)
            weights_list.append(w)
        indices = torch.stack(indices_list, dim=0)
        weights = torch.stack(weights_list, dim=0)
        return indices, weights

    def _interp_kernel(self, t: Tensor) -> Tensor:
        r"""Interpolation kernel value at signed distance ``t`` from a tap.

        - nearest: :math:`\mathbf{1}_{|t| < 0.5}` — never queried in practice
          since :meth:`_interp_taps` picks the single nearest tap and always
          weights it 1.
        - linear: :math:`\max(1 - |t|, 0)` — the Besson triangular kernel.
        - Keys cubic (:math:`a=-1/2`):

          .. math::

              K(t) = \begin{cases}
                  1.5 |t|^3 - 2.5 |t|^2 + 1        & |t| \le 1\\
                  -0.5 |t|^3 + 2.5 |t|^2 - 4|t| + 2 & 1 < |t| \le 2\\
                  0                                 & \text{otherwise}
              \end{cases}
        """
        if self.interp == "nearest":
            return torch.ones_like(t)
        abs_t = torch.abs(t)
        if self.interp == "linear":
            return (1.0 - abs_t).clamp(min=0.0)
        # Keys cubic (a = -1/2).
        t2 = abs_t * abs_t
        t3 = t2 * abs_t
        inner = 1.5 * t3 - 2.5 * t2 + 1.0
        outer = -0.5 * t3 + 2.5 * t2 - 4.0 * abs_t + 2.0
        w = torch.where(abs_t <= 1.0, inner, outer)
        return torch.where(abs_t <= 2.0, w, torch.zeros_like(w))

    def _scatter_interp_1d(self, s: Tensor, contrib: Tensor, n_samples: int) -> Tensor:
        """Splat ``contrib`` into a length-``n_samples`` axis at fractional ``s``.

        :param torch.Tensor s: fractional sample positions, shape ``(n_e, N)``.
        :param torch.Tensor contrib: complex contributions, shape ``(B, n_e, N)``.
        :return: complex tensor of shape ``(B, n_e, n_samples)``.
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

    def _gather_interp_1d(self, y_ke: Tensor, s: Tensor) -> Tensor:
        """Interpolation of ``y_ke`` at fractional positions ``s``.

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

    # -------------------------------------------------------------------------
    # Shape checks.
    # -------------------------------------------------------------------------
    def _check_image_shape(self, x: Tensor) -> None:
        Z, X = self.img_size_spatial
        C = self.n_channels
        if x.ndim != 4 or x.shape[1] != C or x.shape[-2:] != (Z, X):
            raise ValueError(
                f"Expected image of shape (B, {C}, {Z}, {X}), got {tuple(x.shape)}."
            )

    def _check_measurement_shape(self, y: Tensor) -> None:
        expected = (
            self.n_channels,
            self.n_transmits,
            self.element_positions.shape[0],
            self.n_samples,
        )
        if y.ndim != 5 or y.shape[1:] != expected:
            raise ValueError(
                f"Expected measurement of shape (B, {', '.join(str(v) for v in expected)}), "
                f"got {tuple(y.shape)}."
            )


class UltrasoundPlaneWave(UltrafastUltrasound):
    r"""
    Ultrafast plane-wave (PW) ultrasound imaging operator.

    Linearized measurement model for coherent plane-wave compounding, following the
    matrix-free parameterization of :footcite:t:`besson2018ultrafast`. It models the standard
    linear inverse problem system:

    .. math::

        \mathbf{y} = \mathbf{A}\mathbf{x},

    where:

    - :math:`\mathbf{x} \in \mathbb{R}^{ZX}` (or :math:`\mathbb{C}^{ZX}`) is the vectorized reflectivity image consisting of :math:`Z \times X` spatial pixels,
    - :math:`\mathbf{y} \in \mathbb{R}^{M}` (or :math:`\mathbb{C}^{M}`) is the vectorized channel-data measurements of dimension :math:`M = n_{\text{angles}} \times n_{\text{elements}} \times n_{\text{samples}}`,
    - :math:`\mathbf{A} \in \mathbb{R}^{M \times ZX}` (or :math:`\mathbb{C}^{M \times ZX}`) is the sparse linearized forward measurement matrix representing steered plane-wave propagation.

    For each individual measurement sample index :math:`n` corresponding to the :math:`i`-th receiver element and :math:`k`-th steered plane-wave angle :math:`\theta_k`, the matrix-vector product is written element-wise as:

    .. math::

        y_{k,i,n} = \sum_{j} A_{(k,i,n), j} \, x_j,

    where :math:`y_{k,i,n}` is the recorded channel-data sample at time :math:`t_n = n / f_s`, and the
    forward matrix elements :math:`A_{(k,i,n), j}` are given by:

    .. math::

        A_{(k,i,n), j} = a_{k,i}(\mathbf{r}_j) \, K\Big(f_s (t_n - \tau_{k,i}(\mathbf{r}_j))\Big) \, \varphi_{k,i}(\mathbf{r}_j).

    Here:

    - :math:`a_{k,i}(\mathbf{r}_j)` is the combined transmit-receive apodization,
    - :math:`K(\cdot)` is the interpolation kernel (nearest-neighbor, linear, or cubic),
    - :math:`\varphi_{k,i}(\mathbf{r}_j) = \exp(-i 2\pi f_{\text{demod}} (\tau_{k,i}(\mathbf{r}_j) - 2 z_j / c))` is the demodulation phase rotation (active in ``"iq"`` mode, and :math:`1.0` in ``"rf"`` mode),
    - and the round-trip propagation delay :math:`\tau_{k,i}(\mathbf{r}_j) = \tau_{\mathrm{tx}}(x_j, z_j; \theta_k) + \tau_{\mathrm{rx}}(x_j, z_j; x_{e,i}, z_{e,i})` is defined by the steered plane-wave transmit delay and receive element spherical delay:

    .. math::

        \tau_{\mathrm{tx}}(x, z; \theta_k) = \frac{x \sin\theta_k + z \cos\theta_k}{c},
        \qquad
        \tau_{\mathrm{rx}}(x, z; x_e, z_e) = \frac{\sqrt{(x-x_e)^2 + (z-z_e)^2}}{c}.

    The interpolation kernel :math:`K(\cdot)` can be configured (via ``interp``) to be:

    - ``"linear"`` (default): the triangular kernel :math:`K(t) = \max(1 - |t|, 0)`.
    - ``"nearest"``: nearest-neighbor interpolation :math:`K(t) = \mathbf{1}_{|t| < 0.5}`.
    - ``"keys"``: Keys' 4-tap cubic convolution kernel.

    The adjoint of this operator is delay-and-sum (DAS) beamforming with the chosen
    interpolation, coherently compounded over transmit angles and receive elements.
    The class implements standard PyTorch DAS conventions of :footcite:t:`hyun2021deep`,
    including the IQ demodulation phase compensation
    :math:`\varphi = \exp(-i 2\pi f_{\text{demod}}\,(\tau - 2z/c))` on the forward
    path (and its conjugate on the adjoint) and the receive f-number rectangular
    apodization :math:`|z_p - z_e| / |x_p - x_e| > f_{\#}`.

    Signals are represented as real tensors with an ``(I, Q)`` channel pair in
    channel dimension 1:

    - image ``x`` of shape ``(B, 2, Z, X)`` on a cartesian grid,
    - measurement ``y`` of shape ``(B, 2, n_angles, n_elements, n_samples)``.

    .. note::

        The scatter step used in the forward pass calls ``scatter_add`` on a
        freshly-zeroed tensor. On CUDA this op is nondeterministic unless global
        determinism is enabled via ``torch.use_deterministic_algorithms(True)``
        and ``CUBLAS_WORKSPACE_CONFIG``. Adjointness is exact in float64 and
        agrees with DAS at the interpolation-kernel level.

    :param tuple img_size: spatial image size ``(Z, X)`` in pixels.
    :param torch.Tensor, Iterable[float], int angles: transmit steering angles in radians.
        If ``int``, angles are sampled uniformly and symmetrically in
        ``[-16 deg, 16 deg]``.
    :param torch.Tensor element_positions: receive element positions in meters,
        shape ``(n_elements, 2)`` with columns ``(x, z)``.
    :param int n_samples: number of RF time samples per channel.
    :param float sampling_frequency: RF sampling frequency :math:`f_s` in Hz.
    :param float sound_speed: assumed speed of sound in m/s. Default 1540.
    :param torch.Tensor pixel_grid: optional pixel grid of shape ``(Z, X, 2)`` in
        meters, with columns ``(x, z)``. If ``None``, a grid is built from
        ``pixel_size`` and ``pixel_origin``.
    :param tuple pixel_size: optional pixel spacing ``(dz, dx)`` in meters.
        Default is half the wavelength :math:`c / (2 f_{\text{demod}})` along both axes.
    :param tuple pixel_origin: optional grid origin ``(z0, x0)`` in meters.
        Default is ``(0, x_aperture_center)``.
    :param float, torch.Tensor t0: acquisition-start offset, scalar or per-angle
        tensor of shape ``(n_angles,)``. The sample index of a pixel is
        ``s = (τ_tx + τ_rx + t0)·fs``.
    :param float fdemod: demodulation frequency in Hz used for IQ phase
        compensation (default: 5 MHz). If 0.0, no demodulation phase rotation
        is applied (RF-mode fallback).
    :param float f_number: receive f-number defining the aperture half-width
        :math:`|x_e - x_p| \le z_p / f_\#` used for per-pixel apodization.
        ``None`` disables receive apodization entirely.
    :param str rx_apod_window: shape of the receive apodization window applied
        inside the f-number aperture. One of ``"rect"``, ``"hann"``,
        ``"hamming"``, ``"tukey0.25"``. Ignored when ``f_number`` is ``None``.
        Defaults to ``"rect"``.
    :param str tx_apod_window: optional shape of the transmit apodization window applied
        to the aperture-projected pixel position. One of ``"rect"``, ``"hann"``,
        ``"hamming"``, ``"tukey0.25"``. If ``None``, transmit apodization is disabled.
        (default: ``None``).
    :param str interp: interpolation kernel used for both scatter (forward)
        and gather (adjoint). One of:

        - ``"nearest"``: 1-tap round-to-nearest sample.
        - ``"linear"``: 2-tap linear interpolation (default).
        - ``"keys"``: 4-tap Keys cubic convolution :footcite:t:`keys1981cubic`
          with :math:`a=-1/2`, equivalent to MATLAB's ``imresize('bicubic')``.
    :param str signal_kind: signal representation. ``"iq"`` (default) treats
        images and measurements as IQ with an ``(I, Q)`` channel pair on
        dim 1 and applies IQ demodulation phase compensation. ``"rf"`` treats
        them as real single-channel RF signals (channel dim = 1); no phase
        compensation, and ``fdemod`` is ignored.
    :param torch.device, str device: device for buffers and modules.
    :param torch.dtype dtype: real dtype; ``float32`` → internal ``complex64``,
        ``float64`` → internal ``complex128``.

    |sep|

    :Examples:

        A small IQ operator on a 32x32 image with 4 elements and 3 angles:

        >>> import torch
        >>> from deepinv.physics import UltrasoundPlaneWave
        >>> torch.manual_seed(0)
        <torch._C.Generator object at ...>
        >>> ele_pos = torch.stack([torch.linspace(-1e-3, 1e-3, 4),
        ...                        torch.zeros(4)], dim=-1)
        >>> physics = UltrasoundPlaneWave(
        ...     img_size=(32, 32),
        ...     angles=3,
        ...     element_positions=ele_pos,
        ...     n_samples=256,
        ...     sampling_frequency=40e6,
        ...     fdemod=5e6,
        ... )
        >>> x = torch.randn(1, 2, 32, 32)
        >>> y = physics(x)
        >>> y.shape
        torch.Size([1, 2, 3, 4, 256])
        >>> x_das = physics.A_adjoint(y)
        >>> x_das.shape
        torch.Size([1, 2, 32, 32])

        Same operator on native RF signals (single channel):

        >>> physics_rf = UltrasoundPlaneWave(
        ...     img_size=(32, 32),
        ...     angles=3,
        ...     element_positions=ele_pos,
        ...     n_samples=256,
        ...     sampling_frequency=40e6,
        ...     signal_kind="rf",
        ... )
        >>> x_rf = torch.randn(1, 1, 32, 32)
        >>> physics_rf(x_rf).shape
        torch.Size([1, 1, 3, 4, 256])
    """

    def __init__(
        self,
        img_size: tuple[int, int],
        angles: int | Iterable[float] | Tensor,
        element_positions: Tensor,
        n_samples: int,
        sampling_frequency: float,
        sound_speed: float = 1540.0,
        pixel_grid: Tensor | None = None,
        pixel_size: tuple[float, float] | None = None,
        pixel_origin: tuple[float, float] | None = None,
        t0: float | Tensor = 0.0,
        fdemod: float = 5e6,
        f_number: float | None = None,
        rx_apod_window: str = "rect",
        tx_apod_window: str | None = None,
        interp: str = "linear",
        signal_kind: str = "iq",
        pulse: Tensor | None = None,
        normalize: bool | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        # Mirror :class:`deepinv.physics.Tomography`'s ``angles`` handling.
        if isinstance(angles, int):
            half = math.radians(16.0)
            theta = torch.linspace(-half, half, steps=angles, dtype=dtype)
        elif isinstance(angles, (list, tuple, ndarray)):
            theta = torch.tensor(angles, dtype=dtype)
        elif isinstance(angles, torch.Tensor):
            theta = angles.to(dtype=dtype)
        else:
            raise ValueError(
                f"angles must be int, iterable or Tensor, but got {type(angles)}"
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
            fdemod=fdemod,
            f_number=f_number,
            rx_apod_window=rx_apod_window,
            tx_apod_window=tx_apod_window,
            interp=interp,
            signal_kind=signal_kind,
            pulse=pulse,
            normalize=normalize,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        # Store the steering angles as ``theta`` for parity with
        # :class:`deepinv.physics.Tomography`.
        self.register_buffer("theta", theta.contiguous())
        self.to(device)
        self._finalize_normalization()

    def transmit_delay(self) -> Tensor:
        r"""Plane-wave transmit delay ``(n_angles, Z*X)`` in seconds.

        :math:`\tau_{\mathrm{tx}}(x, z; \theta_k) = (x \sin\theta_k + z \cos\theta_k)/c`.
        """
        grid = self._grid_flat()
        x = grid[:, 0]
        z = grid[:, 1]
        sin_t = torch.sin(self.theta).unsqueeze(-1)
        cos_t = torch.cos(self.theta).unsqueeze(-1)
        return (x * sin_t + z * cos_t) / self.c

    def transmit_apod(self) -> Tensor | None:
        r"""Plane-wave transmit apodization ``(n_angles, Z*X)`` or ``None``.

        Plane-wave transmit apodization: for each transmit angle :math:`\theta_k`
        each pixel is projected back onto the aperture line,
        :math:`x_{\text{proj}} = x_p - z_p \tan\theta_k`. Pixels whose
        projection falls outside the aperture (with a 20% margin) are
        masked to zero. A tapered window is applied to the normalized position
        inside the aperture, matching :attr:`tx_apod_window`.
        """
        if self.tx_apod_window is None:
            return None
        grid = self._grid_flat()
        x = grid[:, 0].unsqueeze(0)
        z = grid[:, 1].unsqueeze(0)
        tan_t = torch.tan(self.theta).unsqueeze(-1)
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

    def select_transmits(
        self, indices: Sequence[int] | Tensor
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
        return UltrasoundPlaneWave(
            img_size=self.img_size_spatial,
            angles=self.theta[idx].detach().clone(),
            element_positions=self.element_positions.detach().clone(),
            n_samples=self.n_samples,
            sampling_frequency=self.fs,
            sound_speed=self.c,
            pixel_grid=self.pixel_grid.detach().clone(),
            t0=self.t0[idx].detach().clone(),
            fdemod=self.fdemod,
            f_number=self.f_number,
            rx_apod_window=self.rx_apod_window,
            tx_apod_window=self.tx_apod_window,
            interp=self.interp,
            signal_kind=self.signal_kind,
            normalize=self.normalize,
            device=self.pixel_grid.device,
            dtype=self.dtype,
        )

    # Backward-compatible alias.
    def select_angles(self, indices: Sequence[int] | Tensor) -> "UltrasoundPlaneWave":
        """Alias for :meth:`select_transmits` — kept for readability at the call site."""
        return self.select_transmits(indices)
