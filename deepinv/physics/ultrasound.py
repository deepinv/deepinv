from __future__ import annotations
from typing import Iterable, Sequence
from warnings import warn

import math

from numpy import ndarray
import torch
from torch import Tensor

from deepinv.physics.forward import LinearPhysics


class UltrafastUltrasound(LinearPhysics):
    r"""Abstract base class for pulse-echo ultrafast ultrasound imaging
    operators

    .. note::

        This class is **not meant to be instantiated directly** — it only
        defines the transmit-mode-independent core (receive delay and
        apodization, IQ demodulation phase, interpolation, and the outer
        loops) and leaves the transmit scheme abstract. End users should
        use the concrete :class:`UltrasoundPlaneWave` operator for steered
        plane-wave imaging. Subclass this class only to implement a new
        transmit scheme (focused, diverging, Hadamard-coded, ...).

    Mathematically, concrete subclasses realize a quadratic Radon transform
    :math:`A` which projects an object :math:`x` along conics whose shapes
    depend on the transmit and receive configuration of the experiment

    .. math::
        y = \forw{x}

    where, for each transmit event :math:`k` and receive element :math:`i`,
    the channel-data sample at time :math:`t_n = n / f_s` is

    .. math::

        y_{k,i,n} = \sum_{j} a_{k,i}(\mathbf{r}_j)\,
            K\!\big(f_s (t_n - \tau_{k,i}(\mathbf{r}_j))\big)\,
            \varphi_{k,i}(\mathbf{r}_j)\, x_j.

    Here :math:`\tau_{k,i} = \tau_{\mathrm{tx},k} + \tau_{\mathrm{rx},i}` is
    the round-trip delay, :math:`a_{k,i}` combines transmit and receive
    apodizations, :math:`K(\cdot)` is the interpolation kernel, and
    :math:`\varphi_{k,i} = \exp(-i 2\pi f_{\mathrm{demod}}(\tau_{k,i} - 2 z_j / c))`
    is the IQ demodulation phase (equal to :math:`1` in ``"rf"`` mode).

    A concrete subclass supplies the transmit scheme by overriding
    :meth:`transmit_delay` (returns transmit delays of shape
    ``(n_transmits, Z*X)`` in seconds) and :meth:`transmit_apod` (returns
    ``None`` or a per-pixel weight tensor of the same shape). The base
    ``__init__`` accepts the shared measurement parameters
    (``img_size``, ``element_positions``, ``n_transmits``, ``n_samples``,
    ``sampling_frequency``, ``sound_speed``, ``signal_kind``,
    ``normalize``, ...) — see :class:`UltrasoundPlaneWave` for the full
    list of shared kwargs and their defaults.
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
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        if sound_speed <= 0.0:
            raise ValueError(
                f"sound_speed must be strictly positive, got {sound_speed}.")
        if interp not in self._VALID_INTERP:
            raise ValueError(
                f"interp must be one of {self._VALID_INTERP}, got {interp!r}.")
        if signal_kind not in self._VALID_SIGNAL_KINDS:
            raise ValueError(
                f"signal_kind must be one of {self._VALID_SIGNAL_KINDS}, "
                f"got {signal_kind!r}.")
        if rx_apod_window not in self._VALID_WINDOWS:
            raise ValueError(
                f"rx_apod_window must be one of {self._VALID_WINDOWS}, "
                f"got {rx_apod_window!r}.")
        if tx_apod_window is not None and tx_apod_window not in self._VALID_WINDOWS:
            raise ValueError(
                f"tx_apod_window must be None or one of {self._VALID_WINDOWS}, "
                f"got {tx_apod_window!r}.")

        Z, X = int(img_size[0]), int(img_size[1])

        ele_pos = torch.as_tensor(element_positions, dtype=dtype)
        if ele_pos.ndim != 2 or ele_pos.shape[1] != 2:
            raise ValueError(
                "element_positions must have shape (n_elements, 2), got "
                f"{tuple(ele_pos.shape)}.")

        if pixel_grid is not None:
            grid = torch.as_tensor(pixel_grid, dtype=dtype)
            if grid.shape != (Z, X, 2):
                raise ValueError(
                    "pixel_grid must have shape (Z, X, 2), got "
                    f"{tuple(grid.shape)} for img_size=({Z}, {X}).")
        else:
            if pixel_size is None:
                ref_freq = fdemod if fdemod > 0.0 else sampling_frequency
                lam = sound_speed / ref_freq
                pixel_size = (lam / 2.0, lam / 2.0)
            dz, dx = float(pixel_size[0]), float(pixel_size[1])
            if pixel_origin is None:
                x_center = 0.5 * (ele_pos[:, 0].min() +
                                  ele_pos[:, 0].max()).item()
                pixel_origin = (0.0, x_center - dx * (X - 1) / 2.0)
            z0, x0 = float(pixel_origin[0]), float(pixel_origin[1])
            z_axis = z0 + dz * torch.arange(Z, dtype=dtype)
            x_axis = x0 + dx * torch.arange(X, dtype=dtype)
            zz, xx = torch.meshgrid(z_axis, x_axis, indexing="ij")
            grid = torch.stack([xx, zz], dim=-1)  # (Z, X, 2), columns (x, z)

        t0_tensor = torch.as_tensor(t0, dtype=dtype)
        if t0_tensor.ndim == 0:
            t0_tensor = t0_tensor.expand(n_transmits).clone()
        elif t0_tensor.shape != (n_transmits, ):
            raise ValueError(
                f"t0 must be scalar or shape ({n_transmits},), got "
                f"{tuple(t0_tensor.shape)}.")

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
        self._complex_dtype = (torch.complex128
                               if dtype == torch.float64 else torch.complex64)

        if pulse is not None:
            h = torch.as_tensor(pulse, dtype=self.dtype)
            h = h / torch.linalg.norm(h)
            self.register_buffer("pulse_echo_ir", h.contiguous())
        else:
            self.pulse_echo_ir = None

    def transmit_delay(self) -> Tensor:
        """Transmit delay ``(n_transmits, Z*X)`` in seconds. Subclass hook."""
        raise NotImplementedError(
            "Subclass must implement transmit_delay for its transmit mode.")

    def transmit_apod(self) -> Tensor | None:
        """Transmit apodization ``(n_transmits, Z*X)`` or ``None``. Subclass hook."""
        raise NotImplementedError(
            "Subclass must implement transmit_apod for its transmit mode.")

    def _receive_delay(self) -> Tensor:
        r"""Receive delay ``(n_elements, Z*X)`` in seconds.

        :math:`\tau_{\mathrm{rx}}(x, z; x_e, z_e) = \|(x,z) - (x_e, z_e)\|/c`.
        """
        grid = self.pixel_grid.reshape(-1, 2)
        dx = grid[:, 0].unsqueeze(0) - self.element_positions[:,
                                                              0].unsqueeze(1)
        dz = grid[:, 1].unsqueeze(0) - self.element_positions[:,
                                                              1].unsqueeze(1)
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
        grid = self.pixel_grid.reshape(-1, 2)

        if self.element_positions.shape[0] > 1:
            x_pos = torch.sort(self.element_positions[:, 0])[0]
            pitch = torch.diff(x_pos).mean().item()
            min_width = max(0.5 * pitch, 1e-6)
        else:
            min_width = 1e-3

        dx = grid[:, 0].unsqueeze(0) - self.element_positions[:,
                                                              0].unsqueeze(1)
        dz = grid[:, 1].unsqueeze(0) - self.element_positions[:,
                                                              1].unsqueeze(1)

        w_half = torch.abs(dz) / self.f_number
        eps = torch.full_like(dx, torch.finfo(self.pixel_grid.dtype).eps)
        u = dx / torch.maximum(w_half, eps)
        inside = torch.abs(u) <= 1.0

        window = self._window_shape(u, self.rx_apod_window)
        apod = torch.where(inside, window, torch.zeros_like(window))
        near_axis = torch.abs(dx) <= min_width
        apod = torch.where(near_axis, torch.ones_like(apod), apod)
        return apod.to(self.pixel_grid.dtype)

    def _normalize(self, normalize: bool | None) -> None:
        r"""Compute the operator norm and enable normalization of :meth:`A`
        and :meth:`A_adjoint`. Mirrors the pattern of
        :class:`deepinv.physics.Tomography`.

        Concrete subclasses call this once at the end of their ``__init__``,
        after registering all transmit-mode-specific buffers so that the
        power iteration inside :meth:`compute_norm` sees the full operator.
        """
        if normalize is None:
            warn("The default value of `normalize` is not specified and will "
                 "be automatically set to `True`. Set `normalize` explicitly "
                 "to `True` or `False` to avoid this warning.")
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
            self.register_buffer(
                "operator_norm",
                self.compute_norm(x, squared=False, verbose=False),
            )
            self.normalize = True

    def A(self, x: Tensor, **kwargs) -> Tensor:
        r"""Forward operator :math:`y = A x` (Besson kernel).

        :param torch.Tensor x: image ``(B, 2, Z, X)`` in IQ mode, or
            ``(B, 1, Z, X)`` in RF mode.
        :return: measurement ``(B, 2, n_transmits, n_elements, n_samples)`` or
            ``(B, 1, n_transmits, n_elements, n_samples)`` respectively.
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
            x_flat = torch.view_as_complex(x.moveaxis(
                1, -1).contiguous()).reshape(B, Z * X)
            out_dtype = self._complex_dtype
        else:
            x_flat = x[:, 0].reshape(B, Z * X)
            out_dtype = self.dtype

        tau_rx = self._receive_delay()  # (n_e, Z*X)
        tau_tx_all = self.transmit_delay()  # (n_t, Z*X)
        apod_rx = self._receive_apod()  # (n_e, Z*X) or None
        apod_tx_all = self.transmit_apod()  # (n_t, Z*X) or None
        z_pix = self.pixel_grid.reshape(-1, 2)[:, 1] if is_iq else None

        y_out = torch.zeros((B, n_t, n_e, n_s),
                            dtype=out_dtype,
                            device=x.device)
        for k in range(n_t):
            tau_tx_k = tau_tx_all[k]  # (Z*X,)
            tau_full = tau_tx_k.unsqueeze(0) + tau_rx + self.t0[k]

            if is_iq:
                phase_arg = 2.0 * math.pi * self.fdemod * (
                    tau_full - 2.0 * z_pix / self.c)
                weight = torch.exp(
                    torch.complex(torch.zeros_like(phase_arg), -phase_arg))
                if apod_rx is not None:
                    weight = weight * apod_rx
            else:
                weight = apod_rx if apod_rx is not None else torch.ones_like(
                    tau_full)

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
                y_real_padded = torch.nn.functional.pad(
                    y_real, (pad_left, pad_right))
                y_imag_padded = torch.nn.functional.pad(
                    y_imag, (pad_left, pad_right))
                conv_real = torch.nn.functional.conv1d(
                    y_real_padded, self.pulse_echo_ir.view(1, 1, -1))
                conv_imag = torch.nn.functional.conv1d(
                    y_imag_padded, self.pulse_echo_ir.view(1, 1, -1))
                y_out = torch.complex(conv_real,
                                      conv_imag).reshape(B, n_t, n_e, n_s)
            else:
                y_padded = torch.nn.functional.pad(y_flat,
                                                   (pad_left, pad_right))
                conv = torch.nn.functional.conv1d(
                    y_padded, self.pulse_echo_ir.view(1, 1, -1))
                y_out = conv.reshape(B, n_t, n_e, n_s)

        if is_iq:
            out = torch.view_as_real(y_out).moveaxis(-1, 1)
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
            # Adjoint padding is reversed: (pad_right, pad_left)
            y_padded = torch.nn.functional.pad(y_flat_conv,
                                               (pad_right, pad_left))
            conv = torch.nn.functional.conv1d(
                y_padded,
                self.pulse_echo_ir.view(1, 1, -1).flip(-1))
            y = conv.reshape(B, self.n_channels, n_t, n_e, n_s)

        if is_iq:
            y_flat = torch.view_as_complex(y.moveaxis(1, -1).contiguous())
            out_dtype = self._complex_dtype
        else:
            y_flat = y[:, 0]  # (B, n_t, n_e, n_s) real
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
                phase_arg = 2.0 * math.pi * self.fdemod * (
                    tau_full - 2.0 * z_pix / self.c)
                weight = torch.exp(
                    torch.complex(torch.zeros_like(phase_arg), phase_arg))
                if apod_rx is not None:
                    weight = weight * apod_rx
            else:
                weight = apod_rx if apod_rx is not None else torch.ones_like(
                    tau_full)

            if apod_tx_all is not None:
                weight = weight * apod_tx_all[k].unsqueeze(0)

            s_k = tau_full * self.fs
            gathered = self._gather_interp_1d(y_flat[:, k], s_k)
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

        The interpolation kernel :math:`K(t)` at signed distance ``t`` from
        each tap is: :math:`1` for ``nearest`` (single nearest tap);
        :math:`\max(1 - |t|, 0)` for ``linear``;
        Keys 4-tap cubic convolution with :math:`a=-1/2` for ``keys``.
        Out-of-range taps are zeroed.

        :param torch.Tensor s: fractional sample positions, shape ``S``.
        :param int n_samples: length of the time axis.
        :return: ``indices`` of shape ``(K, *S)`` (long, clamped) and
            ``weights`` of shape ``(K, *S)`` (float, OOB neighbors zeroed),
            where ``K`` is 1 for ``nearest``, 2 for ``linear``, 4 for ``keys``.
        """
        if self.interp == "nearest":
            offsets = (0, )
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
        return torch.stack(indices_list, dim=0), torch.stack(weights_list,
                                                             dim=0)

    def _scatter_interp_1d(self, s: Tensor, contrib: Tensor,
                           n_samples: int) -> Tensor:
        """Splat ``contrib`` into a length-``n_samples`` axis at fractional ``s``.

        :param torch.Tensor s: fractional sample positions, shape ``(n_e, N)``.
        :param torch.Tensor contrib: complex contributions, shape ``(B, n_e, N)``.
        :return: complex tensor of shape ``(B, n_e, n_samples)``.
        """
        B, n_e, N = contrib.shape
        indices, weights = self._interp_taps(s, n_samples)
        out = torch.zeros((B, n_e, n_samples),
                          dtype=contrib.dtype,
                          device=contrib.device)
        for k in range(indices.shape[0]):
            idx_b = indices[k].unsqueeze(0).expand(B, n_e, N)
            w_c = weights[k].to(contrib.dtype)
            out = out.scatter_add(2, idx_b, contrib * w_c)
        return out

    def _gather_interp_1d(self, y_ke: Tensor, s: Tensor) -> Tensor:
        """Performs interpolation of ``y_ke`` at fractional positions ``s``.

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
    r"""Ultrafast plane-wave (PW) ultrasound imaging operator, following the
    matrix-free parameterization of :footcite:t:`besson2018ultrafast`.

    
    Mathematically, it is described as a linear operator :math:`A` mapping a reflectivity image
    :math:`x` to transducer measurements :math:`y`

    .. math::
        y = \forw{x}

    where :math:`y` gathers, for each steered plane-wave angle :math:`\theta_k`
    and receive element :math:`i`, the samples recorded at time
    :math:`t_n = n / f_s`. The adjoint :meth:`A_adjoint` is delay-and-sum
    (DAS) beamforming coherently compounded over transmit angles and receive
    elements, following the PyTorch DAS conventions of
    :footcite:t:`hyun2021deep`. The operator is parameterized by the transmit
    and receive delays

    .. math::

        \tau_{\mathrm{tx}}(x, z; \theta_k) = \frac{x \sin\theta_k + z \cos\theta_k}{c},
        \qquad
        \tau_{\mathrm{rx}}(x, z; x_e, z_e) = \frac{\sqrt{(x-x_e)^2 + (z-z_e)^2}}{c},

    and by an interpolation kernel :math:`K` chosen via ``interp``:

    * ``"linear"``. (default)
        Triangular kernel :math:`K(t) = \max(1 - |t|, 0)`.
    * ``"nearest"``.
        Nearest-neighbor kernel :math:`K(t) = \mathbf{1}_{|t| < 0.5}`.
    * ``"keys"``.
        4-tap Keys cubic convolution :footcite:t:`keys1981cubic` with
        :math:`a = -1/2`, equivalent to MATLAB's ``imresize('bicubic')``.

    Signals are represented as batched real tensors, with layout depending
    on ``signal_kind``:

    * ``"iq"``. (default)
        Image ``x`` of shape ``(B, 2, Z, X)`` and measurement ``y`` of shape
        ``(B, 2, n_angles, n_elements, n_samples)``, with an ``(I, Q)``
        channel pair on dim 1 and IQ demodulation-phase compensation
        :math:`\varphi = \exp(-i 2\pi f_{\mathrm{demod}}(\tau - 2 z / c))`
        applied on the forward path (and its conjugate on the adjoint).
    * ``"rf"``.
        Single-channel RF signals: image ``x`` of shape ``(B, 1, Z, X)``
        and measurement ``y`` of shape
        ``(B, 1, n_angles, n_elements, n_samples)``. No phase compensation
        is applied and ``fdemod`` is ignored.

    .. note::

        The scatter step used in the forward pass calls ``scatter_add`` on a
        freshly-zeroed tensor. On CUDA this op is nondeterministic unless
        global determinism is enabled via
        ``torch.use_deterministic_algorithms(True)`` and
        ``CUBLAS_WORKSPACE_CONFIG``. Adjointness is exact in ``float64`` and
        agrees with DAS at the interpolation-kernel level.

    .. warning::

        By default, ``normalize`` is set to ``True`` if not specified.
        Initializing the operator without specifying the normalization
        behavior will issue a warning. Note that normalizing the operator
        affects the reconstruction dynamics, which may not always be
        suitable for real-world applications.

    :param tuple img_size: spatial image size ``(Z, X)`` in pixels.
    :param int, Iterable[float], torch.Tensor angles: transmit steering
        angles in radians. If ``int``, angles are sampled uniformly and
        symmetrically in ``[-16°, 16°]``.
    :param torch.Tensor element_positions: receive element positions in
        meters, shape ``(n_elements, 2)`` with columns ``(x, z)``.
    :param int n_samples: number of RF time samples per channel.
    :param float sampling_frequency: RF sampling frequency :math:`f_s` in Hz.
    :param float sound_speed: speed of sound in m/s. (default: 1540)
    :param torch.Tensor pixel_grid: optional pixel grid of shape
        ``(Z, X, 2)`` in meters, with columns ``(x, z)``. If ``None``, a
        grid is built from ``pixel_size`` and ``pixel_origin``. (default: ``None``)
    :param tuple pixel_size: optional pixel spacing ``(dz, dx)`` in meters.
        (default: half the wavelength :math:`c / (2 f_{\mathrm{demod}})` along both axes)
    :param tuple pixel_origin: optional grid origin ``(z0, x0)`` in meters.
        (default: ``(0, x_aperture_center)``)
    :param float, torch.Tensor t0: acquisition-start offset, scalar or
        per-angle tensor of shape ``(n_angles,)``. The sample index of a
        pixel is :math:`s = (\tau_{\mathrm{tx}} + \tau_{\mathrm{rx}} + t_0)\, f_s`.
        (default: ``0.0``)
    :param float fdemod: demodulation frequency in Hz used for IQ phase
        compensation. Ignored when ``signal_kind="rf"``. (default: 5e6)
    :param float f_number: receive f-number defining the aperture half-width
        :math:`|x_e - x_p| \le z_p / f_\#` used for per-pixel apodization.
        ``None`` disables receive apodization entirely. (default: ``None``)
    :param str rx_apod_window: shape of the receive apodization window
        applied inside the f-number aperture. One of ``"rect"``, ``"hann"``,
        ``"hamming"``, ``"tukey0.25"``. Ignored when ``f_number`` is
        ``None``. (default: ``"rect"``)
    :param str tx_apod_window: optional shape of the transmit apodization
        window applied to the aperture-projected pixel position. One of
        ``"rect"``, ``"hann"``, ``"hamming"``, ``"tukey0.25"``. ``None``
        disables transmit apodization. (default: ``None``)
    :param str interp: interpolation kernel used for both scatter (forward)
        and gather (adjoint). One of ``"nearest"``, ``"linear"``,
        ``"keys"``. (default: ``"linear"``)
    :param str signal_kind: signal representation. One of ``"iq"`` or
        ``"rf"``. (default: ``"iq"``)
    :param torch.Tensor pulse: optional pulse-echo impulse response applied
        by convolution along the time axis on both forward and adjoint.
        (default: ``None``)
    :param bool normalize: If ``True`` :func:`A <deepinv.physics.UltrasoundPlaneWave.A>`
        and :func:`A_adjoint <deepinv.physics.UltrasoundPlaneWave.A_adjoint>`
        are normalized so that the operator has unit norm. (default: ``True``)
    :param torch.device, str device: device for buffers and modules.
        (default: ``"cpu"``)
    :param torch.dtype dtype: real dtype; ``float32`` uses internal
        ``complex64``, ``float64`` uses internal ``complex128``.
        (default: ``torch.float32``)

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
            ...     angles=3,
            ...     element_positions=ele_pos,
            ...     n_samples=256,
            ...     sampling_frequency=40e6,
            ...     fdemod=5e6,
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
            ...     angles=3,
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
            raise ValueError(
                f"angles must be 1-D, got shape {tuple(theta.shape)}.")
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
            device=device,
            dtype=dtype,
            **kwargs,
        )
        self.register_buffer("theta", theta.contiguous())
        self.to(device)
        self._normalize(normalize)

    def transmit_delay(self) -> Tensor:
        r"""Plane-wave transmit delay ``(n_angles, Z*X)`` in seconds.

        :math:`\tau_{\mathrm{tx}}(x, z; \theta_k) = (x \sin\theta_k + z \cos\theta_k)/c`.
        """
        grid = self.pixel_grid.reshape(-1, 2)
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
        grid = self.pixel_grid.reshape(-1, 2)
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
