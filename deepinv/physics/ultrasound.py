from __future__ import annotations
from typing import Iterable

from numpy import ndarray
import torch
from torch import Tensor

from deepinv.physics.forward import LinearPhysics


class UltrafastUltrasound(LinearPhysics):
    r"""
    2D Pulse-echo ultrafast ultrasound imaging operator (abstract base class).

    Models the linear operator mapping an image (or a tissue reflectivity map) :math:`x` to the per-channel element raw data :math:`y`

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

    where :math:`\tilde{h}(t) = \overline{h(-t)}` is the time-reversed conjugated pulse and :math:`\overline{\varphi_{k,i}}` the conjugate modulation phase.

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
    :param float sound_speed: speed of sound in m/s.
    :param torch.Tensor pixel_grid: optional pixel positions in meters, shape `(Z, X, 2)` with columns `(x, z)`. If `None`, built from `pixel_size` and `pixel_origin`. (default: `None`)
    :param tuple[float, float] pixel_size: pixel spacing `(dz, dx)` in meters. (default: :math:`c / (2 f_s)` along both axes, i.e. half the wavelength at the sampling frequency)
    :param tuple[float, float] pixel_origin: grid origin `(z0, x0)` in meters. (default: `(0, x_aperture_center)`)
    :param float, torch.Tensor t0: acquisition-start offset :math:`t_0` in seconds, scalar or per-transmit tensor of shape `(n_transmits,)`.
    :param float demodulation_frequency: demodulation frequency in Hz, required in `"iq"` mode and ignored in `"rf"` mode. (default: `None`)
    :param float f_number: receive f-number defining the aperture half-width :math:`|x_i - x_j| \le z_j / f_\#` at each pixel. `None` disables receive apodization. (default: `None`)
    :param str receive_apod_window: receive apodization window inside the f-number aperture, one of `"rect"`, `"hann"`, `"hamming"`, `"tukey25"`. Ignored if `f_number` is `None`. (default: `"rect"`)
    :param str transmit_apod_window: transmit apodization window, one of `"rect"`, `"hann"`, `"hamming"`, `"tukey25"`. `None` disables transmit apodization. (default: `None`)
    :param str interp: interpolation kernel :math:`K`, one of `"nearest"`, `"linear"`, `"keys"`.
    :param str signal_kind: signal representation, `"iq"` or `"rf"`.
    :param torch.Tensor pulse: optional 1D pulse-echo impulse response :math:`h`, normalized to unit :math:`\ell_2` norm and convolved along the time axis in both :meth:`A` and :meth:`A_adjoint`. In `"iq"` mode it may be complex, i.e. the baseband response of a pulse that is not zero-phase or whose center frequency differs from `demodulation_frequency`; the adjoint then correlates with :math:`\overline{h}`. In `"rf"` mode it must be real. (default: `None`)
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

    def __init__(
        self,
        img_size: tuple[int, int],
        element_positions: Tensor,
        n_transmits: int,
        n_samples: int,
        sampling_frequency: float,
        sound_speed: float,
        *,
        pixel_grid: Tensor | None = None,
        pixel_size: tuple[float, float] | None = None,
        pixel_origin: tuple[float, float] | None = None,
        t0: float | Tensor,
        demodulation_frequency: float | None = None,
        f_number: float | None = None,
        receive_apod_window: str = "rect",
        transmit_apod_window: str | None = None,
        interp: str,
        signal_kind: str,
        pulse: Tensor | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        # Apodization windows, as a function of the normalized position u in [-1, 1]; the
        # tukey taper is flat over |u| <= 1 - alpha with alpha = 0.25.
        windows = {
            "rect": lambda u: torch.ones_like(u),
            "hann": lambda u: 0.5 * (1.0 + torch.cos(torch.pi * u)),
            "hamming": lambda u: 0.54 + 0.46 * torch.cos(torch.pi * u),
            "tukey25": lambda u: torch.where(
                u.abs() <= 0.75,
                torch.ones_like(u),
                0.5 * (1.0 + torch.cos(torch.pi * (u.abs() - 0.75) / 0.25)),
            ),
        }
        # Interpolation kernels K, as (index shift, tap offsets, weight of one tap at
        # signed distance t from it).
        kernels = {
            "nearest": (
                0.5,
                (0,),
                lambda t: torch.ones((), dtype=t.dtype, device=t.device),
            ),
            "linear": (0.0, (0, 1), lambda t: (1.0 - t.abs()).clamp(min=0.0)),
            "keys": (
                0.0,
                (-1, 0, 1, 2),
                lambda t: torch.where(
                    (a := t.abs()) <= 1.0,
                    ((1.5 * a - 2.5) * a) * a + 1.0,
                    torch.where(a <= 2.0, ((-0.5 * a + 2.5) * a - 4.0) * a + 2.0, 0.0),
                ),
            ),
        }

        if signal_kind == "iq" and demodulation_frequency is None:
            raise ValueError("demodulation_frequency must be provided in 'iq' mode.")
        if sound_speed <= 0.0:
            raise ValueError(
                f"sound_speed must be strictly positive, got {sound_speed}."
            )
        if interp not in kernels:
            raise ValueError(f"interp must be one of {tuple(kernels)}, got {interp!r}.")
        if signal_kind not in ("iq", "rf"):
            raise ValueError(
                f"signal_kind must be one of {('iq', 'rf')}, got {signal_kind!r}."
            )
        if receive_apod_window not in windows:
            raise ValueError(
                f"receive_apod_window must be one of {tuple(windows)}, "
                f"got {receive_apod_window!r}."
            )
        if transmit_apod_window is not None and transmit_apod_window not in windows:
            raise ValueError(
                f"transmit_apod_window must be None or one of {tuple(windows)}, "
                f"got {transmit_apod_window!r}."
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
                lam = sound_speed / sampling_frequency
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
        self.receive_apod_window = receive_apod_window
        self.transmit_apod_window = transmit_apod_window
        self.interp = interp
        self.signal_kind = signal_kind
        self._interp_kernel = kernels[interp]
        self._rx_window = windows[receive_apod_window]
        self._tx_window = (
            None if transmit_apod_window is None else windows[transmit_apod_window]
        )
        self.n_channels = n_channels
        self.dtype = dtype
        self._complex_dtype = (
            torch.complex128 if dtype == torch.float64 else torch.complex64
        )

        self.register_buffer("_tau_rx", None, persistent=False)
        self.register_buffer("_apod_rx", None, persistent=False)

        if pulse is not None:
            h = torch.as_tensor(pulse)
            if h.is_complex():
                if signal_kind != "iq":
                    raise ValueError(
                        "A complex pulse is only meaningful in 'iq' mode; in "
                        "'rf' mode the impulse response must be real."
                    )
                h = h.to(self._complex_dtype)
            else:
                h = h.to(self.dtype)
            self.register_buffer("pulse_echo_ir", h / torch.linalg.norm(h))
        else:
            self.pulse_echo_ir = None

    def transmit_delay(
        self, k: int | None = None, params: Tensor | None = None
    ) -> Tensor:
        """Transmit delay in seconds. Subclass hook.

        :param int k: if ``None`` (default), return the full ``(n_transmits, Z*X)``
            tensor; if an integer, return the ``(Z*X,)`` delay for transmit ``k``.
            :meth:`A` and :meth:`A_adjoint` call the per-transmit form to keep
            peak memory independent of ``n_transmits``. ``k`` indexes whichever
            parameter list is in use, i.e. ``params`` when it is given.
        :param torch.Tensor params: transmit parameters to use, one per transmit event.
            ``None`` (default) uses the stored ones.
        """
        raise NotImplementedError(
            "Subclass must implement transmit_delay for its transmit mode."
        )

    def transmit_apod(
        self, k: int | None = None, params: Tensor | None = None
    ) -> Tensor | None:
        """Transmit apodization. Subclass hook.

        :param int k: if ``None`` (default), return the full
            ``(n_transmits, Z*X)`` tensor (or ``None``); if an integer, return
            the ``(Z*X,)`` apodization for transmit ``k`` (or ``None``). ``k`` indexes
            whichever parameter list is in use, i.e. ``params`` when it is given.
        :param torch.Tensor params: transmit parameters to use, as in
            :meth:`transmit_delay`. ``None`` (default) uses the stored ones.
        """
        raise NotImplementedError(
            "Subclass must implement transmit_apod for its transmit mode."
        )

    def _transmit_plan(
        self,
        transmits: Iterable[int] | Tensor | slice | None = None,
        transmit_params=None,
        t0: float | Tensor | None = None,
    ) -> tuple[list[int] | range, Tensor, Tensor | None]:
        r"""Resolve the transmit events one :meth:`A` / :meth:`A_adjoint` call runs over.

        Covers the three ways a call can name its transmits, which are mutually exclusive:
        all of the stored ones (the default), a subset of them (`transmits`, as indices in
        `[0, n_transmits[)`, or parameters given
        outright (`transmit_params`, which the operator need not have been built with, with
        their own `t0`).

        :return: the transmit positions to iterate, the matching :math:`t_0` offsets, and
            the runtime parameters, or ``None`` to read the stored ones.
        """
        device = self.pixel_grid.device
        if transmit_params is not None:
            if transmits is not None:
                raise ValueError(
                    "transmits and runtime transmit parameters are mutually exclusive: "
                    "the parameters already define the transmit list. Pass the subset of "
                    "parameters you want instead."
                )
            params = torch.as_tensor(transmit_params, dtype=self.dtype, device=device)
            if params.ndim != 1 or params.shape[0] == 0:
                raise ValueError(
                    "transmit parameters must be 1-D with at least one transmit, got "
                    f"shape {tuple(params.shape)}."
                )
            n_t = params.shape[0]

            if t0 is None:
                offsets = torch.unique(self.t0)
                if offsets.numel() != 1:
                    raise ValueError(
                        "t0 differs across the stored transmits, so it cannot be reused "
                        "for runtime transmit parameters; pass t0 explicitly."
                    )
            else:
                offsets = torch.as_tensor(t0, dtype=self.dtype, device=device)
                if offsets.ndim and offsets.shape != (n_t,):
                    raise ValueError(
                        f"t0 must be scalar or shape ({n_t},) to match the given "
                        f"transmit parameters, got {tuple(offsets.shape)}."
                    )
            return range(n_t), offsets.reshape(-1).expand(n_t), params

        if t0 is not None:
            raise ValueError(
                "t0 only applies to runtime transmit parameters; it is fixed at "
                "construction otherwise."
            )
        if transmits is None:
            return range(self.n_transmits), self.t0, None

        if isinstance(transmits, slice):
            transmits = range(*transmits.indices(self.n_transmits))
        idx = torch.as_tensor(
            transmits if isinstance(transmits, Tensor) else list(transmits),
            dtype=torch.long,
        ).reshape(-1)
        if idx.numel() == 0:
            raise ValueError("transmits must select at least one transmit.")
        if int(idx.min()) < 0 or int(idx.max()) >= self.n_transmits:
            raise ValueError(
                f"transmits must index into [0, {self.n_transmits}), got "
                f"[{int(idx.min())}, {int(idx.max())}]."
            )
        return idx.tolist(), self.t0, None

    def _receive_config(self) -> tuple[Tensor, Tensor | None]:
        r"""Receive delays and apodization, both of shape ``(n_elements, Z*X)``. Cached.

        The delay is :math:`\tau_{\mathrm{rx}}(x, z; x_e, z_e) = \|(x,z) - (x_e, z_e)\|/c`. The
        apodization combines the f-number cone (aperture half-width :math:`w = z_p / f_\#` at each pixel
        depth) with a tapered window of the normalized position :math:`u = (x_e - x_p) / w \in [-1, 1]`;
        elements within ``min_width`` of the pixel in :math:`x` always get weight 1, guarding against the
        divide-by-zero at boresight. It is ``None`` when ``f_number`` is ``None``.
        """
        if self._tau_rx is not None:
            return self._tau_rx, self._apod_rx

        grid = self.pixel_grid.reshape(-1, 2)
        dx = grid[:, 0].unsqueeze(0) - self.element_positions[:, 0].unsqueeze(1)
        dz = grid[:, 1].unsqueeze(0) - self.element_positions[:, 1].unsqueeze(1)
        self._tau_rx = torch.hypot(dx, dz) / self.c

        if self.f_number is not None:
            if self.element_positions.shape[0] > 1:
                x_pos = torch.sort(self.element_positions[:, 0])[0]
                min_width = max(0.5 * torch.diff(x_pos).mean().item(), 1e-6)
            else:
                min_width = 1e-3

            w_half = torch.abs(dz) / self.f_number
            eps = torch.full_like(dx, torch.finfo(self.pixel_grid.dtype).eps)
            u = dx / torch.maximum(w_half, eps)

            apod = torch.where(
                torch.abs(u) <= 1.0,
                self._rx_window(u),
                torch.zeros_like(u),
            )
            apod = torch.where(torch.abs(dx) <= min_width, torch.ones_like(apod), apod)
            self._apod_rx = apod.to(self.pixel_grid.dtype)

        return self._tau_rx, self._apod_rx

    def _apply_pulse(self, sig: Tensor, adjoint: bool = False) -> Tensor:
        r"""Convolve along the time axis with the pulse-echo impulse response.

        Handles a real or complex :math:`h`; in the adjoint the kernel is time-reversed *and* conjugated,
        since the adjoint of a convolution by :math:`h` is a correlation with :math:`\overline{h}`. The
        real and imaginary parts are stacked on the batch axis and the components of :math:`h` on the
        output-channel axis, so a complex convolution is one :func:`torch.nn.functional.conv1d` call.

        :param torch.Tensor sig: real or complex signal of shape ``(N, 1, n_samples)``.
        :param bool adjoint: if ``True``, apply :math:`\tilde{h}(t) = \overline{h(-t)}`.
        :return: tensor of shape ``(N, 1, n_samples)``, same dtype as ``sig``.
        """
        h = self.pulse_echo_ir
        L = len(h)
        if adjoint:
            h = h.flip(-1).conj()

        n = sig.shape[0]
        if sig.is_complex():
            parts = torch.cat([sig.real, sig.imag])
        else:
            parts = sig

        if L % 2:
            padded = parts
            padding = "same"
        else:
            pad = (L // 2, L - 1 - L // 2)
            if adjoint:
                pad = pad[::-1]
            padded = torch.nn.functional.pad(parts, pad)
            padding = 0

        if h.is_complex():
            kernel = torch.stack([h.real, h.imag]).reshape(2, 1, -1)
        else:
            kernel = h.reshape(1, 1, -1)
        out = torch.nn.functional.conv1d(padded, kernel, padding=padding)

        if not sig.is_complex():
            return out
        real, imag = out[:n], out[n:]
        if not h.is_complex():
            return torch.complex(real, imag)
        return torch.complex(real[:, 0:1] - imag[:, 1:2], real[:, 1:2] + imag[:, 0:1])

    def A(
        self,
        x: Tensor,
        transmits: Iterable[int] | Tensor | slice | None = None,
        transmit_params=None,
        t0: float | Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        r"""Forward operator :math:`y = \forw{x} = \left(h \ast_t G\right) \left(x\right)`.

        :param torch.Tensor x: image of shape `(B, 2, Z, X)` in `"iq"` mode or `(B, 1, Z, X)` in `"rf"` mode.
        :param transmits: optional transmit subset, as a 1-D iterable/tensor of indices into `[0, n_transmits)` or a slice. Indices may repeat, and their order is the order of the transmit axis of the returned channel data. `None` (default) uses every transmit.
        :param transmit_params: optional transmit parameters (steering angles for :class:`UltrasoundPlaneWave`) to use in place of the stored ones, with a leading transmit dimension. Mutually exclusive with `transmits`, which indexes the stored parameters.
        :param t0: acquisition offsets paired with `transmit_params`, scalar or one per given transmit. Only valid together with `transmit_params`.
        :return: channel data of shape `(B, C, n_transmits, n_elements, n_samples)`, with the number of selected or given transmits in place of `n_transmits`, divided by the operator norm if `normalize=True`.
        """
        B = x.shape[0]
        Z, X = self.img_size_spatial
        C = self.n_channels
        if x.ndim != 4 or x.shape[1] != C or x.shape[-2:] != (Z, X):
            raise ValueError(
                f"Expected image of shape (B, {C}, {Z}, {X}), got {tuple(x.shape)}."
            )
        transmit_ids, t0_vec, params = self._transmit_plan(
            transmits, transmit_params, t0
        )
        n_t = len(transmit_ids)
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

        tau_rx, apod_rx = self._receive_config()
        z_pix = self.pixel_grid.reshape(-1, 2)[:, 1] if is_iq else None

        y_out = torch.zeros((B, n_t, n_e, n_s), dtype=out_dtype, device=x.device)
        for pos, k in enumerate(transmit_ids):
            tau_tx_k = self.transmit_delay(k=k, params=params)
            apod_tx_k = self.transmit_apod(k=k, params=params)
            tau_full = tau_tx_k.unsqueeze(0) + tau_rx + t0_vec[k]

            if is_iq:
                phase_arg = (
                    2.0
                    * torch.pi
                    * self.demodulation_frequency
                    * (tau_full - 2.0 * z_pix / self.c)
                )
                if apod_rx is not None:
                    magnitude = apod_rx
                else:
                    magnitude = torch.ones_like(phase_arg)
                weight = torch.polar(magnitude, -phase_arg)
            else:
                weight = apod_rx if apod_rx is not None else torch.ones_like(tau_full)

            if apod_tx_k is not None:
                weight = weight * apod_tx_k.unsqueeze(0)

            s_k = tau_full * self.fs
            contrib = x_flat.unsqueeze(1) * weight.unsqueeze(0)
            y_out[:, pos] = self._interp1d(s_k, contrib, n_s, adjoint=True)

        if self.pulse_echo_ir is not None:
            y_out = self._apply_pulse(y_out.reshape(-1, 1, n_s)).reshape(
                B, n_t, n_e, n_s
            )

        if is_iq:
            out = torch.view_as_real(y_out).moveaxis(-1, 1)
        else:
            out = y_out.unsqueeze(1)
        if self.normalize:
            out = out / self.operator_norm
        return out

    def A_adjoint(
        self,
        y: Tensor,
        transmits: Iterable[int] | Tensor | slice | None = None,
        transmit_params=None,
        t0: float | Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        r"""Adjoint operator :math:`x = A^\top y = G^\top \! \left(\tilde{h} \ast_t y\right)`, with :math:`\tilde{h}(t) = \overline{h(-t)}`.

        :param torch.Tensor y: channel data of shape `(B, 2, n_transmits, n_elements, n_samples)` in `"iq"` mode or `(B, 1, n_transmits, n_elements, n_samples)` in `"rf"` mode, with the number of selected or given transmits in place of `n_transmits`.
        :param transmits: optional transmit subset, as a 1-D iterable/tensor of indices into `[0, n_transmits)` or a slice. Indices may repeat, and their order is the order of the transmit axis of `y`. `None` (default) uses every transmit.
        :param transmit_params: optional transmit parameters (steering angles for :class:`UltrasoundPlaneWave`) to use in place of the stored ones, with a leading transmit dimension. Mutually exclusive with `transmits`, which indexes the stored parameters.
        :param t0: acquisition offsets paired with `transmit_params`, scalar or one per given transmit. Only valid together with `transmit_params`.
        :return: beamformed image of shape `(B, 2, Z, X)` or `(B, 1, Z, X)` respectively, divided by the operator norm if `normalize=True`.
        """
        B = y.shape[0]
        Z, X = self.img_size_spatial
        transmit_ids, t0_vec, params = self._transmit_plan(
            transmits, transmit_params, t0
        )
        n_t = len(transmit_ids)
        n_e = self.element_positions.shape[0]
        expected = (self.n_channels, n_t, n_e, self.n_samples)
        if y.ndim != 5 or y.shape[1:] != expected:
            raise ValueError(
                f"Expected measurement of shape (B, "
                f"{', '.join(str(v) for v in expected)}), got {tuple(y.shape)}."
            )
        is_iq = self.signal_kind == "iq"

        if is_iq:
            y_flat = torch.view_as_complex(y.moveaxis(1, -1).contiguous())
            out_dtype = self._complex_dtype
        else:
            y_flat = y[:, 0]
            out_dtype = self.dtype

        if self.pulse_echo_ir is not None:
            n_s = y_flat.shape[-1]
            y_flat = self._apply_pulse(
                y_flat.reshape(-1, 1, n_s), adjoint=True
            ).reshape(B, n_t, n_e, n_s)

        tau_rx, apod_rx = self._receive_config()
        z_pix = self.pixel_grid.reshape(-1, 2)[:, 1] if is_iq else None

        x_out = torch.zeros((B, Z * X), dtype=out_dtype, device=y.device)
        for pos, k in enumerate(transmit_ids):
            tau_tx_k = self.transmit_delay(k=k, params=params)
            apod_tx_k = self.transmit_apod(k=k, params=params)
            tau_full = tau_tx_k.unsqueeze(0) + tau_rx + t0_vec[k]

            if is_iq:
                phase_arg = (
                    2.0
                    * torch.pi
                    * self.demodulation_frequency
                    * (tau_full - 2.0 * z_pix / self.c)
                )
                if apod_rx is not None:
                    magnitude = apod_rx
                else:
                    magnitude = torch.ones_like(phase_arg)
                weight = torch.polar(magnitude, phase_arg)
            else:
                weight = apod_rx if apod_rx is not None else torch.ones_like(tau_full)

            if apod_tx_k is not None:
                weight = weight * apod_tx_k.unsqueeze(0)

            s_k = tau_full * self.fs
            gathered = self._interp1d(s_k, y_flat[:, pos], self.n_samples)
            x_out = x_out + (gathered * weight.unsqueeze(0)).sum(dim=1)

        if is_iq:
            out = torch.view_as_real(x_out.reshape(B, Z, X)).moveaxis(-1, 1)
        else:
            out = x_out.reshape(B, 1, Z, X)
        if self.normalize:
            out = out / self.operator_norm
        return out

    def _interp1d(
        self, s: Tensor, values: Tensor, n_samples: int, adjoint: bool = False
    ) -> Tensor:
        r"""Interpolate along the time axis at fractional positions, or its adjoint.

        :param torch.Tensor s: fractional sample positions, shape ``(n_elements, Z*X)``.
        :param torch.Tensor values: signal to read, shape ``(B, n_elements, n_samples)``, or the
            contributions to accumulate, shape ``(B, n_elements, Z*X)``, if ``adjoint``.
        :param int n_samples: length of the time axis.
        :param bool adjoint: if ``True``, scatter-add ``values`` onto the time axis instead of gathering.
        :return: shape ``(B, n_elements, Z*X)``, or ``(B, n_elements, n_samples)`` if ``adjoint``.
        """
        shift, offsets, kernel = self._interp_kernel
        base = torch.floor(s + shift)
        idx_0 = base.to(torch.long)
        frac = s - base
        weights = [kernel(frac - off) for off in offsets]

        B = values.shape[0]
        if adjoint:
            out = torch.zeros(
                (B, *s.shape[:-1], n_samples + 2),
                dtype=values.dtype,
                device=values.device,
            )
        else:
            values = torch.nn.functional.pad(values, (1, 1))
            out = None

        for off, w in zip(offsets, weights, strict=True):
            idx = idx_0 + (off + 1)
            idx_b = idx.clamp(0, n_samples + 1).unsqueeze(0).expand(B, *idx.shape)
            if adjoint:
                out = out.scatter_add(2, idx_b, values * w)
            else:
                term = torch.gather(values, 2, idx_b) * w
                out = term if out is None else out + term

        if adjoint:
            return out[..., 1:-1]
        return out


class UltrasoundPlaneWave(UltrafastUltrasound):
    r"""
    2D plane-wave (PW) ultrafast ultrasound imaging operator.

    Models the linear operator :math:`A` mapping an image (tissue reflectivity map) :math:`x` to the per-channel element raw data :math:`y` recorded by a probe insonifying the medium with plane waves

    .. math::
        y = \forw{x} = \left( h \ast_t G \right) \left( x \right)

    as a parabolic Radon transform :math:`G`, i.e. a set of projections of :math:`x` along parabolas whose shape is set by the transmit/receive configuration,
    followed by a convolution with the pulse-echo impulse response :math:`h`.

    For a steering angle :math:`\theta_k` and a receive element at :math:`\mathbf{r}_i`, the parabola has focus :math:`\mathbf{r}_i` and directrix perpendicular to :math:`(\sin\theta_k, \cos\theta_k)` (eccentricity 1), i.e. the level sets of the round-trip time-of-flight

    .. math::
        \tau_{k,i}(\mathbf{r}) = \frac{x \sin\theta_k + z \cos\theta_k}{c} + \frac{\|\mathbf{r} - \mathbf{r}_i\|}{c}.

    The adjoint :math:`A^\top = G^\top \left(\tilde{h} \ast_t \cdot\right)`, with :math:`\tilde{h}(t) = \overline{h(-t)}` the time-reversed conjugated pulse, is delay-and-sum beamforming, coherently compounded over angles and elements.
    See :class:`UltrafastUltrasound` for the in-depth definition of :math:`G` and :math:`G^\top`, including apodization, interpolation and modulation phase.

    Signals are batched real tensors whose layout depends on `signal_kind`: in `"iq"` mode :math:`x` has shape `(B, 2, Z, X)` with an `(I, Q)` channel pair on dim 1
    and :math:`y` has shape `(B, 2, n_angles, n_elements, n_samples)`; in `"rf"` mode both have a single channel.

    .. seealso::
        :class:`deepinv.physics.UltrafastUltrasound` for the base class, to implement another transmit scheme, and :class:`deepinv.physics.LinearPhysics` for the linear operator interface.

    .. warning::
        `normalize` has no default and must be set explicitly: normalization affects reconstruction dynamics, which may not be suitable for real-world applications.

    :param tuple[int, int] img_size: spatial image size `(Z, X)` in pixels.
    :param Iterable[float], torch.Tensor angles: transmit steering angles in radians.
    :param torch.Tensor element_positions: receive element positions in meters, shape `(n_elements, 2)` with columns `(x, z)`.
    :param int n_samples: number of time samples per channel.
    :param float sampling_frequency: sampling frequency in Hz.
    :param float sound_speed: speed of sound :math:`c` in m/s.
    :param torch.Tensor pixel_grid: optional pixel positions in meters, shape `(Z, X, 2)` with columns `(x, z)`. If `None`, built from `pixel_size` and `pixel_origin`. (default: `None`)
    :param tuple[float, float] pixel_size: pixel spacing `(dz, dx)` in meters. (default: :math:`c / (2 f_s)` along both axes, i.e. half the wavelength at the sampling frequency)
    :param tuple[float, float] pixel_origin: grid origin `(z0, x0)` in meters. (default: `(0, x_aperture_center)`)
    :param float, torch.Tensor t0: acquisition-start offset :math:`t_0` in seconds, scalar or per-angle tensor of shape `(n_angles,)`.
    :param float demodulation_frequency: demodulation frequency in Hz, required in `"iq"` mode and ignored in `"rf"` mode. (default: `None`)
    :param float f_number: receive f-number defining the aperture half-width :math:`|x_i - x_j| \le z_j / f_\#` at each pixel. `None` disables receive apodization. (default: `None`)
    :param str receive_apod_window: receive apodization window inside the f-number aperture, one of `"rect"`, `"hann"`, `"hamming"`, `"tukey25"`. Ignored if `f_number` is `None`. (default: `"rect"`)
    :param str transmit_apod_window: transmit apodization window, one of `"rect"`, `"hann"`, `"hamming"`, `"tukey25"`. `None` disables transmit apodization. (default: `None`)
    :param str interp: interpolation kernel :math:`K`, one of `"nearest"`, `"linear"` and `"keys"` (4-tap Keys cubic convolution).
    :param str signal_kind: signal representation, `"iq"` or `"rf"`.
    :param torch.Tensor pulse: optional 1D pulse-echo impulse response :math:`h`, normalized to unit :math:`\ell_2` norm and convolved along the time axis in both :meth:`A` and :meth:`A_adjoint`. In `"iq"` mode it may be complex, i.e. the baseband response of a pulse that is not zero-phase or whose center frequency differs from `demodulation_frequency`; the adjoint then correlates with :math:`\overline{h}`. In `"rf"` mode it must be real. (default: `None`)
    :param bool normalize: if `True`, :meth:`A` and :meth:`A_adjoint` are divided by the operator's spectral norm so it has unit norm.
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
            ...     sound_speed=1540.0,
            ...     t0=0.0,
            ...     demodulation_frequency=5e6,
            ...     interp="linear",
            ...     signal_kind="iq",
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
            ...     sound_speed=1540.0,
            ...     t0=0.0,
            ...     interp="linear",
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
        sound_speed: float,
        *,
        pixel_grid: Tensor | None = None,
        pixel_size: tuple[float, float] | None = None,
        pixel_origin: tuple[float, float] | None = None,
        t0: float | Tensor,
        demodulation_frequency: float | None = None,
        f_number: float | None = None,
        receive_apod_window: str = "rect",
        transmit_apod_window: str | None = None,
        interp: str,
        signal_kind: str,
        pulse: Tensor | None = None,
        normalize: bool,
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
            receive_apod_window=receive_apod_window,
            transmit_apod_window=transmit_apod_window,
            interp=interp,
            signal_kind=signal_kind,
            pulse=pulse,
            device=device,
            dtype=dtype,
        )
        self.register_buffer("angles", theta.contiguous())
        self.to(device)

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

    def A(
        self,
        x: Tensor,
        transmits: Iterable[int] | Tensor | slice | None = None,
        angles: Iterable[float] | Tensor | None = None,
        t0: float | Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        r"""Forward operator, optionally on a transmit subset or on given steering angles.

        Transmits the selected angles. Passing `angles` transmits steering angles the
        operator was not constructed with, and the transmit axis of the channel data then
        holds as many transmits as there are angles.

        :param torch.Tensor x: image, see :meth:`UltrafastUltrasound.A`.
        :param transmits: optional subset of the stored angles, as indices or a slice.
        :param angles: optional steering angles in radians to transmit instead of the
            stored ones, shape `(n_angles,)`. Mutually exclusive with `transmits`.
        :param t0: acquisition offsets paired with `angles`. Only valid together with them.
        :return: channel data whose transmit axis holds the selected or given angles.
        """
        return super().A(
            x, transmits=transmits, transmit_params=angles, t0=t0, **kwargs
        )

    def A_adjoint(
        self,
        y: Tensor,
        transmits: Iterable[int] | Tensor | slice | None = None,
        angles: Iterable[float] | Tensor | None = None,
        t0: float | Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        r"""Delay-and-sum, optionally on a transmit subset or on given steering angles.

        Compounds coherently over the selected transmits. Passing `angles` beamforms with
        steering angles the operator was not constructed with, which requires as many
        angles as `y` has transmits.

        :param torch.Tensor y: channel data, see :meth:`UltrafastUltrasound.A_adjoint`.
        :param transmits: optional subset of the stored angles, as indices or a slice.
        :param angles: optional steering angles in radians describing the transmits of `y`,
            shape `(n_angles,)`. Mutually exclusive with `transmits`.
        :param t0: acquisition offsets paired with `angles`. Only valid together with them.
        :return: beamformed image of shape `(B, C, Z, X)`.
        """
        return super().A_adjoint(
            y, transmits=transmits, transmit_params=angles, t0=t0, **kwargs
        )

    def select_transmits(
        self, indices: Iterable[int] | Tensor
    ) -> "UltrasoundPlaneWave":
        """Return a new operator restricted to a subset of transmit angles.

        Useful for angle-subset splitting in self-supervised losses such as
        :class:`deepinv.loss.Noise2InverseLoss`.

        .. note::
            This builds a standalone operator, which recomputes its receive delays and
            apodization (two `(n_elements, Z*X)` tensors) and, when `normalize=True`, its
            own operator norm. To beamform many transmit subsets of one acquisition, pass
            `transmits=` to :meth:`A_adjoint` instead: it restricts the transmit sum
            in place, reusing the cached receive configuration.

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
            receive_apod_window=self.receive_apod_window,
            transmit_apod_window=self.transmit_apod_window,
            interp=self.interp,
            signal_kind=self.signal_kind,
            pulse=pulse,
            normalize=self.normalize,
            device=self.pixel_grid.device,
            dtype=self.dtype,
        )

    def transmit_delay(
        self, k: int | None = None, params: Tensor | None = None
    ) -> Tensor:
        r"""Plane-wave transmit delay in seconds.

        :math:`\tau_{\mathrm{tx}}(x, z; \theta_k) = (x \sin\theta_k + z \cos\theta_k)/c`.

        :param int k: if ``None`` (default), returns ``(n_angles, Z*X)``; if an
            integer, returns ``(Z*X,)`` for angle ``k`` only. The per-transmit
            form is used by :meth:`A` / :meth:`A_adjoint` to keep memory
            independent of ``n_angles``.
        :param torch.Tensor params: steering angles in radians to use instead of the
            stored ones, shape ``(n_angles,)``. ``None`` (default) uses :attr:`angles`.
        """
        angles = self.angles if params is None else params
        grid = self.pixel_grid.reshape(-1, 2)
        x = grid[:, 0]
        z = grid[:, 1]
        if k is None:
            sin_t = torch.sin(angles).unsqueeze(-1)
            cos_t = torch.cos(angles).unsqueeze(-1)
            return (x * sin_t + z * cos_t) / self.c
        theta_k = angles[k]
        return (x * torch.sin(theta_k) + z * torch.cos(theta_k)) / self.c

    def transmit_apod(
        self, k: int | None = None, params: Tensor | None = None
    ) -> Tensor | None:
        r"""Plane-wave transmit apodization.

        For each transmit angle :math:`\theta_k`
        each pixel is projected back onto the aperture line,
        :math:`x_{\text{proj}} = x_p - z_p \tan\theta_k`. Pixels whose
        projection falls outside the aperture (with a 20% margin) are
        masked to zero. A tapered window is applied to the normalized position
        inside the aperture, matching :attr:`transmit_apod_window`.

        :param int k: if ``None`` (default), returns ``(n_angles, Z*X)`` (or
            ``None``); if an integer, returns ``(Z*X,)`` for angle ``k``
            (or ``None``).
        :param torch.Tensor params: steering angles in radians to use instead of the
            stored ones, shape ``(n_angles,)``. ``None`` (default) uses :attr:`angles`.
        """
        if self._tx_window is None:
            return None
        angles = self.angles if params is None else params
        grid = self.pixel_grid.reshape(-1, 2)
        x_min = self.element_positions[:, 0].min() * 1.2
        x_max = self.element_positions[:, 0].max() * 1.2
        x_center = 0.5 * (x_min + x_max)
        x_half = 0.5 * (x_max - x_min)
        if k is None:
            x = grid[:, 0].unsqueeze(0)
            z = grid[:, 1].unsqueeze(0)
            tan_t = torch.tan(angles).unsqueeze(-1)
        else:
            x = grid[:, 0]
            z = grid[:, 1]
            tan_t = torch.tan(angles[k])
        x_proj = x - z * tan_t
        u = (x_proj - x_center) / x_half
        inside = torch.abs(u) <= 1.0
        window = self._tx_window(u)
        apod = torch.where(inside, window, torch.zeros_like(window))
        return apod.to(self.pixel_grid.dtype)
