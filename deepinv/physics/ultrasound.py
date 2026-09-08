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

    for the steering angle :math:`\theta_k`. Its image-domain level sets are the
    parabolas :math:`G` integrates over (focus :math:`\mathbf{r}_i`, directrix
    perpendicular to :math:`(\sin\theta_k, \cos\theta_k)`).

    The adjoint :meth:`A_adjoint` follows the same formalism with a time-reversed pulse
    :math:`\tilde{h}(t) = h(-t)` and the transpose quadratic Radon transform:

    .. math::
        \left[A^\top y\right]_j = \left[G^\top \! \left(\tilde{h} \ast_t y\right)\right]_j, \qquad
        \left[G^\top y\right]_j = \sum_{k,i,n} a_{k,i}(\mathbf{r}_j)\, K\!\big(f_s\,(t_n - \tau_{k,i}(\mathbf{r}_j))\big)\, y_{k,i,n}.

    Signals are real RF tensors: :math:`x` has shape ``(B, 1, Z, X)`` and :math:`y` has shape
    ``(B, 1, n_angles, n_elements, n_samples)``. Time interpolation is linear.

    .. warning::
        ``normalize`` has no default and must be set explicitly: normalization affects
        reconstruction dynamics, which may not be suitable for real-world applications.

    :param tuple[int, int] img_size: spatial image size ``(Z, X)`` in pixels.
    :param Iterable[float], torch.Tensor angles: transmit steering angles in radians.
    :param torch.Tensor element_positions: receive element positions in meters, shape
        ``(n_elements, 2)`` with columns ``(x, z)``.
    :param int n_samples: number of time samples per channel.
    :param float sampling_frequency: sampling frequency in Hz.
    :param float sound_speed: speed of sound :math:`c` in m/s.
    :param torch.Tensor pixel_grid: optional pixel positions in meters, shape
        ``(Z, X, 2)`` with columns ``(x, z)``. If ``None``, built from ``pixel_size`` and
        ``pixel_origin``. (default: ``None``)
    :param tuple[float, float] pixel_size: pixel spacing ``(dz, dx)`` in meters.
        (default: :math:`c / (2 f_s)` along both axes)
    :param tuple[float, float] pixel_origin: grid origin ``(z0, x0)`` in meters.
        (default: ``(0, x_aperture_center)``)
    :param float, torch.Tensor t0: acquisition-start offset :math:`t_0` in seconds,
        scalar or per-angle tensor of shape ``(n_angles,)``.
    :param float f_number: receive f-number defining the aperture half-width
        :math:`|x_i - x_j| \le z_j / f_\#` at each pixel. ``None`` disables receive
        apodization. (default: ``None``)
    :param str receive_apod_window: receive apodization window inside the f-number
        aperture, one of ``"rect"`` or ``"hann"``. Ignored if ``f_number`` is ``None``.
        (default: ``"rect"``)
    :param str transmit_apod_window: transmit apodization window, one of ``"rect"`` or
        ``"hann"``. ``None`` disables transmit apodization. (default: ``None``)
    :param torch.Tensor pulse: optional real 1D pulse-echo impulse response :math:`h`,
        normalized to unit :math:`\ell_2` norm and convolved along the time axis in both
        :meth:`A` and :meth:`A_adjoint`. (default: ``None``)
    :param bool normalize: if ``True``, :meth:`A` and :meth:`A_adjoint` are divided by
        the operator's spectral norm.
    :param torch.device, str device: device for buffers. (default: ``"cpu"``)

    All buffers are stored in :class:`torch.float32`.

    |sep|

    :Examples:

        RF operator on a 32x32 image with 4 receive elements and 3 steering angles:

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
            ...     normalize=False,
            ... )
            >>> x = torch.randn(1, 1, 32, 32)
            >>> print(physics(x).shape)
            torch.Size([1, 1, 3, 4, 256])
            >>> print(physics.A_adjoint(physics(x)).shape)
            torch.Size([1, 1, 32, 32])
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
        f_number: float | None = None,
        receive_apod_window: str = "rect",
        transmit_apod_window: str | None = None,
        pulse: Tensor | None = None,
        normalize: bool,
        device: torch.device | str = "cpu",
    ):
        theta = torch.as_tensor(angles, dtype=torch.float32).reshape(-1)
        n_transmits = theta.numel()

        ele_pos = torch.as_tensor(element_positions, dtype=torch.float32)
        if ele_pos.ndim != 2 or ele_pos.shape[1] != 2:
            raise ValueError(
                f"element_positions must have shape (n_e, 2), got {tuple(ele_pos.shape)}."
            )

        Z, X = int(img_size[0]), int(img_size[1])
        if pixel_grid is not None:
            grid = torch.as_tensor(pixel_grid, dtype=torch.float32)
        else:
            if pixel_size is None:
                lam = sound_speed / sampling_frequency
                pixel_size = (lam / 2.0, lam / 2.0)
            dz, dx = float(pixel_size[0]), float(pixel_size[1])
            if pixel_origin is None:
                x_center = 0.5 * (ele_pos[:, 0].min() + ele_pos[:, 0].max()).item()
                pixel_origin = (0.0, x_center - dx * (X - 1) / 2.0)
            z0, x0 = float(pixel_origin[0]), float(pixel_origin[1])
            z_ax = z0 + dz * torch.arange(Z, dtype=torch.float32)
            x_ax = x0 + dx * torch.arange(X, dtype=torch.float32)
            zz, xx = torch.meshgrid(z_ax, x_ax, indexing="ij")
            grid = torch.stack([xx, zz], dim=-1)

        t0_t = torch.as_tensor(t0, dtype=torch.float32)
        if t0_t.ndim == 0:
            t0_t = t0_t.expand(n_transmits).contiguous()

        if pulse is not None:
            h = torch.as_tensor(pulse, dtype=torch.float32).reshape(-1)
            h = (h / torch.linalg.norm(h)).contiguous()

        super().__init__(img_size=(1, Z, X), device=device)
        self.register_buffer("element_positions", ele_pos.contiguous())
        self.register_buffer("pixel_grid", grid.contiguous())
        self.register_buffer("t0", t0_t.contiguous())
        self.register_buffer("angles", theta.contiguous())
        self.register_buffer("pulse_echo_ir", h if pulse is not None else None)

        self.img_size_spatial = (Z, X)
        self.n_transmits = int(n_transmits)
        self.n_samples = int(n_samples)
        self.fs = float(sampling_frequency)
        self.c = float(sound_speed)
        self.f_number = None if f_number is None else float(f_number)
        self.receive_apod_window = receive_apod_window
        self.transmit_apod_window = transmit_apod_window
        self.to(device)

        self.normalize = False
        if normalize:
            gdev = self.pixel_grid.device
            x = torch.randn(
                (1, 1, Z, X),
                generator=torch.Generator(gdev).manual_seed(0),
                device=gdev,
                dtype=torch.float32,
            )
            self.register_buffer(
                "operator_norm", self.compute_norm(x, squared=False, verbose=False)
            )
            self.normalize = True

    def _receive_delays(self) -> Tensor:
        r"""Receive time-of-flight :math:`\tau_\mathrm{rx}(x, z; x_e, z_e) = \|(x, z) - (x_e, z_e)\|/c`,
        shape ``(n_elements, Z*X)``.
        """
        grid = self.pixel_grid.reshape(-1, 2)
        dx = grid[:, 0].unsqueeze(0) - self.element_positions[:, 0].unsqueeze(1)
        dz = grid[:, 1].unsqueeze(0) - self.element_positions[:, 1].unsqueeze(1)
        return torch.hypot(dx, dz) / self.c

    def _receive_apod(self) -> Tensor:
        r"""Receive apodization, shape ``(n_elements, Z*X)``."""
        Z, X = self.img_size_spatial
        n_e = self.element_positions.shape[0]
        if self.f_number is None:
            return torch.ones(
                (n_e, Z * X), dtype=torch.float32, device=self.pixel_grid.device
            )
        grid = self.pixel_grid.reshape(-1, 2)
        dx = grid[:, 0].unsqueeze(0) - self.element_positions[:, 0].unsqueeze(1)
        dz = grid[:, 1].unsqueeze(0) - self.element_positions[:, 1].unsqueeze(1)
        min_width = (
            max(
                0.5
                * torch.diff(torch.sort(self.element_positions[:, 0])[0]).mean().item(),
                1e-6,
            )
            if n_e > 1
            else 1e-3
        )
        u = dx / torch.clamp(
            dz.abs() / self.f_number, min=torch.finfo(self.pixel_grid.dtype).eps
        )
        win = (
            0.5 * (1.0 + torch.cos(math.pi * u))
            if self.receive_apod_window == "hann"
            else torch.ones_like(u)
        )
        apod = torch.where(u.abs() <= 1.0, win, torch.zeros_like(u))
        apod = torch.where(dx.abs() <= min_width, torch.ones_like(apod), apod)
        return apod.to(torch.float32)

    def _transmit_apod(self, theta_k: Tensor) -> Tensor:
        r"""Transmit apodization for a given steering angle :math:`\theta_k`, shape ``(Z*X,)``."""
        Z, X = self.img_size_spatial
        if self.transmit_apod_window is None:
            return torch.ones(Z * X, dtype=torch.float32, device=self.pixel_grid.device)
        grid = self.pixel_grid.reshape(-1, 2)
        x_min = self.element_positions[:, 0].min() * 1.2
        x_max = self.element_positions[:, 0].max() * 1.2
        u = (grid[:, 0] - grid[:, 1] * torch.tan(theta_k) - 0.5 * (x_min + x_max)) / (
            0.5 * (x_max - x_min)
        )
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

    def _interp1d(self, s: Tensor, values: Tensor, n_s: int) -> Tensor:
        r"""Linear-interpolation gather along the time axis.

        Reads ``values`` of shape ``(B, n_elements, n_samples)`` at fractional positions
        ``s`` of shape ``(n_elements, Z*X)`` and returns ``(B, n_elements, Z*X)``.
        """
        B = values.shape[0]
        idx0 = torch.floor(s).to(torch.long)
        frac = s - idx0.to(s.dtype)
        w0, w1 = (1.0 - frac).clamp(min=0.0), frac.clamp(min=0.0)
        i0 = (idx0 + 1).clamp(0, n_s + 1).unsqueeze(0).expand(B, *idx0.shape)
        i1 = (idx0 + 2).clamp(0, n_s + 1).unsqueeze(0).expand(B, *idx0.shape)
        p = pad_fn(values, (1, 1))
        return torch.gather(p, 2, i0) * w0.unsqueeze(0) + torch.gather(
            p, 2, i1
        ) * w1.unsqueeze(0)

    def _interp1d_adjoint(self, s: Tensor, values: Tensor, n_s: int) -> Tensor:
        r"""Adjoint of :meth:`_interp1d`: scatter-add along the time axis.

        Accumulates ``values`` of shape ``(B, n_elements, Z*X)`` into a
        length-``n_samples`` time axis, returning ``(B, n_elements, n_samples)``.
        """
        B = values.shape[0]
        idx0 = torch.floor(s).to(torch.long)
        frac = s - idx0.to(s.dtype)
        w0, w1 = (1.0 - frac).clamp(min=0.0), frac.clamp(min=0.0)
        i0 = (idx0 + 1).clamp(0, n_s + 1).unsqueeze(0).expand(B, *idx0.shape)
        i1 = (idx0 + 2).clamp(0, n_s + 1).unsqueeze(0).expand(B, *idx0.shape)
        padded = torch.zeros(
            (B, *s.shape[:-1], n_s + 2), dtype=values.dtype, device=values.device
        )
        padded.scatter_add_(2, i0, values * w0.unsqueeze(0))
        padded.scatter_add_(2, i1, values * w1.unsqueeze(0))
        return padded[..., 1:-1]

    def A(self, x: Tensor, **kwargs) -> Tensor:
        r"""Forward operator :math:`y = \forw{x} = \left(h \ast_t G\right)(x)`.

        :param torch.Tensor x: image of shape ``(B, 1, Z, X)``.
        :return: RF per-channel raw data of shape ``(B, 1, n_transmits, n_elements, n_samples)``,
            divided by the operator norm if ``normalize=True``.
        """
        Z, X = self.img_size_spatial
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[-2:] != (Z, X):
            raise ValueError(
                f"Expected image of shape (B, 1, {Z}, {X}), got {tuple(x.shape)}."
            )
        B, n_t, n_e, n_s = (
            x.shape[0],
            self.n_transmits,
            self.element_positions.shape[0],
            self.n_samples,
        )
        tau_rx, apod_rx = self._receive_delays(), self._receive_apod()
        grid = self.pixel_grid.reshape(-1, 2)
        gx, gz = grid[:, 0], grid[:, 1]
        x_flat = x[:, 0].reshape(B, Z * X)
        y = torch.zeros((B, n_t, n_e, n_s), dtype=torch.float32, device=x.device)
        for k in range(n_t):
            theta_k = self.angles[k]
            tau_full = (
                (gx * torch.sin(theta_k) + gz * torch.cos(theta_k)).unsqueeze(0)
                / self.c
                + tau_rx
                + self.t0[k]
            )
            weight = apod_rx * self._transmit_apod(theta_k).unsqueeze(0)
            contrib = x_flat.unsqueeze(1) * weight.unsqueeze(0)
            y[:, k] = self._interp1d_adjoint(tau_full * self.fs, contrib, n_s)
        if self.pulse_echo_ir is not None:
            y = self._apply_pulse(y.reshape(-1, 1, n_s)).reshape(B, n_t, n_e, n_s)
        out = y.unsqueeze(1)
        return out / self.operator_norm if self.normalize else out

    def A_adjoint(self, y: Tensor, **kwargs) -> Tensor:
        r"""Adjoint operator :math:`x = A^\top y = G^\top(\tilde{h} \ast_t y)`.

        :param torch.Tensor y: per-channel raw data of shape
            ``(B, 1, n_transmits, n_elements, n_samples)``.
        :return: beamformed image of shape ``(B, 1, Z, X)``, divided by the operator
            norm if ``normalize=True``.
        """
        Z, X = self.img_size_spatial
        n_t, n_e, n_s = (
            self.n_transmits,
            self.element_positions.shape[0],
            self.n_samples,
        )
        if y.ndim != 5 or y.shape[1:] != (1, n_t, n_e, n_s):
            raise ValueError(
                f"Expected measurement of shape (B, 1, {n_t}, {n_e}, {n_s}), got {tuple(y.shape)}."
            )
        B = y.shape[0]
        y_flat = y[:, 0]
        if self.pulse_echo_ir is not None:
            y_flat = self._apply_pulse(
                y_flat.reshape(-1, 1, n_s), adjoint=True
            ).reshape(B, n_t, n_e, n_s)
        tau_rx, apod_rx = self._receive_delays(), self._receive_apod()
        grid = self.pixel_grid.reshape(-1, 2)
        gx, gz = grid[:, 0], grid[:, 1]
        x_out = torch.zeros((B, Z * X), dtype=torch.float32, device=y.device)
        for k in range(n_t):
            theta_k = self.angles[k]
            tau_full = (
                (gx * torch.sin(theta_k) + gz * torch.cos(theta_k)).unsqueeze(0)
                / self.c
                + tau_rx
                + self.t0[k]
            )
            weight = apod_rx * self._transmit_apod(theta_k).unsqueeze(0)
            g = self._interp1d(
                tau_full * self.fs, y_flat[:, k], n_s
            ) * weight.unsqueeze(0)
            x_out = x_out + g.sum(dim=1)
        out = x_out.reshape(B, 1, Z, X)
        return out / self.operator_norm if self.normalize else out

    def update_parameters(
        self,
        angles=None,
        ele_pos=None,
        time_zero=None,
        fs=None,
        c=None,
        dx=None,
        dz=None,
        xlims=None,
        zlims=None,
        n_samp=None,
        fnum=None,
        **kwargs,
    ):
        if any(
            p is not None
            for p in (
                angles,
                ele_pos,
                time_zero,
                fs,
                c,
                dx,
                dz,
                xlims,
                zlims,
                n_samp,
                fnum,
            )
        ):
            raise NotImplementedError("TODO")
        return super().update_parameters(**kwargs)
