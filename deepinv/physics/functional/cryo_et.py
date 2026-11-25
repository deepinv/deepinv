r"""
Cryo-electron tomography image formation. Units: frequencies in :math:`\text{Å}^{-1}`, defocus in µm, spherical aberration in mm, voltage in kV, exposure in :math:`e^-/\text{Å}^2`, angles in degrees.
"""

from __future__ import annotations

import math

import torch


def electron_wavelength(voltage: float | torch.Tensor) -> float | torch.Tensor:
    r"""
    Relativistic wavelength of an electron accelerated by a given voltage.

    .. math::

        \lambda = \frac{h}{\sqrt{2 m_0 e V \left(1 + \frac{e V}{2 m_0 c^2}\right)}}

    :param float, torch.Tensor voltage: acceleration voltage in kV.
    :return: (float, torch.Tensor) wavelength in Å.

    |sep|

    :Example:

        >>> from deepinv.physics.functional import electron_wavelength
        >>> round(electron_wavelength(300.0), 5)
        0.01969
    """
    volts = voltage * 1e3
    if isinstance(volts, torch.Tensor):
        return 12.2642598 / torch.sqrt(volts * (1 + 0.978476e-6 * volts))
    return 12.2642598 / math.sqrt(volts * (1 + 0.978476e-6 * volts))


def ctf(
    kx: torch.Tensor,
    ky: torch.Tensor,
    defocus: float | torch.Tensor,
    voltage: float = 300.0,
    spherical_aberration: float = 2.7,
    amplitude_contrast: float = 0.07,
    astigmatism: float | torch.Tensor = 0.0,
    astigmatism_angle: float | torch.Tensor = 0.0,
    phase_shift: float | torch.Tensor = 0.0,
    b_factor: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    r"""
    Contrast transfer function (CTF) of a transmission electron microscope.

    Following :footcite:t:`rohou2015ctffind4`, the CTF at frequency :math:`\mathbf{k}=(k_x,k_y)` with modulus :math:`k` and azimuth :math:`\alpha` is

    .. math::

        \mathrm{CTF}(\mathbf{k}) = -\sin\left(\chi(\mathbf{k}) + \arcsin(A)\right) \exp\left(-\frac{B k^2}{4}\right),

    with the phase aberration

    .. math::

        \chi(\mathbf{k}) = \pi \lambda \Delta f(\alpha) k^2 - \frac{\pi}{2} C_s \lambda^3 k^4 - \varphi,

    where :math:`\lambda` is the :func:`electron wavelength <deepinv.physics.functional.electron_wavelength>`, :math:`C_s` the spherical aberration, :math:`A` the amplitude contrast, :math:`\varphi` a phase shift, :math:`B` an envelope B-factor and

    .. math::

        \Delta f(\alpha) = \Delta f + \frac{\Delta f_{\mathrm{ast}}}{2} \cos\left(2(\alpha - \alpha_{\mathrm{ast}})\right)

    the astigmatic defocus. Positive defocus is underfocus, for which the CTF is negative at low frequencies (positive potential appears dark).
    All parameters broadcast with `kx` and `ky`.

    :param torch.Tensor kx: horizontal spatial frequencies in :math:`\text{Å}^{-1}`.
    :param torch.Tensor ky: vertical spatial frequencies in :math:`\text{Å}^{-1}`.
    :param float, torch.Tensor defocus: mean defocus :math:`\Delta f` in µm (positive for underfocus).
    :param float voltage: acceleration voltage in kV.
    :param float spherical_aberration: spherical aberration :math:`C_s` in mm.
    :param float amplitude_contrast: amplitude contrast ratio :math:`A \in [0, 1)`.
    :param float, torch.Tensor astigmatism: defocus difference :math:`\Delta f_{\mathrm{ast}}` between the principal axes in µm.
    :param float, torch.Tensor astigmatism_angle: azimuth :math:`\alpha_{\mathrm{ast}}` of the astigmatism in degrees.
    :param float, torch.Tensor phase_shift: additional phase shift :math:`\varphi` in degrees.
    :param float, torch.Tensor b_factor: envelope B-factor :math:`B` in :math:`\text{Å}^2`.
    :return: (:class:`torch.Tensor`) real-valued CTF.

    |sep|

    :Example:

        >>> import torch
        >>> from deepinv.physics.functional import ctf
        >>> k = torch.fft.fftfreq(8, d=4.0)
        >>> ctf(k, torch.zeros_like(k), defocus=2.0).shape
        torch.Size([8])
    """
    lam = electron_wavelength(voltage)
    cs = spherical_aberration * 1e7  # mm -> Å
    k2 = kx**2 + ky**2

    defocus = _as_tensor(defocus, k2) * 1e4  # µm -> Å
    astigmatism = _as_tensor(astigmatism, k2) * 1e4
    if torch.any(astigmatism != 0):
        alpha = torch.atan2(ky, kx)
        ast_angle = torch.deg2rad(_as_tensor(astigmatism_angle, k2))
        defocus = defocus + 0.5 * astigmatism * torch.cos(2 * (alpha - ast_angle))

    phase_shift = torch.deg2rad(_as_tensor(phase_shift, k2))
    chi = (
        math.pi * lam * defocus * k2
        - 0.5 * math.pi * cs * lam**3 * k2**2
        - phase_shift
        + math.asin(amplitude_contrast)
    )
    out = -torch.sin(chi)

    b_factor = _as_tensor(b_factor, k2)
    if torch.any(b_factor != 0):
        out = out * torch.exp(-0.25 * b_factor * k2)
    return out


def critical_exposure(
    k: torch.Tensor, voltage: float = 300.0, eps: float = 1e-6
) -> torch.Tensor:
    r"""
    Critical electron exposure as a function of spatial frequency.

    Empirical radiation damage model of :footcite:t:`grant2015measuring`, calibrated at 300 kV and scaled by 0.8 at 200 kV:

    .. math::

        N_c(k) = 0.245 k^{-1.665} + 2.81.

    :param torch.Tensor k: modulus of the spatial frequency in :math:`\text{Å}^{-1}`.
    :param float voltage: acceleration voltage in kV (200 or 300).
    :param float eps: lower bound on :math:`k` to avoid division by zero.
    :return: (:class:`torch.Tensor`) critical exposure in :math:`e^-/\text{Å}^2`.
    """
    scale = 0.8 if abs(voltage - 200.0) < 1e-3 else 1.0
    return scale * (0.245 * torch.clamp(k, min=eps) ** (-1.665) + 2.81)


def dose_filter(
    k: torch.Tensor, exposure: float | torch.Tensor, voltage: float = 300.0
) -> torch.Tensor:
    r"""
    Signal attenuation caused by radiation damage.

    The amplitude at frequency :math:`k` after an accumulated exposure :math:`N` is attenuated by :footcite:p:`grant2015measuring`

    .. math::

        q(k, N) = \exp\left(-\frac{N}{2 N_c(k)}\right),

    where :math:`N_c` is the :func:`critical exposure <deepinv.physics.functional.critical_exposure>`.

    :param torch.Tensor k: modulus of the spatial frequency in :math:`\text{Å}^{-1}`.
    :param float, torch.Tensor exposure: accumulated exposure :math:`N` in :math:`e^-/\text{Å}^2`, broadcastable with `k`.
    :param float voltage: acceleration voltage in kV.
    :return: (:class:`torch.Tensor`) attenuation factors in :math:`[0, 1]`.

    |sep|

    :Example:

        >>> import torch
        >>> from deepinv.physics.functional import dose_filter
        >>> k = torch.tensor([0.0, 0.05, 0.1])
        >>> dose_filter(k, exposure=20.0)
        tensor([1.0000, 0.7725, 0.4930])
    """
    exposure = _as_tensor(exposure, k)
    return torch.exp(-exposure / (2 * critical_exposure(k, voltage)))


def tilt_acquisition_order(
    angles: torch.Tensor,
    scheme: str = "dose_symmetric",
    group_size: int = 2,
    start_angle: float = 0.0,
) -> torch.Tensor:
    r"""
    Acquisition rank of each tilt of a tilt series.

    The acquisition order determines the accumulated exposure, hence the radiation damage, of each tilt. Schemes:

    * `"dose_symmetric"` :footcite:p:`hagen2017implementation`: starts at the tilt closest to `start_angle`, then alternates between positive and negative tilts in groups of `group_size` (e.g. :math:`0, 3, 6, -3, -6, 9, 12, -9, \dots` for `group_size=2`).
    * `"bidirectional"`: from `start_angle` to the maximum tilt, then from `start_angle` to the minimum tilt.
    * `"unidirectional"`: from the minimum to the maximum tilt.

    :param torch.Tensor angles: tilt angles in degrees of shape `(N,)`, in any order.
    :param str scheme: `"dose_symmetric"`, `"bidirectional"` or `"unidirectional"`.
    :param int group_size: number of consecutive tilts on the same side for the dose-symmetric scheme.
    :param float start_angle: starting tilt angle in degrees for the dose-symmetric and bidirectional schemes.
    :return: (:class:`torch.Tensor`) integer tensor of shape `(N,)` giving the acquisition rank of each tilt (0 for the first).

    |sep|

    :Example:

        >>> import torch
        >>> from deepinv.physics.functional import tilt_acquisition_order
        >>> angles = torch.tensor([-6.0, -3.0, 0.0, 3.0, 6.0])
        >>> tilt_acquisition_order(angles, "dose_symmetric", group_size=1)
        tensor([4, 2, 0, 1, 3])
    """
    if angles.dim() != 1:
        raise ValueError(
            f"angles must be a 1D tensor, got shape {tuple(angles.shape)}."
        )
    if group_size < 1:
        raise ValueError(f"group_size must be a positive integer, got {group_size}.")

    n = angles.numel()
    sorted_idx = torch.argsort(angles.detach().cpu()).tolist()
    sorted_angles = angles.detach().cpu()[sorted_idx]
    start = int(torch.argmin((sorted_angles - start_angle).abs()))

    if scheme == "unidirectional":
        sequence = list(range(n))
    elif scheme == "bidirectional":
        sequence = list(range(start, n)) + list(range(start - 1, -1, -1))
    elif scheme == "dose_symmetric":
        sequence = [start]
        sides = [list(range(start + 1, n)), list(range(start - 1, -1, -1))]
        side = 0
        while sides[0] or sides[1]:
            if not sides[side]:
                side = 1 - side
            sequence += sides[side][:group_size]
            sides[side] = sides[side][group_size:]
            side = 1 - side
    else:
        raise ValueError(
            f"Unknown tilt scheme '{scheme}'. Choose among 'dose_symmetric', 'bidirectional' and 'unidirectional'."
        )

    rank = torch.empty(n, dtype=torch.long)
    for r, j in enumerate(sequence):
        rank[sorted_idx[j]] = r
    return rank.to(angles.device)


def _as_tensor(value, like: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.tensor(value, device=like.device, dtype=like.dtype)


def missing_wedge_mask(
    shape: tuple[int, int, int],
    angle_range: tuple[float, float] = (-60.0, 60.0),
    spherical_support: bool = True,
    low_frequency_support: float = 0.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    r"""
    Fourier-domain mask of the frequencies measured by a single-axis tilt series.

    By the Fourier slice theorem, the projection at tilt angle :math:`\theta` (rotation around :math:`y`, see :class:`deepinv.physics.CryoET`) measures the central Fourier plane spanned by :math:`k_y` and :math:`(k_x, k_z) = (\cos\theta, \sin\theta)`.
    A tilt series in :math:`[\theta_{\min}, \theta_{\max}]` measures the frequencies whose azimuth in the :math:`(k_x, k_z)` plane lies in this range; the others form the *missing wedge*.

    The mask is defined on the unshifted :func:`torch.fft.fftn` grid of volumes of shape `(D, H, W)`, with frequencies normalized to :math:`[-1, 1)`, and is symmetric so that filtered real volumes stay real.

    :param tuple[int, int, int] shape: Fourier grid shape `(D, H, W)`.
    :param tuple[float, float] angle_range: minimum and maximum tilt angles in degrees.
    :param bool spherical_support: if `True`, keeps only frequencies inside the unit ball.
    :param float low_frequency_support: radius (normalized frequency) of a ball of low frequencies considered measured.
    :param torch.device, str device: device of the mask.
    :return: (:class:`torch.Tensor`) binary mask of shape `shape`.

    |sep|

    :Example:

        >>> from deepinv.physics.functional import missing_wedge_mask
        >>> mask = missing_wedge_mask((8, 8, 8), angle_range=(-60, 60))
        >>> mask.shape
        torch.Size([8, 8, 8])
    """
    kz, ky, kx = (2 * torch.fft.fftfreq(n, device=device) for n in shape)
    kz, ky, kx = torch.meshgrid(kz, ky, kx, indexing="ij")
    phi = torch.rad2deg(torch.atan2(kz, kx))
    phi = torch.where(phi > 90, phi - 180, phi)
    phi = torch.where(phi <= -90, phi + 180, phi)
    mask = (phi >= angle_range[0]) & (phi <= angle_range[1])
    mask |= (kz == 0) & (kx == 0)
    radius = torch.sqrt(kx**2 + ky**2 + kz**2)
    if spherical_support:
        mask &= radius <= 1
    if low_frequency_support > 0:
        mask |= radius < low_frequency_support
    # symmetrize k -> -k, which only matters at Nyquist frequencies
    reflected = torch.roll(
        torch.flip(mask, dims=(0, 1, 2)), shifts=(1, 1, 1), dims=(0, 1, 2)
    )
    return (mask | reflected).float()


def apply_fourier_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    r"""
    Filters volumes with a real Fourier mask, possibly defined on a zero-padded grid to avoid wrap-around.

    Volumes are zero-padded at the end of each dimension to the mask size, filtered in Fourier and cropped back.

    :param torch.Tensor x: volumes of shape `(..., D, H, W)`.
    :param torch.Tensor mask: mask on the unshifted :func:`torch.fft.fftn` grid of shape `(..., pD, pH, pW)`, with `pD >= D`, `pH >= H`, `pW >= W`, broadcastable with `x`.
    :return: (:class:`torch.Tensor`) filtered volumes of the same shape as `x`.
    """
    D, H, W = x.shape[-3:]
    spectrum = torch.fft.fftn(x, s=mask.shape[-3:], dim=(-3, -2, -1))
    out = torch.fft.ifftn(spectrum * mask, dim=(-3, -2, -1)).real
    return out[..., :D, :H, :W]
