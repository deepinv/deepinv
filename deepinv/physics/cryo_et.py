from __future__ import annotations

from collections.abc import Iterable
import math

import torch
import torch.nn.functional as F

from deepinv.physics.forward import LinearPhysics, adjoint_function
from deepinv.physics.functional.cryo_et import (
    ctf as ctf_fn,
    dose_filter,
    tilt_acquisition_order,
    missing_wedge_mask,
    apply_fourier_mask,
)


class CryoET(LinearPhysics):
    r"""
    Cryo-electron tomography (cryo-ET) forward operator.

    Maps a 3D volume :math:`x` (Coulomb potential of the specimen) to a tilt series of :math:`N` 2D projections acquired by rotating the specimen around a single tilt axis.
    Under the weak-phase approximation, the :math:`i`-th tilt image is

    .. math::

        y_i = \frac{s_i}{\|A\|} \left( b + \sum_{k=1}^{K} h_{i,k} * \mathcal{W}^{2D}_i \Pi_k R_{\theta_i} \mathcal{W}^{3D}_i x \right)

    where:

    * :math:`R_{\theta_i}` rotates the volume by the tilt angle :math:`\theta_i` around the :math:`y`-axis (`H` dimension) and :math:`\Pi_k` integrates along the beam over the :math:`k`-th of :math:`K` = `n_defocus_planes` depth slabs.
    * :math:`\hat h_{i,k}(\mathbf{k}) = \mathrm{CTF}(\mathbf{k}; \Delta f_i + z_k) \, q(\mathbf{k}, N_i)` is the transfer function of slab :math:`k` at height :math:`z_k`, with :func:`CTF <deepinv.physics.functional.ctf>` and :func:`dose filter <deepinv.physics.functional.dose_filter>` :math:`q` for the accumulated exposure :math:`N_i` (see :func:`deepinv.physics.functional.tilt_acquisition_order`). :math:`K > 1` models the defocus gradient across the tilted specimen and through its thickness.
    * :math:`s_i = \frac{D_i}{\bar{D}} \exp\left(-\frac{t}{\lambda_{\mathrm{in}} |\cos \theta_i|}\right)` accounts for the relative exposure :math:`D_i / \bar D` of the tilt and for inelastic scattering in an ice layer of thickness :math:`t`.
    * :math:`b` is the normalized background of the unscattered beam, only added by :meth:`forward` (or `A(x, add_background=True)`), so that :meth:`A` stays linear.
    * :math:`\mathcal{W}^{3D}_i` and :math:`\mathcal{W}^{2D}_i` are optional per-tilt deformations of the volume (beam-induced motion) and of the projections (tilt-axis orientation, misalignment), each made of an affine transform (shift, rotation, shear, scaling) and a local displacement field.

    All transforms are combined into a single trilinear resampling per tilt. The adjoint is computed exactly with automatic differentiation, and the operator is differentiable with respect to its floating-point parameters (e.g. for joint alignment and reconstruction).

    |sep|

    :Conventions:

    Volumes have shape `(B, C, D, H, W)`, with `D` the beam direction at zero tilt (:math:`z`), `H` the tilt axis (:math:`y`) and `W` the perpendicular direction (:math:`x`). Measurements have shape `(B, C, N, H_d, W_d)`.
    Vector parameters follow the tensor dimensions, i.e. `(z, y, x)` in 3D and `(y, x)` in 2D, in voxels/pixels. A point at height :math:`z` (in Å) above the tilt axis has defocus :math:`\Delta f_i + z`.

    Per-tilt parameters can be scalars (or vectors) shared by all tilts, tensors with leading dimension `N`, or tensors with leading dimensions `(B, N)`. They are buffers updated with :meth:`update_parameters` or passed as keyword arguments to :meth:`forward`, :meth:`A` and :meth:`A_adjoint`.

    For realistic noise, use :class:`deepinv.physics.PoissonNoise` or :class:`deepinv.physics.PoissonGaussianNoise` with gain `1 / physics.electrons_per_pixel` and `background=1`, so that measurements are normalized electron counts.
    Remove the background with :meth:`remove_background` before calling :meth:`A_dagger`. For normalized tilt series, :class:`deepinv.physics.GaussianNoise` with `background=0` is a common simplification.

    .. note::

        The CTF is applied in the Fourier domain with periodic boundary conditions; use `fft_padding` to reduce wrap-around effects. The memory of the backward pass grows with the number of tilts and the volume size.

    :param tuple[int, ...] img_size: volume size `(D, H, W)` or `(C, D, H, W)`.
    :param int, torch.Tensor, Iterable[float] angles: number of tilts uniformly spaced in `angle_range`, or tilt angles in degrees of shape `(N,)` or `(B, N)`.
    :param tuple[float, float] angle_range: minimum and maximum tilt angles in degrees, used when `angles` is an `int`.
    :param tuple[int, int] detector_size: projection size `(H_d, W_d)`. Default: `(H, W)`.
    :param float pixel_size: voxel and detector pixel size in Å.
    :param float, torch.Tensor tilt_axis_angle: in-plane rotation of the tilt axis on the detector in degrees, counter-clockwise from the `H` axis.
    :param bool ctf: if `True`, modulates the projections by the CTF.
    :param float, torch.Tensor defocus: defocus at the tilt axis height in µm (positive for underfocus).
    :param float, torch.Tensor astigmatism: defocus difference between the principal axes of astigmatism in µm.
    :param float, torch.Tensor astigmatism_angle: azimuth of the astigmatism in degrees.
    :param float voltage: acceleration voltage in kV.
    :param float spherical_aberration: spherical aberration in mm.
    :param float amplitude_contrast: amplitude contrast ratio.
    :param float, torch.Tensor phase_shift: additional phase shift in degrees (e.g. Volta phase plate).
    :param float, torch.Tensor b_factor: B-factor of the CTF envelope in Å².
    :param int n_defocus_planes: number :math:`K` of depth slabs with distinct defocus. `1` is the thin-specimen approximation.
    :param float, torch.Tensor dose_per_tilt: electron exposure of each tilt in :math:`e^-/\text{Å}^2`.
    :param str, torch.Tensor tilt_scheme: acquisition scheme `"dose_symmetric"`, `"bidirectional"` or `"unidirectional"`, or integer tensor of shape `(N,)` giving the acquisition rank of each tilt.
    :param int dose_symmetric_group: group size of the dose-symmetric scheme.
    :param float start_angle: starting angle of the dose-symmetric and bidirectional schemes in degrees.
    :param bool dose_weighting: if `True`, applies the dose filter for the accumulated exposure of each tilt.
    :param float, torch.Tensor, None ice_thickness: ice thickness :math:`t` in Å, `0` to disable attenuation. Default: `D * pixel_size`.
    :param float inelastic_mean_free_path: inelastic mean free path :math:`\lambda_{\mathrm{in}}` in Å.
    :param float, torch.Tensor background: normalized background :math:`b`, `1` for electron counts, `0` for background-free measurements.
    :param torch.Tensor volume_shift: volume translation `(dz, dy, dx)` in voxels.
    :param torch.Tensor volume_rotation: volume rotation angles `(a_z, a_y, a_x)` in degrees, composed as :math:`R_z(a_z) R_y(a_y) R_x(a_x)`.
    :param torch.Tensor volume_shear: volume shear `(g_xy, g_xz, g_yz)`, i.e. :math:`x \leftarrow x + g_{xy} y + g_{xz} z` and :math:`y \leftarrow y + g_{yz} z`.
    :param torch.Tensor volume_scale: volume scaling `(s_z, s_y, s_x)`.
    :param torch.Tensor volume_displacement: local displacement field `(dz, dy, dx)` in voxels of shape `(3, d, h, w)` (optionally with leading `N` or `(B, N)`), on a coarse grid interpolated trilinearly. Pull convention: the deformed volume at :math:`p` is the volume at :math:`p + u(p)`.
    :param torch.Tensor projection_shift: projection translation `(dy, dx)` in pixels.
    :param torch.Tensor projection_rotation: in-plane projection rotation in degrees, on top of `tilt_axis_angle`.
    :param torch.Tensor projection_shear: projection shear :math:`g`, i.e. :math:`x \leftarrow x + g y`.
    :param torch.Tensor projection_scale: projection magnification.
    :param torch.Tensor projection_displacement: local displacement field `(dy, dx)` in pixels of shape `(2, h, w)` (optionally with leading `N` or `(B, N)`), pull convention.
    :param float depth_step: sampling step along the rays in voxels.
    :param int depth_padding: extra samples added at both ends of each ray in voxels.
    :param int fft_padding: zero-padding of the projections on each side before applying the CTF.
    :param bool normalize: if `True`, normalizes the operator to unit spectral norm.
    :param torch.device, str device: device of the operator.
    :param torch.dtype dtype: floating-point type of the parameters.

    |sep|

    :Example:

        Simulate a noisy tilt series with a dose-symmetric scheme:

        >>> import torch
        >>> import deepinv as dinv
        >>> x = torch.rand(1, 1, 8, 16, 16) * 0.1
        >>> physics = dinv.physics.CryoET(img_size=(8, 16, 16), angles=11, angle_range=(-60, 60), background=1.0)
        >>> physics.noise_model = dinv.physics.PoissonNoise(gain=1 / physics.electrons_per_pixel, clip_positive=True)
        >>> y = physics(x)
        >>> y.shape
        torch.Size([1, 1, 11, 16, 16])

        Add per-tilt misalignment and a drift of the volume during acquisition:

        >>> shifts = torch.randn(11, 2)
        >>> drift = physics.acquisition_order.float()[:, None] * torch.tensor([0.0, 0.1, 0.2])
        >>> y = physics.A(x, projection_shift=shifts, volume_shift=drift)
        >>> y.shape
        torch.Size([1, 1, 11, 16, 16])
    """

    def __init__(
        self,
        img_size: tuple[int, ...],
        angles: int | torch.Tensor | Iterable[float] = 41,
        angle_range: tuple[float, float] = (-60.0, 60.0),
        detector_size: tuple[int, int] | None = None,
        pixel_size: float = 10.0,
        tilt_axis_angle: float | torch.Tensor = 0.0,
        ctf: bool = True,
        defocus: float | torch.Tensor = 3.0,
        astigmatism: float | torch.Tensor = 0.0,
        astigmatism_angle: float | torch.Tensor = 0.0,
        voltage: float = 300.0,
        spherical_aberration: float = 2.7,
        amplitude_contrast: float = 0.07,
        phase_shift: float | torch.Tensor = 0.0,
        b_factor: float | torch.Tensor = 0.0,
        n_defocus_planes: int = 1,
        dose_per_tilt: float | torch.Tensor = 3.0,
        tilt_scheme: str | torch.Tensor = "dose_symmetric",
        dose_symmetric_group: int = 2,
        start_angle: float = 0.0,
        dose_weighting: bool = True,
        ice_thickness: float | torch.Tensor | None = None,
        inelastic_mean_free_path: float = 3500.0,
        background: float | torch.Tensor = 0.0,
        volume_shift: torch.Tensor | None = None,
        volume_rotation: torch.Tensor | None = None,
        volume_shear: torch.Tensor | None = None,
        volume_scale: torch.Tensor | None = None,
        volume_displacement: torch.Tensor | None = None,
        projection_shift: torch.Tensor | None = None,
        projection_rotation: torch.Tensor | None = None,
        projection_shear: torch.Tensor | None = None,
        projection_scale: torch.Tensor | None = None,
        projection_displacement: torch.Tensor | None = None,
        depth_step: float = 1.0,
        depth_padding: int = 2,
        fft_padding: int = 0,
        normalize: bool = False,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        if len(img_size) not in (3, 4):
            raise ValueError(
                f"img_size must be (D, H, W) or (C, D, H, W), got {tuple(img_size)}."
            )
        super().__init__(img_size=tuple(img_size), device=device, **kwargs)

        self.volume_size = tuple(int(s) for s in img_size[-3:])
        D, H, W = self.volume_size
        self.detector_size = (
            (H, W) if detector_size is None else tuple(int(s) for s in detector_size)
        )
        if n_defocus_planes < 1:
            raise ValueError("n_defocus_planes must be a positive integer.")
        if depth_step <= 0:
            raise ValueError("depth_step must be positive.")

        self.pixel_size = float(pixel_size)
        self.ctf = ctf
        self.voltage = float(voltage)
        self.spherical_aberration = float(spherical_aberration)
        self.amplitude_contrast = float(amplitude_contrast)
        self.n_defocus_planes = int(n_defocus_planes)
        self.dose_weighting = dose_weighting
        self.inelastic_mean_free_path = float(inelastic_mean_free_path)
        self.depth_step = float(depth_step)
        self.depth_padding = int(depth_padding)
        self.fft_padding = int(fft_padding)

        tensor = lambda v: torch.as_tensor(v, dtype=dtype, device=device)

        if isinstance(angles, int):
            angles = torch.linspace(
                angle_range[0], angle_range[1], angles, dtype=dtype, device=device
            )
        elif not isinstance(angles, torch.Tensor):
            angles = tensor(list(angles))
        angles = angles.to(device=device, dtype=dtype)
        if angles.dim() not in (1, 2):
            raise ValueError("angles must have shape (N,) or (B, N).")
        self.register_buffer("angles", angles)
        self.register_buffer("tilt_axis_angle", tensor(tilt_axis_angle))

        self.register_buffer("defocus", tensor(defocus))
        self.register_buffer("astigmatism", tensor(astigmatism))
        self.register_buffer("astigmatism_angle", tensor(astigmatism_angle))
        self.register_buffer("phase_shift", tensor(phase_shift))
        self.register_buffer("b_factor", tensor(b_factor))

        self.register_buffer("dose_per_tilt", tensor(dose_per_tilt))
        if isinstance(tilt_scheme, str):
            order = tilt_acquisition_order(
                angles if angles.dim() == 1 else angles[0],
                scheme=tilt_scheme,
                group_size=dose_symmetric_group,
                start_angle=start_angle,
            )
        else:
            order = torch.as_tensor(tilt_scheme, dtype=torch.long)
            if order.shape != (self.n_tilts,):
                raise ValueError(
                    f"tilt_scheme tensor must have shape ({self.n_tilts},), got {tuple(order.shape)}."
                )
        self.register_buffer("acquisition_order", order.to(device))

        if ice_thickness is None:
            ice_thickness = D * self.pixel_size
        self.register_buffer("ice_thickness", tensor(ice_thickness))
        self.register_buffer("background", tensor(background))

        def _default(value, default):
            return tensor(default) if value is None else tensor(value)

        N = self.n_tilts
        self.register_buffer("volume_shift", _default(volume_shift, [0.0] * 3))
        self.register_buffer("volume_rotation", _default(volume_rotation, [0.0] * 3))
        self.register_buffer("volume_shear", _default(volume_shear, [0.0] * 3))
        self.register_buffer("volume_scale", _default(volume_scale, [1.0] * 3))
        self.register_buffer(
            "volume_displacement",
            None if volume_displacement is None else tensor(volume_displacement),
        )
        self.register_buffer(
            "projection_shift", _default(projection_shift, torch.zeros(N, 2))
        )
        self.register_buffer(
            "projection_rotation", _default(projection_rotation, torch.zeros(N))
        )
        self.register_buffer(
            "projection_shear", _default(projection_shear, torch.zeros(N))
        )
        self.register_buffer(
            "projection_scale", _default(projection_scale, torch.ones(N))
        )
        self.register_buffer(
            "projection_displacement",
            (
                None
                if projection_displacement is None
                else tensor(projection_displacement)
            ),
        )

        self.register_buffer("operator_norm", tensor(1.0))
        self.to(device)

        if normalize:
            x0 = torch.randn(
                (1, 1, *self.volume_size),
                generator=torch.Generator(device).manual_seed(0),
                device=device,
                dtype=dtype,
            )
            norm = self.compute_norm(x0, squared=False, verbose=False)
            self.operator_norm = norm.to(dtype).reshape(())

    @property
    def n_tilts(self) -> int:
        r"""Number of tilts :math:`N` of the tilt series."""
        return self.angles.shape[-1]

    @property
    def electrons_per_pixel(self) -> torch.Tensor:
        r"""
        Mean number of electrons per detector pixel and per tilt. Use `1 / physics.electrons_per_pixel` as gain of :class:`deepinv.physics.PoissonNoise` when `background=1`.
        """
        return self._per_tilt(self.dose_per_tilt).mean(dim=-1).squeeze(0) * (
            self.pixel_size**2
        )

    def accumulated_exposure(self) -> torch.Tensor:
        r"""
        Accumulated electron exposure of each tilt in :math:`e^-/\text{Å}^2`, at the middle of the tilt acquisition.

        :return: (:class:`torch.Tensor`) exposures of shape `(Bp, N)`, with `Bp` 1 or the parameter batch size.
        """
        dose = self._per_tilt(self.dose_per_tilt)
        rank = self.acquisition_order.to(dose.device)
        before = (rank[None, :] < rank[:, None]).to(dose.dtype)
        return dose @ before.T + 0.5 * dose

    def tilt_weights(self) -> torch.Tensor:
        r"""
        Per-tilt intensity factors :math:`s_i` (relative exposure and ice attenuation).

        :return: (:class:`torch.Tensor`) factors of shape `(Bp, N)`, with `Bp` 1 or the parameter batch size.
        """
        dose = self._per_tilt(self.dose_per_tilt)
        weights = dose / dose.mean(dim=-1, keepdim=True)
        thickness = self._per_tilt(self.ice_thickness)
        if math.isfinite(self.inelastic_mean_free_path):
            cos = torch.cos(torch.deg2rad(self._per_tilt(self.angles))).abs()
            weights = weights * torch.exp(
                -thickness / (self.inelastic_mean_free_path * cos.clamp(min=1e-3))
            )
        return weights

    def remove_background(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Removes the background :math:`s_i b / \|A\|` added by :meth:`forward`.

        :param torch.Tensor y: measurements of shape `(B, C, N, H_d, W_d)`.
        :return: (:class:`torch.Tensor`) background-free measurements.
        """
        self.update_parameters(**kwargs)
        return y - self._background_term()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Simulates noisy measurements :math:`y = \eta(N(Ax + b))`, including the background :math:`b`.

        :param torch.Tensor x: volume of shape `(B, C, D, H, W)`.
        :return: (:class:`torch.Tensor`) tilt series of shape `(B, C, N, H_d, W_d)`.
        """
        return self.sensor(
            self.noise(self.A(x, add_background=True, **kwargs), **kwargs)
        )

    def A(
        self, x: torch.Tensor, add_background: bool = False, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the noiseless tilt series :math:`Ax`.

        :param torch.Tensor x: volume of shape `(B, C, D, H, W)`.
        :param bool add_background: if `True`, adds the background :math:`s_i b / \|A\|`.
        :param dict kwargs: operator parameters to update, e.g. `defocus`.
        :return: (:class:`torch.Tensor`) tilt series of shape `(B, C, N, H_d, W_d)`.
        """
        self.update_parameters(**kwargs)
        y = self._forward(x, transfer=True, weights=True)
        if add_background:
            y = y + self._background_term()
        return y

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Computes the exact adjoint :math:`A^{\top} y` with automatic differentiation.

        :param torch.Tensor y: tilt series of shape `(B, C, N, H_d, W_d)`.
        :param dict kwargs: operator parameters to update.
        :return: (:class:`torch.Tensor`) volume of shape `(B, C, D, H, W)`.
        """
        self.update_parameters(**kwargs)
        return self._adjoint(y, transfer=True, weights=True)

    def fbp(self, y: torch.Tensor, phase_flip: bool = True, **kwargs) -> torch.Tensor:
        r"""
        Weighted back-projection (WBP), the standard analytical reconstruction in cryo-ET.

        Tilts are optionally phase-flipped with the sign of the CTF, divided by :math:`s_i`, ramp-filtered perpendicular to the tilt axis, weighted by the angular spacing and back-projected.
        CTF amplitudes, dose filter and missing wedge are not compensated. Remove the background first with :meth:`remove_background`.

        :param torch.Tensor y: tilt series of shape `(B, C, N, H_d, W_d)`.
        :param bool phase_flip: if `True` and the CTF is enabled, applies phase flipping.
        :param dict kwargs: operator parameters to update.
        :return: (:class:`torch.Tensor`) volume of shape `(B, C, D, H, W)`.
        """
        self.update_parameters(**kwargs)
        Hd, Wd = self.detector_size
        Hp, Wp = 2 * Hd, 2 * Wd

        Y = torch.fft.rfft2(y, s=(Hp, Wp))
        ky = torch.fft.fftfreq(Hp, device=y.device, dtype=y.dtype)
        kx = torch.fft.rfftfreq(Wp, device=y.device, dtype=y.dtype)
        KY, KX = torch.meshgrid(ky, kx, indexing="ij")
        phi = torch.deg2rad(self.tilt_axis_angle).to(y.dtype)
        ramp = (KX * torch.cos(phi) + KY * torch.sin(phi)).abs()
        filt = ramp.expand(1, self.n_tilts, Hp, Wp // 2 + 1)

        if phase_flip and self.ctf:
            kx_a = KX / self.pixel_size
            ky_a = KY / self.pixel_size
            c = ctf_fn(
                kx_a,
                ky_a,
                defocus=self._per_tilt(self.defocus)[..., None, None],
                voltage=self.voltage,
                spherical_aberration=self.spherical_aberration,
                amplitude_contrast=self.amplitude_contrast,
                astigmatism=self._per_tilt(self.astigmatism)[..., None, None],
                astigmatism_angle=self._per_tilt(self.astigmatism_angle)[
                    ..., None, None
                ],
                phase_shift=self._per_tilt(self.phase_shift)[..., None, None],
            )
            filt = filt * torch.sign(c)

        dtheta = self._angular_weights()
        weights = dtheta / self.tilt_weights() * self.operator_norm
        filt = filt * weights[..., None, None]
        y_f = torch.fft.irfft2(Y * filt[:, None], s=(Hp, Wp))[..., :Hd, :Wd]
        return self._adjoint(y_f, transfer=False, weights=False)

    def A_dagger(self, y: torch.Tensor, fbp: bool = False, **kwargs) -> torch.Tensor:
        r"""
        Least squares solution of :math:`y = Ax`, or its fast analytical approximation :meth:`fbp` if `fbp=True`.

        :param torch.Tensor y: background-free tilt series of shape `(B, C, N, H_d, W_d)`.
        :param bool fbp: if `True`, uses the weighted back-projection.
        :param dict kwargs: operator parameters to update.
        :return: (:class:`torch.Tensor`) volume of shape `(B, C, D, H, W)`.
        """
        if fbp:
            return self.fbp(y, **kwargs)
        return super().A_dagger(y, **kwargs)

    def _per_tilt(self, value: torch.Tensor, n_trailing: int = 0) -> torch.Tensor:
        r"""Broadcasts a parameter to shape `(Bp, N, *trailing)`, with `Bp` 1 or the parameter batch size."""
        N = self.n_tilts
        if value.dim() == n_trailing:
            value = value.expand(N, *value.shape)
        if value.dim() == n_trailing + 1:
            value = value.unsqueeze(0)
        if value.dim() != n_trailing + 2 or value.shape[1] != N:
            raise ValueError(
                f"Parameter of shape {tuple(value.shape)} cannot be broadcast to {N} tilts."
            )
        return value

    def _background_term(self) -> torch.Tensor:
        b = self._per_tilt(self.background) * self.tilt_weights() / self.operator_norm
        return b[:, None, :, None, None]

    def _check_input(self, x: torch.Tensor):
        if x.dim() != 5 or tuple(x.shape[-3:]) != self.volume_size:
            raise ValueError(
                f"Input volume must have shape (B, C, {', '.join(map(str, self.volume_size))}), got {tuple(x.shape)}."
            )

    def _forward(self, x: torch.Tensor, transfer: bool, weights: bool) -> torch.Tensor:
        self._check_input(x)
        out = []
        for i in range(self.n_tilts):
            planes, centers = self._project_tilt(x, i)
            if transfer:
                out.append(self._apply_transfer(planes, centers, i))
            else:
                out.append(planes.sum(dim=2))
        y = torch.stack(out, dim=2)
        if weights:
            y = y * self.tilt_weights()[:, None, :, None, None]
            y = y / self.operator_norm
        return y

    def _adjoint(self, y: torch.Tensor, transfer: bool, weights: bool) -> torch.Tensor:
        x_shape = (y.shape[0], y.shape[1], *self.volume_size)
        adjoint = adjoint_function(
            lambda x: self._forward(x, transfer=transfer, weights=weights),
            x_shape,
            device=y.device,
            dtype=y.dtype,
        )
        return adjoint(y)

    def _angular_weights(self) -> torch.Tensor:
        r"""Angular spacing of each tilt in radians, of shape `(Bp, N)`."""
        angles = torch.deg2rad(self._per_tilt(self.angles))
        if self.n_tilts == 1:
            return torch.full_like(angles, math.pi)
        sorted_angles, idx = torch.sort(angles, dim=-1)
        diff = sorted_angles.diff(dim=-1)
        w = torch.zeros_like(sorted_angles)
        w[..., 1:] += 0.5 * diff
        w[..., :-1] += 0.5 * diff
        w[..., 0] += 0.5 * diff[..., 0]
        w[..., -1] += 0.5 * diff[..., -1]
        return torch.empty_like(w).scatter_(-1, idx, w)

    @staticmethod
    def _rot2(a):
        c, s = torch.cos(a), torch.sin(a)
        return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)

    @staticmethod
    def _rot3(a, axis):
        c, s = torch.cos(a), torch.sin(a)
        o, z = torch.ones_like(a), torch.zeros_like(a)
        if axis == "x":
            rows = [[o, z, z], [z, c, -s], [z, s, c]]
        elif axis == "y":
            rows = [[c, z, s], [z, o, z], [-s, z, c]]
        else:
            rows = [[c, -s, z], [s, c, z], [z, z, o]]
        return torch.stack([torch.stack(r, -1) for r in rows], -2)

    def _project_tilt(self, x: torch.Tensor, i: int):
        r"""
        Integrates the deformed volume along the rays of tilt `i` over `K` depth slabs.

        :return: slab projections of shape `(B, C, K, Hd, Wd)` and slab heights in voxels of shape `(K,)`.
        """
        B = x.shape[0]
        D, H, W = self.volume_size
        Hd, Wd = self.detector_size
        dev, dt = x.device, x.dtype
        tp = lambda v, n=0: self._per_tilt(v.to(dt), n)[:, i]

        # inverses of the push maps F2 = R(rho) Sh(g) m (projections) and F3 = R Sh S (volume), in (x, y, z) order
        theta = torch.deg2rad(tp(self.angles))
        rho = torch.deg2rad(tp(self.projection_rotation) + self.tilt_axis_angle.to(dt))
        g2 = tp(self.projection_shear)
        m2 = tp(self.projection_scale)
        o2, z2 = torch.ones_like(g2), torch.zeros_like(g2)
        sh2_inv = torch.stack(
            [torch.stack([o2, -g2], -1), torch.stack([z2, o2], -1)], -2
        )
        F2inv = sh2_inv @ self._rot2(-rho) / m2[..., None, None]
        s2 = tp(self.projection_shift, 1).flip(-1)  # (dx, dy)

        rot = torch.deg2rad(tp(self.volume_rotation, 1))  # (az, ay, ax)
        R3_inv = (
            self._rot3(-rot[..., 2], "x")
            @ self._rot3(-rot[..., 1], "y")
            @ self._rot3(-rot[..., 0], "z")
        )
        g3 = tp(self.volume_shear, 1)  # (g_xy, g_xz, g_yz)
        a, b, c3 = g3[..., 0], g3[..., 1], g3[..., 2]
        o3, z3 = torch.ones_like(a), torch.zeros_like(a)
        sh3_inv = torch.stack(
            [
                torch.stack([o3, -a, a * c3 - b], -1),
                torch.stack([z3, o3, -c3], -1),
                torch.stack([z3, z3, o3], -1),
            ],
            -2,
        )
        S3_inv = torch.diag_embed(1 / tp(self.volume_scale, 1).flip(-1))
        F3inv = S3_inv @ sh3_inv @ R3_inv
        s3 = tp(self.volume_shift, 1).flip(-1)  # (dx, dy, dz)

        # pull map from the beam frame (q_x, q_y, t) to the volume: p = M [q; t] + c
        Rt = self._rot3(theta, "y").transpose(-1, -2)
        Bm = torch.broadcast_shapes(F2inv.shape[:-2], F3inv.shape[:-2], Rt.shape[:-2])
        E = torch.zeros(*Bm, 3, 3, device=dev, dtype=dt)
        E[..., :2, :2] = F2inv
        E[..., 2, 2] = 1.0
        e = torch.zeros(*Bm, 3, device=dev, dtype=dt)
        e[..., :2] = -(F2inv @ s2.unsqueeze(-1)).squeeze(-1)
        M = F3inv @ Rt @ E
        c = (F3inv @ ((Rt @ e.unsqueeze(-1)).squeeze(-1) - s3).unsqueeze(-1)).squeeze(
            -1
        )

        th = theta.detach().abs().max()
        length = D * torch.cos(th).abs().item() + W * torch.sin(th).abs().item()
        T = max(1, math.ceil((length + 2 * self.depth_padding) / self.depth_step))
        t = (torch.arange(T, device=dev, dtype=dt) - (T - 1) / 2) * self.depth_step

        qy = torch.arange(Hd, device=dev, dtype=dt) - (Hd - 1) / 2
        qx = torch.arange(Wd, device=dev, dtype=dt) - (Wd - 1) / 2
        QY, QX = torch.meshgrid(qy, qx, indexing="ij")
        QX, QY = QX[None], QY[None]
        if self.projection_displacement is not None:
            u2 = tp(self.projection_displacement, 3)  # (dy, dx)
            grid2 = torch.stack([QX / (Wd / 2), QY / (Hd / 2)], -1)
            u2 = F.grid_sample(
                u2,
                grid2.expand(u2.shape[0], -1, -1, -1),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            QY = QY + u2[:, 0]
            QX = QX + u2[:, 1]
        QX, QY = QX[:, None], QY[:, None]
        Tz = t[None, :, None, None]

        coef = lambda a, b: M[:, a, b][:, None, None, None]
        p = torch.stack(
            [
                coef(j, 0) * QX
                + coef(j, 1) * QY
                + coef(j, 2) * Tz
                + c[:, j][:, None, None, None]
                for j in range(3)
            ],
            -1,
        )  # (x, y, z) in voxels

        half = torch.tensor([W / 2, H / 2, D / 2], device=dev, dtype=dt)
        grid = p / half

        if self.volume_displacement is not None:
            u3 = tp(self.volume_displacement, 4)  # (dz, dy, dx)
            Bc = max(u3.shape[0], grid.shape[0])
            u3 = F.grid_sample(
                u3.expand(Bc, -1, -1, -1, -1),
                grid.expand(Bc, -1, -1, -1, -1),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            grid = grid + u3.permute(0, 2, 3, 4, 1).flip(-1) / half

        if grid.shape[0] not in (1, B):
            raise ValueError(
                f"Parameter batch size {grid.shape[0]} does not match input batch size {B}."
            )
        samples = F.grid_sample(
            x,
            grid.expand(B, -1, -1, -1, -1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        K = min(self.n_defocus_planes, T)
        planes = (
            torch.stack(
                [chunk.sum(dim=2) for chunk in torch.tensor_split(samples, K, dim=2)], 2
            )
            * self.depth_step
        )
        centers = torch.stack([chunk.mean() for chunk in torch.tensor_split(t, K)])
        return planes, centers

    def _apply_transfer(self, planes: torch.Tensor, centers: torch.Tensor, i: int):
        r"""
        Applies the CTF and dose filter of tilt `i` to each slab and sums the slabs.

        :param torch.Tensor planes: slab projections of shape `(B, C, K, Hd, Wd)`.
        :param torch.Tensor centers: slab heights in voxels of shape `(K,)`.
        :return: (:class:`torch.Tensor`) projection of shape `(B, C, Hd, Wd)`.
        """
        if not self.ctf and not self.dose_weighting:
            return planes.sum(dim=2)

        dev, dt = planes.device, planes.dtype
        Hd, Wd = self.detector_size
        pad = self.fft_padding
        Hp, Wp = Hd + 2 * pad, Wd + 2 * pad
        tp = lambda v: self._per_tilt(v.to(dt))[:, i]

        ky = torch.fft.fftfreq(Hp, d=self.pixel_size, device=dev, dtype=dt)
        kx = torch.fft.rfftfreq(Wp, d=self.pixel_size, device=dev, dtype=dt)
        KY, KX = torch.meshgrid(ky, kx, indexing="ij")

        H = torch.ones(1, 1, *KX.shape, device=dev, dtype=dt)
        if self.ctf:
            df = tp(self.defocus)[:, None] + centers[None, :] * self.pixel_size * 1e-4
            col = lambda v: tp(v)[:, None, None, None]
            H = ctf_fn(
                KX,
                KY,
                defocus=df[..., None, None],
                voltage=self.voltage,
                spherical_aberration=self.spherical_aberration,
                amplitude_contrast=self.amplitude_contrast,
                astigmatism=col(self.astigmatism),
                astigmatism_angle=col(self.astigmatism_angle),
                phase_shift=col(self.phase_shift),
                b_factor=col(self.b_factor),
            )
        if self.dose_weighting:
            exposure = self.accumulated_exposure().to(dt)[:, i]
            k = torch.sqrt(KX**2 + KY**2)
            H = H * dose_filter(k, exposure[:, None, None, None], self.voltage)

        if pad > 0:
            planes = F.pad(planes, (pad, pad, pad, pad))
        Y = torch.fft.rfft2(planes)
        Y = (Y * H[:, None]).sum(dim=2)
        y = torch.fft.irfft2(Y, s=(Hp, Wp))
        if pad > 0:
            y = y[..., pad : pad + Hd, pad : pad + Wd]
        return y


class MissingWedge(LinearPhysics):
    r"""
    Missing-wedge operator for cryo-electron tomography volumes.

    A tomogram reconstructed from a tilt series acquired in :math:`[\theta_{\min}, \theta_{\max}]` only contains the Fourier coefficients inside a double wedge (see :func:`deepinv.physics.functional.missing_wedge_mask`). The operator is

    .. math::

        y = \forw{x} = C \mathcal{F}^{-1} M \mathcal{F} P x,

    where :math:`P` zero-pads the volume to `padding_factor` times its size to reduce wrap-around, :math:`\mathcal{F}` is the 3D Fourier transform, :math:`M` is the binary mask and :math:`C = P^{\top}` crops, as in :footcite:t:`kishore2026icecream`.
    It is self-adjoint, and an orthogonal projection when `padding_factor=1`. Volumes have shape `(B, C, D, H, W)` with the tilt axis along `H`, as in :class:`deepinv.physics.CryoET`.

    :param tuple[int, ...] img_size: volume size `(D, H, W)` or `(C, D, H, W)`.
    :param tuple[float, float] angle_range: minimum and maximum tilt angles in degrees.
    :param int padding_factor: zero-padding factor of the Fourier grid of the mask.
    :param bool spherical_support: if `True`, keeps only frequencies inside the unit ball.
    :param float low_frequency_support: radius (normalized frequency) of a ball of low frequencies considered measured.
    :param torch.device, str device: device of the operator.

    |sep|

    :Example:

        >>> import torch
        >>> from deepinv.physics import MissingWedge
        >>> physics = MissingWedge(img_size=(16, 16, 16), angle_range=(-60, 60))
        >>> x = torch.randn(1, 1, 16, 16, 16)
        >>> physics(x).shape
        torch.Size([1, 1, 16, 16, 16])
    """

    def __init__(
        self,
        img_size: tuple[int, ...],
        angle_range: tuple[float, float] = (-60.0, 60.0),
        padding_factor: int = 2,
        spherical_support: bool = True,
        low_frequency_support: float = 0.0,
        device: torch.device | str = "cpu",
        **kwargs,
    ):
        if len(img_size) not in (3, 4):
            raise ValueError(
                f"img_size must be (D, H, W) or (C, D, H, W), got {tuple(img_size)}."
            )
        if padding_factor < 1:
            raise ValueError("padding_factor must be a positive integer.")
        super().__init__(img_size=tuple(img_size), device=device, **kwargs)
        self.volume_size = tuple(int(s) for s in img_size[-3:])
        self.angle_range = (float(angle_range[0]), float(angle_range[1]))
        self.padding_factor = int(padding_factor)
        mask = missing_wedge_mask(
            tuple(padding_factor * s for s in self.volume_size),
            angle_range=self.angle_range,
            spherical_support=spherical_support,
            low_frequency_support=low_frequency_support,
            device=device,
        )
        self.register_buffer("mask", mask)
        self.to(device)

    def A(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        r"""
        Applies the missing-wedge filter.

        :param torch.Tensor x: volumes of shape `(B, C, D, H, W)`.
        :param torch.Tensor mask: new Fourier mask on the padded grid, stored for later calls.
        :return: (:class:`torch.Tensor`) filtered volumes of shape `(B, C, D, H, W)`.
        """
        self.update_parameters(mask=mask)
        return apply_fourier_mask(x, self.mask.to(x.dtype))

    def A_adjoint(
        self, y: torch.Tensor, mask: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        r"""
        Applies the adjoint of the missing-wedge filter, i.e. the filter itself.

        :param torch.Tensor y: volumes of shape `(B, C, D, H, W)`.
        :param torch.Tensor mask: new Fourier mask on the padded grid, stored for later calls.
        :return: (:class:`torch.Tensor`) filtered volumes of shape `(B, C, D, H, W)`.
        """
        return self.A(y, mask=mask)
