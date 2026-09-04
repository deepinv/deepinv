from __future__ import annotations

from .forward import LinearPhysics
from .noise import PoissonNoise

import torch
import math
from functools import partial
from typing import Callable
from deepinv.physics.functional import (
    conv2d,
    conv_transpose2d,
    rotate,
    rotate_grid,
    rotate_adjoint,
)


def _linear_cdr(d: torch.Tensor, a: float, c: float) -> torch.Tensor:
    r"""
    Collimator-detector response width :math:`\text{FWHM}(d) = ad + c`, in mm.

    Linear fit to the geometric response :math:`D(L_\text{eff} + d + b)/L_\text{eff}`
    combined in quadrature with the intrinsic detector resolution, so
    :math:`a \approx D/L_\text{eff}` (hole diameter / effective hole length) and
    :math:`c` absorbs the zero-distance geometric term and the intrinsic resolution.
    Calibrate per scanner.

    :param torch.Tensor d: source-to-collimator distances in mm.
    :param float a: slope, dimensionless.
    :param float c: intercept in mm.
    """
    return a * d + c


class SPECT(LinearPhysics):
    r"""
    Single photon emission computed tomography (SPECT) physics model.

    Rotation-based projector with depth-dependent collimator-detector response and
    attenuation correction, following :footcite:t:`<INSERTBIBTEXKEY>`.

    .. math::
        v(i,k,l) = \sum_j p_{j,l} \circledast \left[ \bar\mu(i,j,k;l)\, \tilde x(i,j,k;l) \right]

    where :math:`\tilde x` is the activity rotated into the detector frame at view
    :math:`\theta_l`, :math:`\bar\mu` the accumulated attenuation and :math:`p_{j,l}`
    the depth-dependent point spread function.

    .. note::
        The Gaussian collimator-detector response is a low-E approximation
        (Tc-99m, I-123). It neglects septal penetration and collimator scatter, so it
        is not appropriate for I-131, Lu-177, Ac-225 or In-111 without a
        penetration-tail kernel. Pass `cdr_fwhm` to substitute your own model.

    :param tuple img_size: volume shape `(n_x, n_y, n_z)`, where `n_y` is depth toward
        the detector and `n_z` the axial axis. Requires a square isotropic transaxial grid,
        `n_x == n_y` and `d_x == d_y`, so that rotation maps the voxel grid onto itself.
    :param int n_view: number of projection views over :math:`2\pi`. `n_view=1` gives a
        single stationary view (planar scintigraphy), still with attenuation and
        depth-dependent blur.
    :param tuple voxel_size: voxel size in mm, `(d_x, d_y, d_z)`. A scalar is broadcast.
    :param float radius: distance in mm from the rotation centre to the collimator face,
        setting the depth of each plane and hence the blur width.
    :param Callable cdr_fwhm: collimator-detector response width in mm as a function of
        depth in mm. Defaults to :math:`ad + c` with `(a, c) = cdr_params`. Replace it to
        substitute a measured or non-Gaussian response.
    :param tuple cdr_params: `(a, c)` of the default linear response, see :func:`_linear_cdr`.
        The defaults are placeholders; calibrate per scanner.
    :param int view_chunk: number of views projected per iteration. `None` processes all
        views at once (fastest, highest peak memory, which scales as
        `B * n_view * n_x * n_y * n_z`). Lower it to 8-16 for large volumes.
    :param float gain: gain :math:`\gamma` of the Poisson noise model.
    :param torch.Tensor attenuation: attenuation map :math:`\mu` in 1/mm, image space with
        spatial shape `img_size`, typically from an auxiliary CT. Defaults to zeros, i.e.
        no attenuation. Unlike :class:`deepinv.physics.PET`, it cannot be given in
        projection space: a SPECT photon's path length depends on its emission depth.
    :param str, torch.device device: device to run the computations on.

    |sep|

    :Example:

    >>> from deepinv.physics import SPECT
    >>> import torch
    >>> physics = SPECT(img_size=(32, 32, 8), n_view=8)
    >>> x = torch.rand(1, 1, 32, 32, 8)
    >>> physics.A(x).shape
    torch.Size([1, 1, 8, 32, 8])
    """

    def __init__(
        self,
        img_size: tuple,
        n_view: int,
        voxel_size: tuple = (4.42, 4.42, 4.42),
        radius: float = 250.0,
        cdr_fwhm: Callable[[torch.Tensor], torch.Tensor] | None = None,
        cdr_params: tuple = (0.03, 6.0),
        view_chunk: int | None = None,
        gain: float = 1.0,
        attenuation: torch.Tensor | None = None,
        device: str | torch.device = "cpu",
        **kwargs,
    ):

        super().__init__(**kwargs)
        if len(img_size) != 3:
            raise ValueError(f"img_size must be (n_x, n_y, n_z), got {img_size}.")
        if n_view < 1:
            raise ValueError(f"n_view must be at least 1, got {n_view}.")

        if isinstance(voxel_size, (int, float)):
            voxel_size = (voxel_size,) * 3

        if img_size[0] != img_size[1] or voxel_size[0] != voxel_size[1]:
            raise ValueError(
                f"SPECT requires a square isotropic transaxial grid, got "
                f"n=({img_size[0]}, {img_size[1]}), d=({voxel_size[0]}, {voxel_size[1]}). "
                "Rotation must map the voxel grid onto itself, or the reconstruction "
                "FOV becomes view-dependent."
            )

        self.img_size = tuple(img_size)
        self.voxel_size = tuple(float(v) for v in voxel_size)
        self.n_view = int(n_view)
        self.radius = float(radius)
        self.view_chunk = view_chunk

        if cdr_fwhm is None:
            cdr_fwhm = partial(_linear_cdr, a=cdr_params[0], c=cdr_params[1])
        self.cdr_fwhm = cdr_fwhm

        dtype = torch.get_default_dtype()
        theta = (
            2 * math.pi * torch.arange(self.n_view, dtype=torch.float64) / self.n_view
        )
        self.register_buffer("theta", theta.to(dtype), persistent=False)
        self.register_buffer(
            "grid", rotate_grid(img_size[0], theta, dtype=dtype), persistent=False
        )

        if attenuation is None:
            attenuation = torch.zeros((1, 1) + self.img_size)
        self.register_buffer("attenuation", attenuation)

        self.noise_model = PoissonNoise(gain=gain)
        self.update_parameters(attenuation=attenuation)
        self.to(device)

    def A(
        self, x: torch.Tensor, attenuation: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        r"""
        Project an activity volume to SPECT views.

        :param torch.Tensor x: activity volume of shape `(B, 1, n_x, n_y, n_z)`, with `n_y`
            the depth toward the detector and `n_z` the axial axis.
        :param torch.Tensor attenuation: if not `None`, update the attenuation map
            :math:`\mu` (image space, 1/mm).
        :return: projection views of shape `(B, 1, n_view, n_x, n_z)`, one `(n_x, n_z)`
            planar view per angle.
        """
        if x.shape[1] != 1:
            raise ValueError(f"Input volume must have 1 channel, got {x.shape[1]}.")
        self.update_parameters(attenuation=attenuation)

        n_x, _, n_z = self.img_size
        v = x.new_zeros(x.shape[0], 1, self.n_view, n_x, n_z)
        for sl in self._view_chunks():
            v[:, :, sl] = self._project(x, sl).unsqueeze(1)
        return v

    def A_adjoint(
        self, y: torch.Tensor, attenuation: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        r"""
        Backproject SPECT views, apply :math:`A^{\top}y`.
        :param torch.Tensor y: projection views of shape `(B, 1, n_view, n_x, n_z)`.
        :param torch.Tensor attenuation: if not `None`, update the attenuation map
            :math:`\mu` (image space, 1/mm).
        :return: volume of shape `(B, 1, n_x, n_y, n_z)`.
        """

        if y.shape[1] != 1:
            raise ValueError(
                f"Input measurements must have 1 channel, got {y.shape[1]}"
            )
        self.update_parameters(attenuation=attenuation)

        x = y.new_zeros((y.shape[0], 1) + self.img_size)
        for sl in self._view_chunks():
            x = x + self._backproject(y[:, :, sl], sl)
        return x

    def _view_chunks(self):
        step = self.view_chunk or self.n_view
        for lo in range(0, self.n_view, step):
            yield slice(lo, min(lo + step, self.n_view))

    def _project(self, x: torch.Tensor, views: slice) -> torch.Tensor:
        """Rotate, attenuate, blur and sum a chunk of views. Returns `(B, L, n_x, n_z)`."""
        batch_size, n_view = x.shape[0], self._n_views(views)
        grid = self.grid[views].to(x.dtype).repeat(batch_size, 1, 1, 1)

        vol = self._unfold(rotate(self._fold(x, n_view), grid), batch_size, n_view)
        mu_bar = self._mu_bar(views, x.dtype)
        if mu_bar is not None:
            vol = vol * mu_bar
        return self._blur(vol, views).sum(-2)

    def _backproject(self, v: torch.Tensor, views: slice) -> torch.Tensor:
        batch_size, n_view = x.shape[0], self._n_views(views)
        n_x, n_y, n_z = self.img_size

        vol = v.squeeze(1).unsqueeze(-2).expand(batch_size, n_view, n_x, n_y, n_z)
        vol = self._blur_adjoint(vol, views)
        mu_bar = self._mu_bar(views, v.dtype)
        if mu_bar is not None:
            vol = vol * mu_bar

        grid = self.grid[views].t(v.dtype).repeat(batch_size, 1, 1, 1)
        planes = rotate_adjoint(self._to_planes(vol), grid)
        return self._unfold(planes, batch_size, n_view).sum(1).unsqueeze(1)

    def _n_views(self, views: slice) -> int:
        return len(range(*views.indices(self.n_view)))

    def _blur(self, vol: torch.Tensor, views: slice) -> torch.Tensor:
        r"""Depth-dependent collimator response in the `(i, k)` detector plane."""
        return vol

    def _blur_adjoint(self, vol: torch.Tensor, views: slice) -> torch.Tensor:
        r"""Adjoint of :meth:`_blur`. Self-adjoint for symmetric kernels."""
        return vol

    def _mu_bar(self, view_slice: slice, dtype: torch.dtype) -> torch.Tensor | None:
        r"""
        Accumulated attenuation factor for a chunk of views, or ``None`` if
        :math:`\mu \equiv 0`.

        .. math::
            \bar\mu(i,j,k;l) = e^{-\Delta y\left(\tfrac12\tilde\mu(i,j,k;l)
                                + \sum_{s>j}\tilde\mu(i,s,k;l)\right)}

        The :math:`\tfrac12` as the photon travels through only half of its own voxel

        :return: ``(1, L, n_x, n_y, n_z)``
        """
        if self._no_attenuation:
            return None

        n_view = self._n_views
        grid = self.grid[view_slice].to(dtype)

        mu = self._unfold(
            rotate(self._fold(self.attenuation.to(dtype), n_view), grid), 1, n_view
        )
        tail = mu.flip(-2).cumsum(-2).flip(-2)  # suffix sum: Σ_{s ≥ j}
        return torch.exp(-self.voxel_size[1] * (tail - 0.5 * mu))

    def update_parameters(self, attenuation: torch.Tensor | None = None, **kwargs):
        r"""
        Update the attenuation map :math:`\mu`.

        Unlike :class:`deepinv.physics.PET`, the attenuation is not precomputed
        into projection space (in SPECT: attenuation depends on depth)
        :math:`\bar\mu` is a per-voxel, per-view factor applied
        inside the depth sum. Recomputed per view chunk in ``A``/``A_adjoint``.
        """
        if attenuation is not None:
            if tuple(attenuation.shape[-3:]) != self.img_size:
                raise ValueError(
                    f"attenuation must have spatial shape {self.img_size}, "
                    f"got {tuple(attenuation.shape[-3:])}"
                )
            while attenuation.ndim < 5:
                attenuation = attenuation.unsqueeze(0)
            super().update_parameters(attenuation=attenuation, **kwargs)
            self._no_attenuation = bool((self.attenuation == 0).all())
        else:
            super().update_parameters(**kwargs)

    def _fold(self, x: torch.Tensor, n_view: int) -> torch.Tensor:
        r"""
        Replicate a volume across views and lay it out as planes.
        ``(B, 1, n_x, n_y, n_z) -> (B*L, n_z, n_x, n_y)``, flat batch index ``b*L + l``.

        ``(B, 1, n_x, n_y, n_z) -> (B*L, n_z, n_x, n_y)``. This is the peak-memory step
        (``B * L * n_x * n_y * n_z``), and what ``view_chunk`` slices.
        """
        return self._to_planes(x.expand(-1, n_view, -1, -1, -1))

    def _unfold(self, s: torch.Tensor, batch_size: int, n_view: int) -> torch.Tensor:
        r"""
        Inverse layout of :func:`_fold`:
        ``(B*L, n_z, n_x, n_y) -> (B, L, n_x, n_y, n_z)``.
        """
        n_z, n_x, n_y = s.shape[-3:]
        return s.reshape(batch_size, n_view, n_z, n_x, n_y).permute(0, 1, 3, 4, 2)

    def _to_planes(self, vol: torch.Tensor) -> torch.Tensor:
        r"""
        adjoint counterpart of _fold: ``(B, L, n_x, n_y, n_z) -> (B*L, n_z, n_x, n_y)``
        """
        n_x, n_y, n_z = vol.shape[-3:]
        return vol.permute(0, 1, 4, 2, 3).reshape(-1, n_z, n_x, n_y)
