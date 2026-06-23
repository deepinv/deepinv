from __future__ import annotations
import torch
import numpy as np
from math import ceil, floor
from deepinv.physics.generator import PhysicsGenerator
from deepinv.physics.functional import gaussian_blur, random_uniform
from deepinv.physics.functional.hist import histogramdd
from deepinv.physics.functional.convolution import conv2d
from deepinv.physics.functional.interp import ThinPlateSpline
from deepinv.utils.decorators import _deprecated_alias
from deepinv.transform.rotate import rotate_via_shear
from deepinv.utils.mixins import TiledMixin2d
from deepinv.utils._internal import _check_pairwise_leq, _as_sequence
from .zernike import Zernike


class PSFGenerator(PhysicsGenerator):
    r"""
    Base class for generating Point Spread Functions (PSFs).


    :param tuple psf_size: the shape of the generated PSF in 2D
        ``(kernel_size, kernel_size)``. If an `int` is given, it will be used for both dimensions.
    :param int num_channels: number of images channels. Defaults to 1.
    """

    def __init__(
        self,
        psf_size: tuple[int] = (31, 31),
        num_channels: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(psf_size, int):
            psf_size = (psf_size, psf_size)

        self.shape = (num_channels,) + psf_size
        self.psf_size = psf_size
        self.num_channels = num_channels


class GaussianBlurGenerator(PSFGenerator):
    r"""
    Random Gaussian blur generator. Generates 1D, 2D, or 3D Gaussian kernels with random standard deviations and rotation angles.

    :param tuple[int, ...] psf_size: the shape of the generated point spread function (PSF). The dimension (1D, 2D, or 3D) of the kernel is determined by the length of the ``psf_size`` tuple.
    :param float | tuple[float, ...] sigma_min: the minimum standard deviation(s) for the Gaussian kernel. If a single value is provided, it is applied to all dimensions. If a tuple is provided, it should have the same length as the number of dimensions and specify the minimum sigma for each dimension.
    :param float | tuple[float, ...] sigma_max: the maximum standard deviation(s) for the Gaussian kernel. Follows the same format as ``sigma_min``.
    :param bool isotropic: If True, the generated Gaussian kernels will be isotropic (same sigma for all dimensions). If False, the kernels can be anisotropic (different sigma for each dimension). Defaults to True.
    :param float | tuple[float, ...] angle_min: the minimum rotation angle(s) for the Gaussian kernel in degrees. For 2D kernels, this is a single angle of rotation in the plane. For 3D kernels, this can be a tuple of three angles (alpha, beta, gamma) representing minimum rotation values around the x, y, and z axes respectively. In 3D, if a single angle is provided, it is used as minimum value for all axes.
    :param float | tuple[float, ...] angle_max: the maximum rotation angle(s) for the Gaussian kernel in degrees. Follows the same format as ``angle_min``.
    :param int num_channels: number of images channels. Defaults to 1.
    :param torch.Generator rng: PyTorch random number generator for reproducibility. If ``None``, a torch.Generator will be created on the specified device.
    :param str device: the device to create the tensors on. Defaults to "cpu".
    :param type dtype: the data type of the generated tensors. Defaults to torch.float32.

    |sep|

    :Examples:

    >>> import deepinv as dinv
    >>> rng = torch.Generator(device="cpu").manual_seed(0)
    >>> generator = dinv.physics.generator.GaussianBlurGenerator((7, 7), device="cpu", rng=rng, isotropic=False)
    >>> params = generator.step(batch_size=4)  # dict_keys(['filter'])
    >>> dinv.utils.plot(params['filter'])  # doctest: +SKIP

    .. plot::

        import torch
        import deepinv as dinv
        rng = torch.Generator(device="cpu").manual_seed(0)
        generator = dinv.physics.generator.GaussianBlurGenerator((7, 7), device="cpu", rng=rng, isotropic=False)
        params = generator.step(batch_size=4)
        dinv.utils.plot(params['filter'])

    """

    def __init__(
        self,
        psf_size: tuple[int, ...],
        sigma_min: float | tuple[float, ...] = 0.5,
        sigma_max: float | tuple[float, ...] = 5.0,
        isotropic: bool = True,
        angle_min: float | tuple[float, ...] = 0.0,
        angle_max: float | tuple[float, ...] = 360.0,
        num_channels: int = 1,
        rng: torch.Generator = None,
        device: str = "cpu",
        dtype: type = torch.float32,
    ):

        dim = len(psf_size)
        if dim not in {1, 2, 3}:
            raise ValueError("Only 1D, 2D, and 3D kernels are supported.")

        sigma_min, sigma_max = self._resolve_sequences_dimensionality(
            sigma_min, sigma_max, dim, name_min="sigma_min", name_max="sigma_max"
        )
        angle_min, angle_max = self._resolve_sequences_dimensionality(
            angle_min,
            angle_max,
            dim=(
                3 if dim == 3 else 1
            ),  # For 2D, the angle is a single value of rotation in the plane, for 3D it can be a tuple of 3 values
            name_min="angle_min",
            name_max="angle_max",
        )

        kwargs = {
            "dim": dim,
            "psf_size": psf_size,
            "num_channels": num_channels,
            "isotropic": isotropic,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "angle_min": angle_min,
            "angle_max": angle_max,
        }
        super().__init__(device=device, dtype=dtype, rng=rng, **kwargs)

    @staticmethod
    def _resolve_sequences_dimensionality(
        min_vals: tuple[float, ...],
        max_vals: tuple[float, ...],
        dim: int,
        name_min: str = None,
        name_max: str = None,
    ):

        min_vals = _as_sequence(min_vals)
        max_vals = _as_sequence(max_vals)

        if len(min_vals) == 1:
            min_vals = min_vals * dim
        if len(max_vals) == 1:
            max_vals = max_vals * dim

        if len(min_vals) != dim or len(max_vals) != dim:
            raise ValueError(
                f"Length of {name_min} and {name_max} should be either 1 or {dim}. Got len({name_min})={len(min_vals)} and len({name_max})={len(max_vals)}."
            )

        _check_pairwise_leq(min_vals, max_vals, name_min, name_max)

        return min_vals, max_vals

    def _generate_parameters(
        self, batch_size: int, min_vals, max_vals, isotropic: bool = False, **kwargs
    ):

        if isotropic:
            params = torch.stack(
                [
                    random_uniform(
                        batch_size,
                        low=min_vals[0],
                        high=max_vals[0],
                        generator=self.rng,
                        **self.factory_kwargs,
                    )
                ]
                * len(min_vals),
                dim=-1,
            )
        else:
            params = torch.stack(
                [
                    random_uniform(
                        batch_size,
                        low=min_val,
                        high=max_val,
                        generator=self.rng,
                        **self.factory_kwargs,
                    )
                    for min_val, max_val in zip(min_vals, max_vals, strict=True)
                ],
                dim=-1,
            )  # Shape: (batch_size, dim)

        return params

    def step(
        self,
        batch_size: int = 1,
        sigma: torch.Tensor = None,
        angle: torch.Tensor = None,
        seed: int = None,
        **kwargs,
    ):
        self.rng_manual_seed(seed)
        dim = len(self.psf_size)

        sigma = (
            self._generate_parameters(
                batch_size, self.sigma_min, self.sigma_max, isotropic=self.isotropic
            )
            if sigma is None
            else sigma
        )
        angle = (
            self._generate_parameters(batch_size, self.angle_min, self.angle_max, False)
            if angle is None
            else angle
        )

        # filter.shape = (batch_size, 1, *psf_size)
        filters = gaussian_blur(self.psf_size, sigma, angle, **self.factory_kwargs)
        return {"filter": filters.expand(-1, self.num_channels, *(-1,) * dim)}


class MotionBlurGenerator(PSFGenerator):
    r"""
    Random motion blur generator.

    See :footcite:t:`schuler2015learning` for more details.

    A blur trajectory is generated by sampling both its x- and y-coordinates independently
    from a Gaussian Process with a Matérn 3/2 covariance function.

    .. math::

        f_x(t), f_y(t) \sim \mathcal{GP}(0, k(t, t'))

    where :math:`k` is defined as

    .. math::

        k(t, s) = \sigma^2 \left( 1 + \frac{\sqrt{5} |t -s|}{l} + \frac{5 (t-s)^2}{3 l^2} \right) \exp \left(-\frac{\sqrt{5} |t-s|}{l}\right)

    :param int, tuple[int] psf_size: the shape of the generated PSF in 2D, should be `(kernel_size, kernel_size)`. If an `int` is given, the same value will be used for both dimensions.
    :param int num_channels: number of images channels. Defaults to 1.
    :param float l: the length scale of the trajectory, defaults to 0.3
    :param float sigma: the standard deviation of the Gaussian Process, defaults to 0.25
    :param int n_steps: the number of points in the trajectory, defaults to 1000

    |sep|

    :Examples:

    >>> from deepinv.physics.generator import MotionBlurGenerator
    >>> generator = MotionBlurGenerator((5, 5), num_channels=1)
    >>> blur = generator.step()  # dict_keys(['filter'])
    >>> print(blur['filter'].shape)
    torch.Size([1, 1, 5, 5])
    """

    def __init__(
        self,
        psf_size: tuple,
        num_channels: int = 1,
        rng: torch.Generator = None,
        device: str = "cpu",
        dtype: type = torch.float32,
        l: float = 0.3,
        sigma: float = 0.25,
        n_steps: int = 1000,
    ) -> None:
        kwargs = {"l": l, "sigma": sigma, "n_steps": n_steps}
        if isinstance(psf_size, int):
            psf_size = (psf_size, psf_size)

        if len(psf_size) != 2:
            raise ValueError(
                "psf_size must 2D. Add channels via num_channels parameter"
            )
        super().__init__(
            psf_size=psf_size,
            num_channels=num_channels,
            device=device,
            dtype=dtype,
            rng=rng,
            **kwargs,
        )

    def matern_kernel(self, diff, sigma: float = None, l: float = None):
        r"""
        Compute the Matérn 3/2 covariance.

        :param torch.Tensor diff: the difference `t - s`
        :param float sigma: the standard deviation of the Gaussian Process
        :param float l: the length scale of the trajectory
        """
        if sigma is None:
            sigma = self.sigma
        if l is None:
            l = self.l
        fraction = 5**0.5 * diff.abs() / l
        return sigma**2 * (1 + fraction + fraction**2 / 3) * torch.exp(-fraction)

    def f_matern(self, batch_size: int = 1, sigma: float = None, l: float = None):
        r"""
        Generates the trajectory.

        :param int batch_size: batch_size.
        :param float sigma: the standard deviation of the Gaussian Process.
        :param float l: the length scale of the trajectory.
        :return: the trajectory of shape `(batch_size, n_steps)`
        """
        vec = torch.randn(
            batch_size, self.n_steps, generator=self.rng, **self.factory_kwargs
        )
        time = torch.linspace(-torch.pi, torch.pi, self.n_steps, **self.factory_kwargs)[
            None
        ]

        kernel = self.matern_kernel(time, sigma, l)
        kernel_fft = torch.fft.rfft(kernel)
        vec_fft = torch.fft.rfft(vec)
        return torch.fft.irfft(vec_fft * torch.sqrt(kernel_fft)).real[
            :,
            torch.arange(self.n_steps // (2 * torch.pi), **self.factory_kwargs).type(
                torch.int
            ),
        ]

    def step(
        self,
        batch_size: int = 1,
        sigma: float = None,
        l: float = None,
        seed: int = None,
        **kwargs,
    ):
        r"""
        Generate a random motion blur PSF with parameters :math:`\sigma` and :math:`l`

        :param int batch_size: batch_size.
        :param float sigma: the standard deviation of the Gaussian Process
        :param float l: the length scale of the trajectory
        :param int seed: the seed for the random number generator.

        :return: dictionary with keys

            - `filter`: the generated PSF of shape `(batch_size, 1, psf_size[0], psf_size[1])`
        """
        self.rng_manual_seed(seed)
        f_x = self.f_matern(batch_size, sigma, l)[..., None]
        f_y = self.f_matern(batch_size, sigma, l)[..., None]

        trajectories = torch.cat(
            (
                f_x - torch.mean(f_x, dim=1, keepdim=True),
                f_y - torch.mean(f_y, dim=1, keepdim=True),
            ),
            dim=-1,
        )
        kernels = [
            histogramdd(trajectory, bins=list(self.psf_size), low=[-1, -1], upp=[1, 1])[
                None, None
            ]
            for trajectory in trajectories
        ]
        kernel = torch.cat(kernels, dim=0).to(**self.factory_kwargs)
        kernel = kernel / (torch.sum(kernel, dim=(-2, -1), keepdim=True) + 1e-6)
        return {
            "filter": kernel.expand(
                -1,
                self.num_channels,
                -1,
                -1,
            )
        }


class DiffractionBlurGenerator(PSFGenerator):
    r"""
    Diffraction limited blur generator.

    Generates 2D diffraction PSFs in optics using Zernike decomposition of the phase mask (Fresnel/Fraunhoffer diffraction theory).

    Zernike polynomials are a sequence of orthogonal polynomials defined on the unit disk.
    They are commonly used in optical systems to describe wavefront aberrations.

    The PSF :math:`h(\theta)` is defined as the squared magnitude of the Fourier transform of the pupil function :math:`p_{\theta} = \exp(- i 2 \pi \phi_{\theta})`:

    .. math::

        h(\boldsymbol{x}; \lambda) = \left| \mathcal{F} \left[ \mathbb{1}_{|\boldsymbol{\rho}| \leq 1} \cdot \exp \left( - i 2 \pi \sum_k \frac{a_k}{\lambda} z_k(\boldsymbol{\rho}) \right) \right](\boldsymbol{x}) \right|^2

    where :math:`\boldsymbol{\rho}` are normalised pupil-plane coordinates (on the unit disk),
    :math:`a_k` are the Zernike coefficients **in physical units** (nm OPD, wavelength-independent),
    :math:`\lambda` is the emission wavelength (nm), and :math:`z_k` are the Zernike polynomials.

    The phase in waves is therefore :math:`\theta_k(\lambda) = a_k / \lambda`, so the same
    physical aberration produces a stronger wavefront error (in waves) at shorter wavelengths.

    For multi-channel (multi-colour) imaging the generator supports a perturbation model:

    .. math::

        \theta_k^{(b,c)} = \underbrace{\theta_k^{(b)} \cdot \frac{\lambda_{\text{ref}}}{\lambda_c}}_{\text{monochromatic, rescaled}} + \underbrace{\Delta\theta_k^{(b,c)}}_{\text{chromatic perturbation}}

    where :math:`\theta_k^{(b)}` are base coefficients (in waves at :math:`\lambda_{\text{ref}}`)
    shared across channels, and :math:`\Delta\theta_k^{(b,c)}` are small per-channel perturbations
    (e.g. sample-induced dispersion). The cutoff frequency is also wavelength-dependent:

    .. math::

        f_c^{(c)} = \frac{\mathrm{NA} \cdot p}{\lambda_c}

    where :math:`\mathrm{NA}` is the numerical aperture and :math:`p` is the pixel size (nm).

    See :footcite:t:`lakshminarayanan2011zernike`
    `or this link <https://e-l.unifi.it/pluginfile.php/1055875/mod_resource/content/1/Appunti_2020_Lezione%2014_4_Zernikepolynomialsaguidefinal.pdf>`_
    or :class:`deepinv.physics.generator.Zernike` for more details.

    In the ideal diffraction-limited case (i.e., no aberrations), the PSF corresponds to the Airy pattern.

    The Zernike polynomials :math:`z_k` are indexed using the ``'noll'`` or ``'ansi'`` convention (defined by `index_convention` parameter).
    Conversion from the two conventions to the standard radial-angular indexing is done internally (see `wikipedia page <https://en.wikipedia.org/wiki/Zernike_polynomials>`_).

    :param tuple psf_size: the shape ``H x W`` of the generated PSF in 2D
    :param int num_channels: number of images channels. Defaults to 1.
    :param tuple[int, ...], tuple[tuple[int, int], ...] zernike_index: activated Zernike coefficients in the following `index_convention` convention.
        It can be either:

            - a tuple of `int` corresponding to the Noll or ANSI indices, in which case the `index_convention` parameter is required to interpret them correctly.
            - a tuple of `tuple[int, int]` corresponding to the standard radial-angular indexing :math:`(n,m)`. In this case, the `index_convention` parameter is ignored.

        Defaults to ``(4, 5, 6, 7, 8, 9, 10, 11)``, correspond to radial order `n` from 2 to 3 (included) and the spherical aberration.
        These correspond to the following aberrations: defocus, astigmatism, coma, trefoil and spherical aberration.
    :param float fc: default cutoff frequency ``(NA/emission_wavelength) * pixel_size``. Should be in ``[0, 0.25]``
        to respect the Shannon-Nyquist sampling theorem, defaults to ``0.2``. Used when neither ``fc`` nor
        ``wavelengths`` is passed to :meth:`step`.

        .. deprecated::
            Prefer passing ``wavelengths`` (together with ``NA`` and ``pixel_size``) to :meth:`step`
            for physically consistent multi-channel PSFs, or pass ``fc`` directly for manual control.
            ``fc`` here is kept for backward compatibility and as a default fallback.
    :param float NA: numerical aperture of the objective. Required when using ``wavelengths`` in
        :meth:`step`. Can be overridden per :meth:`step` call. Defaults to ``1.4``.
    :param float pixel_size: camera pixel size in **nm**. Required when using ``wavelengths`` in
        :meth:`step`. Can be overridden per :meth:`step` call. Defaults to ``100.0``.
    :param float lambda_ref: reference wavelength in **nm** used to express the base Zernike
        coefficients returned by :meth:`generate_coeff`. The :math:`1/\lambda` rescaling in the
        perturbation model is performed relative to this value. Defaults to ``450.0`` nm.
    :param float max_zernike_amplitude: maximum amplitude of the Zernike coefficients **in waves
        at** ``lambda_ref``, defaults to ``0.15``.
        The amplitude of each Zernike coefficient is sampled uniformly in ``[-max_zernike_amplitude/2, max_zernike_amplitude/2]``.
    :param float zernike_perturbation_amplitude: amplitude of per-channel perturbations relative
        to ``max_zernike_amplitude``, defaults to ``1e-3``.
    :param tuple[int] pupil_size: pixel size used to synthesize the super-resolved pupil.
        The higher the more precise, defaults to ``(256, 256)``.
        If a single ``int`` is given, a square pupil is considered.
    :param bool apodize: whether to apodize the PSF to reduce ringing artifacts, defaults to ``False``.
    :param bool random_rotate: whether to randomly rotate the PSF, defaults to ``False``.
    :param str index_convention: the convention for the Zernike polynomials indexing. Can be either ``'noll'`` (default) or ``'ansi'``.
    :param str, torch.device device: device where the tensors are allocated and processed, defaults to ``'cpu'``.
    :param torch.dtype dtype: data type of the tensors, defaults to ``torch.float32``.
    :param torch.Generator rng: pseudo random number generator for reproducibility. Defaults to ``None``.

    |sep|

    :Examples:

    >>> from deepinv.physics.generator import DiffractionBlurGenerator
    >>> generator = DiffractionBlurGenerator((5, 5), num_channels=1)
    >>> print("\n".join(generator.zernike_polynomials)) # list of Zernike polynomials used
    Zernike(n = 2, m = 0) -- Defocus
    Zernike(n = 2, m = -2) -- Oblique Astigmatism
    Zernike(n = 2, m = 2) -- Vertical Astigmatism
    Zernike(n = 3, m = -1) -- Vertical Coma
    Zernike(n = 3, m = 1) -- Horizontal Coma
    Zernike(n = 3, m = -3) -- Vertical Trefoil
    Zernike(n = 3, m = 3) -- Oblique Trefoil
    Zernike(n = 4, m = 0) -- Primary Spherical
    >>> blur = generator.step()  # dict_keys(['filter', 'coeff', 'pupil'])
    >>> print(blur['filter'].shape)
    torch.Size([1, 1, 5, 5])

    Multi-channel PSF with per-channel cutoff frequencies (e.g. RGB):

    >>> import torch
    >>> generator = DiffractionBlurGenerator((5, 5), num_channels=3)
    >>> fc_rgb = torch.tensor([[0.18, 0.20, 0.22]])   # shape (1, 3): 1 batch x 3 channels
    >>> blur = generator.step(batch_size=1, fc=fc_rgb)
    >>> print(blur['filter'].shape)
    torch.Size([1, 3, 5, 5])

    Physically consistent multi-colour PSF via wavelengths (recommended for fluorescence microscopy):

    >>> generator = DiffractionBlurGenerator(
    ...     (5, 5), num_channels=3, NA=1.2, pixel_size=65.0, lambda_ref=450.0
    ... )
    >>> wavelengths = torch.tensor([450.0, 520.0, 640.0])  # nm, one per channel
    >>> blur = generator.step(batch_size=2, wavelengths=wavelengths)
    >>> print(blur['filter'].shape)
    torch.Size([2, 3, 5, 5])
    >>> print(blur['coeff'].shape)   # (B, C, K) wavelength-rescaled + perturbations
    torch.Size([2, 3, 8])

    """

    @_deprecated_alias(list_param="zernike_index")
    def __init__(
        self,
        psf_size: tuple,
        num_channels: int = 1,
        zernike_index: tuple[int, ...] | tuple[tuple[int, int], ...] = tuple(
            range(4, 12)
        ),
        fc: float = 0.2,
        NA: float = 1.4,
        pixel_size: float = 100.0,
        lambda_ref: float = 450.0,
        max_zernike_amplitude: float = 0.15,
        zernike_perturbation_amplitude: float = 1e-3,
        pupil_size: tuple[int, ...] = (256, 256),
        apodize: bool = False,
        random_rotate: bool = False,
        index_convention: str = "noll",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        rng: torch.Generator = None,
    ):
        super().__init__(
            psf_size=psf_size,
            num_channels=num_channels,
            device=device,
            dtype=dtype,
            rng=rng,
        )
        # For backward compatibility if a list / tuple of a string is given
        # Should be removed in future versions
        zernike_index = list(zernike_index)
        for i, index in enumerate(zernike_index):
            if isinstance(index, str):
                if not index.upper().startswith("Z"):
                    raise ValueError(f"Zernike index must start with 'Z', got {index}")
                index = int(index[1:])  # Convert "Z4" to 4
            zernike_index[i] = index

        self.zernike_index = sorted(zernike_index)
        self.fc = fc  # kept as default fallback for backward compatibility
        self.NA = NA
        self.pixel_size = pixel_size
        self.lambda_ref = lambda_ref
        self.max_zernike_amplitude = max_zernike_amplitude
        self.zernike_perturbation_amplitude = zernike_perturbation_amplitude
        self.apodize = apodize
        self.random_rotate = random_rotate
        self.index_convention = index_convention
        self.n_zernike = len(self.zernike_index)

        if self.apodize:
            lin_0 = torch.linspace(
                -psf_size[0] // 2, psf_size[0] // 2, psf_size[0], **self.factory_kwargs
            )
            lin_1 = torch.linspace(
                -psf_size[1] // 2, psf_size[1] // 2, psf_size[1], **self.factory_kwargs
            )
            XX0, XX1 = torch.meshgrid(lin_0, lin_1, indexing="ij")
            dist = (XX0**2 + XX1**2) ** 0.5
            radius = min(psf_size) / 2
            apodize_length = min(10, radius)
            self.apodize_mask = bump_function(
                dist, a=radius - apodize_length, b=apodize_length
            )
        else:
            self.apodize_mask = None

        pupil_size = (
            max(pupil_size[0], self.psf_size[0]),
            max(pupil_size[1], self.psf_size[1]),
        )
        self.pupil_size = pupil_size

        lin_x = torch.linspace(-0.5, 0.5, self.pupil_size[0], **self.factory_kwargs)
        self.step_rho = lin_x[1] - lin_x[0]

        # In order to avoid layover in Fourier convolution we need to zero pad and then extract a part of image
        # computed from pupil_size and psf_size
        self.pad_pre = (
            ceil((self.pupil_size[0] - self.psf_size[0]) / 2),
            ceil((self.pupil_size[1] - self.psf_size[1]) / 2),
        )
        self.pad_post = (
            floor((self.pupil_size[0] - self.psf_size[0]) / 2),
            floor((self.pupil_size[1] - self.psf_size[1]) / 2),
        )

        # Store the (n, m) pairs for Zernike evaluation — resolved once at init,
        # used at every _compute_psf call to rebuild Z on the correct fc-scaled grid.
        self._zernike_nm = []
        for index in self.zernike_index:
            if isinstance(index, int):
                n, m = Zernike.index_conversion(index, convention=index_convention)
            elif isinstance(index, tuple) and len(index) == 2:
                n, m = index
            else:
                raise ValueError(
                    f"Zernike index must be either int or tuple of (n,m), got {index!r}"
                )
            self._zernike_nm.append((n, m))

        self.to(device=device, dtype=dtype)

    def _build_pupil_grid(self, fc: torch.Tensor) -> tuple:
        r"""
        Build the fc-normalised Fourier-plane grid and Zernike tensor for a given fc.

        The Zernike polynomials :math:`z_k(\boldsymbol{\rho})` are defined on the unit
        disk, where :math:`\boldsymbol{\rho} = \mathbf{u} / f_c` and :math:`\mathbf{u}`
        lives on :math:`[-0.5, 0.5]^2`. Building both the indicator mask and ``Z`` on
        the same fc-normalised grid ensures they are consistent: the pupil boundary
        (``rho_normalised = 1``) always coincides with the edge of the Zernike support.

        :param torch.Tensor fc: scalar cutoff frequency, shape ``()``.  Only a single
            fc value is accepted; the caller is responsible for iterating over distinct
            fc values when needed (see :meth:`_compute_psf`).

        :return: ``(XX_fc, YY_fc, indicator_circ, Z)`` where

            - ``XX_fc``, ``YY_fc``: ``(H, W)`` grids on ``[-0.5/fc, 0.5/fc]^2``.
            - ``indicator_circ``: ``(H, W)`` pupil support mask.
            - ``Z``: ``(H, W, K)`` Zernike tensor evaluated on the unit disk.
        """
        lin_x = torch.linspace(-0.5, 0.5, self.pupil_size[0],
                               device=fc.device, dtype=fc.dtype)
        lin_y = torch.linspace(-0.5, 0.5, self.pupil_size[1],
                               device=fc.device, dtype=fc.dtype)
        # Normalised coordinates: rho=1 at the pupil edge regardless of fc
        XX_fc = lin_x[:, None] / fc   # (H, 1) / scalar → (H, 1)  [broadcast below]
        YY_fc = lin_y[None, :] / fc   # (1, W)

        rho = cart2pol(XX_fc, YY_fc)                              # (H, W)
        step = self.step_rho / (2.0 * fc)
        indicator_circ = bump_function(rho, 1.0 - step, b=step)  # (H, W)

        Z = torch.zeros(
            (self.pupil_size[0], self.pupil_size[1], self.n_zernike),
            device=fc.device, dtype=fc.dtype,
        )
        for k, (n, m) in enumerate(self._zernike_nm):
            Z[:, :, k] = Zernike.cartesian_evaluate(n, m, XX_fc, YY_fc)

        return indicator_circ, Z

    def step(
        self,
        batch_size: int = 1,
        coeff: torch.Tensor = None,
        angle: torch.Tensor = None,
        seed: int = None,
        fc: float | torch.Tensor | None = None,
        wavelengths: torch.Tensor | None = None,
        NA: float | None = None,
        pixel_size: float | None = None,
        **kwargs,
    ) -> dict:
        r"""
        Generate a batch of PSFs with a batch of Zernike coefficients.

        :param int batch_size: batch_size.
        :param torch.Tensor coeff: Zernike coefficients. Accepted shapes:

            - ``None`` (default): sampled randomly, producing ``(B, n_zernike)``
              (same coefficients shared across channels) when ``wavelengths`` is ``None``,
              or ``(B, C, n_zernike)`` (with chromatic perturbations) otherwise.
            - ``(B, n_zernike)``: one set of coefficients per batch element,
              shared across channels.
            - ``(B, C, n_zernike)``: independent coefficients per batch element
              **and** per channel, enabling wavelength-dependent aberrations.

        :param torch.Tensor angle: ``(batch_size,)`` angles in degrees for PSF rotation
            (default ``None``).
        :param int seed: seed for the random number generator.
        :param float or torch.Tensor fc: cutoff frequency ``(NA/emission_wavelength) *
            pixel_size``. All values must be in ``[0, 0.25]``. Accepted shapes:

            - ``None`` (default): uses ``self.fc`` set at construction (backward-compatible).
            - ``float``: same cutoff for all batch elements and channels.
            - ``(B,)`` tensor: one cutoff per batch element, shared across channels.
            - ``(B, C)`` tensor: one cutoff per batch element **and** per channel.

            Mutually exclusive with ``wavelengths``.
        :param torch.Tensor wavelengths: per-channel emission wavelengths in **nm**,
            shape ``(C,)``. When provided, ``fc`` is derived as
            ``NA * pixel_size / wavelengths`` and the base Zernike coefficients are
            rescaled by :math:`\lambda_{\text{ref}} / \lambda_c`. Mutually exclusive
            with ``fc``.
        :param float NA: numerical aperture override for this call. Falls back to
            ``self.NA`` when ``None``. Only used when ``wavelengths`` is provided.
        :param float pixel_size: pixel size override (nm) for this call. Falls back to
            ``self.pixel_size`` when ``None``. Only used when ``wavelengths`` is provided.

        :return: dictionary with keys

            - ``filter``: tensor ``(batch_size, num_channels, H, W)`` of PSFs.
            - ``coeff``: Zernike coefficients, shape ``(B, n_zernike)`` or ``(B, C, n_zernike)``.
            - ``pupil``: pupil function, shape ``(B, C_eff, H_pupil, W_pupil)``.
            - ``angle``: random rotation angles in degrees (only if ``random_rotate``).
        """
        self.rng_manual_seed(seed)

        # ------------------------------------------------------------------
        # Resolve NA / pixel_size (per-call override or fall back to self)
        # ------------------------------------------------------------------
        NA = NA if NA is not None else self.NA
        pixel_size = pixel_size if pixel_size is not None else self.pixel_size

        # ------------------------------------------------------------------
        # Mutually exclusive fc / wavelengths paths
        # ------------------------------------------------------------------
        if fc is not None and wavelengths is not None:
            raise ValueError("fc and wavelengths are mutually exclusive.")

        if wavelengths is not None:
            # Derive per-channel fc from physical parameters
            wavelengths = torch.as_tensor(
                wavelengths, device=self.step_rho.device, dtype=self.step_rho.dtype
            )  # (C,)

            assert wavelengths.shape[1] == self.num_channels, (
                f"Expected {self.num_channels} emission wavelengths, "
                f"got {wavelengths.shape[1]}."
            )

            if wavelengths.shape[0] != batch_size:
                raise ValueError(
                    f"Expected {batch_size} emission wavelengths, "
                    f"got {wavelengths.shape[0]}."
                )

            fc = NA * pixel_size / wavelengths  # (B, C,) → will become (1, C) in _parse_fc
            if torch.any(fc > 0.25):
                raise ValueError(
                    f"NA={NA}, pixel_size={pixel_size} nm and the provided wavelengths "
                    f"yield fc values up to {fc.max().item():.4f} > 0.25 (Nyquist limit). "
                    "Reduce NA, increase pixel_size, or use longer wavelengths."
                )
            # Generate base coeff (B, K) + delta (B, C, K) if coeff not supplied
            if coeff is None:
                coeff_base, coeff_delta = self.generate_coeff(
                    batch_size, num_channels=self.num_channels
                )
            else:
                # User supplied coeff: treat as base (B, K) or full (B, C, K)
                if coeff.ndim == 2:
                    coeff_base, coeff_delta = coeff, 0.0
                else:
                    # (B, C, K) already fully specified — skip rescaling
                    coeff_base = coeff_delta = None

            if coeff_base is not None:
                # Apply 1/lambda rescaling: (B, K) * (C,) → (B, C, K)
                lambda_scale = self.lambda_ref / wavelengths          # (C,)
                coeff = (
                    coeff_base[:, None, :] * lambda_scale[:, :, None]
                    + coeff_delta
                )   # (B, C, K)

        else:
            # Plain fc path (or backward-compat default)
            if coeff is None:
                coeff, _ = self.generate_coeff(batch_size, num_channels=1)

            if fc is not None:
                fc_check = torch.as_tensor(fc, dtype=self.step_rho.dtype)
                if torch.any(fc_check > 0.25):
                    raise ValueError(
                        f"At least one fc value ({fc_check.max().item():.4f}) exceeds "
                        "0.25 (Nyquist limit)."
                    )

        # ------------------------------------------------------------------
        # Normalise coeff → (B*C_eff, K) and fc → (B, C, 1, 1)
        # ------------------------------------------------------------------
        coeff_flat, n_coeff_channels = self._parse_coeff(coeff, batch_size)

        if fc is None:
            # Backward-compatible single-fc path: use self.fc for everything
            fc_tensor = torch.tensor(
                [[[[self.fc]]]], device=self.step_rho.device, dtype=self.step_rho.dtype
            )
        else:
            fc_tensor = self._parse_fc(fc, batch_size)  # (B, C_fc, 1, 1)

        n_fc_channels = fc_tensor.shape[1]

        # ------------------------------------------------------------------
        # Consistency check & expand so both sides see the same B*C
        # ------------------------------------------------------------------
        n_channels = max(n_coeff_channels, n_fc_channels)

        if n_coeff_channels != n_fc_channels:
            if n_coeff_channels == 1:
                coeff_flat = coeff_flat.repeat_interleave(n_fc_channels, dim=0)
            elif n_fc_channels == 1:
                fc_tensor = fc_tensor.expand(batch_size, n_coeff_channels, 1, 1)
            else:
                raise ValueError(
                    f"coeff implies {n_coeff_channels} channel(s) and fc implies "
                    f"{n_fc_channels} channel(s); they must match or one must be 1."
                )

        # fc_tensor is now (B, C, 1, 1); flatten to (B*C,) for iteration
        fc_flat = fc_tensor.reshape(batch_size * n_channels)  # (B*C,)

        # ------------------------------------------------------------------
        # Build pupil grid(s) and run FFT
        #
        # We group elements that share the same fc to avoid redundant grid
        # builds. In the common cases (scalar fc, or wavelengths path where
        # every batch element uses the same C fc values) this means exactly
        # C unique builds regardless of batch size.
        # ------------------------------------------------------------------
        unique_fc, inverse_idx = torch.unique(fc_flat, return_inverse=True)

        psf_flat_list  = [None] * (batch_size * n_channels)
        pupil_flat_list = [None] * (batch_size * n_channels)

        for i, fc_val in enumerate(unique_fc):
            indicator_circ, Z = self._build_pupil_grid(fc_val)  # (H,W), (H,W,K)
            mask = inverse_idx == i                              # which B*C elements
            psf_i, pupil_i = self._compute_psf(
                coeff_flat[mask], indicator_circ, Z
            )  # (n_i, 1, H_psf, W_psf)
            for j, orig_idx in enumerate(mask.nonzero(as_tuple=True)[0].tolist()):
                psf_flat_list[orig_idx]   = psf_i[j]
                pupil_flat_list[orig_idx] = pupil_i[j]

        psf_flat   = torch.stack(psf_flat_list,   dim=0)  # (B*C, 1, H_psf, W_psf)
        pupil_flat = torch.stack(pupil_flat_list, dim=0)  # (B*C, H_pup, W_pup)

        # Unfold back to (B, C, H_psf, W_psf)
        H_psf, W_psf = psf_flat.shape[-2], psf_flat.shape[-1]
        psf_filter = psf_flat.reshape(batch_size, n_channels, H_psf, W_psf)
        pupil      = pupil_flat.reshape(batch_size, n_channels, *pupil_flat.shape[-2:])

        # ------------------------------------------------------------------
        # Post-processing
        # ------------------------------------------------------------------
        if self.random_rotate:
            if angle is None:
                angle = self.generate_angles(batch_size)
            psf_filter = rotate_via_shear(psf_filter, angle)

        if self.apodize:
            psf_filter = self.apodize_mask * psf_filter

        psf_filter = psf_filter / torch.sum(psf_filter, dim=(-1, -2), keepdim=True)

        params = {
            "filter": psf_filter.expand(-1, self.shape[0], -1, -1),
            "coeff": coeff,
            "pupil": pupil,
        }
        if self.random_rotate:
            params["angle"] = angle
        return params

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_psf(
        self,
        coeff: torch.Tensor,
        indicator_circ: torch.Tensor,
        Z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Core PSF computation for a single fc value.

        :param torch.Tensor coeff: ``(B_eff, K)`` Zernike coefficients.
        :param torch.Tensor indicator_circ: ``(H, W)`` pupil support mask.
        :param torch.Tensor Z: ``(H, W, K)`` Zernike polynomials on the fc-normalised grid.

        :return: ``(psf, pupil)`` where ``psf`` is ``(B_eff, 1, H_psf, W_psf)``
            and ``pupil`` is ``(B_eff, H_pupil, W_pupil)``.
        """
        # (H, W, K) @ (K, B_eff) → (H, W, B_eff) → (B_eff, H, W)
        pupil = (Z @ coeff[:, : self.n_zernike].T).permute(2, 0, 1)
        pupil = torch.exp(-2.0j * torch.pi * pupil) * indicator_circ  # (B_eff, H, W)

        psf = torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(pupil)))
        psf = psf.abs().pow(2)

        psf = psf[
            :,
            self.pad_pre[0] : self.pupil_size[0] - self.pad_post[0],
            self.pad_pre[1] : self.pupil_size[1] - self.pad_post[1],
        ].unsqueeze(1)  # (B_eff, 1, H_psf, W_psf)

        return psf, pupil

    def _parse_coeff(
        self, coeff: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, int]:
        r"""
        Normalise ``coeff`` to a flat ``(B * n_channels, n_zernike)`` tensor.

        :param torch.Tensor coeff: ``(B, K)`` or ``(B, C, K)``.
        :param int batch_size: expected batch size ``B``.

        :return: ``(coeff_flat, n_channels)`` where ``coeff_flat`` is
            ``(B * n_channels, K)`` and ``n_channels`` is 1 or C.
        """
        if coeff.ndim == 2:
            if coeff.shape[0] != batch_size:
                raise ValueError(
                    f"coeff has shape {tuple(coeff.shape)} but batch_size={batch_size}."
                )
            return coeff, 1
        if coeff.ndim == 3:
            B, C, K = coeff.shape
            if B != batch_size:
                raise ValueError(
                    f"coeff has shape {tuple(coeff.shape)} but batch_size={batch_size}."
                )
            return coeff.reshape(B * C, K), C
        raise ValueError(
            f"coeff must be 2-D (B, K) or 3-D (B, C, K). Got shape {tuple(coeff.shape)}."
        )

    def _parse_fc(self, fc: float | torch.Tensor, batch_size: int) -> torch.Tensor:
        r"""
        Normalise ``fc`` to a ``(B, C, 1, 1)`` tensor on the correct device/dtype.

        Accepted shapes:

        - scalar ``float``: ``(1, 1, 1, 1)``.
        - ``(B,)`` tensor: ``(B, 1, 1, 1)``.
        - ``(B, C)`` or ``(C,)`` tensor: ``(B, C, 1, 1)``.
        """
        dev, dt = self.step_rho.device, self.step_rho.dtype

        if isinstance(fc, (int, float)):
            return torch.tensor([[[[fc]]]], device=dev, dtype=dt)

        if not isinstance(fc, torch.Tensor):
            fc = torch.tensor(fc, device=dev, dtype=dt)

        fc = fc.to(device=dev, dtype=dt)

        if fc.ndim == 0:
            return fc.reshape(1, 1, 1, 1)
        if fc.ndim == 1:
            C = fc.shape[0]
            if C == batch_size:
                # Ambiguous: treat as (B,) — one fc per batch, shared across channels
                return fc.reshape(batch_size, 1, 1, 1)
            else:
                # Treat as (C,) — one fc per channel, shared across batch
                return fc.reshape(1, C, 1, 1)
        if fc.ndim == 2:
            B, C = fc.shape
            if B != batch_size:
                raise ValueError(
                    f"fc has shape {fc.shape} but batch_size={batch_size}. "
                    "A 2-D fc tensor must have shape (batch_size, num_channels)."
                )
            return fc.reshape(B, C, 1, 1)

        raise ValueError(
            f"fc must be a scalar, 1-D, or 2-D tensor. Got shape {tuple(fc.shape)}."
        )

    @property
    def zernike_polynomials(self) -> list[str]:
        r"""
        List of Zernike polynomials used in the decomposition, with the corresponding aberration if available.
        """
        return [Zernike.get_name(n, m) for n, m in self._zernike_nm]

    def generate_coeff(self, batch_size: int, num_channels: int = 1) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""
        Generate random Zernike coefficients.

        Base coefficients are sampled uniformly in
        ``[-max_zernike_amplitude/2, max_zernike_amplitude/2]`` (in waves at
        ``lambda_ref``). When ``num_channels > 1``, per-channel perturbations
        ``coeff_delta`` of shape ``(B, C, K)`` are also returned, sampled from
        a zero-mean Gaussian with std
        ``zernike_perturbation_amplitude * max_zernike_amplitude``.

        :param int batch_size: number of independent aberration realisations.
        :param int num_channels: number of spectral channels. When ``> 1``,
            also returns ``coeff_delta``.

        :return: ``(coeff_base, coeff_delta)`` where

            - ``coeff_base``: ``(B, K)`` base coefficients in waves at ``lambda_ref``.
            - ``coeff_delta``: ``(B, C, K)`` per-channel perturbations, or ``None``
              when ``num_channels == 1``.
        """
        coeff_base = (
            torch.rand(
                (batch_size, self.n_zernike),
                generator=self.rng,
                **self.factory_kwargs,
            )
            - 0.5
        ) * self.max_zernike_amplitude

        if num_channels > 1:
            coeff_delta = (
                torch.randn(
                    (batch_size, num_channels, self.n_zernike),
                    generator=self.rng,
                    **self.factory_kwargs,
                )
                * self.zernike_perturbation_amplitude
                * self.max_zernike_amplitude
            )
        else:
            coeff_delta = None

        return coeff_base, coeff_delta

    def generate_angles(self, batch_size: int) -> torch.Tensor:
        r"""
        Generate random rotation angles for the PSF.

        :param int batch_size: batch_size.
        :return: ``(batch_size,)`` angles in degrees.
        """
        return torch.rand(batch_size, generator=self.rng, **self.factory_kwargs) * 360


def cart2pol(x, y):
    r"""
    Cartesian to polar coordinates

    :param torch.Tensor x: x coordinates
    :param torch.Tensor y: y coordinates

    :return: rho of torch.Tensor of radius
    :rtype: tuple
    """

    rho = torch.sqrt(x**2 + y**2)
    return rho

def bump_function(x, a=1.0, b=1.0):
    r"""
    Defines a function which is 1 on the interval [-a,a]
    and goes to 0 smoothly on [-a-b,-a]U[a,a+b] using a bump function.
    For the discretization of indicator functions, we advise b=1, so that
    a=0, b=1 yields a bump.

    Supports arbitrary tensor shapes and broadcasting between x, a, and b.

    :param torch.Tensor x: tensor of arbitrary size.
    :param float or torch.Tensor a: radius (default is 1).
    :param float or torch.Tensor b: interval on which the function goes to 0 (default is 1).

    :return: the bump function sampled at points x.
    :rtype: torch.Tensor

    :Examples:

    >>> import deepinv as dinv
    >>> x = torch.linspace(-15, 15, 31)
    >>> X, Y = torch.meshgrid(x, x, indexing = 'ij')
    >>> R = torch.sqrt(X**2 + Y**2)
    >>> Z = bump_function(R, 3, 1)
    >>> Z = Z / torch.sum(Z)
    """
    abs_x = torch.abs(x)

    # Transition value: exp(-1 / (1 - t^2)) / exp(-1), where t = (|x| - a) / b
    # Clamped to avoid NaN from sqrt outside [a, a+b]: safe_t is always in [0, 1)
    t = (abs_x - a) / b
    safe_t = t.clamp(0.0, 1.0 - 1e-6)
    transition = torch.exp(-1.0 / (1.0 - safe_t ** 2)) / np.exp(-1.0)

    return torch.where(abs_x <= a,
               torch.ones_like(x),
               torch.where(abs_x < a + b, transition, torch.zeros_like(x))
           )

class ProductConvolutionBlurGenerator(PhysicsGenerator):
    r"""
    Generates parameters of space-varying blurs.

    Parameters generated:

    -`'filters'`: tensor of shape ``(B, C, n_eigen_psf, psf_size, psf_size)``
    - 'multipliers': tensor of shape ``(B, C, n_eigen_psf, H, W)``

    See :class:`deepinv.physics.SpaceVaryingBlur` for more details.

    :param deepinv.physics.generator.PSFGenerator psf_generator: A PSF generator, such as :class:`motion blur <deepinv.physics.generator.MotionBlurGenerator>` or
        :class:`diffraction blur generator <deepinv.physics.generator.DiffractionBlurGenerator>`.
    :param tuple img_size: image size ``(H,W)``.
    :param int n_eigen_psf: each PSF in the field of view will be a linear combination of ``n_eigen_psf`` eigen PSF grids.
        Defaults to 10.
    :param tuple spacing: steps between the PSF grids used for interpolation (defaults ``(H//8, W//8)``).
    :param str padding: boundary conditions in (options = ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``).
        Defaults to ``'valid'``.

    |sep|

    :Examples:

    >>> from deepinv.physics.generator import DiffractionBlurGenerator
    >>> from deepinv.physics.generator import ProductConvolutionBlurGenerator
    >>> psf_size = 7
    >>> psf_generator = DiffractionBlurGenerator((psf_size, psf_size), fc=0.25)
    >>> pc_generator = ProductConvolutionBlurGenerator(psf_generator, img_size=(64, 64), n_eigen_psf=8)
    >>> params = pc_generator.step(1)
    >>> print(params.keys())
    dict_keys(['filters', 'multipliers'])

    """

    def __init__(
        self,
        psf_generator: PSFGenerator,
        img_size: tuple[int],
        n_eigen_psf: int = 10,
        spacing: tuple[int] = None,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__(device=device, **kwargs)
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(spacing, int):
            spacing = (spacing, spacing)

        self.psf_generator = psf_generator
        self.img_size = img_size
        self.n_eigen_psf = n_eigen_psf
        self.spacing = (
            spacing
            if spacing is not None
            else (self.img_size[0] // 8, self.img_size[1] // 8)
        )

        self.n_psf_prid = (self.img_size[0] // self.spacing[0]) * (
            self.img_size[1] // self.spacing[1]
        )
        if self.n_psf_prid < self.n_eigen_psf:  # pragma: no cover
            raise ValueError(
                f"n_eigen_psf={n_eigen_psf} must be smaller than the number of PSF grid points = {self.n_psf_prid}"
            )

        # Interpolating the psf_grid coefficients with thin plate splines
        T0 = torch.linspace(
            0, 1, self.img_size[0] // self.spacing[0], **self.factory_kwargs
        )
        T1 = torch.linspace(
            0, 1, self.img_size[1] // self.spacing[1], **self.factory_kwargs
        )
        yy, xx = torch.meshgrid(T0, T1, indexing="ij")
        self.X = torch.stack((yy.flatten(), xx.flatten()), dim=1)

        T0 = torch.linspace(0, 1, self.img_size[0], **self.factory_kwargs)
        T1 = torch.linspace(0, 1, self.img_size[1], **self.factory_kwargs)
        yy, xx = torch.meshgrid(T0, T1, indexing="ij")
        self.XX = torch.stack((yy.flatten(), xx.flatten()), dim=1)

        self.tps = ThinPlateSpline(0.0, **self.factory_kwargs)

    def step(self, batch_size: int = 1, seed: int = None, **kwargs):
        r"""
        Generates a random set of filters and multipliers for space-varying blurs.

        :param int batch_size: number of space-varying blur parameters to generate.
        :param int seed: the seed for the random number generator.

        :returns: a dictionary containing filters, multipliers and paddings.
            filters: a tensor of shape (B, C, n_eigen_psf, psf_size, psf_size).
            multipliers: a tensor of shape (B, C, n_eigen_psf, H, W).
        """
        self.rng_manual_seed(seed)
        self.psf_generator.rng_manual_seed(seed)

        # Generating psf_grid on a grid
        psf_grid = self.psf_generator.step(self.n_psf_prid * batch_size)["filter"]
        psf_size = psf_grid.shape[-2:]
        channels = psf_grid.shape[1]
        psf_grid = psf_grid.view(
            batch_size, self.n_psf_prid, channels, *psf_size
        )  # B x n_psf_prid x C x psf_size x psf_size

        # Computing the eigen-PSF
        psf_grid = psf_grid.flatten(-2, -1).transpose(
            1, 2
        )  # B x C x n_psf_prid x (psf_size*psf_size)
        _, _, V = torch.linalg.svd(psf_grid, full_matrices=False)
        n_eigen_chosen = min(self.n_eigen_psf, V.size(-2))
        V = V[..., :n_eigen_chosen, :]  # B x C x n_eigen_psf x (psf_size*psf_size)
        coeffs = torch.matmul(
            psf_grid, V.transpose(-1, -2)
        )  # B x C x n_psf_prid x n_eigen_psf
        eigen_psf = V.reshape(V.size(0), channels, n_eigen_chosen, *psf_size)

        # compute multipliers by interpolating the coeffs with thin-plate splines
        self.tps.fit(self.X, coeffs)
        w = self.tps.transform(self.XX).transpose(-1, -2)
        w = w.reshape(w.size(0), channels, n_eigen_chosen, *self.img_size)

        # Ending
        params_blur = {"filters": eigen_psf, "multipliers": w}
        return params_blur


class DiffractionBlurGenerator3D(PSFGenerator):
    r"""
    3D diffraction limited kernels using Zernike decomposition of the phase mask (Fresnel/Fraunhoffer diffraction theory).

    This method simulates the propagation of the wavefront from the pupil plane
    (frequency domain) to multiple defocus planes in the image space.
    The pupil function is constructed using a Zernike polynomial decomposition
    of the wavefront aberrations, see :class:`deepinv.physics.generator.DiffractionBlurGenerator` for more details.

    At each depth :math:`z`, the pupil function is modulated by a phase term
    corresponding to the axial wave vector :math:`k_z`,
    which is derived from the dispersion relation of light in free space.

    .. math::

        k_z = \sqrt{k_{\text{total}}^2 - k_{\text{lateral}}^2}

    where :math:`k_{\text{total}}` is the total wave number (`kb`) and :math:`k_{\text{lateral}}` is the lateral wave vector component.
    The pupil function at depth :math:`z` is given by:

    .. math::

        P(x, y, z) = P(x, y, 0) \cdot \exp \left( - i 2 \pi \cdot k_z \cdot z \right)

    And the depth planes are sampled according to the `stepz_pixel` parameter, which defines the ratio between the physical size of the :math:`z` direction to that in the :math:`x/y` direction of the voxels in the 3D image.


    The 3D PSF is then computed by square modulus of Fourier transform of the modulated pupil function at each depth plane, followed by normalization across the spatial dimensions.


    .. note::

        This class uses :class:`deepinv.physics.generator.DiffractionBlurGenerator` under the hood to generate the pupil function at :math:`z=0`. Refer to its documentation for more details.

    :param tuple psf_size: give in the order `(depth, height, width)` the size of the PSF to generate.
    :param int num_channels: number of channels. Default to `1`.
    :param tuple[int, ...], tuple[tuple[int, int], ...] zernike_index: activated Zernike coefficients in the following `index_convention` convention.
        It can be either:

            - a tuple of `int` corresponding to the Noll or ANSI indices, in which case the `index_convention` parameter is required to interpret them correctly.
            - a tuple of `tuple[int, int]` corresponding to the standard radial-angular indexing :math:`(n,m)`. In this case, the `index_convention` parameter is ignored.

        Defaults to ``(4, 5, 6, 7, 8, 9, 10, 11)``, correspond to radial order `n` from 2 to 3 (included) and the spherical aberration.
        These correspond to the following aberrations: defocus, astigmatism, coma, trefoil and spherical aberration.
    :param float fc: cutoff frequency `(NA/emission_wavelength) * pixel_size`. Should be in `[0, 1/4]` to respect Shannon, defaults to `0.2`.
    :param float kb: wave number `(NI/emission_wavelength) * pixel_size` or `(NA/NI) * fc`. Must be greater than `fc`. Defaults to `0.3`.
    :param float max_zernike_amplitude: maximum amplitude of Zernike coefficients. Defaults to 0.15.
    :param tuple[int] pupil_size: pixel size to synthesize the super-resolved pupil. The higher the more precise, defaults to `(512, 512)`.
        If an `int` is given, a square pupil is considered.
    :param bool apodize: whether to apodize the PSF to reduce ringing effects. Defaults to `False`.
    :param bool random_rotate: whether to randomly rotate the PSF in the xy plane. Defaults to `False`.
    :param float stepz_pixel: Ratio between the physical size of the :math:`z` direction to that in the :math:`x/y` direction of the voxels in the 3D image.
        Defaults to `1.0`.
    :param str index_convention: convention for the Zernike indices, either ``'noll'`` (default) or ``'ansi'``.
    :param torch.Generator rng: random number generator (default to `None`).
    :param str device: device (default to ``'cpu'``).
    :param type dtype: data type (default to `torch.float32`).
    :param kwargs: additional arguments for :class:`deepinv.physics.generator.DiffractionBlurGenerator`.

    .. note::

        - `NA`: numerical aperture,
        - `NI`: refraction index of the immersion medium,
        - `emission_wavelength`: wavelength of the light,
        - `pixel_size`: physical size of the pixels in the :math:`xy` plane in the same unit as `emission_wavelength`.

    |sep|

    :Examples:

    >>> import torch
    >>> from deepinv.physics.generator import DiffractionBlurGenerator3D
    >>> generator = DiffractionBlurGenerator3D((21, 51, 51), stepz_pixel = 2, zernike_index=(3,), index_convention='ansi')
    >>> print(generator.zernike_polynomials) # list of Zernike polynomials used
    ['Zernike(n = 2, m = -2) -- Oblique Astigmatism']
    >>> dict = generator.step()
    >>> filter = dict['filter']
    >>> print(filter.shape)
    torch.Size([1, 1, 21, 51, 51])
    >>> batch_size = 2
    >>> n_zernike = len(generator.generator2d.zernike_index)
    >>> dict = generator.step(batch_size=batch_size, coeff=0.1 * torch.rand(batch_size, n_zernike))
    >>> dict.keys()
    dict_keys(['filter', 'pupil', 'coeff'])


    """

    @_deprecated_alias(list_param="zernike_index")
    def __init__(
        self,
        psf_size: tuple,
        num_channels: int = 1,
        zernike_index: tuple[int, ...] | tuple[tuple[int, int], ...] = tuple(
            range(4, 12)
        ),
        fc: float = 0.2,
        kb: float = 0.25,
        max_zernike_amplitude: float = 0.15,
        pupil_size: tuple[int] = (512, 512),
        apodize: bool = False,
        random_rotate: bool = False,
        stepz_pixel: float = 1.0,
        index_convention: str = "noll",
        rng: torch.Generator = None,
        device: str = "cpu",
        dtype: type = torch.float32,
        **kwargs,
    ):
        if len(psf_size) != 3:
            raise ValueError(
                "You should provide a tuple of len == 3 to generate 3D PSFs."
            )

        super().__init__(
            psf_size=psf_size,
            num_channels=num_channels,
            device=device,
            dtype=dtype,
            rng=rng,
        )

        self.generator2d = DiffractionBlurGenerator(
            psf_size=psf_size[1:],
            num_channels=num_channels,
            zernike_index=zernike_index,
            fc=fc,
            max_zernike_amplitude=max_zernike_amplitude,
            pupil_size=pupil_size,
            apodize=apodize,
            device=device,
            dtype=dtype,
            rng=rng,
            index_convention=index_convention,
            **kwargs,
        )
        self.apodize = apodize
        self.random_rotate = random_rotate
        self.stepz_pixel = stepz_pixel
        self.kb = kb
        self.psf_size = psf_size
        self.nzs = psf_size[0]
        self.fc = fc
        self.zernike_index = zernike_index
        self.n_zernike = len(self.zernike_index)
        self._defocus = (
            torch.linspace(
                -self.nzs / 2, self.nzs / 2, self.nzs, device=device, dtype=dtype
            )[:, None, None]
            * self.stepz_pixel
        )
        self.to(device=device, dtype=dtype)

    def step(
        self,
        batch_size: int = 1,
        coeff: torch.Tensor = None,
        angle: torch.Tensor = None,
        seed: int = None,
        **kwargs,
    ) -> dict:
        r"""
        Generate a batch of PSF with a batch of Zernike coefficients

        :param int batch_size: number of PSFs to generate.
        :param torch.Tensor coeff: tensor of size (batch_size x len(zernike_index)) containing the Zernike coefficients.
            If `None`, random coefficients are generated.
        :param int seed: the seed for the random number generator.

        :return: dictionary with keys

            - `filter`: tensor of size `(batch_size x num_channels x psf_size[0] x psf_size[1])` batch of PSFs,
            - `pupil`: the pupil function,
            - `coeff`: list of sampled Zernike coefficients in this realization,
            - `angle`: the random rotation angles in degrees if `random_rotate` is `True`, nothing otherwise.
        """
        gen_dict = self.generator2d.step(
            batch_size=batch_size, coeff=coeff, seed=seed, **kwargs
        )

        pupil = gen_dict["pupil"]
        d = ((self.kb) ** 2 - (self.generator2d.rho * self.fc) ** 2 + 0j) ** 0.5

        propKer = torch.exp(-1j * 2 * torch.pi * d * self._defocus) + 0j
        p = pupil[:, None, ...] * propKer[None, ...]
        p = torch.nan_to_num(p, nan=0.0)
        pshift = torch.fft.fftshift(p, dim=(-2, -1))
        pfft = torch.fft.fft2(pshift, dim=(-2, -1))
        psf = torch.fft.ifftshift(pfft, dim=(-2, -1))
        psf = psf.abs().pow(2)

        psf = psf[
            :,
            :,
            self.generator2d.pad_pre[0] : self.generator2d.pupil_size[0]
            - self.generator2d.pad_post[0],
            self.generator2d.pad_pre[1] : self.generator2d.pupil_size[1]
            - self.generator2d.pad_post[1],
        ].unsqueeze(1)
        if self.random_rotate:
            from einops import rearrange

            if angle is None:
                angle = self.generator2d.generate_angles(batch_size)

            psf = rotate_via_shear(rearrange(psf, "b c d h w -> b (c d) h w"), angle)
            psf = rearrange(psf, "b (c d) h w -> b c d h w", d=self.psf_size[0])

        if self.apodize:
            psf = self.generator2d.apodize_mask[None, None, None] * psf
        psf = psf / torch.sum(psf, dim=(-3, -2, -1), keepdim=True)

        params = {
            "filter": psf.expand(-1, self.shape[0], -1, -1, -1),
            "pupil": pupil,
            "coeff": gen_dict["coeff"],
        }
        if self.random_rotate:
            params["angle"] = angle
        return params

    @property
    def zernike_polynomials(self) -> list[str]:
        r"""
        List of Zernike polynomials used in the decomposition, with the corresponding aberration if available.
        """
        return self.generator2d.zernike_polynomials


class ConfocalBlurGenerator3D(PSFGenerator):
    r"""
    Generates the 3D point spread function (PSF) of a confocal laser scanning microsope.

    :param tuple psf_size: give in the order `(depth, height, width)`
    :param int num_channels: number of channels. Default to `1`.
    :param tuple[int, ...], tuple[tuple[int, int], ...] zernike_index: activated Zernike coefficients in the following `index_convention` convention.
        It can be either:

            - a tuple of `int` corresponding to the Noll or ANSI indices, in which case the `index_convention` parameter is required to interpret them correctly.
            - a tuple of `tuple[int, int]` corresponding to the standard radial-angular indexing :math:`(n,m)`. In this case, the `index_convention` parameter is ignored.

        Defaults to ``(4, 5, 6, 7, 8, 9, 10, 11)``, correspond to radial order `n` from 2 to 3 (included) and the spherical aberration.
        These correspond to the following aberrations: defocus, astigmatism, coma, trefoil and spherical aberration.

    :param float NI: Refractive index of  the immersion medium. Defaults to `1.51` (oil),
    :param float NA: Numerical aperture. Should be less than NI. Defaults to `1.37`.
    :param float lambda_ill: Wavelength of the illumination light (fluorescence excitation). Defaults to `489e-9`.
    :param float lambda_coll: Wavelength of the collection light (fluorescence emission). Defaults to `395e-9`.
    :param float pixelsize_XY: Physical pixel size in the lateral direction (height, width). Defaults to `50e-9`.
    :param float pixelsize_Z:  Physical pixel size in the axial direction (depth). Defaults to `100e-9`.
    :param float pinhole_radius: Radius of pinhole in Airy units. Defaults to `1`.
    :param float max_zernike_amplitude: maximum amplitude of Zernike coefficients. Defaults to `0.1`.
    :param tuple[int] pupil_size: pixel size to synthesize the super-resolved pupil. The higher the more precise, defaults to `(512, 512)`.
            If an `int` is given, a square pupil is considered.
    :param str index_convention: convention for the Zernike indices, either ``'noll'`` (default) or ``'ansi'``.
    :param torch.Generator rng: random number generator (default to `None`).
    :param str device: device (default to `cpu`).
    :param type dtype: data type (default to `torch.float32`).

    |sep|

    :Examples:

    >>> import torch
    >>> from deepinv.physics.generator import ConfocalBlurGenerator3D
    >>> generator = ConfocalBlurGenerator3D((21, 51, 51), zernike_index=(3,))
    >>> print(generator.zernike_polynomials)
    ['Zernike(n = 1, m = -1) -- Vertical Tilt']
    >>> dict = generator.step()
    >>> filter = dict['filter']
    >>> print(filter.shape)
    torch.Size([1, 1, 21, 51, 51])
    >>> batch_size = 2
    >>> n_zernike = len(generator.generator_ill.generator2d.zernike_index)
    >>> dict = generator.step(batch_size=batch_size,
    ...                       coeff_ill = 0.1 * torch.rand(batch_size, n_zernike),
    ...                       coeff_coll = 0.1 * torch.rand(batch_size, n_zernike))
    >>> dict.keys()
    dict_keys(['filter', 'coeff_ill', 'coeff_coll'])

    """

    @_deprecated_alias(list_param="zernike_index")
    def __init__(
        self,
        psf_size: tuple,
        num_channels: int = 1,
        zernike_index: tuple[int, ...] | tuple[tuple[int, int], ...] = tuple(
            range(4, 12)
        ),
        NI: float = 1.51,
        NA: float = 1.37,
        lambda_ill: float = 489e-9,
        lambda_coll: float = 395e-9,
        pixelsize_XY: float = 50e-9,
        pixelsize_Z: float = 100e-9,
        pinhole_radius: float = 1,
        max_zernike_amplitude: float = 0.1,
        pupil_size: tuple[int] = (512, 512),
        index_convention: str = "noll",
        device: str = "cpu",
        dtype: type = torch.float32,
        rng: torch.Generator = None,
        **kwargs,
    ):
        if len(psf_size) != 3:
            raise ValueError(
                "You should provide a tuple of len == 3 to generate 3D PSFs."
            )

        super().__init__()

        self.fc_ill = (
            NA / lambda_ill
        ) * pixelsize_XY  # cutoff frequency for illumination
        self.kb_ill = (NI / lambda_ill) * pixelsize_XY  # wavenumber for illumination

        self.fc_coll = (
            NA / lambda_coll
        ) * pixelsize_XY  # cutoff freauency for collection
        # wavenumber for collection
        self.kb_coll = (NI / lambda_coll) * pixelsize_XY  # wavenumber for collection
        self.pinhole_radius = pinhole_radius
        self.pixelsize_XY = pixelsize_XY
        self.pixel_size_Z = pixelsize_Z

        self.lambda_ill = lambda_ill
        self.lambda_coll = lambda_coll
        self.NI = NI
        self.NA = NA

        # Initialize generator for the Illumniation PSF
        self.generator_ill = DiffractionBlurGenerator3D(
            psf_size=psf_size,
            num_channels=num_channels,
            fc=self.fc_ill,
            kb=self.kb_ill,
            stepz_pixel=int(pixelsize_Z / pixelsize_XY),
            zernike_index=zernike_index,
            max_zernike_amplitude=max_zernike_amplitude,
            pupil_size=pupil_size,
            index_convention=index_convention,
            rng=rng,
            device=device,
            dtype=dtype,
        )

        # Initialize generator for the Collection PSF
        self.generator_coll = DiffractionBlurGenerator3D(
            psf_size=psf_size,
            num_channels=num_channels,
            fc=self.fc_coll,
            kb=self.kb_coll,
            stepz_pixel=int(pixelsize_Z / pixelsize_XY),
            zernike_index=zernike_index,
            max_zernike_amplitude=max_zernike_amplitude,
            pupil_size=pupil_size,
            index_convention=index_convention,
            rng=rng,
            device=device,
            dtype=dtype,
        )
        self.to(device=device, dtype=dtype)

    def step(
        self,
        batch_size: int = 1,
        coeff_ill: torch.Tensor = None,
        coeff_coll: torch.Tensor = None,
        **kwargs,
    ) -> dict:
        r"""
        Generate a batch of 3D confocal PSF with a batch of Zernike coefficients
        for illumination and collection

        :param int batch_size: number of PSFs to generate.
        :param torch.Tensor coeff_ill: tensor of size `batch_size x len(zernike_index)` containing the Zernike coefficients for illumination.
            If `None`, random coefficients are generated.
        :param torch.Tensor coeff_coll: tensor of size `batch_size x len(zernike_index)` containing the Zernike coefficients for collection.
            If `None`, random coefficients are generated.

        :return: dictionary with keys

            - `filter`: tensor of size `batch_size x num_channels x psf_size[0] x psf_size[1]` batch of PSFs,
            - `coeff_ill`: list of sampled Zernike coefficients in this realization of illumination,
            - `coeff_coll`: list of sampled Zernike coefficients in this realization of collection,

        """
        dict_ill = self.generator_ill.step(
            batch_size=batch_size, coeff=coeff_ill
        )  # generate illumuinition PSF
        psf_ill = dict_ill["filter"]
        coeff_ill = dict_ill["coeff"]
        dict_coll = self.generator_coll.step(
            batch_size=batch_size, coeff=coeff_coll
        )  # generate collection PSF
        psf_coll = dict_coll["filter"]
        coeff_coll = dict_coll["coeff"]

        # convolution of the collection PSF by pinhole
        # 1. Define the pinhole D
        airy_unit = 0.61 * self.lambda_coll / self.NA
        PH_radius = self.pinhole_radius * airy_unit
        lin_x = torch.linspace(
            -1.5 * PH_radius,
            1.5 * PH_radius,
            int(3 * PH_radius / self.pixelsize_XY),
            **self.factory_kwargs,
        )
        lin_y = torch.linspace(
            -1.5 * PH_radius,
            1.5 * PH_radius,
            int(3 * PH_radius / self.pixelsize_XY),
            **self.factory_kwargs,
        )
        PH_step_rho = lin_x[1] - lin_x[0]
        # The plane is discretized on [-1.5 * r_pinhole, 1.5 * r_pinhole] x  [-1.5 * r_pinhole, 1.5 * r_pinhole]
        XX, YY = torch.meshgrid(lin_x, lin_y, indexing="ij")
        PH_rho = torch.sqrt(XX**2 + YY**2)  # Cartesian coordinates
        D = bump_function(
            PH_rho, PH_radius - PH_step_rho / 2, b=PH_step_rho / 2
        )  # D(r) in equation

        # 2. Apply 2D convolution in all z planes
        psf_coll_convolved = torch.zeros(psf_coll.shape, **self.factory_kwargs)
        for i in range(psf_coll.shape[-3]):
            psf_coll_convolved[:, :, i] = conv2d(
                psf_coll[:, :, i], filter=D[None, None], padding="constant"
            )

        psf_confocal = psf_ill * psf_coll_convolved  # final PSF of confocal microscope

        psf = psf_confocal / torch.sum(psf_confocal, dim=(-3, -2, -1), keepdim=True)

        return {
            "filter": psf.expand(-1, self.shape[0], -1, -1, -1),
            "coeff_ill": coeff_ill,
            "coeff_coll": coeff_coll,
        }

    @property
    def zernike_polynomials(self) -> list[str]:
        r"""
        List of Zernike polynomials used in the decomposition, with the corresponding aberration if available.
        """
        return self.generator_ill.zernike_polynomials


class TiledBlurGenerator(TiledMixin2d, PSFGenerator):
    r"""
    Generates parameters of the :class:`deepinv.physics.TiledSpaceVaryingBlur` operator.
    The image is divided into overlapping patches, each local patch is convolved with a different PSF.

    This generates a dict with key `'filter'`, which is tensor of shape `(B, C, K, psf_size, psf_size)`
    where `K` is the number of patches in which the image is divided.
    It is computed based on the `patch_size`, `stride` and the given `img_size` during the `step()` function call.

    :param deepinv.physics.generator.PSFGenerator psf_generator: A PSF generator, such as :class:`motion blur <deepinv.physics.generator.MotionBlurGenerator>` or :class:`diffraction blur generator <deepinv.physics.generator.DiffractionBlurGenerator>`.

    :param int | tuple[int, int] patch_size: size of the patches (height, width) in which the image is divided.
    :param int | tuple[int, int] stride: stride between adjacent patches (height, width). Defaults to `patch_size`.
    """

    def __init__(
        self,
        psf_generator: PSFGenerator,
        patch_size: int | tuple[int, int],
        stride: int | tuple[int, int] = None,
        rng: torch.Generator = None,
        device: str | torch.device = "cpu",
        **kwargs,
    ):
        super().__init__(
            patch_size=patch_size, stride=stride, rng=rng, device=device, **kwargs
        )
        self.psf_generator = psf_generator
        self.psf_size = psf_generator.psf_size

    def step(
        self,
        batch_size: int = 1,
        img_size: int | tuple[int, int] = None,
        seed: int | None = None,
        **kwargs,
    ) -> dict:
        r"""
        Generates a random set of filters for the tiled space-varying blur.

        :param int batch_size: batch size of the PSF parameters to generate. Should be equal to the batch size of the images to be blurred.
        :param int | tuple[int, int] img_size: size of the image to be blurred (height, width).
        :param int | None seed: the seed for the random number generator.

        :returns: a dictionary containing filters, with key:

            - `filters`: a tensor of shape `(B, C, K, psf_size, psf_size)`, where `K` is the number of patches in which the image is divided.

        """

        num_patches = self.get_num_patches(img_size=img_size)
        num_patches = num_patches[0] * num_patches[1]

        params = self.psf_generator.step(
            batch_size=batch_size * num_patches, seed=seed, **kwargs
        )
        psf = (
            params["filter"]
            .view(batch_size, num_patches, -1, *self.psf_size)
            .transpose(1, 2)
        )  # B x C x num_patches x psf_size x psf_size

        return dict(filters=psf)
