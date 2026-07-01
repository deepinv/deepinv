from __future__ import annotations
import torch
import numpy as np
from math import ceil, floor
from deepinv.physics.generator import PhysicsGenerator
from deepinv.physics.functional import gaussian_blur, random_uniform
from deepinv.physics.functional.hist import histogramdd
from deepinv.physics.functional.convolution import conv2d
from deepinv.physics.functional.interp import ThinPlateSpline
from deepinv.utils.decorators import _deprecated_alias, _deprecated_argument
from deepinv.transform.rotate import rotate_via_shear
from deepinv.utils.mixins import TiledMixin2d
from deepinv.utils._internal import _check_pairwise_leq, _as_sequence
from .zernike import Zernike


class PSFGenerator(PhysicsGenerator):
    r"""
    Base class for generating Point Spread Functions (PSFs).


    :param tuple psf_size: the shape of the generated PSF in 2D
        ``(kernel_size, kernel_size)``. If an `int` is given, it will be used for both dimensions.
    """

    @_deprecated_argument("num_channels")
    def __init__(
        self,
        psf_size: tuple[int] = (31, 31),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(psf_size, int):
            psf_size = (psf_size, psf_size)

        self.shape = psf_size
        self.psf_size = psf_size


class GaussianBlurGenerator(PSFGenerator):
    r"""
    Random Gaussian blur generator. Generates 1D, 2D, or 3D Gaussian kernels with random standard deviations and rotation angles.

    :param tuple[int, ...] psf_size: the shape of the generated point spread function (PSF). The dimension (1D, 2D, or 3D) of the kernel is determined by the length of the ``psf_size`` tuple.
    :param float | tuple[float, ...] sigma_min: the minimum standard deviation(s) for the Gaussian kernel. If a single value is provided, it is applied to all dimensions. If a tuple is provided, it should have the same length as the number of dimensions and specify the minimum sigma for each dimension.
    :param float | tuple[float, ...] sigma_max: the maximum standard deviation(s) for the Gaussian kernel. Follows the same format as ``sigma_min``.
    :param bool isotropic: If True, the generated Gaussian kernels will be isotropic (same sigma for all dimensions). If False, the kernels can be anisotropic (different sigma for each dimension). Defaults to True.
    :param float | tuple[float, ...] angle_min: the minimum rotation angle(s) for the Gaussian kernel in degrees. For 2D kernels, this is a single angle of rotation in the plane. For 3D kernels, this can be a tuple of three angles (alpha, beta, gamma) representing minimum rotation values around the x, y, and z axes respectively. In 3D, if a single angle is provided, it is used as minimum value for all axes.
    :param float | tuple[float, ...] angle_max: the maximum rotation angle(s) for the Gaussian kernel in degrees. Follows the same format as ``angle_min``.
    :param torch.Generator rng: PyTorch random number generator for reproducibility. If ``None``, a torch.Generator will be created on the specified device.
    :param str device: the device to create the tensors on. Defaults to "cpu".
    :param type dtype: the data type of the generated tensors. Defaults to torch.float32.

    .. note::
        Always generates single-channel PSFs of shape ``(B, 1, H, W)``.

    |sep|

    :Examples:

    >>> import deepinv as dinv
    >>> rng = torch.Generator(device="cpu").manual_seed(0)
    >>> generator = dinv.physics.generator.GaussianBlurGenerator((7, 7), device="cpu", rng=rng, isotropic=False)
    >>> params = generator.step(batch_size=4)  # dict_keys(['filter'])
    >>> print(params['filter'].shape)
    torch.Size([4, 1, 7, 7])
    >>> dinv.utils.plot(params['filter'])  # doctest: +SKIP

    .. plot::

        import torch
        import deepinv as dinv
        rng = torch.Generator(device="cpu").manual_seed(0)
        generator = dinv.physics.generator.GaussianBlurGenerator((7, 7), device="cpu", rng=rng, isotropic=False)
        params = generator.step(batch_size=4)
        dinv.utils.plot(params['filter'])

    """

    @_deprecated_argument("num_channels")
    def __init__(
        self,
        psf_size: tuple[int, ...],
        sigma_min: float | tuple[float, ...] = 0.5,
        sigma_max: float | tuple[float, ...] = 5.0,
        isotropic: bool = True,
        angle_min: float | tuple[float, ...] = 0.0,
        angle_max: float | tuple[float, ...] = 360.0,
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
        return {"filter": filters}


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
    :param float l: the length scale of the trajectory, defaults to 0.3
    :param float sigma: the standard deviation of the Gaussian Process, defaults to 0.25
    :param int n_steps: the number of points in the trajectory, defaults to 1000

    .. note::
        Always generates single-channel PSFs of shape ``(B, 1, H, W)``.

    |sep|

    :Examples:

    >>> from deepinv.physics.generator import MotionBlurGenerator
    >>> generator = MotionBlurGenerator((5, 5))
    >>> blur = generator.step()  # dict_keys(['filter'])
    >>> print(blur['filter'].shape)
    torch.Size([1, 1, 5, 5])
    """

    @_deprecated_argument("num_channels")
    def __init__(
        self,
        psf_size: tuple,
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
            raise ValueError("psf_size must 2D.")
        super().__init__(
            psf_size=psf_size,
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
        return {"filter": kernel}


class DiffractionBlurGenerator(PSFGenerator):
    r"""
    Diffraction limited blur generator.

    Generates 2D diffraction PSFs in optics using Zernike decomposition of the phase mask (Fresnel/Fraunhoffer diffraction theory, Fourier optics).

    Zernike polynomials are a sequence of orthogonal polynomials defined on the unit disk.
    They are commonly used in optical systems to describe wavefront aberrations.

    The PSF is modeled as:

    .. math::

        h(\cdot; \lambda) = \left| \mathcal{F} \left[ \mathbb{1}_{|\boldsymbol{\rho}| \leq 1} \cdot \exp \left( - i 2 \pi \sum_k \frac{a_k}{\lambda} z_k(\boldsymbol{\rho}) \right) \right](\cdot) \right|^2

    where :math:`\boldsymbol{\rho}` are normalised pupil-plane coordinates (on the unit disk),
    :math:`a_k` are the Zernike coefficients **in physical units** (nm OPD, wavelength-independent),
    :math:`\lambda` is the emission wavelength (nm), and :math:`z_k` are the Zernike polynomials.

    The phase in waves is therefore :math:`\theta_k(\lambda) = a_k / \lambda`, so the same
    physical aberration produces a stronger wavefront error (in waves) at shorter wavelengths.

    For multi-channel (multi-colour) imaging the generator supports a perturbation model:

    .. math::

        \theta_k^{(b,c)} = \underbrace{\theta_k^{(b)} \cdot \frac{\lambda_{\text{ref}}}{\lambda_c}}_{\text{monochromatic, rescaled}} + \underbrace{\Delta\theta_k^{(b,c)}}_{\text{chromatic perturbation}}

    where :math:`\theta_k^{(b)}` are base coefficients (in waves at :math:`\lambda_{\text{ref}}`,
    i.e. channel 0) shared across channels, and :math:`\Delta\theta_k^{(b,c)}` are small
    per-channel perturbations (e.g. sample-induced dispersion). The cutoff frequency is also
    wavelength-dependent:

    .. math::

        f_c^{(c)} = \frac{\mathrm{NA} \cdot p}{\lambda_c}

    where :math:`\mathrm{NA}` is the numerical aperture and :math:`p` is the pixel size.

    See :footcite:t:`lakshminarayanan2011zernike`
    `or this link <https://e-l.unifi.it/pluginfile.php/1055875/mod_resource/content/1/Appunti_2020_Lezione%2014_4_Zernikepolynomialsaguidefinal.pdf>`_
    or :class:`deepinv.physics.generator.Zernike` for more details.

    In the ideal diffraction-limited case (i.e., no aberrations), the PSF corresponds to the Airy pattern.

    The Zernike polynomials :math:`z_k` are indexed using the ``'noll'`` or ``'ansi'`` convention (defined by `index_convention` parameter).
    Conversion from the two conventions to the standard radial-angular indexing is done internally (see `wikipedia page <https://en.wikipedia.org/wiki/Zernike_polynomials>`_).

    :param tuple psf_size: the shape ``H x W`` of the generated PSF in 2D
    :param tuple[int, ...], tuple[tuple[int, int], ...] zernike_index: activated Zernike coefficients in the following `index_convention` convention.
        It can be either:

            - a tuple of `int` corresponding to the Noll or ANSI indices, in which case the `index_convention` parameter is required to interpret them correctly.
            - a tuple of `tuple[int, int]` corresponding to the standard radial-angular indexing :math:`(n,m)`. In this case, the `index_convention` parameter is ignored.

        Defaults to ``(4, 5, 6, 7, 8, 9, 10, 11)``, correspond to radial order `n` from 2 to 3 (included) and the spherical aberration.
        These correspond to the following aberrations: defocus, astigmatism, coma, trefoil and spherical aberration.
    :param float, tuple[float, ...], list[float], torch.Tensor fc: default cutoff frequency
        ``(NA/emission_wavelength) * pixel_size``. Should be in ``[0, 0.25]`` to respect the
        Shannon-Nyquist sampling theorem, defaults to ``0.2``.

        At **construction time**, only a scalar ``float`` or a 1D tensor/sequence of length ``C``
        are accepted. A 2D tensor raises a ``ValueError``.

        At **step time** (passed to :meth:`step`), ``fc`` may additionally be a 2D tensor of
        shape ``(B, C)`` for full per-(batch, channel) control. The output PSF shape is then
        driven by ``fc`` as follows:

            - ``float`` / scalar: ``(batch_size, 1, H, W)``.
            - ``(C,)`` 1D tensor/sequence: ``(batch_size, C, H, W)``.
            - ``(B, C)`` 2D tensor: ``(B, C, H, W)``.
            - ``(1, C)`` 2D tensor: ``(batch_size, C, H, W)``.
            - ``(B, 1)`` 2D tensor: ``(B, 1, H, W)``.

    :param float max_zernike_amplitude: default amplitude of the base Zernike coefficients (in
        waves at the channel-0/reference cutoff frequency), defaults to ``0.15``. The amplitude
        of each coefficient is sampled uniformly in ``[-max_zernike_amplitude/2, max_zernike_amplitude/2]``.
        Can be overridden per :meth:`step` call.
    :param float zernike_perturbation_amplitude: default amplitude of the per-channel chromatic
        perturbations, defaults to ``0``. Can be overridden per :meth:`step` call.
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
    >>> generator = DiffractionBlurGenerator((5, 5))
    >>> print("\n".join(generator.zernike_polynomials))
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

    >>> generator = DiffractionBlurGenerator((5, 5), fc=(0.18, 0.20, 0.22))
    >>> blur = generator.step(batch_size=2)
    >>> print(blur['filter'].shape)
    torch.Size([2, 3, 5, 5])
    >>> print(blur['coeff'].shape)   # (B, C, K): wavelength-rescaled base + chromatic perturbations
    torch.Size([2, 3, 8])

    """

    @_deprecated_alias(list_param="zernike_index")
    @_deprecated_argument("num_channels")
    def __init__(
        self,
        psf_size: tuple,
        zernike_index: tuple[int, ...] | tuple[tuple[int, int], ...] = tuple(
            range(4, 12)
        ),
        fc: float | tuple[float, ...] | list[float] | torch.Tensor = 0.2,
        max_zernike_amplitude: float = 0.15,
        zernike_perturbation_amplitude: float = 0.0,
        pupil_size: tuple[int, ...] = (256, 256),
        apodize: bool = False,
        random_rotate: bool = False,
        index_convention: str = "noll",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        rng: torch.Generator = None,
    ):

        if isinstance(fc, float):
            self.fc = fc
        else:
            self.fc = torch.as_tensor(fc, dtype=dtype)
            if self.fc.ndim != 1:
                raise ValueError(
                    f"fc must be a scalar or 1D tensor/list/tuple at construction time, got {self.fc.ndim}D."
                )

        super().__init__(
            psf_size=psf_size,
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
        lin_y = torch.linspace(-0.5, 0.5, self.pupil_size[1], **self.factory_kwargs)
        self.register_buffer("lin_x", lin_x)
        self.register_buffer("lin_y", lin_y)
        self.step_rho = (lin_x[1] - lin_x[0]).item()

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

        # (n, m) per active Zernike term -- independent of fc, computed once.
        self._nm_list = self._zernike_index_to_nm_list(
            self.zernike_index, index_convention
        )

        self.to(device=device, dtype=dtype)

    @staticmethod
    def _zernike_index_to_nm_list(zernike_index, index_convention="noll"):
        r"""Convert an iterable of Zernike indices to a list of ``(n, m)`` pairs.

        Each element may be an ``int`` (interpreted via ``index_convention``)
        or a ``(n, m)`` tuple.

        :param zernike_index: iterable of ``int`` or ``(n, m)`` tuples.
        :param str index_convention: ``"noll"`` or ``"ansi"`` (ignored for tuple inputs).
        :return: list of ``(n, m)`` pairs.
        """
        nm_list = []
        for index in zernike_index:
            if isinstance(index, int):
                n, m = Zernike.index_conversion(index, convention=index_convention)
            elif isinstance(index, tuple) and len(index) == 2:
                n, m = index
            else:
                raise ValueError(
                    f"Zernike index must be either int or tuple of (n, m), got {index!r}"
                )
            nm_list.append((n, m))
        return nm_list

    def _format_fc(self, fc, batch_size: int) -> torch.Tensor:
        r"""
        Normalises ``fc`` into a fully resolved 2D tensor of shape ``(B, C)``.

        - scalar / 0-D → ``(batch_size, 1)``
        - 1D of length ``C`` → ``(batch_size, C)``
        - 2D ``(B, C)`` → returned as-is; ``batch_size`` is ignored.

        :param fc: cutoff frequency input.
        :param int batch_size: used to expand scalar and 1D inputs.
        :return: ``torch.Tensor`` of shape ``(B, C)``.
        """
        if isinstance(fc, torch.Tensor):
            t = fc.to(**self.factory_kwargs)
        else:
            try:
                t = torch.as_tensor(fc, **self.factory_kwargs)
            except (TypeError, ValueError):
                raise TypeError(
                    f"fc must be a float, list/tuple, or Tensor, got {type(fc)}."
                )

        if t.ndim == 2:
            return t
        if t.ndim == 0:
            return t.reshape(1, 1).expand(batch_size, -1)
        if t.ndim == 1:
            return t.unsqueeze(0).expand(batch_size, -1)
        raise ValueError(f"fc must be 0D, 1D or 2D, got {t.ndim}D.")

    def _zernike_basis(self, fc: torch.Tensor, nm_list=None):
        r"""
        Synthesizes the Zernike polynomials and pupil indicator function for
        the given cutoff frequencies.

        :param torch.Tensor fc: tensor of shape ``(Bf, Cf)``.
        :param list nm_list: list of ``(n, m)`` pairs to use. Defaults to
            ``self._nm_list`` (the full set built at init).
        :return: tuple ``(Z, indicator_circ)`` of shapes ``(Bf, Cf, H, W, K)``
            and ``(Bf, Cf, H, W)``.
        """
        if nm_list is None:
            nm_list = self._nm_list

        Bf, Cf = fc.shape
        fc_r = fc.reshape(Bf, Cf, 1, 1)

        XX, YY = torch.meshgrid(self.lin_x, self.lin_y, indexing="ij")
        XX = XX.reshape(1, 1, *XX.shape) / fc_r
        YY = YY.reshape(1, 1, *YY.shape) / fc_r

        rho = cart2pol(XX, YY)

        # step spacing in the rescaled (rho) coordinates is step_rho / fc, not
        # step_rho -- using the un-rescaled spacing would make the pupil edge
        # transition narrower than one pixel for fc < 1 (essentially always),
        # aliasing the pupil boundary.
        step_rho_eff = self.step_rho / fc_r  # (Bf, Cf, 1, 1)
        indicator_circ = bump_function(rho, 1 - step_rho_eff / 2, step_rho_eff / 2)

        Z = torch.stack(
            [Zernike.cartesian_evaluate(n, m, XX, YY) for n, m in nm_list],
            dim=-1,
        )
        return Z, indicator_circ

    def step(
        self,
        batch_size: int = 1,
        coeff: torch.Tensor = None,
        angle: torch.Tensor = None,
        max_zernike_amplitude: float = None,
        zernike_perturbation_amplitude: float = None,
        seed: int = None,
        fc: float | tuple[float, ...] | list[float] | torch.Tensor = None,
        used_zernike_index=None,
        **kwargs,
    ) -> dict:
        r"""
        Generate a batch of PSFs with a batch of Zernike coefficients.

        The shape of the output PSF is determined by ``fc`` as follows:

            - ``None``: ``(batch_size, 1, *self.psf_size)`` or ``(batch_size, len(self.fc), *self.psf_size)``.
            - ``float`` / scalar: ``(batch_size, 1, *self.psf_size)``.
            - ``(C,)`` 1D tensor/sequence: ``(batch_size, C, *self.psf_size)``.
            - ``(B, C)`` 2D tensor: ``(B, C, *self.psf_size)``.

        :param int batch_size: number of PSFs to generate. Ignored when ``fc`` is a 2D
            tensor with ``B > 1`` (batch size is then read from ``fc``). Defaults to ``1``.
        :param torch.Tensor coeff: Zernike coefficients. Accepted shapes:

            - ``None`` (default): sampled via :meth:`generate_coeff`.
            - ``(B, n_zernike_used)``: base coefficients per batch element, shared across
              channels. No rescaling applied. No chromatic perturbation is added.
            - ``(B, C, n_zernike_used)``: fully specified per channel. No rescaling applied.

        :param torch.Tensor angle: ``(batch_size,)`` angles in degrees for PSF rotation.
        :param float max_zernike_amplitude: overrides ``self.max_zernike_amplitude``
            for this call only. Only used when ``coeff`` is ``None``.
        :param float zernike_perturbation_amplitude: overrides
            ``self.zernike_perturbation_amplitude`` for this call only.
        :param int seed: the seed for the random number generator.
        :param float, tuple[float, ...], list[float], torch.Tensor fc: overrides ``self.fc``
            for this call only (does not mutate ``self.fc``). Defaults to ``None``, in which
            case ``self.fc`` is used. See class docstring for accepted shapes and their effect
            on the output PSF shape.
        :param used_zernike_index: subset of Zernike indices to activate for this call.
            Must be a sub-sequence of ``self.zernike_index`` (same format: ints or ``(n, m)``
            tuples). ``None`` (default) uses the full set ``self.zernike_index``. Useful for
            varying the active polynomial set without re-instantiating the generator::

                gen = DiffractionBlurGenerator((31, 31), zernike_index=range(3, 37))
                p1 = gen.step(used_zernike_index=range(3, 16))
                p2 = gen.step(used_zernike_index=range(3, 28))

        :return: dictionary with keys

            - `filter`: tensor of size ``(B, C, H, W)`` where ``B`` and ``C`` are
              determined by ``fc`` as described above,
            - `coeff`: the Zernike coefficients actually used, shape
              ``(B, n_zernike_used)`` or ``(B, C, n_zernike_used)`` where
              ``n_zernike_used = len(used_zernike_index)`` if specified, else
              ``self.n_zernike``,
            - `pupil`: the pupil function,
            - `angle`: the random rotation angle in degrees if `random_rotate` is ``True``,
            - `fc`: tensor of shape ``(Bf, Cf)`` with the cutoff frequencies actually used.
        """

        self.rng_manual_seed(seed)

        # Resolve the active Zernike set for this call.
        if used_zernike_index is not None:
            nm_list_used = self._zernike_index_to_nm_list(
                used_zernike_index, self.index_convention
            )
            invalid = [nm for nm in nm_list_used if nm not in self._nm_list]
            if invalid:
                raise ValueError(
                    f"used_zernike_index contains (n, m) entries {invalid} that are not "
                    f"in self.zernike_index. Initialise with a larger zernike_index set."
                )
            n_zernike_used = len(nm_list_used)
        else:
            nm_list_used = self._nm_list
            n_zernike_used = self.n_zernike

        if max_zernike_amplitude is None:
            max_zernike_amplitude = self.max_zernike_amplitude
        if zernike_perturbation_amplitude is None:
            zernike_perturbation_amplitude = self.zernike_perturbation_amplitude

        fc = self.fc if fc is None else fc
        fc_check = self._format_fc(fc, 1)  ##for checks on coeff shape

        if coeff is not None:

            if coeff.shape[-1] != n_zernike_used:
                raise ValueError(
                    f"The number of Zernike coefficients {coeff.shape[-1]} "
                    f"in input coeff does not match n_zernike_used={n_zernike_used}"
                )

            fc_used = self._format_fc(fc, coeff.shape[0])
            B, C = fc_used.shape

            if coeff.ndim == 2:
                if fc_check.shape[0] == 1:  ###case input fc was float or (C,)
                    pass
                else:  ### case input fc was (B, C)
                    if coeff.shape[0] != fc_check.shape[0]:
                        if coeff.shape[0] == 1:
                            raise ValueError(
                                f"coeff shape {tuple(coeff.shape)} does not match fc inferred shape (B={B}, K)."
                                f"If you wanted to simulate the same Zernike coefficients with different fc "
                                f"across batches then pass a 2D coeff tensor with shape (B={B}, K) "
                                f"not (B={coeff.shape[0]}, K)."
                            )
                        else:
                            raise ValueError(
                                f"coeff shape {tuple(coeff.shape)} does not match fc inferred shape (B={B}, K)."
                            )

            elif coeff.ndim == 3:
                if (
                    coeff.shape[0] != fc_used.shape[0]
                    or coeff.shape[1] != fc_used.shape[1]
                ):
                    raise ValueError(
                        f"coeff shape {tuple(coeff.shape)} does not match fc inferred shape (B={B}, C={C}, K)."
                    )

            else:
                raise ValueError(
                    f"coeff must be 2D (B, K) or 3D (B, C, K), got {coeff.ndim}D."
                )
        else:
            fc_used = self._format_fc(fc, batch_size)
            B, C = fc_used.shape

            coeff = self.generate_coeff(
                batch_size=B,
                fc=fc_used,
                max_zernike_amplitude=max_zernike_amplitude,
                zernike_perturbation_amplitude=zernike_perturbation_amplitude,
                n_zernike=n_zernike_used,
            )

        if coeff.ndim == 2:
            coeff = coeff.unsqueeze(1).expand(-1, C, -1)
        _, C_coeff, _ = coeff.shape

        Z, indicator_circ = self._zernike_basis(fc_used, nm_list=nm_list_used)

        if Z.shape[1] == 1 and C_coeff > 1:
            Z = Z.expand(-1, C_coeff, -1, -1, -1)
            indicator_circ = indicator_circ.expand(-1, C_coeff, -1, -1)

        pupil = torch.einsum("bchwk,bck->bchw", Z, coeff.to(Z.dtype))
        pupil = torch.exp(-2.0j * torch.pi * pupil)
        pupil = pupil * indicator_circ

        psf = torch.fft.ifftshift(
            torch.fft.fft2(torch.fft.fftshift(pupil, dim=(-2, -1)), dim=(-2, -1)),
            dim=(-2, -1),
        )
        psf = psf.abs().pow(2)
        psf = psf[
            ...,
            self.pad_pre[0] : self.pupil_size[0] - self.pad_post[0],
            self.pad_pre[1] : self.pupil_size[1] - self.pad_post[1],
        ]

        psf = psf / torch.sum(psf, dim=(-1, -2), keepdim=True)

        if self.random_rotate:
            if angle is None:
                angle = self.generate_angles(B)
            psf = rotate_via_shear(psf, angle)

        if self.apodize:
            psf = self.apodize_mask * psf
            psf = psf / torch.sum(psf, dim=(-1, -2), keepdim=True)

        params = {
            "filter": psf,
            "coeff": coeff,
            "pupil": pupil,
            "fc": fc_used,
        }
        if self.random_rotate:
            params["angle"] = angle
        return params

    @property
    def zernike_polynomials(self) -> list[str]:
        r"""
        List of Zernike polynomials used in the decomposition, with the corresponding aberration if available.
        """
        return [Zernike.get_name(n, m) for n, m in self._nm_list]

    def generate_coeff(
        self,
        batch_size: int,
        fc: torch.Tensor = None,
        max_zernike_amplitude: float | None = None,
        zernike_perturbation_amplitude: float | None = None,
        n_zernike: int | None = None,
    ) -> torch.Tensor:
        r"""
        Generate random Zernike coefficients, scaled by cutoff frequency per channel.

        :param int batch_size: number of independent aberration realisations.
        :param torch.Tensor fc: already-formatted ``(B, C)`` tensor from
            class method ``_format_fc()``. If ``None``, ``self.fc`` is used with ``batch_size``,
            producing a ``(batch_size, K)`` output (backward-compatible behaviour).
        :param float max_zernike_amplitude: amplitude of the base coefficients.
            Defaults to ``self.max_zernike_amplitude``.
        :param float zernike_perturbation_amplitude: amplitude of per-channel
            perturbations. Defaults to ``self.zernike_perturbation_amplitude``.
        :param int n_zernike: number of Zernike coefficients to generate. Defaults to
            ``self.n_zernike``. Set to ``len(used_zernike_index)`` when called from
            :meth:`step` with a ``used_zernike_index`` argument.
        :return: ``(batch_size, K)`` if ``C == 1``, else ``(B, C, K)``.
        """
        if max_zernike_amplitude is None:
            max_zernike_amplitude = self.max_zernike_amplitude
        if zernike_perturbation_amplitude is None:
            zernike_perturbation_amplitude = self.zernike_perturbation_amplitude
        if fc is None:
            fc = self._format_fc(self.fc, batch_size)
        if n_zernike is None:
            n_zernike = self.n_zernike

        _, C = fc.shape

        coeff_base = (
            torch.rand(
                (batch_size, n_zernike),
                generator=self.rng,
                **self.factory_kwargs,
            )
            - 0.5
        ) * max_zernike_amplitude

        if C == 1:
            return coeff_base

        fc_ref = fc[:, 0:1]
        color_scale = fc / fc_ref

        coeff_delta = (
            torch.randn(
                (batch_size, C, n_zernike),
                generator=self.rng,
                **self.factory_kwargs,
            )
            * zernike_perturbation_amplitude
        )
        return coeff_base.unsqueeze(1) * color_scale.unsqueeze(-1) + coeff_delta

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
    t = (abs_x - a) / b
    safe_t = t.clamp(0.0, 1.0 - 1e-6)
    transition = torch.exp(-1.0 / (1.0 - safe_t**2)) / np.exp(-1.0)
    return torch.where(
        abs_x <= a,
        torch.ones_like(x),
        torch.where(abs_x < a + b, transition, torch.zeros_like(x)),
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
        psf_grid = self.psf_generator.step(self.n_psf_prid * batch_size, **kwargs)[
            "filter"
        ]
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
    :param tuple[int, ...], tuple[tuple[int, int], ...] zernike_index: activated Zernike coefficients in the following `index_convention` convention.
        It can be either:

            - a tuple of `int` corresponding to the Noll or ANSI indices, in which case the `index_convention` parameter is required to interpret them correctly.
            - a tuple of `tuple[int, int]` corresponding to the standard radial-angular indexing :math:`(n,m)`. In this case, the `index_convention` parameter is ignored.

        Defaults to ``(4, 5, 6, 7, 8, 9, 10, 11)``, correspond to radial order `n` from 2 to 3 (included) and the spherical aberration.
        These correspond to the following aberrations: defocus, astigmatism, coma, trefoil and spherical aberration.
    :param float, tuple[float, ...], list[float], torch.Tensor fc: cutoff frequency `(NA/emission_wavelength) * pixel_size`. Should be in `[0, 1/4]` to respect Shannon, defaults to `0.2`.
    :param float, tuple[float, ...], list[float], torch.Tensor kb: wave number `(NI/emission_wavelength) * pixel_size` or `(NA/NI) * fc`. Must be greater than `fc`. Defaults to `0.3`.
    :param float max_zernike_amplitude: maximum amplitude of Zernike coefficients. Defaults to 0.15.
    :param float zernike_perturbation_amplitude: amplitude of per-channel chromatic perturbations, defaults to ``0``.
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
    dict_keys(['filter', 'pupil', 'coeff', 'fc'])


    """

    @_deprecated_alias(list_param="zernike_index")
    @_deprecated_argument("num_channels")
    def __init__(
        self,
        psf_size: tuple,
        zernike_index: tuple[int, ...] | tuple[tuple[int, int], ...] = tuple(
            range(4, 12)
        ),
        fc: float | tuple[float, ...] | list[float] | torch.Tensor = 0.2,
        kb: float | tuple[float, ...] | list[float] | torch.Tensor = 0.25,
        max_zernike_amplitude: float = 0.15,
        zernike_perturbation_amplitude: float = 0.0,
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

        if isinstance(fc, float):
            self.fc = fc
        else:
            self.fc = torch.as_tensor(fc, dtype=dtype)
            if self.fc.ndim != 1:
                raise ValueError(
                    f"fc must be a scalar or 1D tensor/list/tuple at construction time, got {self.fc.ndim}D."
                )

        super().__init__(
            psf_size=psf_size,
            device=device,
            dtype=dtype,
            rng=rng,
        )

        self.generator2d = DiffractionBlurGenerator(
            psf_size=psf_size[1:],
            fc=fc,
            zernike_index=zernike_index,
            max_zernike_amplitude=max_zernike_amplitude,
            zernike_perturbation_amplitude=zernike_perturbation_amplitude,
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
        fc: float | tuple[float, ...] | list[float] | torch.Tensor = None,
        kb: float | tuple[float, ...] | list[float] | torch.Tensor = None,
        max_zernike_amplitude: float | None = None,
        zernike_perturbation_amplitude: float | None = None,
        **kwargs,
    ) -> dict:
        r"""
        Generate a batch of PSF with a batch of Zernike coefficients

        :param int batch_size: number of PSFs to generate.
        :param torch.Tensor coeff: tensor containing the Zernike coefficients.
            If `None`, random coefficients are generated. Accepts ``(B, K)`` or ``(B, C, K)``, exactly as in
            :class:`deepinv.physics.generator.DiffractionBlurGenerator`.
        :param torch.Tensor angle: ``(batch_size,)`` angles in degrees for PSF rotation.
        :param int seed: the seed for the random number generator.
        :param float, tuple[float, ...], list[float], torch.Tensor fc: overrides ``self.fc``
            for this call only. Accepts the same types as the constructor's ``fc``.
        :param float, tuple[float, ...], list[float], torch.Tensor kb: overrides ``self.kb``
            for this call only. Accepts the same types as ``fc``.
        :param float max_zernike_amplitude: overrides ``self.max_zernike_amplitude`` for this call only. Only used when ``coeff`` is ``None``.
        :param float zernike_perturbation_amplitude: overrides ``self.zernike_perturbation_amplitude`` for this call only.

        :return: dictionary with keys

            - `filter`: tensor of size `(B, C, depth, H, W)` batch of 3D PSFs,
            - `pupil`: the pupil function,
            - `coeff`: list of sampled Zernike coefficients in this realization,
            - `angle`: the random rotation angles in degrees if `random_rotate` is `True`, nothing otherwise.
            - `fc`: tensor of shape ``(B, C)`` with the cutoff frequencies used.
        """
        # Delegate 2D pupil generation (handles fc, coeff, multi-channel logic).
        gen_dict = self.generator2d.step(
            batch_size=batch_size,
            coeff=coeff,
            seed=seed,
            fc=fc,
            max_zernike_amplitude=max_zernike_amplitude,
            zernike_perturbation_amplitude=zernike_perturbation_amplitude,
            **kwargs,
        )

        pupil = gen_dict["pupil"]  # (B, C, H, W) complex
        fc_used = gen_dict["fc"]  # (B, C)

        kb_val = self.kb if kb is None else kb
        kb_used = self.generator2d._format_fc(kb_val, batch_size=fc_used.shape[0])
        kb_used = kb_used.expand_as(fc_used)  # (B, C)

        lin_x = self.generator2d.lin_x
        lin_y = self.generator2d.lin_y
        XX_norm, YY_norm = torch.meshgrid(lin_x, lin_y, indexing="ij")
        k_lat = cart2pol(XX_norm, YY_norm)  # (H, W)

        B, C = fc_used.shape
        kb_hw = kb_used.reshape(B, C, 1, 1)

        d = (kb_hw**2 - k_lat**2 + 0j) ** 0.5  # (B, C, H, W)
        propKer = torch.exp(
            -1j * 2 * torch.pi * d.unsqueeze(2) * self._defocus[None, None, :, :, :]
        )  # (B, C, D, H, W)

        p = pupil.unsqueeze(2) * propKer  # (B, C, D, H, W) complex
        p = torch.nan_to_num(p, nan=0.0)

        pshift = torch.fft.fftshift(p, dim=(-2, -1))
        pfft = torch.fft.fft2(pshift, dim=(-2, -1))
        psf = torch.fft.ifftshift(pfft, dim=(-2, -1))
        psf = psf.abs().pow(2)

        # Crop from pupil_size back to psf_size (lateral dims only).
        psf = psf[
            ...,
            self.generator2d.pad_pre[0] : self.generator2d.pupil_size[0]
            - self.generator2d.pad_post[0],
            self.generator2d.pad_pre[1] : self.generator2d.pupil_size[1]
            - self.generator2d.pad_post[1],
        ]  # (B, C, D, H, W)

        if self.random_rotate:
            from einops import rearrange

            if angle is None:
                angle = self.generator2d.generate_angles(B)
            psf = rotate_via_shear(rearrange(psf, "b c d h w -> b (c d) h w"), angle)
            psf = rearrange(psf, "b (c d) h w -> b c d h w", d=self.psf_size[0])

        if self.apodize:
            psf = self.generator2d.apodize_mask[None, None, None] * psf

        psf = psf / torch.sum(psf, dim=(-3, -2, -1), keepdim=True)

        params = {
            "filter": psf,
            "pupil": pupil,
            "coeff": gen_dict["coeff"],
            "fc": fc_used,
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
    :param tuple[int, ...], tuple[tuple[int, int], ...] zernike_index: activated Zernike coefficients in the following `index_convention` convention.
        It can be either:

            - a tuple of `int` corresponding to the Noll or ANSI indices, in which case the `index_convention` parameter is required to interpret them correctly.
            - a tuple of `tuple[int, int]` corresponding to the standard radial-angular indexing :math:`(n,m)`. In this case, the `index_convention` parameter is ignored.

        Defaults to ``(4, 5, 6, 7, 8, 9, 10, 11)``, correspond to radial order `n` from 2 to 3 (included) and the spherical aberration.
        These correspond to the following aberrations: defocus, astigmatism, coma, trefoil and spherical aberration.

    :param float NI: Refractive index of  the immersion medium. Defaults to `1.51` (oil),
    :param float NA: Numerical aperture. Should be less than NI. Defaults to `1.37`.
    :param float, list[float] lambda_ill: Wavelength(s) of the illumination light (fluorescence excitation). Defaults to `489e-9`.
        Pass a list of ``C`` values to generate multi-colour PSFs (one channel per wavelength).
    :param float, list[float] lambda_coll: Wavelength(s) of the collection light (fluorescence emission). Defaults to `395e-9`.
        Must have the same length as ``lambda_ill`` when a list is provided.
    :param float pixelsize_XY: Physical pixel size in the lateral direction (height, width). Defaults to `50e-9`.
    :param float pixelsize_Z:  Physical pixel size in the axial direction (depth). Defaults to `100e-9`.
    :param float pinhole_radius: Radius of pinhole in Airy units. Defaults to `1`.
    :param float max_zernike_amplitude: maximum amplitude of Zernike coefficients. Defaults to `0.1`.
    :param float zernike_perturbation_amplitude: amplitude of per-channel chromatic perturbations, defaults to ``0``.
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
    dict_keys(['filter', 'pupil_ill', 'pupil_coll', 'coeff_ill', 'coeff_coll', 'fc_ill', 'fc_coll'])

    Multi-colour example (one channel per excitation/emission wavelength pair):

    >>> generator = ConfocalBlurGenerator3D(
    ...     (21, 51, 51),
    ...     lambda_ill=[489e-9, 561e-9],
    ...     lambda_coll=[525e-9, 620e-9],
    ...     zernike_index=(3,),
    ... )
    >>> dict = generator.step()
    >>> print(dict['filter'].shape)
    torch.Size([1, 2, 21, 51, 51])

    """

    @_deprecated_alias(list_param="zernike_index")
    @_deprecated_argument("num_channels")
    def __init__(
        self,
        psf_size: tuple,
        zernike_index: tuple[int, ...] | tuple[tuple[int, int], ...] = tuple(
            range(4, 12)
        ),
        NI: float = 1.51,
        NA: float = 1.37,
        lambda_ill: float | list[float] = 489e-9,
        lambda_coll: float | list[float] = 395e-9,
        pixelsize_XY: float = 50e-9,
        pixelsize_Z: float = 100e-9,
        pinhole_radius: float = 1,
        max_zernike_amplitude: float = 0.1,
        zernike_perturbation_amplitude: float = 0.0,
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

        # Normalise lambda_ill / lambda_coll to lists for uniform handling.
        # A scalar is wrapped in a length-1 list; backward compat is preserved.
        if isinstance(lambda_ill, (int, float)):
            lambda_ill = [lambda_ill]
        if isinstance(lambda_coll, (int, float)):
            lambda_coll = [lambda_coll]

        if len(lambda_ill) != len(lambda_coll):
            raise ValueError(
                f"lambda_ill and lambda_coll must have the same length, "
                f"got {len(lambda_ill)} and {len(lambda_coll)}."
            )

        super().__init__(
            psf_size=psf_size,
            device=device,
            dtype=dtype,
            rng=rng,
        )

        # Compute per-channel fc and kb from physical parameters.
        fc_ill = [NA / lam * pixelsize_XY for lam in lambda_ill]
        kb_ill = [NI / lam * pixelsize_XY for lam in lambda_ill]
        fc_coll = [NA / lam * pixelsize_XY for lam in lambda_coll]
        kb_coll = [NI / lam * pixelsize_XY for lam in lambda_coll]

        # Unwrap to scalar for the single-channel case (backward compat: the
        # sub-generators receive a plain float, exactly as in the original code).
        self.fc_ill = fc_ill[0] if len(fc_ill) == 1 else fc_ill
        self.kb_ill = kb_ill[0] if len(kb_ill) == 1 else kb_ill
        self.fc_coll = fc_coll[0] if len(fc_coll) == 1 else fc_coll
        self.kb_coll = kb_coll[0] if len(kb_coll) == 1 else kb_coll

        self.pinhole_radius = pinhole_radius
        self.pixelsize_XY = pixelsize_XY
        self.pixel_size_Z = pixelsize_Z
        self.lambda_ill = lambda_ill
        self.lambda_coll = lambda_coll
        self.NI = NI
        self.NA = NA

        # Initialize generator for the Illumination PSF
        self.generator_ill = DiffractionBlurGenerator3D(
            psf_size=psf_size,
            fc=self.fc_ill,
            kb=self.kb_ill,
            stepz_pixel=int(pixelsize_Z / pixelsize_XY),
            zernike_index=zernike_index,
            max_zernike_amplitude=max_zernike_amplitude,
            zernike_perturbation_amplitude=zernike_perturbation_amplitude,
            pupil_size=pupil_size,
            index_convention=index_convention,
            rng=rng,
            device=device,
            dtype=dtype,
        )

        # Initialize generator for the Collection PSF
        self.generator_coll = DiffractionBlurGenerator3D(
            psf_size=psf_size,
            fc=self.fc_coll,
            kb=self.kb_coll,
            stepz_pixel=int(pixelsize_Z / pixelsize_XY),
            zernike_index=zernike_index,
            max_zernike_amplitude=max_zernike_amplitude,
            zernike_perturbation_amplitude=zernike_perturbation_amplitude,
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
        seed: int = None,
        coeff_ill: torch.Tensor = None,
        coeff_coll: torch.Tensor = None,
        fc_ill: float | tuple[float, ...] | list[float] | torch.Tensor = None,
        kb_ill: float | tuple[float, ...] | list[float] | torch.Tensor = None,
        fc_coll: float | tuple[float, ...] | list[float] | torch.Tensor = None,
        kb_coll: float | tuple[float, ...] | list[float] | torch.Tensor = None,
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
        :param float, tuple[float, ...], list[float], torch.Tensor fc_ill: overrides the illumination cutoff frequency for this call only.
        :param float, tuple[float, ...], list[float], torch.Tensor kb_ill: overrides the illumination wave number for this call only.
        :param float, tuple[float, ...], list[float], torch.Tensor fc_coll: overrides the collection cutoff frequency for this call only.
        :param float, tuple[float, ...], list[float], torch.Tensor kb_coll: overrides the collection wave number for this call only.

        :return: dictionary with keys

            - `filter`: tensor of size `batch_size x C x psf_size[0] x psf_size[1] x psf_size[2]` batch of PSFs,
            - `coeff_ill`: list of sampled Zernike coefficients in this realization of illumination,
            - `coeff_coll`: list of sampled Zernike coefficients in this realization of collection,
            - `pupil_ill`: the illumination pupil function,
            - `pupil_coll`: the collection pupil function,
            - `fc_ill`: tensor of shape ``(B, C)`` with the illumination cutoff frequencies used,
            - `fc_coll`: tensor of shape ``(B, C)`` with the collection cutoff frequencies used.
        """
        dict_ill = self.generator_ill.step(
            batch_size=batch_size,
            seed=seed,
            coeff=coeff_ill,
            fc=fc_ill,
            kb=kb_ill,
        )  # generate illumination PSF
        psf_ill = dict_ill["filter"]
        coeff_ill = dict_ill["coeff"]
        dict_coll = self.generator_coll.step(
            batch_size=batch_size,
            seed=seed,
            coeff=coeff_coll,
            fc=fc_coll,
            kb=kb_coll,
        )  # generate collection PSF
        psf_coll = dict_coll["filter"]
        coeff_coll = dict_coll["coeff"]

        # Convolution of the collection PSF by pinhole.
        D_list = []
        for lam_c in self.lambda_coll:
            airy_unit = 0.61 * lam_c / self.NA
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
            XX, YY = torch.meshgrid(lin_x, lin_y, indexing="ij")
            PH_rho = torch.sqrt(XX**2 + YY**2)
            D_list.append(
                bump_function(PH_rho, PH_radius - PH_step_rho / 2, b=PH_step_rho / 2)
            )

        # 2. Apply 2D convolution in all z planes, per channel.
        psf_coll_convolved = torch.zeros_like(psf_coll)
        for c, D_c in enumerate(D_list):
            for i in range(psf_coll.shape[-3]):
                psf_coll_convolved[:, c, i] = conv2d(
                    psf_coll[:, c : c + 1, i],
                    filter=D_c[None, None],
                    padding="constant",
                )[:, 0]

        psf_confocal = psf_ill * psf_coll_convolved

        psf = psf_confocal / torch.sum(psf_confocal, dim=(-3, -2, -1), keepdim=True)

        return {
            "filter": psf,
            "pupil_ill": dict_ill["pupil"],
            "pupil_coll": dict_coll["pupil"],
            "coeff_ill": coeff_ill,
            "coeff_coll": coeff_coll,
            "fc_ill": dict_ill["fc"],
            "fc_coll": dict_coll["fc"],
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
