import torch
import numpy as np
from typing import List, Tuple
from math import ceil, floor
from deepinv.physics.generator import PhysicsGenerator
from deepinv.physics.functional import histogramdd
from deepinv.physics.functional.interp import ThinPlateSpline


class PSFGenerator(PhysicsGenerator):
    r"""
    Base class for generating Point Spread Functions (PSFs).


    :param tuple psf_size: the shape of the generated PSF in 2D
        `(kernel_size, kernel_size)`.
    :param int num_channels: number of images channels. Defaults to 1.
    """

    def __init__(
        self,
        psf_size: tuple = (31, 31),
        num_channels: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.shape = (num_channels,) + psf_size
        self.psf_size = psf_size
        self.num_channels = num_channels


class MotionBlurGenerator(PSFGenerator):
    r"""
    Random motion blur generator.

    See https://arxiv.org/pdf/1406.7444.pdf for more details.

    A blur trajectory is generated by sampling both its x- and y-coordinates independently
    from a Gaussian Process with a Matérn 3/2 covariance function.

    .. math::

        f_x(t), f_y(s) \sim \mathcal{GP}(0, k(t, s))

    where :math:`k(t,s)` is defined as

    .. math::

        k(t, s) = \sigma^2 \left( 1 + \frac{\sqrt{5} |t -s|}{l} + \frac{5 (t-s)^2}{3 l^2} \right) \exp \left(-\frac{\sqrt{5} |t-s|}{l}\right)

    :param tuple psf_size: the shape of the generated PSF in 2D, should be `(kernel_size, kernel_size)`
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
        device: str = "cpu",
        dtype: type = torch.float32,
        l: float = 0.3,
        sigma: float = 0.25,
        n_steps: int = 1000,
    ) -> None:
        kwargs = {"l": l, "sigma": sigma, "n_steps": n_steps}
        if len(psf_size) != 2:
            raise ValueError(
                "psf_size must 2D. Add channels via num_channels parameter"
            )
        super().__init__(
            psf_size=psf_size,
            num_channels=num_channels,
            device=device,
            dtype=dtype,
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
        """
        vec = torch.randn(batch_size, self.n_steps, **self.factory_kwargs)
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

    def step(self, batch_size: int = 1, sigma: float = None, l: float = None):
        r"""
        Generate a random motion blur PSF with parameters :math:`\sigma` and :math:`l`

        :param int batch_size: batch_size.
        :param float sigma: the standard deviation of the Gaussian Process
        :param float l: the length scale of the trajectory

        :return: dictionary with key **'filter'**: the generated PSF of shape `(batch_size, 1, psf_size[0], psf_size[1])`
        :rtype: dict
        """

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
        kernel = torch.cat(kernels, dim=0)
        kernel = kernel / torch.sum(kernel, dim=(-2, -1), keepdim=True)

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

    Generates 2D diffraction kernels in optics using Zernike decomposition of the phase mask (Fresnel/Fraunhoffer diffraction theory).

    :param tuple psf_size: the shape of the generated PSF in 2D
    :param int num_channels: number of images channels. Defaults to 1.
    :param list[str] list_param: list of activated Zernike coefficients, defaults to `["Z4", "Z5", "Z6","Z7", "Z8", "Z9", "Z10", "Z11"]`
    :param float fc: cutoff frequency (NA/emission_wavelength) * pixel_size. Should be in `[0, 1/4]` to respect Shannon, defaults to `0.2`

    :param tuple[int] pupil_size: this is used to synthesize the super-resolved pupil. The higher the more precise, defaults to (256, 256).
            If a int is given, a square pupil is considered.

    |sep|

    :Examples:

    >>> from deepinv.physics.generator import DiffractionBlurGenerator
    >>> generator = DiffractionBlurGenerator((5, 5), num_channels=3)
    >>> blur = generator.step()  # dict_keys(['filter', 'coeff', 'pupil'])
    >>> print(blur['filter'].shape)
    torch.Size([1, 3, 5, 5])


    """

    def __init__(
        self,
        psf_size: tuple,
        num_channels: int = 1,
        device: str = "cpu",
        dtype: type = torch.float32,
        list_param: List[str] = [
            "Z4",
            "Z5",
            "Z6",
            "Z7",
            "Z8",
            "Z9",
            "Z10",
            "Z11",
        ],
        fc: float = 0.2,
        max_zernike_amplitude: float = 0.15,
        pupil_size: Tuple[int] = (256, 256),
    ):
        kwargs = {
            "list_param": list_param,
            "fc": fc,
            "pupil_size": pupil_size,
            "max_zernike_amplitude": max_zernike_amplitude,
        }
        super().__init__(
            psf_size=psf_size,
            num_channels=num_channels,
            device=device,
            dtype=dtype,
            **kwargs,
        )

        self.list_param = list_param  # list of parameters to provide

        pupil_size = (
            max(self.pupil_size[0], self.psf_size[0]),
            max(self.pupil_size[1], self.psf_size[1]),
        )
        self.pupil_size = pupil_size

        lin_x = torch.linspace(-0.5, 0.5, self.pupil_size[0], **self.factory_kwargs)
        lin_y = torch.linspace(-0.5, 0.5, self.pupil_size[1], **self.factory_kwargs)

        # Fourier plane is discretized on [-0.5,0.5]x[-0.5,0.5]
        XX, YY = torch.meshgrid(lin_x / self.fc, lin_y / self.fc, indexing="ij")
        self.rho = cart2pol(XX, YY)  # Cartesian coordinates

        # The list of Zernike polynomial functions
        list_zernike_polynomial = define_zernike()

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

        # a list of indices of the parameters
        self.index_params = np.sort([int(param[1:]) for param in list_param])
        assert (
            np.max(self.index_params) <= 38
        ), "The Zernike polynomial index can not be exceed 38"

        # the number of Zernike coefficients
        self.n_zernike = len(self.index_params)

        # the tensor of Zernike polynomials in the pupil plane
        self.Z = torch.zeros(
            (self.pupil_size[0], self.pupil_size[1], self.n_zernike),
            **self.factory_kwargs,
        )
        for k in range(len(self.index_params)):
            self.Z[:, :, k] = list_zernike_polynomial[self.index_params[k]](
                XX, YY
            )  # defining the k-th Zernike polynomial

    def __update__(self):
        r"""
        Update the device and dtype of Zernike polynomials and the coordinates
        """
        self.rho = self.rho.to(**self.factory_kwargs)
        self.Z = self.Z.to(**self.factory_kwargs)

    def step(self, batch_size: int = 1, coeff: torch.Tensor = None):
        r"""
        Generate a batch of PFS with a batch of Zernike coefficients

        :param int batch_size: batch_size.
        :param torch.Tensor coeff: batch_size x len(list_param) coefficients of the Zernike decomposition (defaults is None)

        :return: dictionary with keys **'filter'**: tensor of size (batch_size x num_channels x psf_size[0] x psf_size[1]) batch of psfs,
            **'coeff'**: list of sampled Zernike coefficients in this realization, **'pupil'**: the pupil function
        :rtype: dict
        """
        self.__update__()

        if coeff is None:
            coeff = self.generate_coeff(batch_size)

        pupil1 = (self.Z @ coeff[:, : self.n_zernike].T).transpose(2, 0)
        pupil2 = torch.exp(-2.0j * torch.pi * pupil1)
        indicator = bump_function(self.rho, 1.0)
        pupil3 = pupil2 * indicator
        psf1 = torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(pupil3)))
        psf2 = torch.real(psf1 * torch.conj(psf1))

        psf3 = psf2[
            :,
            self.pad_pre[0] : self.pupil_size[0] - self.pad_post[0],
            self.pad_pre[1] : self.pupil_size[1] - self.pad_post[1],
        ].unsqueeze(1)

        psf = psf3 / torch.sum(psf3, dim=(-1, -2), keepdim=True)
        return {
            "filter": psf.expand(-1, self.shape[0], -1, -1),
            "coeff": coeff,
            "pupil": pupil3,
        }

    def generate_coeff(self, batch_size):
        r"""Generates random coefficients of the decomposition in the Zernike polynomials.

        :param int batch_size: batch_size.

        :return: a tensor of shape `(batch_size, len(list_param))` coefficients in the Zernike decomposition.

        """
        coeff = torch.rand((batch_size, len(self.list_param)), **self.factory_kwargs)
        coeff = (coeff - 0.5) * self.max_zernike_amplitude
        return coeff


def define_zernike():
    r"""
    Returns a list of Zernike polynomials lambda functions in Cartesian coordinates.

    :param list[func]: list of 37 lambda functions with the Zernike Polynomials. They are ordered as follows:

        Z1:Z00 Piston or Bias
        Z2:Z11 x Tilt
        Z3:Z11 y Tilt
        Z4:Z20 Defocus
        Z5:Z22 Primary Astigmatism at 45
        Z6:Z22 Primary Astigmatism at 0
        Z7:Z31 Primary y Coma
        Z8:Z31 Primary x Coma
        Z9:Z33 y Trefoil
        Z10:Z33 x Trefoil
        Z11:Z40 Primary Spherical
        Z12:Z42 Secondary Astigmatism at 0
        Z13:Z42 Secondary Astigmatism at 45
        Z14:Z44 x Tetrafoil
        Z15:Z44 y Tetrafoil
        Z16:Z51 Secondary x Coma
        Z17:Z51 Secondary y Coma
        Z18:Z53 Secondary x Trefoil
        Z19:Z53 Secondary y Trefoil
        Z20:Z55 x Pentafoil
        Z21:Z55 y Pentafoil
        Z22:Z60 Secondary Spherical
        Z23:Z62 Tertiary Astigmatism at 45
        Z24:Z62 Tertiary Astigmatism at 0
        Z25:Z64 Secondary x Trefoil
        Z26:Z64 Secondary y Trefoil
        Z27:Z66 Hexafoil Y
        Z28:Z66 Hexafoil X
        Z29:Z71 Tertiary y Coma
        Z30:Z71 Tertiary x Coma
        Z31:Z73 Tertiary y Trefoil
        Z32:Z73 Tertiary x Trefoil
        Z33:Z75 Secondary Pentafoil Y
        Z34:Z75 Secondary Pentafoil X
        Z35:Z77 Heptafoil Y
        Z36:Z77 Heptafoil X
        Z37:Z80 Tertiary Spherical
    """
    Z = [None for k in range(38)]

    def r2(x, y):
        return x**2 + y**2

    sq3 = 3**0.5
    sq5 = 5**0.5
    sq6 = 6**0.5
    sq7 = 7**0.5
    sq8 = 8**0.5
    sq10 = 10**0.5
    sq12 = 12**0.5
    sq14 = 14**0.5

    Z[0] = lambda x, y: torch.ones_like(x)  # piston
    Z[1] = lambda x, y: torch.ones_like(x)  # piston
    Z[2] = lambda x, y: 2 * x  # tilt x
    Z[3] = lambda x, y: 2 * y  # tilt y
    Z[4] = lambda x, y: sq3 * (2 * r2(x, y) - 1)  # defocus
    Z[5] = lambda x, y: 2 * sq6 * x * y
    Z[6] = lambda x, y: sq6 * (x**2 - y**2)
    Z[7] = lambda x, y: sq8 * y * (3 * r2(x, y) - 2)
    Z[8] = lambda x, y: sq8 * x * (3 * r2(x, y) - 2)
    Z[9] = lambda x, y: sq8 * y * (3 * x**2 - y**2)
    Z[10] = lambda x, y: sq8 * x * (x**2 - 3 * y**2)
    Z[11] = lambda x, y: sq5 * (6 * r2(x, y) ** 2 - 6 * r2(x, y) + 1)
    Z[12] = lambda x, y: sq10 * (x**2 - y**2) * (4 * r2(x, y) - 3)
    Z[13] = lambda x, y: 2 * sq10 * x * y * (4 * r2(x, y) - 3)
    Z[14] = lambda x, y: sq10 * (r2(x, y) ** 2 - 8 * x**2 * y**2)
    Z[15] = lambda x, y: 4 * sq10 * x * y * (x**2 - y**2)
    Z[16] = lambda x, y: sq12 * x * (10 * r2(x, y) ** 2 - 12 * r2(x, y) + 3)
    Z[17] = lambda x, y: sq12 * y * (10 * r2(x, y) ** 2 - 12 * r2(x, y) + 3)
    Z[18] = lambda x, y: sq12 * x * (x**2 - 3 * y**2) * (5 * r2(x, y) - 4)
    Z[19] = lambda x, y: sq12 * y * (3 * x**2 - y**2) * (5 * r2(x, y) - 4)
    Z[20] = (
        lambda x, y: sq12 * x * (16 * x**4 - 20 * x**2 * r2(x, y) + 5 * r2(x, y) ** 2)
    )
    Z[21] = (
        lambda x, y: sq12 * y * (16 * y**4 - 20 * y**2 * r2(x, y) + 5 * r2(x, y) ** 2)
    )
    Z[22] = lambda x, y: sq7 * (
        20 * r2(x, y) ** 3 - 30 * r2(x, y) ** 2 + 12 * r2(x, y) - 1
    )
    Z[23] = lambda x, y: 2 * sq14 * x * y * (15 * r2(x, y) ** 2 - 20 * r2(x, y) + 6)
    Z[24] = lambda x, y: sq14 * (x**2 - y**2) * (15 * r2(x, y) ** 2 - 20 * r2(x, y) + 6)
    Z[25] = lambda x, y: 4 * sq14 * x * y * (x**2 - y**2) * (6 * r2(x, y) - 5)
    Z[26] = (
        lambda x, y: sq14
        * (8 * x**4 - 8 * x**2 * r2(x, y) + r2(x, y) ** 2)
        * (6 * r2(x, y) - 5)
    )
    Z[27] = (
        lambda x, y: sq14
        * x
        * y
        * (32 * x**4 - 32 * x**2 * r2(x, y) + 6 * r2(x, y) ** 2)
    )
    Z[28] = lambda x, y: sq14 * (
        32 * x**6 - 48 * x**4 * r2(x, y) + 18 * x**2 * r2(x, y) ** 2 - r2(x, y) ** 3
    )
    Z[29] = (
        lambda x, y: 4
        * y
        * (35 * r2(x, y) ** 3 - 60 * r2(x, y) ** 2 + 30 * r2(x, y) + 10)
    )
    Z[30] = (
        lambda x, y: 4
        * x
        * (35 * r2(x, y) ** 3 - 60 * r2(x, y) ** 2 + 30 * r2(x, y) + 10)
    )
    Z[31] = (
        lambda x, y: 4
        * y
        * (3 * x**2 - y**2)
        * (21 * r2(x, y) ** 2 - 30 * r2(x, y) + 10)
    )
    Z[32] = (
        lambda x, y: 4
        * x
        * (x**2 - 3 * y**2)
        * (21 * r2(x, y) ** 2 - 30 * r2(x, y) + 10)
    )
    Z[33] = (
        lambda x, y: 4
        * (7 * r2(x, y) - 6)
        * (4 * x**2 * y * (x**2 - y**2) + y * (r2(x, y) ** 2 - 8 * x**2 * y**2))
    )
    Z[34] = lambda x, y: (
        4
        * (7 * r2(x, y) - 6)
        * (x * (r2(x, y) ** 2 - 8 * x**2 * y**2) - 4 * x * y**2 * (x**2 - y**2))
    )
    Z[35] = lambda x, y: (
        8 * x**2 * y * (3 * r2(x, y) ** 2 - 16 * x**2 * y**2)
        + 4 * y * (x**2 - y**2) * (r2(x, y) ** 2 - 16 * x**2 * y**2)
    )
    Z[36] = lambda x, y: (
        4 * x * (x**2 - y**2) * (r2(x, y) ** 2 - 16 * x**2 * y**2)
        - 8 * x * y**2 * (3 * r2(x, y) ** 2 - 16 * x**2 * y**2)
    )
    Z[37] = lambda x, y: 3 * (
        70 * r2(x, y) ** 4
        - 140 * r2(x, y) ** 3
        + 90 * r2(x, y) ** 2
        - 20 * r2(x, y)
        + 1
    )
    return Z


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
    and goes to 0 smoothly on [-a-b,-a]U[a,a+b] using a bump function
    For the discretization of indicator functions, we advise b=1, so that
    a=0, b=1 yields a bump.

    :param torch.Tensor x: tensor of arbitrary size
        input.
    :param Float a: radius (default is 1)
    :param Float b: interval on which the function goes to 0. (default is 1)

    :return: the bump function sampled at points x
    :rtype: torch.Tensor

    :Examples:

    >>> import deepinv as dinv
    >>> x = torch.linspace(-15, 15, 31)
    >>> X, Y = torch.meshgrid(x, x, indexing = 'ij')
    >>> R = torch.sqrt(X**2 + Y**2)
    >>> Z = bump_function(R, 3, 1)
    >>> Z = Z / torch.sum(Z)
    >>> dinv.utils.plot(Z[None])
    """
    v = torch.zeros_like(x)
    v[torch.abs(x) <= a] = 1
    I = (torch.abs(x) > a) * (torch.abs(x) < a + b)
    v[I] = torch.exp(-1.0 / (1.0 - ((torch.abs(x[I]) - a) / b) ** 2)) / np.exp(-1.0)
    return v


class ProductConvolutionBlurGenerator(PhysicsGenerator):
    r"""
    Generates a dictionary {'h', 'w'} of parameters to be used within :meth:`deepinv.physics.blur.SpaceVaryingBlur`

    :param deepinv.physics.generator.PSFGenerator psf_generator: A psf generator (e.g. generator = DiffractionBlurGenerator((1, psf_size, psf_size), fc=0.25))
    :param int n_eigen_psf: each psf in the field of view will be a linear combination of n_eigen_psf eigen psfs
    :param tuple img_size: image size HxW (defaults (512, 512))
    :param tuple spacing: steps between the psfs used for interpolation (defaults (H//8, W//8))
    :param str padding: boundary conditions in (options = `valid`, `circular`, `replicate`, `reflect`), defaults `valid`

    :return: a ProductConvolutionBlurGenerator function

    |sep|

    :Examples:

    >>> psf_size = 41
    >>> step = []
    >>> psf_generator = DiffractionBlurGenerator((1, psf_size, psf_size), fc=0.25)
    >>> pc_generator = ProductConvolutionBlurGenerator(psf_generator, 8)

    """

    def __init__(
        self,
        psf_generator=None,
        img_size: tuple = (512, 512),
        n_eigen_psf: int = 10,
        spacing: tuple = (64, 64),
        padding: str = "valid",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.psf_generator = psf_generator
        self.img_size = img_size
        self.n_eigen_psf = n_eigen_psf
        self.spacing = spacing
        self.padding = padding

    def step(self, batch_size: int = 1, sigma: float = None, l: float = None):
        r"""
        Generate a random motion blur PSF with parameters :math: '\sigma' and :math: `l`

        :param float sigma: the standard deviation of the Gaussian Process
        :param float l: the length scale of the trajectory

        :return: the generated PSF of shape `(batch_size, 1, kernel_size, kernel_size)`
        :rtype: torch.Tensor
        """

        # Generating psfs on a grid
        n0, n1 = self.img_size
        s0, s1 = self.spacing

        n_psf = (n0 // s0) * (n1 // s1)
        psfs = self.psf_generator.step(n_psf)["filter"]
        psf_size = psfs.shape[-1]

        # Computing the eigen-psfs
        psfs_reshape = psfs.reshape(n_psf, psf_size * psf_size)
        U, S, V = torch.svd_lowrank(psfs_reshape, q=self.n_eigen_psf)
        eigen_psf = (V.T).reshape(self.n_eigen_psf, psf_size, psf_size)[:, None, None]
        coeffs = psfs_reshape @ V

        # Interpolating the psfs coefficients with Thinplate splines
        T0 = torch.linspace(0, 1, n0 // s0, **self.factory_kwargs)
        T1 = torch.linspace(0, 1, n1 // s1, **self.factory_kwargs)
        yy, xx = torch.meshgrid(T0, T1)
        X = torch.stack((yy.flatten(), xx.flatten()), dim=1)
        tps = ThinPlateSpline(0.0, **self.factory_kwargs)
        tps.fit(X, coeffs)
        T0 = torch.linspace(0, 1, n0, **self.factory_kwargs)
        T1 = torch.linspace(0, 1, n1, **self.factory_kwargs)
        yy, xx = torch.meshgrid(T0, T1)
        w = tps.transform(torch.stack((yy.flatten(), xx.flatten()), dim=1)).T
        w = w.reshape(self.n_eigen_psf, n0, n1)[:, None, None]

        # Ending
        params_blur = {"h": eigen_psf, "w": w, "padding": self.padding}
        return params_blur