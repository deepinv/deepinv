from warnings import warn
from typing import Union
from torchvision.transforms.functional import rotate
import torchvision
import torch
import torch.fft as fft
from torch import Tensor
from deepinv.physics.forward import LinearPhysics, DecomposablePhysics, adjoint_function
from deepinv.physics.functional import (
    conv2d,
    conv_transpose2d,
    filter_fft_2d,
    product_convolution2d,
    product_convolution2d_adjoint,
    conv3d_fft,
    conv_transpose3d_fft,
    imresize_matlab,
)


class Downsampling(LinearPhysics):
    r"""
    Downsampling operator for super-resolution problems.

    It is defined as

    .. math::

        y = S (h*x)

    where :math:`h` is a low-pass filter and :math:`S` is a subsampling operator.

    :param torch.Tensor, str, None filter: Downsampling filter. It can be ``'gaussian'``, ``'bilinear'``, ``'bicubic'``
        , ``'sinc'`` or a custom ``torch.Tensor`` filter. If ``None``, no filtering is applied.
    :param tuple[int], None img_size: optional size of the high resolution image `(C, H, W)`.
        If `tuple`, use this fixed image size.
        If `None`, override on-the-fly using input data size and `factor` (note that here, `A_adjoint` will
        only produce even img shapes).
    :param int factor: downsampling factor
    :param str padding: options are ``'valid'``, ``'circular'``, ``'replicate'`` and ``'reflect'``.
        If ``padding='valid'`` the blurred output is smaller than the image (no padding)
        otherwise the blurred output has the same size as the image.

    |sep|

    :Examples:

        Downsampling operator with a gaussian filter:

        >>> from deepinv.physics import Downsampling
        >>> x = torch.zeros((1, 1, 32, 32)) # Define black image of size 32x32
        >>> x[:, :, 16, 16] = 1 # Define one white pixel in the middle
        >>> physics = Downsampling(filter = "gaussian", img_size=(1, 32, 32), factor=2)
        >>> y = physics(x)
        >>> y[:, :, 7:10, 7:10] # Display the center of the downsampled image
        tensor([[[[0.0146, 0.0241, 0.0146],
                  [0.0241, 0.0398, 0.0241],
                  [0.0146, 0.0241, 0.0146]]]])

    """

    def __init__(
        self,
        img_size=None,
        filter=None,
        factor=2,
        device="cpu",
        padding="circular",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(factor, (int, float)):
            raise ValueError("Downsampling factor must be an integer")

        self.imsize = tuple(img_size) if isinstance(img_size, list) else img_size
        self.imsize_dynamic = (3, 128, 128)  # placeholder
        self.padding = padding
        self.device = device

        self.register_buffer("filter", None)
        self.update_parameters(filter=filter, factor=factor, **kwargs)
        self.to(device)

    def A(self, x, filter=None, factor=None, **kwargs):
        r"""
        Applies the downsampling operator to the input image.

        :param torch.Tensor x: input image.
        :param None, str, torch.Tensor filter: Filter :math:`h` to be applied to the input image before downsampling.
            If not ``None``, it uses this filter and stores it as the current filter.
        :param int, float, torch.Tensor factor: downsampling factor. If not `None`, use this factor and store it as current factor.

        .. warning::

            If `factor` is passed, `filter` must also be passed as a `str` or `Tensor`, in order to update the filter to the new factor.

        """
        self.imsize_dynamic = x.shape[-3:]
        self.update_parameters(filter=filter, factor=factor, **kwargs)

        if self.filter is not None:
            x = conv2d(x, self.filter, padding=self.padding)

        x = x[:, :, :: self.factor, :: self.factor]  # downsample

        return x

    def A_adjoint(self, y, filter=None, factor=None, **kwargs):
        r"""
        Adjoint operator of the downsampling operator.

        :param torch.Tensor y: downsampled image.
        :param None, str, torch.Tensor filter: Filter :math:`h` to be applied to the input image before downsampling.
            If not ``None``, it uses this filter and stores it as the current filter.
        :param int, float, torch.Tensor factor: downsampling factor. If not `None`, use this factor and store it as current factor.

        .. warning::

            If `factor` is passed, `filter` must also be passed as a `str` or `Tensor`, in order to update the filter to the new factor.

        """
        if factor is not None:
            self.factor = self.check_factor(factor)

        self.imsize_dynamic = (
            y.shape[-3],
            y.shape[-2] * self.factor,
            y.shape[-1] * self.factor,
        )

        self.update_parameters(filter=filter, factor=factor, **kwargs)

        imsize = self.imsize if self.imsize is not None else self.imsize_dynamic

        if self.filter is not None and self.padding == "valid":
            imsize = (
                imsize[0],
                imsize[1] - self.filter.shape[-2] + 1,
                imsize[2] - self.filter.shape[-1] + 1,
            )
        else:
            imsize = imsize[:3]

        x = torch.zeros((y.shape[0],) + imsize, device=y.device, dtype=y.dtype)
        x[:, :, :: self.factor, :: self.factor] = y  # upsample
        if self.filter is not None:
            x = conv_transpose2d(
                x, self.filter, padding=self.padding
            )  # Note: this may be slow against x = conv_transpose2d_fft(x, self.filter) in the case of circular padding

        return x

    def prox_l2(self, z, y, gamma, use_fft=True, **kwargs):
        r"""
        If the padding is circular, it computes the proximal operator with the closed-formula of :footcite:t:`zhu2014fast`.

        Otherwise, it computes it using the conjugate gradient algorithm which can be slow if applied many times.
        """

        if use_fft and self.padding == "circular":  # Formula from (Zhao, 2016)
            z_hat = self.A_adjoint(y) + 1 / gamma * z
            Fz_hat = fft.fft2(z_hat)

            def splits(a, sf):
                """split a into sfxsf distinct blocks
                Args:
                    a: NxCxWxH
                    sf: split factor
                Returns:
                    b: NxCx(W/sf)x(H/sf)x(sf^2)
                """
                b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
                b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
                return b

            top = torch.mean(splits(self.Fh * Fz_hat, self.factor), dim=-1)
            below = torch.mean(splits(self.Fh2, self.factor), dim=-1) + 1 / gamma
            rc = self.Fhc * (top / below).repeat(1, 1, self.factor, self.factor)
            r = torch.real(fft.ifft2(rc))
            return (z_hat - r) * gamma
        else:
            return LinearPhysics.prox_l2(self, z, y, gamma, **kwargs)

    def check_factor(self, factor: Union[int, float, Tensor]) -> int:
        """Check new downsampling factor.

        :param int, float, torch.Tensor factor: downsampling factor to be checked and cast to `int`. If :class:`torch.Tensor`,
            it must be 1D and all its elements must be the same, since downsampling only supports one factor per batch.
        :return: `int`: factor
        """
        if isinstance(factor, (int, float)):
            return int(factor)
        elif isinstance(factor, Tensor):
            if factor.ndim > 1:
                raise ValueError("Factor tensor must be 1D.")

            factor = torch.unique(factor)
            if len(factor) > 1:
                raise ValueError(
                    f"Downsampling only supports one unique factor per batch, but got factors {torch.unique(factor).tolist()}."
                )

            return int(factor.item())
        else:
            raise ValueError(
                f"Factor must be an integer, got {factor} of type {type(factor)}."
            )

    def update_parameters(
        self,
        filter: Tensor = None,
        factor: Union[int, float, Tensor] = None,
        **kwargs,
    ):
        r"""
        Updates the current filter and/or factor.

        :param torch.Tensor filter: New filter to be applied to the input image.
        :param int, float, torch.Tensor factor: New downsampling factor to be applied to the input image.
        """
        if factor is not None and filter is None and self.filter is not None:
            warn(
                "Updating factor but not filter. Filter will not be valid for new factor. Pass filter string or new filter to resolve this."
            )

        if factor is not None:
            self.factor = self.check_factor(factor=factor)

        if filter is not None:
            if isinstance(filter, list):
                # Batched filter strings
                if len(set(filter)) == 1 and isinstance(filter[0], str):
                    filter = filter[0]
                else:
                    raise ValueError(
                        f"Downsampling supports filter string lists if they are identical, but got unique filters {set(filter)}."
                    )

            if isinstance(filter, torch.Tensor):
                filter = filter.to(self.device)
            elif filter == "gaussian":
                filter = gaussian_blur(sigma=(self.factor, self.factor)).to(self.device)
            elif filter == "bilinear":
                filter = bilinear_filter(self.factor).to(self.device)
            elif filter == "bicubic":
                filter = bicubic_filter(self.factor).to(self.device)
            elif filter == "sinc":
                filter = sinc_filter(self.factor, length=4 * self.factor).to(
                    self.device
                )

            self.register_buffer("filter", filter)

        if self.filter is not None:
            imsize = self.imsize if self.imsize is not None else self.imsize_dynamic

            self.register_buffer(
                "Fh",
                filter_fft_2d(self.filter, imsize, real_fft=False).to(self.device),
            )
            self.register_buffer("Fhc", torch.conj(self.Fh))
            self.register_buffer("Fh2", self.Fhc * self.Fh)

        super().update_parameters(**kwargs)


class Upsampling(Downsampling):
    r"""
    Upsampling operator.

    This operator performs the operation

    .. math::
        y = h^T * S^T (x)

    where :math:`S^T` is the adjoint of the subsampling operator and :math:`h` is a low-pass filter.

    :param torch.Tensor, str, None filter: Upsampling filter. It can be ``'gaussian'``, ``'bilinear'``, ``'bicubic'``,
        ``'sinc'`` or a custom ``torch.Tensor`` filter. If ``None``, no filtering is applied.
    :param tuple[int] img_size: size of the output image
    :param int factor: upsampling factor
    :param str padding: options are ``'circular'``, ``'replicate'`` and ``'reflect'``.
    :param str device: cpu or cuda
    """

    def __init__(
        self,
        img_size,
        filter=None,
        factor=2,
        padding="circular",
        device="cpu",
        **kwargs,
    ):

        assert (
            padding != "valid"
        ), "Padding 'valid' is not supported for Upsampling operator."

        super().__init__(
            img_size=img_size,
            filter=filter,
            factor=factor,
            padding=padding,
            device=device,
            **kwargs,
        )

    def A(self, x, **kwargs):
        return super().A_adjoint(x, **kwargs)

    def A_adjoint(self, y, **kwargs):
        return super().A(y, **kwargs)

    def prox_l2(self, z, y, gamma, **kwargs):
        return super().prox_l2(z, y, gamma, **kwargs)


class Blur(LinearPhysics):
    r"""

    Blur operator.

    This forward operator performs

    .. math::

        y = w*x

    where :math:`*` denotes convolution and :math:`w` is a filter.

    :param torch.Tensor filter: Tensor of size (b, 1, h, w) or (b, c, h, w) in 2D; (b, 1, d, h, w) or (b, c, d, h, w) in 3D,
        containing the blur filter, e.g., :func:`deepinv.physics.blur.gaussian_blur`.
    :param str padding: options are ``'valid'``, ``'circular'``, ``'replicate'`` and ``'reflect'``.
        If ``padding='valid'`` the blurred output is smaller than the image (no padding)
        otherwise the blurred output has the same size as the image. (default is ``'valid'``).
        Only ``padding='valid'`` and  ``padding = 'circular'`` are implemented in 3D.
    :param str device: cpu or cuda.


    .. note::

        This class makes it possible to change the filter at runtime by passing a new filter to the forward method, e.g.,
        ``y = physics(x, w)``. The new filter :math:`w` is stored as the current filter.

    .. note::

        This class uses the highly optimized :func:`torch.nn.functional.conv2d` for performing the convolutions in 2D
        and FFT for performing the convolutions in 3D as implemented in :func:`deepinv.physics.functional.conv3d_fft`.
        It uses FFT based convolutions in 3D since :func:`torch.nn.functional.conv3d` is slow for large kernels.

    |sep|

    :Examples:

        Blur operator with a basic averaging filter applied to a 16x16 black image with
        a single white pixel in the center:

        >>> from deepinv.physics import Blur
        >>> x = torch.zeros((1, 1, 16, 16)) # Define black image of size 16x16
        >>> x[:, :, 8, 8] = 1 # Define one white pixel in the middle
        >>> w = torch.ones((1, 1, 2, 2)) / 4 # Basic 2x2 averaging filter
        >>> physics = Blur(filter=w)
        >>> y = physics(x)
        >>> y[:, :, 7:10, 7:10] # Display the center of the blurred image
        tensor([[[[0.2500, 0.2500, 0.0000],
                  [0.2500, 0.2500, 0.0000],
                  [0.0000, 0.0000, 0.0000]]]])

    """

    def __init__(self, filter=None, padding="valid", device="cpu", **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.padding = padding
        assert (
            isinstance(filter, Tensor) or filter is None
        ), f"The filter must be a torch.Tensor or None, got filter of type {type(filter)}."
        self.register_buffer("filter", filter)
        self.to(device)

    def A(self, x, filter=None, **kwargs):
        r"""
        Applies the filter to the input image.

        :param torch.Tensor x: input image.
        :param torch.Tensor filter: Filter :math:`w` to be applied to the input image.
            If not ``None``, it uses this filter instead of the one defined in the class, and
            the provided filter is stored as the current filter.
        """
        self.update_parameters(filter=filter, **kwargs)

        if x.dim() == 4:
            return conv2d(x, filter=self.filter, padding=self.padding)
        elif x.dim() == 5:
            return conv3d_fft(x, filter=self.filter, padding=self.padding)

    def A_adjoint(self, y, filter=None, **kwargs):
        r"""
        Adjoint operator of the blur operator.

        :param torch.Tensor y: blurred image.
        :param torch.Tensor filter: Filter :math:`w` to be applied to the input image.
            If not ``None``, it uses this filter instead of the one defined in the class, and
            the provided filter is stored as the current filter.
        """
        self.update_parameters(filter=filter, **kwargs)

        if y.dim() == 4:
            return conv_transpose2d(y, filter=self.filter, padding=self.padding)
        elif y.dim() == 5:
            return conv_transpose3d_fft(y, filter=self.filter, padding=self.padding)


class BlurFFT(DecomposablePhysics):
    """

    FFT-based blur operator.

    It performs the operation

    .. math::

        y = w*x

    where :math:`*` denotes convolution and :math:`w` is a filter.

    Blur operator based on ``torch.fft`` operations, which assumes a circular padding of the input, and allows for
    the singular value decomposition via ``deepinv.Physics.DecomposablePhysics`` and has fast pseudo-inverse and prox operators.



    :param tuple img_size: Input image size in the form (C, H, W).
    :param torch.Tensor filter: torch.Tensor of size (1, c, h, w) containing the blur filter with h<=H, w<=W and c=1 or c=C e.g.,
        :func:`deepinv.physics.blur.gaussian_blur`.
    :param str device: cpu or cuda

    |sep|

    :Examples:

        BlurFFT operator with a basic averaging filter applied to a 16x16 black image with
        a single white pixel in the center:

        >>> from deepinv.physics import BlurFFT
        >>> x = torch.zeros((1, 1, 16, 16)) # Define black image of size 16x16
        >>> x[:, :, 8, 8] = 1 # Define one white pixel in the middle
        >>> filter = torch.ones((1, 1, 2, 2)) / 4 # Basic 2x2 filter
        >>> physics = BlurFFT(filter=filter, img_size=(1, 16, 16))
        >>> y = physics(x)
        >>> y[y<1e-5] = 0.
        >>> y[:, :, 7:10, 7:10] # Display the center of the blurred image
        tensor([[[[0.2500, 0.2500, 0.0000],
                  [0.2500, 0.2500, 0.0000],
                  [0.0000, 0.0000, 0.0000]]]])
    """

    def __init__(self, img_size, filter: Tensor = None, device="cpu", **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        assert (
            isinstance(filter, Tensor) or filter is None
        ), f"The filter must be a torch.Tensor or None, got filter of type {type(filter)}."
        self.update_parameters(filter=filter, **kwargs)
        self.to(device)

    def A(self, x: Tensor, filter: Tensor = None, **kwargs) -> Tensor:
        self.update_parameters(filter=filter, **kwargs)
        return super().A(x)

    def A_adjoint(self, x: Tensor, filter: Tensor = None, **kwargs) -> Tensor:
        self.update_parameters(filter=filter, **kwargs)
        return super().A_adjoint(x)

    def V_adjoint(self, x: Tensor) -> Tensor:
        return torch.view_as_real(
            fft.rfft2(x, norm="ortho")
        )  # make it a true SVD (see J. Romberg notes)

    def U(self, x):
        return fft.irfft2(
            torch.view_as_complex(x) * self.angle,
            norm="ortho",
            s=self.img_size[-2:],
        )

    def U_adjoint(self, x):
        return torch.view_as_real(
            fft.rfft2(x, norm="ortho") * torch.conj(self.angle)
        )  # make it a true SVD (see J. Romberg notes)

    def V(self, x):
        return fft.irfft2(torch.view_as_complex(x), norm="ortho", s=self.img_size[-2:])

    def update_parameters(self, filter: Tensor = None, **kwargs):
        r"""
        Updates the current filter.

        :param torch.Tensor filter: New filter to be applied to the input image.
        """
        if filter is not None and isinstance(filter, Tensor):
            if self.img_size[0] > filter.shape[1]:
                filter = filter.repeat(1, self.img_size[0], 1, 1)
            mask = filter_fft_2d(filter, self.img_size)
            angle = torch.angle(mask)
            mask = torch.abs(mask).unsqueeze(-1)
            mask = torch.cat([mask, mask], dim=-1)

            self.register_buffer("filter", filter)
            self.register_buffer("angle", torch.exp(-1.0j * angle))
            self.register_buffer("mask", mask)

        super().update_parameters(**kwargs)


class SpaceVaryingBlur(LinearPhysics):
    r"""

    Implements a space varying blur via product-convolution.

    This operator performs

    .. math::

        y = \sum_{k=1}^K h_k \star (w_k \odot x)

    where :math:`\star` is a convolution, :math:`\odot` is a Hadamard product,  :math:`w_k` are multipliers :math:`h_k` are filters.

    :param torch.Tensor w: Multipliers :math:`w_k`. Tensor of size (b, c, K, H, W). b in {1, B} and c in {1, C}
    :param torch.Tensor h: Filters :math:`h_k`. Tensor of size (b, c, K, h, w). b in {1, B} and c in {1, C}, h<=H and w<=W.
    :param padding: options = ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``.
        If ``padding = 'valid'`` the blurred output is smaller than the image (no padding),
        otherwise the blurred output has the same size as the image.
    :param str device: cpu or cuda

    |sep|

    :Examples:

        We show how to instantiate a spatially varying blur operator.

        >>> from deepinv.physics.generator import DiffractionBlurGenerator, ProductConvolutionBlurGenerator
        >>> from deepinv.physics.blur import SpaceVaryingBlur
        >>> from deepinv.utils.plotting import plot
        >>> psf_size = 32
        >>> img_size = (256, 256)
        >>> delta = 16
        >>> psf_generator = DiffractionBlurGenerator((psf_size, psf_size))
        >>> pc_generator = ProductConvolutionBlurGenerator(psf_generator=psf_generator, img_size=img_size)
        >>> params_pc = pc_generator.step(1)
        >>> physics = SpaceVaryingBlur(**params_pc)
        >>> dirac_comb = torch.zeros(img_size).unsqueeze(0).unsqueeze(0)
        >>> dirac_comb[0,0,::delta,::delta] = 1
        >>> psf_grid = physics(dirac_comb)
        >>> plot(psf_grid, titles="Space varying impulse responses")

    """

    def __init__(
        self,
        filters: Tensor = None,
        multipliers: Tensor = None,
        padding: str = None,
        device="cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.update_parameters(filters, multipliers, padding, **kwargs)
        self.to(device)

    def A(
        self, x: Tensor, filters=None, multipliers=None, padding=None, **kwargs
    ) -> torch.Tensor:
        r"""
        Applies the space varying blur operator to the input image.

        It can receive new parameters  :math:`w_k`, :math:`h_k` and padding to be used in the forward operator, and stored
        as the current parameters.

        :param torch.Tensor filters: Multipliers :math:`w_k`. Tensor of size (b, c, K, H, W). b in {1, B} and c in {1, C}
        :param torch.Tensor multipliers: Filters :math:`h_k`. Tensor of size (b, c, K, h, w). b in {1, B} and c in {1, C}, h<=H and w<=W
        :param padding: options = ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``.
            If `padding = 'valid'` the blurred output is smaller than the image (no padding),
            otherwise the blurred output has the same size as the image.
        :param str device: cpu or cuda
        """
        self.update_parameters(filters, multipliers, padding, **kwargs)
        return product_convolution2d(x, self.multipliers, self.filters, self.padding)

    def A_adjoint(
        self, y: Tensor, filters=None, multipliers=None, padding=None, **kwargs
    ) -> torch.Tensor:
        r"""
        Applies the adjoint operator.

        It can receive new parameters :math:`w_k`, :math:`h_k` and padding to be used in the forward operator, and stored
        as the current parameters.

        :param torch.Tensor h: Filters :math:`h_k`. Tensor of size (b, c, K, h, w). b in {1, B} and c in {1, C}, h<=H and w<=W
        :param torch.Tensor w: Multipliers :math:`w_k`. Tensor of size (b, c, K, H, W). b in {1, B} and c in {1, C}
        :param padding: options = ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``.
            If `padding = 'valid'` the blurred output is smaller than the image (no padding),
            otherwise the blurred output has the same size as the image.
        :param str device: cpu or cuda
        """
        self.update_parameters(
            filters=filters, multipliers=multipliers, padding=padding, **kwargs
        )
        return product_convolution2d_adjoint(
            y, self.multipliers, self.filters, self.padding
        )

    def update_parameters(
        self,
        filters: Tensor = None,
        multipliers: Tensor = None,
        padding: str = None,
        **kwargs,
    ):
        r"""
        Updates the current parameters.

        :param torch.Tensor filters: Multipliers :math:`w_k`. Tensor of size (b, c, K, H, W). b in {1, B} and c in {1, C}
        :param torch.Tensor multipliers: Filters :math:`h_k`. Tensor of size (b, c, K, h, w). b in {1, B} and c in {1, C}, h<=H and w<=W
        :param padding: options = ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``.
        """
        if filters is not None and isinstance(filters, Tensor):
            self.register_buffer("filters", filters)
        if multipliers is not None and isinstance(filters, Tensor):
            self.register_buffer("multipliers", multipliers)
        if padding is not None:
            self.padding = padding
        super().update_parameters(**kwargs)


def gaussian_blur(sigma=(1, 1), angle=0):
    r"""
    Gaussian blur filter.

    Defined as

    .. math::
        \begin{equation*}
            G(x, y) = \frac{1}{2\pi\sigma_x\sigma_y} \exp{\left(-\frac{x'^2}{2\sigma_x^2} - \frac{y'^2}{2\sigma_y^2}\right)}
        \end{equation*}

    where :math:`x'` and :math:`y'` are the rotated coordinates obtained by rotating $(x, y)$ around the origin
    by an angle :math:`\theta`:

    .. math::

        \begin{align*}
            x' &= x \cos(\theta) - y \sin(\theta) \\
            y' &= x \sin(\theta) + y \cos(\theta)
        \end{align*}

    with :math:`\sigma_x` and :math:`\sigma_y`  the standard deviations along the :math:`x'` and :math:`y'` axes.


    :param float, tuple[float] sigma: standard deviation of the gaussian filter. If sigma is a float the filter is isotropic, whereas
        if sigma is a tuple of floats (sigma_x, sigma_y) the filter is anisotropic.
    :param float angle: rotation angle of the filter in degrees (only useful for anisotropic filters)
    """
    if isinstance(sigma, (int, float)):
        sigma = (sigma, sigma)

    s = max(sigma)
    c = int(s / 0.3 + 1)
    k_size = 2 * c + 1

    delta = torch.arange(k_size)

    x, y = torch.meshgrid(delta, delta, indexing="ij")
    x = x - c
    y = y - c
    filt = (x / sigma[0]).pow(2)
    filt += (y / sigma[1]).pow(2)
    filt = torch.exp(-filt / 2.0)

    filt = (
        rotate(
            filt.unsqueeze(0).unsqueeze(0),
            angle,
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        .squeeze(0)
        .squeeze(0)
    )

    filt = filt / filt.flatten().sum()

    return filt.unsqueeze(0).unsqueeze(0)


def kaiser_window(beta, length, device="cpu"):
    """Return the Kaiser window of length `length` and shape parameter `beta`."""
    if beta < 0:
        raise ValueError("beta must be greater than 0")
    if length < 1:
        raise ValueError("length must be greater than 0")
    if length == 1:
        return torch.tensor([1.0])
    half = (length - 1) / 2
    n = torch.arange(length, device=device)
    beta = torch.tensor(beta, device=device)
    return torch.i0(beta * torch.sqrt(1 - ((n - half) / half) ** 2)) / torch.i0(beta)


def sinc_filter(factor=2, length=11, windowed=True, device="cpu"):
    r"""
    Anti-aliasing sinc filter multiplied by a Kaiser window.

    The kaiser window parameter is computed as follows:

    .. math::

        A = 2.285 \cdot (L - 1) \cdot 3.14 \cdot \Delta f + 7.95

    where :math:`\Delta f = 2 (2 - \sqrt{2}) / \text{factor}`. Then, the beta parameter is computed as:

    .. math::

        \begin{equation*}
            \beta = \begin{cases}
                0 & \text{if } A \leq 21 \\
                0.5842 \cdot (A - 21)^{0.4} + 0.07886 \cdot (A - 21) & \text{if } 21 < A \leq 50 \\
                0.1102 \cdot (A - 8.7) & \text{otherwise}
            \end{cases}
        \end{equation*}

    :param float factor: Downsampling factor.
    :param int length: Length of the filter.
    """
    if isinstance(factor, torch.Tensor):
        factor = factor.cpu().item()

    deltaf = 2 * (2 - 1.4142136) / factor

    n = torch.arange(length, device=device) - (length - 1) / 2
    filter = torch.sinc(n / factor)

    if windowed:
        A = 2.285 * (length - 1) * 3.14159 * deltaf + 7.95
        if A <= 21:
            beta = 0
        elif A <= 50:
            beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21)
        else:
            beta = 0.1102 * (A - 8.7)

        filter = filter * kaiser_window(beta, length, device=device)

    filter = filter.unsqueeze(0)
    filter = filter * filter.T
    filter = filter.unsqueeze(0).unsqueeze(0)
    filter = filter / filter.sum()
    return filter


def bilinear_filter(factor=2):
    r"""
    Bilinear filter.

    It has size (2*factor, 2*factor) and is defined as

    .. math::

        \begin{equation*}
            w(x, y) = \begin{cases}
                (1 - |x|) \cdot (1 - |y|) & \text{if } |x| \leq 1 \text{ and } |y| \leq 1 \\
                0 & \text{otherwise}
            \end{cases}
        \end{equation*}

    for :math:`x, y \in {-\text{factor} + 0.5, -\text{factor} + 0.5 + 1/\text{factor}, \ldots, \text{factor} - 0.5}`.

    :param int factor: downsampling factor
    """
    if isinstance(factor, torch.Tensor):
        factor = factor.cpu().item()
    x = torch.arange(start=-factor + 0.5, end=factor, step=1) / factor
    w = 1 - x.abs()
    w = torch.outer(w, w)
    w = w / torch.sum(w)
    return w.unsqueeze(0).unsqueeze(0)


def bicubic_filter(factor=2):
    r"""
    Bicubic filter.

    It has size (4*factor, 4*factor) and is defined as

    .. math::

        \begin{equation*}
            w(x, y) = \begin{cases}
                (a + 2)|x|^3 - (a + 3)|x|^2 + 1 & \text{if } |x| \leq 1 \\
                a|x|^3 - 5a|x|^2 + 8a|x| - 4a & \text{if } 1 < |x| < 2 \\
                0 & \text{otherwise}
            \end{cases}
        \end{equation*}

    for :math:`x, y \in {-2\text{factor} + 0.5, -2\text{factor} + 0.5 + 1/\text{factor}, \ldots, 2\text{factor} - 0.5}`.

    :param int factor: downsampling factor
    """
    if isinstance(factor, torch.Tensor):
        factor = factor.cpu().item()
    x = torch.arange(start=-2 * factor + 0.5, end=2 * factor, step=1) / factor
    a = -0.5
    x = x.abs()
    w = ((a + 2) * x.pow(3) - (a + 3) * x.pow(2) + 1) * (x <= 1)
    w += (a * x.pow(3) - 5 * a * x.pow(2) + 8 * a * x - 4 * a) * (x > 1) * (x < 2)
    w = torch.outer(w, w)
    w = w / torch.sum(w)
    return w.unsqueeze(0).unsqueeze(0)


class DownsamplingMatlab(Downsampling):
    """Downsampling with MATLAB imresize

    Downsamples with default MATLAB `imresize`, using a bicubic kernel, antialiasing and reflect padding.

    Wraps `imresize` from a modified version of the `original implementation <https://github.com/sanghyun-son/bicubic_pytorch>`_.

    The adjoint is computed using autograd via :func:`deepinv.physics.adjoint_function`.
    This is because `imresize` with reciprocal of scale is not a correct adjoint.
    Note however the adjoint is quite slow.

    :param int, float factor: downsampling factor
    :param str kernel: MATLAB kernel, supports only `cubic` for bicubic downsampling.
    :param str padding: MATLAB padding type, supports only `reflect` for reflect padding.
    :param bool antialiasing: whether to perform antialiasing in MATLAB downsampling.
        Recommended to set to `True` to match MATLAB.
    """

    def __init__(
        self,
        factor: Union[int, float] = 2,
        kernel: str = "cubic",
        padding: str = "reflect",
        antialiasing: bool = True,
        **kwargs,
    ):
        super().__init__(filter=None, factor=factor, **kwargs)

        self.kernel = kernel
        self.padding = padding
        self.antialiasing = antialiasing

    def A(self, x, factor: Union[int, float] = None, **kwargs):
        """Downsample forward operator

        :param torch.Tensor x: input image
        :param int, float factor: downsampling factor. If not `None`, use this factor and store it as current factor.
        """
        self.update_parameters(factor=factor, **kwargs)
        # Clone because of in-place ops
        return imresize_matlab(
            x.clone(),
            scale=1 / self.factor,
            antialiasing=self.antialiasing,
            kernel=self.kernel,
            padding_type=self.padding,
        )

    def A_adjoint(self, y, factor: Union[int, float] = None, **kwargs):
        """Downsample adjoint operator via autograd.

        :param torch.Tensor y: input measurement
        :param int, float factor: downsampling factor. If not `None`, use this factor and store it as current factor.
        """
        self.update_parameters(factor=factor, **kwargs)

        adj = adjoint_function(
            self.A,
            (*y.shape[:2], y.shape[-2] * self.factor, y.shape[-1] * self.factor),
            device=y.device,
            dtype=y.dtype,
        )
        return adj(y)
