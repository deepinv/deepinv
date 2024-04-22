from torchvision.transforms.functional import rotate
import torchvision
import torch
import numpy as np
import torch.fft as fft
from torch import Tensor
from deepinv.physics.forward import LinearPhysics, DecomposablePhysics
from deepinv.physics.functional import (
    conv2d,
    conv_transpose2d,
    filter_fft_2d,
    product_convolution2d,
    product_convolution2d_adjoint,
)


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
    x = np.arange(start=-factor + 0.5, stop=factor, step=1) / factor
    w = 1 - np.abs(x)
    w = np.outer(w, w)
    w = w / np.sum(w)
    return torch.Tensor(w).unsqueeze(0).unsqueeze(0)


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
    x = np.arange(start=-2 * factor + 0.5, stop=2 * factor, step=1) / factor
    a = -0.5
    x = np.abs(x)
    w = ((a + 2) * np.power(x, 3) - (a + 3) * np.power(x, 2) + 1) * (x <= 1)
    w += (
        (a * np.power(x, 3) - 5 * a * np.power(x, 2) + 8 * a * x - 4 * a)
        * (x > 1)
        * (x < 2)
    )
    w = np.outer(w, w)
    w = w / np.sum(w)
    return torch.Tensor(w).unsqueeze(0).unsqueeze(0)


class Downsampling(LinearPhysics):
    r"""
    Downsampling operator for super-resolution problems.

    It is defined as

    .. math::

        y = S (h*x)

    where :math:`h` is a low-pass filter and :math:`S` is a subsampling operator.

    :param torch.Tensor, str, NoneType filter: Downsampling filter. It can be ``'gaussian'``, ``'bilinear'`` or ``'bicubic'`` or a
        custom ``torch.Tensor`` filter. If ``None``, no filtering is applied.
    :param tuple[int] img_size: size of the input image
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
        img_size,
        filter=None,
        factor=2,
        device="cpu",
        padding="circular",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.factor = factor
        assert isinstance(factor, int), "downsampling factor should be an integer"
        assert len(img_size) == 3, "img_size should be a tuple of length 3, C x H x W"
        self.imsize = img_size
        self.padding = padding
        if isinstance(filter, torch.nn.Parameter):
            self.filter = filter.requires_grad_(False).to(device)
        if isinstance(filter, torch.Tensor):
            self.filter = torch.nn.Parameter(filter, requires_grad=False).to(device)
        elif filter is None:
            self.filter = filter
        elif filter == "gaussian":
            self.filter = torch.nn.Parameter(
                gaussian_blur(sigma=(factor, factor)), requires_grad=False
            ).to(device)
        elif filter == "bilinear":
            self.filter = torch.nn.Parameter(
                bilinear_filter(self.factor), requires_grad=False
            ).to(device)
        elif filter == "bicubic":
            self.filter = torch.nn.Parameter(
                bicubic_filter(self.factor), requires_grad=False
            ).to(device)
        else:
            raise Exception("The chosen downsampling filter doesn't exist")

        if self.filter is not None:
            self.Fh = filter_fft_2d(self.filter, img_size, real_fft=False).to(device)
            self.Fhc = torch.conj(self.Fh)
            self.Fh2 = self.Fhc * self.Fh
            self.Fhc = torch.nn.Parameter(self.Fhc, requires_grad=False)
            self.Fh2 = torch.nn.Parameter(self.Fh2, requires_grad=False)

    def A(self, x, filter=None, **kwargs):
        r"""
        Applies the downsampling operator to the input image.

        :param torch.Tensor x: input image.
        :param None, torch.Tensor filter: Filter :math:`h` to be applied to the input image before downsampling.
            If not ``None``, it uses this filter and stores it as the current filter.
        """
        if filter is not None:
            self.filter = torch.nn.Parameter(filter, requires_grad=False)

        if self.filter is not None:
            x = conv2d(x, self.filter, padding=self.padding)

        x = x[:, :, :: self.factor, :: self.factor]  # downsample
        return x

    def A_adjoint(self, y, filter=None, **kwargs):
        r"""
        Adjoint operator of the downsampling operator.

        :param torch.Tensor y: downsampled image.
        :param None, torch.Tensor filter: Filter :math:`h` to be applied to the input image before downsampling.
            If not ``None``, it uses this filter and stores it as the current filter.
        """
        if filter is not None:
            self.filter = torch.nn.Parameter(filter, requires_grad=False)

        imsize = self.imsize
        if self.filter is not None:
            if self.padding == "valid":
                imsize = (
                    self.imsize[0],
                    self.imsize[1] - self.filter.shape[-2] + 1,
                    self.imsize[2] - self.filter.shape[-1] + 1,
                )
            else:
                imsize = (
                    self.imsize[0],
                    self.imsize[1] - (self.filter.shape[-2] + 1) % 2,
                    self.imsize[2] - (self.filter.shape[-1] + 1) % 2,
                )

        x = torch.zeros((y.shape[0],) + imsize, device=y.device)
        x[:, :, :: self.factor, :: self.factor] = y  # upsample
        if self.filter is not None:
            x = conv_transpose2d(x, self.filter, padding=self.padding)
        return x

    def prox_l2(self, z, y, gamma, use_fft=True):
        r"""
        If the padding is circular, it computes the proximal operator with the closed-formula of
        https://arxiv.org/abs/1510.00143.

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
            return LinearPhysics.prox_l2(self, z, y, gamma)


class Blur(LinearPhysics):
    r"""

    Blur operator.

    This forward operator performs

    .. math:: y = w*x

    where :math:`*` denotes convolution and :math:`w` is a filter.

    This class uses :meth:`torch.nn.functional.conv2d` for performing the convolutions.

    :param torch.Tensor filter: Tensor of size (1, 1, H, W) or (1, C, H, W) containing the blur filter, e.g., :meth:`deepinv.physics.blur.gaussian_filter`.
    :param str padding: options are ``'valid'``, ``'circular'``, ``'replicate'`` and ``'reflect'``. If ``padding='valid'`` the blurred output is smaller than the image (no padding)
        otherwise the blurred output has the same size as the image. (default is ``'valid'``)
    :param str device: cpu or cuda.


    .. note::

        This class allows to change the filter at runtime by passing a new filter to the forward method, e.g.,
        ``y = physics(x, w)``. The new filter :math:`w` is stored as the current filter.

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
        self.padding = padding
        if filter is not None:
            self.filter = torch.nn.Parameter(filter, requires_grad=False).to(device)

    def A(self, x, filter=None, **kwargs):
        r"""
        Applies the filter to the input image.

        :param torch.Tensor x: input image.
        :param torch.Tensor filter: Filter :math:`w` to be applied to the input image.
            If not ``None``, it uses this filter instead of the one defined in the class, and
            the provided filter is stored as the current filter.
        """
        if filter is not None:
            self.filter = torch.nn.Parameter(filter, requires_grad=False)

        return conv2d(x, self.filter, self.padding)

    def A_adjoint(self, y, filter=None, **kwargs):
        r"""
        Adjoint operator of the blur operator.

        :param torch.Tensor y: blurred image.
        :param torch.Tensor filter: Filter :math:`w` to be applied to the input image.
            If not ``None``, it uses this filter instead of the one defined in the class, and
            the provided filter is stored as the current filter.
        """

        if filter is not None:
            self.filter = torch.nn.Parameter(filter, requires_grad=False)

        return conv_transpose2d(y, self.filter, self.padding)


class BlurFFT(DecomposablePhysics):
    """

    FFT-based blur operator.

    It performs the operation

    .. math:: y = w*x

    where :math:`*` denotes convolution and :math:`w` is a filter.

    Blur operator based on ``torch.fft`` operations, which assumes a circular padding of the input, and allows for
    the singular value decomposition via ``deepinv.Physics.DecomposablePhysics`` and has fast pseudo-inverse and prox operators.



    :param tuple img_size: Input image size in the form (C, H, W).
    :param torch.Tensor filter: torch.Tensor of size (1, c, h, w) containing the blur filter with h<=H, w<=W and c=1 or c=C e.g.,
        :meth:`deepinv.physics.blur.gaussian_filter`.
    :param str device: cpu or cuda

    |sep|

    :Examples:

        BlurFFT operator with a basic averaging filter applied to a 16x16 black image with
        a single white pixel in the center:

        >>> from deepinv.physics import BlurFFT
        >>> x = torch.zeros((1, 1, 16, 16)) # Define black image of size 16x16
        >>> x[:, :, 8, 8] = 1 # Define one white pixel in the middle
        >>> filter = torch.ones((1, 1, 2, 2)) / 4 # Basic 2x2 filter
        >>> physics = BlurFFT(filter=filter, img_size=(1, 1, 16, 16))
        >>> y = physics(x)
        >>> y[y<1e-5] = 0.
        >>> y[:, :, 7:10, 7:10] # Display the center of the blurred image
        tensor([[[[0.2500, 0.2500, 0.0000],
                  [0.2500, 0.2500, 0.0000],
                  [0.0000, 0.0000, 0.0000]]]])
    """

    def __init__(self, img_size, filter, device="cpu", **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.img_size = img_size
        self.set_mask(filter)

    def set_mask(self, filter):
        if self.img_size[0] > filter.shape[1]:
            filter = filter.repeat(1, self.img_size[0], 1, 1)
        self.filter = torch.nn.Parameter(filter, requires_grad=False).to(self.device)

        mask = filter_fft_2d(filter, self.img_size).to(self.device)
        self.angle = torch.angle(mask)
        self.angle = torch.exp(-1.0j * self.angle).to(self.device)
        mask = torch.abs(mask).unsqueeze(-1)
        mask = torch.cat([mask, mask], dim=-1)
        self.mask = torch.nn.Parameter(mask, requires_grad=False)

    def A(self, x, filter=None, **kwargs):
        if filter is not None:
            self.set_mask(filter)
        return super().A(x)

    def A_adjoint(self, x, filter=None, **kwargs):
        if filter is not None:
            self.set_mask(filter)
        return super().A_adjoint(x)

    def V_adjoint(self, x):
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


class SpaceVaryingBlur(LinearPhysics):
    r"""

    Implements a space varying blur via product-convolution.

    This operator performs

    .. math::

        y = \sum_{k=1}^K h_k \star (w_k \odot x)

    where :math:`\star` is a convolution, :math:`\odot` is a Hadamard product,  :math:`w_k` are multipliers :math:`h_k` are filters.

    :param torch.Tensor w: Multipliers :math:`w_k`. Tensor of size (K, b, c, H, W). b in {1, B} and c in {1, C}
    :param torch.Tensor h: Filters :math:`h_k`. Tensor of size (K, b, c, h, w). b in {1, B} and c in {1, C}, h<=H and w<=W.
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

    def __init__(self, filters=None, multipliers=None, padding=None, **kwargs):
        super().__init__(**kwargs)
        self.method = "product_convolution2d"
        if self.method == "product_convolution2d":
            self.set_params(filters, multipliers, padding)

    def set_params(self, filters, multipliers, padding):
        if filters is not None:
            self.filters = torch.nn.Parameter(filters, requires_grad=False)
        if multipliers is not None:
            self.multipliers = torch.nn.Parameter(multipliers, requires_grad=False)
        if padding is not None:
            self.padding = padding

    def A(
        self, x: Tensor, filters=None, multipliers=None, padding=None, **kwargs
    ) -> Tensor:
        r"""
        Applies the space varying blur operator to the input image.

        It can receive new parameters  :math:`w_k`, :math:`h_k` and padding to be used in the forward operator, and stored
        as the current parameters.

        :param torch.Tensor filters: Multipliers :math:`w_k`. Tensor of size (K, b, c, H, W). b in {1, B} and c in {1, C}
        :param torch.Tensor multipliers: Filters :math:`h_k`. Tensor of size (K, b, c, h, w). b in {1, B} and c in {1, C}, h<=H and w<=W
        :param padding: options = ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``.
            If `padding = 'valid'` the blurred output is smaller than the image (no padding),
            otherwise the blurred output has the same size as the image.
        :param str device: cpu or cuda
        """
        if self.method == "product_convolution2d":
            self.set_params(filters, multipliers, padding)

            return product_convolution2d(
                x, self.multipliers, self.filters, self.padding
            )
        else:
            raise NotImplementedError("Method not implemented in product-convolution")

    def A_adjoint(
        self, y: Tensor, filters=None, multipliers=None, padding=None, **kwargs
    ) -> Tensor:
        r"""
        Applies the adjoint operator.

        It can receive new parameters :math:`w_k`, :math:`h_k` and padding to be used in the forward operator, and stored
        as the current parameters.

        :param torch.Tensor h: Filters :math:`h_k`. Tensor of size (K, b, c, h, w). b in {1, B} and c in {1, C}, h<=H and w<=W
        :param torch.Tensor w: Multipliers :math:`w_k`. Tensor of size (K, b, c, H, W). b in {1, B} and c in {1, C}
        :param padding: options = ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``.
            If `padding = 'valid'` the blurred output is smaller than the image (no padding),
            otherwise the blurred output has the same size as the image.
        :param str device: cpu or cuda
        """
        if self.method == "product_convolution2d":
            self.set_params(filters, multipliers, padding)

            return product_convolution2d_adjoint(
                y, self.multipliers, self.filters, self.padding
            )
        else:
            raise NotImplementedError("Method not implemented in product-convolution")
