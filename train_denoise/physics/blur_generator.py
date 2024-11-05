import torch

import torchvision

from deepinv.physics.blur import rotate
from deepinv.physics.generator import PSFGenerator

def gaussian_blur_padded(sigma=(1, 1), angle=0, filt_size=None):
    r"""
    Padded gaussian blur filter.

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
        device = "cpu"
    elif isinstance(sigma, torch.Tensor):
        device = sigma.device

    s = max(sigma)
    c = int(s / 0.3 + 1)
    k_size = 2 * c + 1

    delta = torch.arange(k_size).to(device)

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

    filt = filt.unsqueeze(0).unsqueeze(0)

    if filt_size is not None:
        filt = torch.nn.functional.pad(
            filt, ((filt_size[0] - filt.shape[-2]) // 2, (filt_size[0] - filt.shape[-2] + 1) // 2,
                   (filt_size[1] - filt.shape[-1]) // 2, (filt_size[1] - filt.shape[-1] + 1) // 2)
        )

    return filt

class GaussianBlurGenerator(PSFGenerator):
    
    def __init__(
        self,
        psf_size: tuple,
        num_channels: int = 1,
        device: str = "cpu",
        dtype: type = torch.float32,
        l: float = 0.3,
        sigma: float = 0.25,
        sigma_min: float = 0.01,
        sigma_max: float = 4.,
    ) -> None:
        kwargs = {"l": l, "sigma": sigma, "sigma_min": sigma_min, "sigma_max": sigma_max}
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

    def step(self, batch_size: int = 1, sigma: float = None):
        r"""
        Generate a random motion blur PSF with parameters :math:`\sigma` and :math:`l`

        :param int batch_size: batch_size.
        :param float sigma: the standard deviation of the Gaussian Process
        :param float l: the length scale of the trajectory

        :return: dictionary with key **'filter'**: the generated PSF of shape `(batch_size, 1, psf_size[0], psf_size[1])`
        """

        sigmas = [self.sigma_min + torch.rand(2, **self.factory_kwargs)*(self.sigma_max-self.sigma_min) for batch in range(batch_size)]
        angles = [(torch.rand(1, **self.factory_kwargs)*180.).item() for batch in range(batch_size)]

        
        kernels = [
            gaussian_blur_padded(sigma, angle, filt_size=self.psf_size) for sigma, angle in zip(sigmas, angles)
        ]
        kernel = torch.cat(kernels, dim=0)

        return {
            "filter": kernel.expand(
                -1,
                self.num_channels,
                -1,
                -1,
            )
        }
