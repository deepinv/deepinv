import torch
from .base import Denoiser

def generalized_anscombe_transform(x : torch.Tensor, gain : float | torch.Tensor, sigma : float | torch.Tensor):
    r"""
    Generalized Anscombe Transform (GAT)

    The transform converts a noisy observation :math:`y` from a :class:`Poisson-Gaussian distribution <deepinv.physics.PoissonGaussianNoise>` with
    gain :math:`\gamma` and :class:`Gaussian noise <deepinv.physics.GaussianNoise>` standard deviation :math:`\sigma` to an approximately Gaussian distribution with variance :math:`\gamma`.

    The transform is defined as:

    .. math::

        x' = 2 \sqrt{\gamma x + \frac{3}{8}\gamma^2 + \sigma^2}

    :param torch.Tensor x: tensor corrupted with Poisson-Gaussian noise
    :param float | torch.Tensor gain: Gain of the Poisson distribution :math:`\gamma`
    :param float | torch.Tensor sigma: Standard deviation of the Gaussian noise :math:`\sigma`
    :return torch.Tensor: Transformed measurements
    """
    if gain <= 0:
        raise ValueError(f"Gain should be positive. Got {gain}.")
    if sigma <= 0:
        raise ValueError(f"Sigma should be positive. Got {sigma}.")

    aux = gain*x + 3./8*gain**2 + sigma**2
    out = torch.where(aux > 0, aux.sqrt(), torch.zeros_like(aux))
    return 2. * out


def inverse_generalized_anscombe_transform(x : torch.Tensor, gain : float | torch.Tensor, sigma : float | torch.Tensor):
    r"""
    Inverse Generalized Anscombe Transform (IGAT)

    The transform converts an approximately Gaussian signal :math:`z` (output of the
    :func:`generalized_anscombe_transform`) back to the original
    :class:`Poisson-Gaussian <deepinv.physics.PoissonGaussianNoise>` domain with
    gain :math:`\gamma` and :class:`Gaussian noise <deepinv.physics.GaussianNoise>` standard deviation :math:`\sigma`.

    The transform is defined as the algebraic inverse of the
    :func:`generalized_anscombe_transform`:

    .. math::

        y = \frac{1}{4}x^2 + \frac{1}{4}\sqrt{\frac{3}{2}}\, x^{-1} - \frac{11}{8} x^{-2} + \frac{5}{8}\sqrt{\frac{3}{2}}\, x^{-3} - \frac{1}{8} + \frac{\sigma^2}{\gamma^2}

    :param torch.Tensor x: Anscombe-transformed tensor.
    :param float | torch.Tensor gain: Gain of the Poisson distribution :math:`\gamma`
    :param float | torch.Tensor sigma: Standard deviation of the Gaussian noise :math:`\sigma`
    :return torch.Tensor: Reconstructed measurements in the original domain
    """
    if gain <= 0:
        raise ValueError(f"Gain should be positive. Got {gain}.")
    if sigma <= 0:
        raise ValueError(f"Sigma should be positive. Got {sigma}.")
    x = x/gain
    return gain * (1/4*x**2 + 1/4*(3/2)**.5 * x**(-1) - 11/8 * x**(-2) + 5/8 * (3/2)**.5 * x**(-3) - 1/8 + sigma**2/gain**2)


class AnscombeDenoiser(Denoiser):
    r"""
    Wraps a Gaussian denoiser into a Poisson-Gaussian denoiser using the
    :func:`Generalized Anscombe Transform (GAT) <deepinv.models.generalized_anscombe_transform>`
    and its :func:`inverse <deepinv.models.inverse_generalized_anscombe_transform>`.

    Given a noisy observation :math:`y` corrupted by
    :class:`Poisson-Gaussian noise <deepinv.physics.PoissonGaussianNoise>` with gain :math:`\gamma`
    and Gaussian standard deviation :math:`\sigma`, the wrapper:

    1. Applies the GAT to stabilize the variance to approximately :math:`\gamma^2`:

    .. math::

        z = 2 \sqrt{\gamma y + \frac{3}{8}\gamma^2 + \sigma^2}

    2. Applies the wrapped Gaussian denoiser :math:`\denoisername` at noise level :math:`\gamma`
       (which matches the standard deviation of the GAT output):

    .. math::

        \hat{z} = \denoisername(z,\; \gamma)

    3. Applies the inverse GAT to return to the original domain.
       Setting :math:`u = \hat{z}/\gamma`:

    .. math::

        \hat{y} = \gamma\left(\frac{1}{4}u^2 + \frac{1}{4}\sqrt{\frac{3}{2}}\,u^{-1} - \frac{11}{8}u^{-2} + \frac{5}{8}\sqrt{\frac{3}{2}}\,u^{-3} - \frac{1}{8} + \frac{\sigma^2}{\gamma^2}\right), \qquad u = \frac{\hat{z}}{\gamma}

    .. note::

        When ``gain = 0`` or ``gain = None`` the noise is purely Gaussian and the GAT/IGAT are bypassed:
        the wrapped denoiser is called directly as :math:`\denoisername(y, \sigma)`.

    |sep|

    :Examples:

        >>> import deepinv as dinv
        >>> import torch
        >>> from deepinv.models import AnscombeDenoiserWrapper, DRUNet
        >>> denoiser = DRUNet(pretrained=None)
        >>> anscombe_denoiser = AnscombeDenoiserWrapper(denoiser, sigma_denoiser=25/255)
        >>> y = torch.rand(1, 3, 32, 32)
        >>> with torch.no_grad():
        ...     y_denoised = anscombe_denoiser(y, sigma=0.05, gain=0.1)

    :param deepinv.models.Denoiser denoiser: Gaussian denoiser :math:`\denoisername` to wrap.
    """

    def __init__(self, denoiser: Denoiser,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoiser = denoiser

    def forward(
        self,
        y: torch.Tensor,
        sigma: float | torch.Tensor,
        gain: float | torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Applies the Poisson-Gaussian denoiser via the Anscombe transform.

        :param torch.Tensor y: Noisy measurements corrupted by Poisson-Gaussian noise.
        :param float | torch.Tensor sigma: Standard deviation of the Gaussian noise :math:`\sigma`.
        :param float | torch.Tensor | None gain: Gain of the Poisson distribution :math:`\gamma`.
        :param args: Additional positional arguments passed to the wrapped denoiser.
        :param kwargs: Additional keyword arguments passed to the wrapped denoiser.
        :return torch.Tensor: Denoised measurements in the original domain.
        """
        # Bypass GAT/IGAT for purely Gaussian noise (gain = 0)
        if gain is None or gain == 0:
            return self.denoiser(y, sigma, *args, **kwargs)

        z = generalized_anscombe_transform(y, gain, sigma)

        z_denoised = self.denoiser(z, sigma=gain, *args, **kwargs)

        return inverse_generalized_anscombe_transform(z_denoised, gain, sigma)
