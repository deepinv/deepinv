import torch


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


