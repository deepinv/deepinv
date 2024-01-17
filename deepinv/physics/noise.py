import torch


class GaussianNoise(torch.nn.Module):
    r"""

    Gaussian noise :math:`y=z+\epsilon` where :math:`\epsilon\sim \mathcal{N}(0,I\sigma^2)`.

    |sep|

    :Examples:

        Adding gaussian noise to a physics operator by setting the ``noise_model``
        attribute of the physics operator:

        >>> from deepinv.physics import Denoising, GaussianNoise
        >>> import torch
        >>> physics = Denoising()
        >>> physics.noise_model = GaussianNoise()
        >>> x = torch.rand(1, 1, 2, 2)
        >>> y = physics(x)

    :param float sigma: Standard deviation of the noise.

    """

    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = torch.nn.Parameter(torch.tensor(sigma), requires_grad=False)

    def forward(self, x):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :returns: noisy measurements
        """
        return x + torch.randn_like(x) * self.sigma


class UniformGaussianNoise(torch.nn.Module):
    r"""
    Gaussian noise :math:`y=z+\epsilon` where
    :math:`\epsilon\sim \mathcal{N}(0,I\sigma^2)` and
    :math:`\sigma \sim\mathcal{U}(\sigma_{\text{min}}, \sigma_{\text{max}})`

    |sep|

    :Examples:

        Adding uniform gaussian noise to a physics operator by setting the ``noise_model``
        attribute of the physics operator:

        >>> from deepinv.physics import Denoising, UniformGaussianNoise
        >>> import torch
        >>> physics = Denoising()
        >>> physics.noise_model = UniformGaussianNoise()
        >>> x = torch.rand(1, 1, 2, 2)
        >>> y = physics(x)


    :param float sigma_min: minimum standard deviation of the noise.
    :param float sigma_max: maximum standard deviation of the noise.
    :param float, torch.Tensor sigma: standard deviation of the noise.
        If ``None``, the noise is sampled uniformly at random
        in :math:`[\sigma_{\text{min}}, \sigma_{\text{max}}]`) during the forward pass. Default: ``None``.

    """

    def __init__(self, sigma_min=0.0, sigma_max=0.5, sigma=None):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma = sigma

    def forward(self, x):
        r"""
        Adds the noise to measurements x.

        :param torch.Tensor x: measurements
        :returns: noisy measurements.
        """
        if self.sigma is None:
            sigma = (
                torch.rand((x.shape[0], 1) + (1,) * (x.dim() - 2))
                * (self.sigma_max - self.sigma_min)
                + self.sigma_min
            )
            self.sigma = sigma.to(x.device)
        noise = torch.randn_like(x) * self.sigma
        return x + noise


class PoissonNoise(torch.nn.Module):
    r"""

    Poisson noise :math:`y = \mathcal{P}(\frac{x}{\gamma})`
    with gain :math:`\gamma>0`.

    If ``normalize=True``, the output is divided by the gain, i.e., :math:`\tilde{y} = \gamma y`.

    |sep|

    :Examples:

        Adding Poisson noise to a physics operator by setting the ``noise_model``
        attribute of the physics operator:

        >>> from deepinv.physics import Denoising, PoissonNoise
        >>> import torch
        >>> physics = Denoising()
        >>> physics.noise_model = PoissonNoise()
        >>> x = torch.rand(1, 1, 2, 2)
        >>> y = physics(x)

    :param float gain: gain of the noise.
    :param bool normalize: normalize the output.
    :param bool clip_positive: clip the input to be positive before adding noise. This may be needed when a NN outputs negative values e.g. when using LeakyReLU.

    """

    def __init__(self, gain=1.0, normalize=True, clip_positive=False):
        super().__init__()
        self.normalize = torch.nn.Parameter(
            torch.tensor(normalize), requires_grad=False
        )
        self.gain = torch.nn.Parameter(torch.tensor(gain), requires_grad=False)
        self.clip_positive = clip_positive

    def forward(self, x):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :returns: noisy measurements
        """
        y = torch.poisson(
            torch.clip(x / self.gain, min=0.0) if self.clip_positive else x / self.gain
        )
        if self.normalize:
            y *= self.gain
        return y


class PoissonGaussianNoise(torch.nn.Module):
    r"""
    Poisson-Gaussian noise :math:`y = \gamma z + \epsilon` where :math:`z\sim\mathcal{P}(\frac{x}{\gamma})`
    and :math:`\epsilon\sim\mathcal{N}(0, I \sigma^2)`.

    |sep|

    :Examples:

        Adding Poisson gaussian noise to a physics operator by setting the ``noise_model``
        attribute of the physics operator:

        >>> from deepinv.physics import Denoising, PoissonGaussianNoise
        >>> import torch
        >>> physics = Denoising()
        >>> physics.noise_model = PoissonGaussianNoise()
        >>> x = torch.rand(1, 1, 2, 2)
        >>> y = physics(x)

    :param float gain: gain of the noise.
    :param float sigma: Standard deviation of the noise.

    """

    def __init__(self, gain=1.0, sigma=0.1):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(gain), requires_grad=False)
        self.sigma = torch.nn.Parameter(torch.tensor(sigma), requires_grad=False)

    def forward(self, x):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :returns: noisy measurements
        """
        y = torch.poisson(x / self.gain) * self.gain

        y += torch.randn_like(x) * self.sigma
        return y


class UniformNoise(torch.nn.Module):
    r"""
    Uniform noise :math:`y = x + \epsilon` where :math:`\epsilon\sim\mathcal{U}(-a,a)`.

    |sep|

    :Examples:

        Adding uniform noise to a physics operator by setting the ``noise_model``
        attribute of the physics operator:

        >>> from deepinv.physics import Denoising, UniformNoise
        >>> import torch
        >>> physics = Denoising()
        >>> physics.noise_model = UniformNoise()
        >>> x = torch.rand(1, 1, 2, 2)
        >>> y = physics(x)

    :param float a: amplitude of the noise.
    """

    def __init__(self, a=0.1):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(a), requires_grad=False)

    def forward(self, x):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :returns: noisy measurements
        """
        return x + (torch.rand_like(x) - 0.5) * 2 * self.a
