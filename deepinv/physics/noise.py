import torch


class GaussianNoise(torch.nn.Module):
    r"""

    Additive gaussian noise with standard deviation :math:`\sigma`, i.e., :math:`y=z+\epsilon` where :math:`\epsilon\sim \mathcal{N}(0,I\sigma^2)`.

    It can be added to a physics operator in its construction or by setting
    ::

        physics.noise_model = GaussianNoise()

    :param float sigma: Standard deviation of the noise.

    """

    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = torch.nn.Parameter(torch.tensor(sigma), requires_grad=False)

    def forward(self, x):
        r"""
        Adds the noise to measurements x

        :param torch.tensor x: measurements
        :returns: noisy measurements
        """
        return x + torch.randn_like(x) * self.sigma


class PoissonNoise(torch.nn.Module):
    r"""

    Poisson noise is defined as
    :math:`y = \mathcal{P}(\frac{x}{\gamma})`
    where :math:`\gamma` is the gain.

    If ``normalize=True``, the output is divided by the gain, i.e., :math:`\tilde{y} = \gamma y`.

    It can be added to a physics operator in its construction or by setting:
    ::

        physics.noise_model = PoissonNoise()

    :param float gain: gain of the noise.
    :param bool normalize: normalize the output.

    """

    def __init__(self, gain=1.0, normalize=True):
        super().__init__()
        self.normalize = torch.nn.Parameter(
            torch.tensor(normalize), requires_grad=False
        )
        self.gain = torch.nn.Parameter(torch.tensor(gain), requires_grad=False)

    def forward(self, x):
        r"""
        Adds the noise to measurements x

        :param torch.tensor x: measurements
        :returns: noisy measurements
        """
        y = torch.poisson(x / self.gain)
        if self.normalize:
            y *= self.gain
        return y


class PoissonGaussianNoise(torch.nn.Module):
    r"""
    Poisson-Gaussian noise is defined as
    :math:`y = \gamma z + \epsilon` where :math:`z\sim\mathcal{P}(\frac{x}{\gamma})`
    and :math:`\epsilon\sim\mathcal{N}(0, I \sigma^2)`.

    It can be added to a physics operator by setting

    ::

        physics.noise_model = PoissonGaussianNoise()

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

        :param torch.tensor x: measurements
        :returns: noisy measurements
        """
        y = torch.poisson(x / self.gain) * self.gain

        y += torch.randn_like(x) * self.sigma
        return y


class UniformNoise(torch.nn.Module):
    r"""
    Uniform noise is defined as
    :math:`y = x + \epsilon` where :math:`\epsilon\sim\mathcal{U}(-a,a)`.

    It can be added to a physics operator by setting
    ::

        physics.noise_model = UniformNoise()

    :param float a: amplitude of the noise.
    """

    def __init__(self, a=0.1):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(a), requires_grad=False)

    def forward(self, x):
        r"""
        Adds the noise to measurements x

        :param torch.tensor x: measurements
        :returns: noisy measurements
        """
        return x + (self.rand_like(x) - 0.5) * 2 * self.a
