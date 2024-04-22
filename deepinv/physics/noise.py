import torch


def to_nn_parameter(x):
    if isinstance(x, torch.Tensor):
        return torch.nn.Parameter(x, requires_grad=False)
    else:
        return torch.nn.Parameter(torch.tensor(x), requires_grad=False)


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
        self.sigma = to_nn_parameter(sigma)

    def forward(self, x, sigma=None, **kwargs):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :param float, torch.Tensor sigma: standard deviation of the noise.
            If not None, it will overwrite the current noise level.
        :returns: noisy measurements
        """
        if sigma is not None:
            self.sigma = to_nn_parameter(sigma)

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

    """

    def __init__(self, sigma_min=0.0, sigma_max=0.5):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, x, **kwargs):
        r"""
        Adds the noise to measurements x.

        :param torch.Tensor x: measurements.
        :returns: noisy measurements.
        """

        sigma = (
            torch.rand((x.shape[0], 1) + (1,) * (x.dim() - 2), device=x.device)
            * (self.sigma_max - self.sigma_min)
            + self.sigma_min
        )
        noise = torch.randn_like(x) * sigma
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
        self.normalize = to_nn_parameter(normalize)
        self.gain = to_nn_parameter(gain)
        self.clip_positive = clip_positive

    def forward(self, x, gain=None, **kwargs):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :param None, float, torch.Tensor gain: gain of the noise. If not None, it will overwrite the current noise level.

        :returns: noisy measurements
        """
        if gain is not None:
            self.gain = to_nn_parameter(gain)

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
        self.gain = to_nn_parameter(gain)
        self.sigma = to_nn_parameter(sigma)

    def forward(self, x, gain=None, sigma=None, **kwargs):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :param None, float, torch.Tensor gain: gain of the noise. If not None, it will overwrite the current gain.
        :param None, float, torch.Tensor sigma: Tensor containing gain and standard deviation.
            If not None, it will overwrite the current gain and standard deviation.
        :returns: noisy measurements
        """
        if gain is not None:
            self.gain = to_nn_parameter(gain)

        if sigma is not None:
            self.sigma = to_nn_parameter(sigma)

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
        self.a = to_nn_parameter(a)

    def forward(self, x, a=None, **kwargs):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :param float, torch.Tensor a: amplitude of the noise. If not None, it will overwrite the current noise level.
        :returns: noisy measurements
        """
        if a is not None:
            self.a = to_nn_parameter(a)

        return x + (torch.rand_like(x) - 0.5) * 2 * self.a


class LogPoissonNoise(torch.nn.Module):
    r"""
    Log-Poisson noise :math:`y = \frac{1}{\mu} \log(\frac{\mathcal{P}(\exp(-\mu x) N_0)}{N_0})`.

    This noise model is mostly used for modelling the noise for (low dose) computed tomography measurements.
    Here, N0 describes the average number of measured photons. It acts as a noise-level parameter, where a
    larger value of N0 corresponds to a lower strength of the noise.
    The value mu acts as a normalization constant of the forward operator. Consequently it should be chosen antiproportionally to the image size.

    For more details on the interpretation of the parameters for CT measurements, we refer to the paper
    `"LoDoPaB-CT, a benchmark dataset for low-dose computed tomography reconstruction" <https://www.nature.com/articles/s41597-021-00893-z>`_.

    :param float N0: number of photons

        |sep|

    :Examples:

        Adding LogPoisson noise to a physics operator by setting the ``noise_model``
        attribute of the physics operator:

        >>> from deepinv.physics import Denoising, LogPoissonNoise
        >>> import torch
        >>> physics = Denoising()
        >>> physics.noise_model = LogPoissonNoise()
        >>> x = torch.rand(1, 1, 2, 2)
        >>> y = physics(x)


    :param float mu: normalization constant
    """

    def __init__(self, N0=1024.0, mu=1 / 50.0):
        super().__init__()
        self.mu = to_nn_parameter(mu)
        self.N0 = to_nn_parameter(N0)

    def forward(self, x, mu=None, N0=None, **kwargs):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :param None, float, torch.Tensor mu: number of photons.
            If not None, it will overwrite the current number of photons.
        :param None, float, torch.Tensor N0: normalization constant.
            If not None, it will overwrite the current normalization constant.
        :returns: noisy measurements
        """
        if N0 is not None:
            self.N0 = to_nn_parameter(N0)

        if mu is not None:
            self.mu = to_nn_parameter(mu)

        N1_tilde = torch.poisson(self.N0 * torch.exp(-x * self.mu))
        y = -torch.log(N1_tilde / self.N0) / self.mu
        return y
