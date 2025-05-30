import torch
import torch.nn as nn
from typing import Callable
import warnings


class NoiseModel(nn.Module):
    r"""
    Base class for noise model.

    Noise models can be combined via :func:`deepinv.physics.NoiseModel.__mul__`.

    :param Callable noise_model: noise model function :math:`N(y)`.
    :param torch.Generator rng: (optional) a pseudorandom random number generator for the parameter generation.
        If provided, it should be on the same device as the input.
    """

    def __init__(self, noise_model: Callable = None, rng: torch.Generator = None):
        super().__init__()
        if noise_model is None:
            noise_model = lambda x: x
        self.noise_model = noise_model
        self.rng = rng
        if rng is not None:
            self.initial_random_state = rng.get_state()

    def forward(self, input: torch.Tensor, seed: int = None) -> torch.Tensor:
        r"""
        Add noise to the input

        :param torch.Tensor input: input tensor
        :param int seed: the seed for the random number generator.
        """
        self.rng_manual_seed(seed)
        return self.noise_model(input)

    def __mul__(self, other):
        r"""
        Concatenates two noise :math:`N = N_1 \circ N_2` via the mul operation

        The resulting operator will add the noise from both noise models and keep the `rng` of :math:`N_1`.

        :param deepinv.physics.NoiseModel other: Physics operator :math:`A_2`
        :return: (deepinv.physics.NoiseModel) concatenated operator

        """
        noise_model = lambda x: self.noise_model(other.noise_model(x))
        return NoiseModel(
            noise_model=noise_model,
            rng=self.rng,
        )

    def rng_manual_seed(self, seed: int = None):
        r"""
        Sets the seed for the random number generator.

        :param int seed: the seed to set for the random number generator.
            If not provided, the current state of the random number generator is used.
            .. note:: The seed will be ignored if the random number generator is not initialized.
        """
        if seed is not None:
            if self.rng is not None:
                self.rng = self.rng.manual_seed(seed)
            else:
                warnings.warn(
                    "Cannot set seed for random number generator because it is not initialized. The `seed` parameter is ignored."
                )

    def reset_rng(self):
        r"""
        Reset the random number generator to its initial state.
        """
        self.rng.set_state(self.initial_random_state)

    def rand_like(self, input: torch.Tensor, seed: int = None):
        r"""
        Equivalent to `torch.rand_like` but supports a pseudorandom number generator argument.

        :param int seed: the seed for the random number generator, if `rng` is provided.
        """
        self.rng_manual_seed(seed)
        return torch.empty_like(input).uniform_(generator=self.rng)

    def randn_like(self, input: torch.Tensor, seed: int = None):
        r"""
        Equivalent to `torch.randn_like` but supports a pseudorandom number generator argument.

        :param int seed: the seed for the random number generator, if `rng` is provided.
        """
        self.rng_manual_seed(seed)
        return torch.empty_like(input).normal_(generator=self.rng)


class GaussianNoise(NoiseModel):
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
        >>> x = torch.rand(3, 1, 2, 2)
        >>> y = physics(x)

        We can sum 2 GaussianNoise instances:

        >>> gaussian_noise_1 = GaussianNoise(sigma=3.0)
        >>> gaussian_noise_2 = GaussianNoise(sigma=4.0)
        >>> gaussian_noise = gaussian_noise_1 + gaussian_noise_2
        >>> y = gaussian_noise(x)
        >>> gaussian_noise.sigma.item()
        5.0

        We can also multiply a GaussianNoise by a float:

        | :math:`scaled\_gaussian\_noise(x) = \lambda \times gaussian\_noise(x)`

        >>> scaled_gaussian_noise = 3.0 * gaussian_noise
        >>> y = scaled_gaussian_noise(x)
        >>> scaled_gaussian_noise.sigma.item()
        15.0

        We can also create a batch of GaussianNoise with different standard deviations:

        | :math:`x=[x_1, ..., x_b]`
        | :math:`t=[[[[\lambda_1]]], ..., [[[\lambda_b]]]]` a batch of scaling factors.
        | :math:`[t \times gaussian](x) = [\lambda_1 \times gaussian(x_1), ..., \lambda_b \times gaussian(x_b)]`

        >>> t = torch.rand((x.size(0),) + (1,) * (x.dim() - 1)) # if x.shape = (b, 3, 32, 32) then t.shape = (b, 1, 1, 1)
        >>> batch_gaussian_noise = t * gaussian_noise
        >>> y = batch_gaussian_noise(x)
        >>> assert (t[0]*gaussian_noise).sigma.item() == batch_gaussian_noise.sigma[0].item(), "Wrong standard deviation value for the first GaussianNoise."

    :param float sigma: Standard deviation of the noise.
    :param torch.Generator rng: (optional) a pseudorandom random number generator for the parameter generation.
    """

    def __init__(self, sigma=0.1, rng: torch.Generator = None):
        super().__init__(rng=rng)
        self.update_parameters(sigma=sigma)

    def __add__(self, other):
        r"""
        Sum of 2 gaussian noises via + operator.

        :math:`\sigma = \sqrt{\sigma_1^2 + \sigma_2^2}`

        :param deepinv.physics.GaussianNoise other: Gaussian with standard deviation :math:`\sigma`
        :return: (:class:`deepinv.physics.GaussianNoise`) -- Gaussian noise with the sum of the linears operators.
        """
        if not isinstance(other, GaussianNoise):
            raise TypeError(
                f"GaussianNoise Add Operator is unsupported for type {type(other)}"
            )
        return GaussianNoise(sigma=(self.sigma**2 + other.sigma**2) ** (0.5))

    def __mul__(self, other):
        r"""
        Element-wise multiplication of a GaussianNoise via `*` operator.

        0) If `other` is a :class:`NoiseModel`, then applies the multiplication from `NoiseModel`.

        1) If `other` is a :class:`float`, then the standard deviation of the GaussianNoise is multiplied by `other`.

            | :math:`x=[x_1, ..., x_b]` a batch of images.
            | :math:`\lambda` a float.
            | :math:`\sigma = [\lambda \times \sigma_1, ..., \lambda \times \sigma_b]`

        2) If `other` is a :class:`torch.Tensor`, then the standard deviation of the GaussianNoise is multiplied by `other`.

            | :math:`x=[x_1, ..., x_b]` a batch of images.
            | :math:`other=[[[[\lambda_1]]], ..., [[[\lambda_b]]]]` a batch of scaling factors.
            | :math:`\sigma = [\lambda \times \sigma_1, ..., \lambda \times \sigma_b]`

        :param float or torch.Tensor other: Scaling factor for the GaussianNoise's standard deviation.
        :return: (:class:`deepinv.physics.GaussianNoise`) -- A new GaussianNoise with the new standard deviation.
        """
        if isinstance(other, NoiseModel):  # standard NoiseModel multiplication
            return super().__mul__(other)
        elif isinstance(other, float) or isinstance(
            other, torch.Tensor
        ):  # should be a float or a torch.Tensor
            if isinstance(self.sigma, torch.Tensor) and self.sigma.dim() > 0:
                self.sigma = to_nn_parameter(
                    self.sigma.reshape((self.sigma.size(0),) + (1,) * (other.dim() - 1))
                )
            return GaussianNoise(sigma=self.sigma * other)
        else:
            raise NotImplementedError(
                "Multiplication with type {} is not supported.".format(type(other))
            )

    def __rmul__(self, other):
        r"""
        Commutativity of the __mul__ operator.

        :param float or torch.Tensor other: Scaling factor for the GaussianNoise's standard deviation.
        :return: (:class:`deepinv.physics.GaussianNoise`) -- A new GaussianNoise with the new standard deviation.
        """
        if not isinstance(other, NoiseModel):
            return self.__mul__(other)
        else:
            raise NotImplementedError(
                "Multiplication (noise_model * gaussian_noise) with type {} is not supported.".format(
                    type(other)
                )
            )

    def forward(self, x, sigma=None, seed=None, **kwargs):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :param float, torch.Tensor sigma: standard deviation of the noise.
            If not `None`, it will overwrite the current noise level.
        :param int seed: the seed for the random number generator, if `rng` is provided.

        :returns: noisy measurements
        """
        self.update_parameters(sigma=sigma)
        if isinstance(self.sigma, torch.Tensor):
            sigma = self.sigma[(...,) + (None,) * (x.dim() - 1)]
        else:
            sigma = self.sigma
        return x + self.randn_like(x, seed=seed) * sigma.to(x.device)

    def update_parameters(self, sigma=None, **kwargs):
        r"""
        Updates the standard deviation of the noise.

        :param float, torch.Tensor sigma: standard deviation of the noise.
        """
        if sigma is not None:
            self.sigma = to_nn_parameter(sigma)


class UniformGaussianNoise(NoiseModel):
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
    :param torch.Generator rng: (optional) a pseudorandom random number generator for the parameter generation.

    """

    def __init__(self, sigma_min=0.0, sigma_max=0.5, rng: torch.Generator = None):
        super().__init__(rng=rng)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, x, seed: int = None, **kwargs):
        r"""
        Adds the noise to measurements x.

        :param torch.Tensor x: measurements.
        :param int seed: the seed for the random number generator, if `rng` is provided.
        :returns: noisy measurements.
        """
        self.rng_manual_seed(seed)
        sigma = (
            torch.rand(
                (x.shape[0], 1) + (1,) * (x.dim() - 2),
                device=x.device,
                dtype=x.dtype,
                generator=self.rng,
            )
            * (self.sigma_max - self.sigma_min)
            + self.sigma_min
        )
        noise = self.randn_like(x) * sigma
        return x + noise


class PoissonNoise(NoiseModel):
    r"""

    Poisson noise :math:`y = \mathcal{P}(\frac{x}{\gamma})`
    with gain :math:`\gamma>0`.

    If ``normalize=True``, the output is multiplied by the gain, i.e., :math:`\tilde{y} = \gamma y`.

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
    :param bool clip_positive: clip the input to be positive before adding noise.
        This may be needed when a NN outputs negative values e.g. when using leaky ReLU.
    :param torch.Generator rng: (optional) a pseudorandom random number generator for the parameter generation.

    """

    def __init__(
        self, gain=1.0, normalize=True, clip_positive=False, rng: torch.Generator = None
    ):
        super().__init__(rng=rng)
        self.normalize = normalize
        self.update_parameters(gain=gain)
        self.clip_positive = clip_positive

    def forward(self, x, gain=None, seed: int = None, **kwargs):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :param None, float, torch.Tensor gain: gain of the noise. If not None, it will overwrite the current noise level.
        :param int seed: the seed for the random number generator, if `rng` is provided.
        :returns: noisy measurements
        """
        self.update_parameters(gain=gain)
        self.rng_manual_seed(seed)
        if isinstance(self.gain, torch.Tensor):
            gain = self.gain[(...,) + (None,) * (x.dim() - 1)].to(x.device)
        else:
            gain = self.gain.to(x.device)
        y = torch.poisson(
            torch.clip(x / gain, min=0.0) if self.clip_positive else x / gain,
            generator=self.rng,
        )
        if self.normalize:
            y = y * gain
        return y

    def update_parameters(self, gain=None, **kwargs):
        r"""
        Updates the gain of the noise.

        :param float, torch.Tensor gain: gain of the noise.
        """
        if gain is not None:
            self.gain = to_nn_parameter(gain)


class GammaNoise(NoiseModel):
    r"""
    Gamma noise :math:`y = \mathcal{G}(\ell, x/\ell)`

    Follows the (shape, scale) parameterization of the Gamma distribution,
    where the mean is given by :math:`x` and the variance is given by :math:`x/\ell`,
    see https://en.wikipedia.org/wiki/Gamma_distribution for more details.

    Distribution for modelling speckle noise (e.g. SAR images),
    where :math:`\ell>0` controls the noise level (smaller values correspond to higher noise).

    .. warning:: This noise model does not support the random number generator.

    :param float, torch.Tensor l: noise level.
    """

    def __init__(self, l=1.0):
        super().__init__(rng=None)
        if isinstance(l, int):
            l = float(l)
        self.update_parameters(l=l)

    def forward(self, x, l=None, **kwargs):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :param None, float, torch.Tensor l: noise level. If not None, it will overwrite the current noise level.
        :returns: noisy measurements
        """
        self.update_parameters(l=l)
        d = torch.distributions.gamma.Gamma(
            self.l.to(x.device), self.l.to(x.device) / x
        )
        return d.sample()

    def update_parameters(self, l=None, **kwargs):
        r"""
        Updates the noise level.

        :param float, torch.Tensor ell: noise level.
        """
        if l is not None:
            self.l = to_nn_parameter(l)


class PoissonGaussianNoise(NoiseModel):
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
    :param torch.Generator rng: (optional) a pseudorandom random number generator for the parameter generation.
    """

    def __init__(
        self, gain=1.0, sigma=0.1, clip_positive=False, rng: torch.Generator = None
    ):
        super().__init__(rng=rng)
        self.clip_positive = clip_positive
        self.update_parameters(gain=gain, sigma=sigma)

    def forward(self, x, gain=None, sigma=None, seed: int = None, **kwargs):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :param None, float, torch.Tensor gain: gain of the noise. If not None, it will overwrite the current gain.
        :param None, float, torch.Tensor sigma: Tensor containing gain and standard deviation.
            If not None, it will overwrite the current gain and standard deviation.
        :param int seed: the seed for the random number generator, if `rng` is provided.

        :returns: noisy measurements
        """
        self.update_parameters(gain=gain, sigma=sigma)
        self.rng_manual_seed(seed)

        if isinstance(self.gain, torch.Tensor):
            gain = self.gain[(...,) + (None,) * (x.dim() - 1)].to(x.device)
        else:
            gain = self.gain.to(x.device)

        if isinstance(self.sigma, torch.Tensor):
            sigma = self.sigma[(...,) + (None,) * (x.dim() - 1)].to(x.device)
        else:
            sigma = self.sigma.to(x.device)

        if self.clip_positive:
            y = torch.poisson(torch.clip(x / gain, min=0.0), generator=self.rng) * gain
        else:
            y = torch.poisson(x / gain, generator=self.rng) * gain

        y = y + self.randn_like(x) * sigma

        return y

    def update_parameters(self, gain=None, sigma=None, **kwargs):
        r"""
        Updates the gain and standard deviation of the noise.

        :param float, torch.Tensor gain: gain of the noise.
        :param float, torch.Tensor sigma: standard deviation of the noise.
        """
        if gain is not None:
            self.gain = to_nn_parameter(gain)

        if sigma is not None:
            self.sigma = to_nn_parameter(sigma)


class UniformNoise(NoiseModel):
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
    :param torch.Generator rng: (optional) a pseudorandom random number generator for the parameter generation.
    """

    def __init__(self, a=0.1, rng: torch.Generator = None):
        super().__init__(rng=rng)
        self.update_parameters(a=a)

    def forward(self, x, a=None, seed: int = None, **kwargs):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :param float, torch.Tensor a: amplitude of the noise. If not None, it will overwrite the current noise level.
        :param int seed: the seed for the random number generator, if `rng` is provided.
        :returns: noisy measurements
        """
        self.update_parameters(a=a)
        return x + (self.rand_like(x, seed=seed) - 0.5) * 2 * self.a

    def update_parameters(self, a=None, **kwargs):
        r"""
        Updates the amplitude of the noise.

        :param float, torch.Tensor a: amplitude of the noise.
        """
        if a is not None:
            self.a = to_nn_parameter(a)


class LogPoissonNoise(NoiseModel):
    r"""
    Log-Poisson noise :math:`y = \frac{1}{\mu} \log(\frac{\mathcal{P}(\exp(-\mu x) N_0)}{N_0})`.

    This noise model is mostly used for modelling the noise for (low dose) computed tomography measurements.
    Here, N0 describes the average number of measured photons. It acts as a noise-level parameter, where a
    larger value of N0 corresponds to a lower strength of the noise.
    The value mu acts as a normalization constant of the forward operator. Consequently it should be chosen antiproportionally to the image size.

    For more details on the interpretation of the parameters for CT measurements, we refer to the paper
    `"LoDoPaB-CT, a benchmark dataset for low-dose computed tomography reconstruction" <https://www.nature.com/articles/s41597-021-00893-z>`_.

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

    :param float N0: number of photons
    :param float mu: normalization constant
    :param torch.Generator rng: (optional) a pseudorandom random number generator for the parameter generation.
    """

    def __init__(self, N0=1024.0, mu=1 / 50.0, rng: torch.Generator = None):
        super().__init__(rng=rng)
        self.update_parameters(mu=mu, N0=N0)

    def forward(self, x, mu=None, N0=None, seed: int = None, **kwargs):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :param None, float, torch.Tensor mu: number of photons.
            If not None, it will overwrite the current number of photons.
        :param None, float, torch.Tensor N0: normalization constant.
            If not None, it will overwrite the current normalization constant.
        :param int seed: the seed for the random number generator, if `rng` is provided.
        :returns: noisy measurements
        """
        self.update_parameters(mu=mu, N0=N0)
        self.rng_manual_seed(seed)
        N1_tilde = torch.poisson(self.N0 * torch.exp(-x * self.mu), generator=self.rng)
        y = -torch.log(N1_tilde / self.N0) / self.mu
        return y

    def update_parameters(self, mu=None, N0=None, **kwargs):
        r"""
        Updates the number of photons and normalization constant.

        :param float, torch.Tensor mu: number of photons.
        :param float, torch.Tensor N0: normalization constant.
        """
        if mu is not None:
            self.mu = to_nn_parameter(mu)

        if N0 is not None:
            self.N0 = to_nn_parameter(N0)


class SaltPepperNoise(NoiseModel):
    r"""
    SaltPepper noise :math:`y = \begin{cases} 0 & \text{if } z < p\\ z & \text{if } z \in [p, 1-s]\\ 1 & \text{if } z > 1 - s\end{cases}` with :math:`z\sim\mathcal{U}(0,1)`

    This noise model is also known as impulse noise, is a form of noise sometimes seen on digital images.
    For black-and-white or grayscale images, it presents as sparsely occurring white and black pixels,
    giving the appearance of an image sprinkled with salt and pepper.

    The parameters s and p control the amount of salt (pixel to 1) and pepper (pixel to 0) noise.

    |sep|

    :Examples:

        Adding LogPoisson noise to a physics operator by setting the ``noise_model``
        attribute of the physics operator:

        >>> from deepinv.physics import Denoising, SaltPepperNoise
        >>> import torch
        >>> physics = Denoising()
        >>> physics.noise_model = SaltPepperNoise()
        >>> x = torch.rand(1, 1, 2, 2)
        >>> y = physics(x)

    :param float s: amount of salt noise.
    :param float p: amount of pepper noise.
    :param torch.Generator rng: (optional) a pseudorandom random number generator for the parameter generation.
    """

    def __init__(self, p=0.025, s=0.025, rng: torch.Generator = None):
        super().__init__(rng=rng)
        self.update_parameters(p=p, s=s)

    def forward(self, x, p=None, s=None, seed: int = None, **kwargs):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :param None, float, torch.Tensor s: amount of salt noise.
            If not None, it will overwrite the current salt noise.
        :param None, float, torch.Tensor p: amount of pepper noise.
            If not None, it will overwrite the current pepper noise.
        :param int seed: the seed for the random number generator, if `rng` is provided.
        :returns: noisy measurements
        """
        self.update_parameters(p=p, s=s)
        self.rng_manual_seed(seed)

        proba_flip = self.s + self.p
        proba_salt_vs_pepper = self.s / (self.s + self.p)

        mask_flipped = (self.rand_like(x) < proba_flip).float()
        mask_salt = (self.rand_like(x) < proba_salt_vs_pepper).float()
        y = x * (1 - mask_flipped) + mask_flipped * mask_salt
        return y

    def update_parameters(self, s=None, p=None, **kwargs):
        r"""
        Updates the number of photons and normalization constant.

        :param None, float, torch.Tensor s: amount of salt noise.
        :param None, float, torch.Tensor p: amount of pepper noise.
        """
        if s is not None:
            self.s = to_nn_parameter(s)

        if p is not None:
            self.p = to_nn_parameter(p)


def to_nn_parameter(x):
    if isinstance(x, torch.Tensor):
        return torch.nn.Parameter(x, requires_grad=False)
    else:
        return torch.nn.Parameter(torch.tensor(x), requires_grad=False)
