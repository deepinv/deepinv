import torch
import torch.nn as nn
from typing import Callable, Union, Optional
from collections.abc import Iterable
import warnings
import deepinv as dinv


class NoiseModel(nn.Module):
    r"""
    Base class for noise model.

    Noise models can be combined via :func:`deepinv.physics.NoiseModel.__mul__`.

    :param Callable noise_model: noise model function :math:`N(y)`.
    :param torch.Generator, None rng: (optional) a pseudorandom random number generator for the parameter generation.
        If provided, it should be on the same device as the input.
    """

    def __init__(self, noise_model: Callable = None, rng: torch.Generator = None):
        super().__init__()
        if noise_model is None:
            noise_model = lambda x: x
        self.noise_model = noise_model
        self.rng = rng
        if rng is not None:
            # NOTE: There is no use in moving RNG states from one device to another
            # as Generator.set_state only supports inputs living on the CPU. Yet,
            # by registering the initial random state as a buffer, it might be
            # moved to another device. This might hinder performance as the tensor
            # will need to be moved back to the CPU if it needs to be used later.
            # We could fix that by letting it be a regular class attribute instead
            # of a buffer but it would prevent it from being included in the
            # state dicts which is undesirable.
            self.register_buffer("initial_random_state", rng.get_state())

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
        if self.rng is not None:
            # NOTE: Generator.set_state expects a tensor living on the CPU.
            self.rng.set_state(self.initial_random_state.cpu())
        else:
            warnings.warn(
                "Cannot reset state for random number generator because it was not initialized. This is ignored."
            )

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

    def update_parameters(self, **kwargs):
        r"""
        Update the parameters of the noise model.

        :param dict kwargs: dictionary of parameters to update.
        """
        if kwargs:
            for key, value in kwargs.items():
                if (
                    value is not None
                    and hasattr(self, key)
                    and isinstance(value, (torch.Tensor, float))
                ):
                    self.register_buffer(key, self._float_to_tensor(value))

    def _float_to_tensor(self, value):
        r"""
        Convert a float or int to a torch.Tensor.

        :param value float or int or torch.Tensor: the input value

        :return: the same value as a torch.Tensor
        :rtype: torch.Tensor
        """
        if value is None:
            return value
        elif isinstance(value, (float, int)):
            return torch.tensor(value, dtype=torch.float32)
        elif isinstance(value, torch.Tensor):
            return value
        else:
            raise ValueError(
                f"Unsupported type for noise level. Expected float, int, or torch.Tensor, got {type(value)}."
            )

    # To handle the transfer between CPU/GPU properly
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = self._get_device_from_args(*args, **kwargs)
        if device is not None and self.rng is not None:
            state = self.rng.get_state()
            # Move the generator to the specified device
            self.rng = torch.Generator(device=device)
            try:
                self.rng.set_state(state)
            except RuntimeError:
                warnings.warn(
                    "Moving the random number generator between CPU/GPU is not possible. Reinitialize the generator on the correct device."
                )

        return self

    # Helper to extract device from .to() arguments
    def _get_device_from_args(self, *args, **kwargs):
        if args:
            if isinstance(args[0], torch.device):
                return args[0]
            elif isinstance(args[0], str):
                return torch.device(args[0])
        if "device" in kwargs:
            return (
                torch.device(kwargs["device"])
                if isinstance(kwargs["device"], str)
                else kwargs["device"]
            )
        return None


class ZeroNoise(NoiseModel):
    r"""
    Zero noise model :math:`y=x`, serves as a placeholder.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        r"""
        Return the same input.

        :param torch.Tensor x: measurements.
        :returns: x.
        """
        return x


class GaussianNoise(NoiseModel):
    r"""
    Gaussian noise :math:`y=z+\epsilon` where :math:`\epsilon\sim \mathcal{N}(0,I\sigma^2)`.

    :param Union[float, torch.Tensor] sigma: Standard deviation of the noise.
    :param torch.Generator, None rng: (optional) a pseudorandom random number generator for the parameter generation.

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

    """

    def __init__(
        self,
        sigma: Union[float, torch.Tensor] = 0.1,
        rng: Optional[torch.Generator] = None,
    ):
        device = _infer_device([sigma, rng])
        super().__init__(rng=rng)
        sigma = self._float_to_tensor(sigma)
        sigma = sigma.to(device)
        self.register_buffer("sigma", sigma)

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
                self.sigma = self.sigma.reshape(
                    (self.sigma.size(0),) + (1,) * (other.dim() - 1)
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
                self.sigma = self.sigma.reshape(
                    (self.sigma.size(0),) + (1,) * (other.dim() - 1)
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
        self.update_parameters(sigma=sigma, **kwargs)
        self.to(x.device)
        return (
            x
            + self.randn_like(x, seed=seed)
            * self.sigma[(...,) + (None,) * (x.dim() - 1)]
        )


class UniformGaussianNoise(NoiseModel):
    r"""
    Gaussian noise :math:`y=z+\epsilon` where
    :math:`\epsilon\sim \mathcal{N}(0,I\sigma^2)` and
    :math:`\sigma \sim\mathcal{U}(\sigma_{\text{min}}, \sigma_{\text{max}})`

    :param Union[float, torch.Tensor] sigma_min: minimum standard deviation of the noise.
    :param Union[float, torch.Tensor] sigma_max: maximum standard deviation of the noise.
    :param torch.Generator, None rng: (optional) a pseudorandom random number generator for the parameter generation.

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

    """

    def __init__(
        self,
        sigma_min: Union[float, torch.Tensor] = 0.0,
        sigma_max: Union[float, torch.Tensor] = 0.5,
        rng: torch.Generator = None,
    ):
        device = _infer_device([sigma_min, sigma_max, rng])
        super().__init__(rng=rng)

        sigma_min = self._float_to_tensor(sigma_min)
        sigma_min = sigma_min.to(device)

        self.register_buffer("sigma_min", sigma_min)

        sigma_max = self._float_to_tensor(sigma_max)
        sigma_max = sigma_max.to(device)

        self.register_buffer("sigma_max", sigma_max)

    def forward(self, x, seed: int = None, **kwargs):
        r"""
        Adds the noise to measurements x.

        :param torch.Tensor x: measurements.
        :param int seed: the seed for the random number generator, if `rng` is provided.
        :returns: noisy measurements.
        """
        self.rng_manual_seed(seed)
        self.update_parameters(**kwargs)
        self.to(x.device)
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

    :param Union[float, torch.Tensor] gain: gain of the noise.
    :param bool normalize: normalize the output.
    :param bool clip_positive: clip the input to be positive before adding noise.
        This may be needed when a NN outputs negative values e.g. when using leaky ReLU.
    :param torch.Generator, None rng: (optional) a pseudorandom random number generator for the parameter generation.

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

    """

    def __init__(
        self,
        gain: Union[float, torch.Tensor] = 1.0,
        normalize: bool = True,
        clip_positive: bool = False,
        rng: Optional[torch.Generator] = None,
    ):
        device = _infer_device([gain, rng])
        super().__init__(rng=rng)

        normalize = torch.tensor(normalize, dtype=torch.bool)
        self.register_buffer("normalize", normalize)
        self.clip_positive = clip_positive

        gain = self._float_to_tensor(gain)
        gain = gain.to(device)
        self.register_buffer("gain", gain)

    def forward(self, x, gain=None, seed: int = None, **kwargs):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :param None, float, torch.Tensor gain: gain of the noise. If not None, it will overwrite the current noise level.
        :param int seed: the seed for the random number generator, if `rng` is provided.
        :returns: noisy measurements
        """
        self.update_parameters(gain=gain, **kwargs)
        self.rng_manual_seed(seed)
        self.to(x.device)
        gain = self.gain[(...,) + (None,) * (x.dim() - 1)]

        if self.clip_positive:
            z = torch.clip(x / gain, min=0.0)
        else:
            # NOTE: PyTorch operations are generally run asynchronously on CUDA
            # devices and the underlying CUDA kernel under
            # torch.poisson typically raises a CUDA-level assertion error
            # when its input has negative entries. Those errors can't be
            # recovered from using Python's exception system due to their
            # asynchronous nature. For this reason we add a manual check if the
            # RNG is on a CUDA device.
            if self.rng is not None and self.rng.device.type == "cuda":
                assert gain > 0, "Gain must be positive"
                assert torch.all(x >= 0), "Input tensor must be non-negative"

            z = x / gain

        y = torch.poisson(z, generator=self.rng)
        if self.normalize:
            y = y * gain
        return y


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
        self.register_buffer("l", self._float_to_tensor(l))

    def forward(self, x, l=None, seed: int = None, **kwargs):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :param None, float, torch.Tensor l: noise level. If not None, it will overwrite the current noise level.
        :returns: noisy measurements
        """
        self.update_parameters(l=l, **kwargs)
        self.to(x.device)
        d = torch.distributions.gamma.Gamma(self.l, self.l / x)
        return d.sample()


class PoissonGaussianNoise(NoiseModel):
    r"""
    Poisson-Gaussian noise :math:`y = \gamma z + \epsilon` where :math:`z\sim\mathcal{P}(\frac{x}{\gamma})`
    and :math:`\epsilon\sim\mathcal{N}(0, I \sigma^2)`.

    This noise model allows to recover the Poisson noise model by setting the standard deviation to zero,
    i.e., :math:`\sigma=0`, and the Gaussian noise model by setting the gain to zero, i.e., :math:`\gamma\to0`.

    .. note::

        If :math:`\gamma=0`, the model will clamp the input to a small value
        to avoid division by zero, i.e., :math:`\gamma=\max(\gamma, \text{min\_gain})`.


    :param Union[float, torch.Tensor] gain: gain of the noise.
    :param Union[float, torch.Tensor] sigma: Standard deviation of the noise.
    :param bool clip_positive: (optional) if True, the input is clipped to be positive before adding noise.
    :param torch.Generator, None rng: (optional) a pseudorandom random number generator for the parameter generation.

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

    """

    def __init__(
        self,
        gain: Union[float, torch.Tensor] = 1.0,
        sigma: Union[float, torch.Tensor] = 0.1,
        clip_positive: bool = False,
        min_gain: [float, torch.Tensor] = 1e-12,
        rng: Union[torch.Generator, None] = None,
    ):
        device = _infer_device([gain, sigma, rng])
        super().__init__(rng=rng)

        self.clip_positive = clip_positive
        self.min_gain = min_gain
        gain = self._float_to_tensor(gain)
        gain = gain.to(device)
        self.register_buffer("gain", gain)

        sigma = self._float_to_tensor(sigma)
        sigma = sigma.to(device)
        self.register_buffer("sigma", sigma)

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
        self.update_parameters(
            gain=gain,
            sigma=sigma,
            **kwargs,
        )
        self.rng_manual_seed(seed)
        self.to(x.device)

        gain = self.gain[(...,) + (None,) * (x.dim() - 1)]
        sigma = self.sigma[(...,) + (None,) * (x.dim() - 1)]

        gain = torch.clip(gain, min=self.min_gain)

        if self.clip_positive:
            y = torch.poisson(torch.clip(x / gain, min=0.0), generator=self.rng) * gain
        else:
            # NOTE: PyTorch operations are generally run asynchronously on CUDA
            # devices and the underlying CUDA kernel under
            # torch.poisson typically raises a CUDA-level assertion error
            # when its input has negative entries. Those errors can't be
            # recovered from using Python's exception system due to their
            # asynchronous nature. For this reason we add a manual check if the
            # RNG is on a CUDA device.
            if self.rng is not None and self.rng.device.type == "cuda":
                assert gain > 0, "Gain must be positive"
                assert torch.all(x >= 0), "Input tensor must be non-negative"

            y = torch.poisson(x / gain, generator=self.rng) * gain

        y = y + self.randn_like(x) * sigma

        return y


class UniformNoise(NoiseModel):
    r"""
    Uniform noise :math:`y = x + \epsilon` where :math:`\epsilon\sim\mathcal{U}(-a,a)`.

    :param Union[float, torch.Generator] a: amplitude of the noise.
    :param torch.Generator, None rng: (optional) a pseudorandom random number generator for the parameter generation.

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

    """

    def __init__(
        self, a: Union[float, torch.Tensor] = 0.1, rng: torch.Generator = None
    ):
        device = _infer_device([a, rng])
        super().__init__(rng=rng)

        a = self._float_to_tensor(a)
        a = a.to(device)
        self.register_buffer("a", a)

    def forward(self, x, a=None, seed: int = None, **kwargs):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :param float, torch.Tensor a: amplitude of the noise. If not None, it will overwrite the current noise level.
        :param int seed: the seed for the random number generator, if `rng` is provided.
        :returns: noisy measurements
        """
        self.update_parameters(a=a, **kwargs)
        self.rng_manual_seed(seed)
        self.to(x.device)
        return (
            x
            + (self.rand_like(x, seed=seed) - 0.5)
            * 2
            * self.a[(...,) + (None,) * (x.dim() - 1)]
        )


class LogPoissonNoise(NoiseModel):
    r"""
    Log-Poisson noise :math:`y = \frac{1}{\mu} \log(\frac{\mathcal{P}(\exp(-\mu x) N_0)}{N_0})`.

    This noise model is mostly used for modelling the noise for (low dose) computed tomography measurements.
    Here, N0 describes the average number of measured photons. It acts as a noise-level parameter, where a
    larger value of N0 corresponds to a lower strength of the noise.
    The value mu acts as a normalization constant of the forward operator. Consequently it should be chosen antiproportionally to the image size.

    For more details on the interpretation of the parameters for CT measurements, we refer to the paper :footcite:t:`leuschner2021lodopab`.

    :param Union[float, torch.Tensor] N0: number of photons
    :param Union[float, torch.Tensor] mu: normalization constant
    :param torch.Generator, None rng: (optional) a pseudorandom random number generator for the parameter generation.

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

    """

    def __init__(
        self,
        N0: Union[float, torch.Tensor] = 1024.0,
        mu: Union[float, torch.Tensor] = 1 / 50.0,
        rng: torch.Generator = None,
    ):
        device = _infer_device([N0, mu, rng])
        super().__init__(rng=rng)

        mu = self._float_to_tensor(mu)
        mu = mu.to(device)
        self.register_buffer("mu", mu)

        N0 = self._float_to_tensor(N0)
        N0 = N0.to(device)
        self.register_buffer("N0", N0)

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
        self.update_parameters(mu=mu, N0=N0, **kwargs)
        self.rng_manual_seed(seed)
        self.to(x.device)
        N1_tilde = torch.poisson(self.N0 * torch.exp(-x * self.mu), generator=self.rng)
        y = -torch.log(N1_tilde / self.N0) / self.mu
        return y


class SaltPepperNoise(NoiseModel):
    r"""
    SaltPepper noise :math:`y = \begin{cases} 0 & \text{if } z < p\\ z & \text{if } z \in [p, 1-s]\\ 1 & \text{if } z > 1 - s\end{cases}` with :math:`z\sim\mathcal{U}(0,1)`

    This noise model is also known as impulse noise, is a form of noise sometimes seen on digital images.
    For black-and-white or grayscale images, it presents as sparsely occurring white and black pixels,
    giving the appearance of an image sprinkled with salt and pepper.

    The parameters s and p control the amount of salt (pixel to 1) and pepper (pixel to 0) noise.

    :param Union[float, torch.Tensor] s: amount of salt noise.
    :param Union[float, torch.Tensor] p: amount of pepper noise.
    :param torch.Generator, None rng: (optional) a pseudorandom random number generator for the parameter generation.

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

    """

    def __init__(
        self,
        p: Union[float, torch.Tensor] = 0.025,
        s: Union[float, torch.Tensor] = 0.025,
        rng: torch.Generator = None,
    ):
        device = _infer_device([p, s, rng])
        super().__init__(rng=rng)

        p = self._float_to_tensor(p)
        p = p.to(device)
        self.register_buffer("p", p)

        s = self._float_to_tensor(s)
        s = s.to(device)
        self.register_buffer("s", s)

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


def _infer_device(
    device_held_candidates: Iterable, *, default: torch.device = torch.device("cpu")
) -> torch.device:
    """Infer the device from a list of candidates.

    Check that all candidates bound to a device are bound to the same device and return that device. If no candidate is bound to a device, then return a default device.

    Supported device-held types are ``torch.Tensor``, ``torch.Generator``, and ``deepinv.utils.TensorList``.

    :param Iterable device_held_candidates: list of tensors or generators to infer the device from.
    :param torch.device default: default device to return if no candidates are bound to a device (default: cpu).
    :raises RuntimeError: if more than one device is found among the inputs.
    :return: the device of the candidates or the default device if no candidates are bound to a device.
    """
    input_devices = set()

    for device_held_candidate in device_held_candidates:
        if isinstance(
            device_held_candidate,
            (torch.Tensor, torch.Generator, dinv.utils.TensorList),
        ):
            input_devices.add(device_held_candidate.device)

    if len(input_devices) > 1:
        raise RuntimeError(
            f"Input tensors and Generator should be on the same device. Found devices: {input_devices}."
        )

    return input_devices.pop() if input_devices else default
