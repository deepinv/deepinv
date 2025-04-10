import torch
from deepinv.physics.generator import PhysicsGenerator


class SigmaGenerator(PhysicsGenerator):
    r"""
    Generator for the noise level :math:`\sigma` in the :class:`Gaussian noise model <deepinv.physics.GaussianNoise>`.

    The noise level is sampled uniformly from the interval :math:`[\sigma_{\text{min}}, \sigma_{\text{max}}]`.

    :param float sigma_min: minimum noise level
    :param float sigma_max: maximum noise level
    :param torch.Generator rng: random number generator
    :param str device: device where the tensor is stored
    :param torch.dtype dtype: data type of the generated tensor.

    |sep|

    :Examples:

    >>> from deepinv.physics.generator import SigmaGenerator
    >>> generator = SigmaGenerator()
    >>> sigma_dict = generator.step(seed=0) # dict_keys(['sigma'])
    >>> print(sigma_dict['sigma'])
    tensor([0.2532])

    """

    def __init__(
        self,
        sigma_min=0.01,
        sigma_max=0.5,
        rng: torch.Generator = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(shape=(1,), rng=rng, device=device, dtype=dtype)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def step(self, batch_size=1, seed: int = None, **kwargs):
        r"""
        Generates a batch of noise levels.

        :param int batch_size: batch size
        :param int seed: the seed for the random number generator.

        :return: dictionary with key **'sigma'**: tensor of size (batch_size,).
        :rtype: dict

        """
        self.rng_manual_seed(seed)
        sigma = (
            torch.rand(batch_size, generator=self.rng, **self.factory_kwargs)
            * (self.sigma_max - self.sigma_min)
            + self.sigma_min
        )
        return {"sigma": sigma}


class GainGenerator(PhysicsGenerator):
    r"""

    Generator for the noise level :math:`\gamma` in the :class:`Poisson noise model <deepinv.physics.PoissonNoise>`.

    The gain is sampled uniformly from the interval :math:`[\gamma_\text{min}, \gamma_\text{max}]`.

    :param float gain_min: minimum noise level
    :param float gain_max: maximum noise level
    :param torch.Generator rng: random number generator
    :param str device: device where the tensor is stored
    :param torch.dtype dtype: data type of the generated tensor.

    |sep|

    :Examples:

    >>> from deepinv.physics.generator import GainGenerator
    >>> generator = GainGenerator()
    >>> params = generator.step(seed=0) # params(['gain'])
    >>> print(params['gain'])
    tensor([0.2489])

    """

    def __init__(
        self,
        gain_min=0.1,
        gain_max=0.4,
        rng: torch.Generator = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(shape=(1,), rng=rng, device=device, dtype=dtype)
        self.gain_min = gain_min
        self.gain_max = gain_max

    def step(self, batch_size=1, seed: int = None, **kwargs):
        r"""
        Generates a batch of noise levels.

        :param int batch_size: batch size
        :param int seed: the seed for the random number generator.

        :return: dictionary with key **'gain'**: tensor of size (batch_size,).
        :rtype: dict

        """
        self.rng_manual_seed(seed)
        gain = (
            torch.rand(batch_size, generator=self.rng, **self.factory_kwargs)
            * (self.gain_max - self.gain_min)
            + self.gain_min
        )
        return {"gain": gain}
