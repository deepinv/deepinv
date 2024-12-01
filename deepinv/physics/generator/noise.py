import torch
from deepinv.physics.generator import PhysicsGenerator


class SigmaGenerator(PhysicsGenerator):
    r"""
    Generator for the noise level :math:`\sigma` in the Gaussian noise model.

    The noise level is sampled uniformly from the interval :math:`[\text{sigma_min}, \text{sigma_max}]`.

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
