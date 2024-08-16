import torch
from deepinv.physics.generator import PhysicsGenerator


class SigmaGenerator(PhysicsGenerator):
    r"""
    Generator for the noise level :math:`\sigma` in the Gaussian noise model.

    The noise level is sampled uniformly from the interval :math:`[\text{sigma_min}, \text{sigma_max}]`.

    :param float sigma_min: minimum noise level
    :param float sigma_max: maximum noise level
    :param str device: device where the tensor is stored

    |sep|

    :Examples:

    >>> from deepinv.physics.generator import SigmaGenerator
    >>> generator = SigmaGenerator()
    >>> _ = torch.manual_seed(0)
    >>> sigma_dict = generator.step() # dict_keys(['sigma'])
    >>> print(sigma_dict['sigma'])
    tensor([0.2532])

    """

    def __init__(
        self,
        sigma_min=0.01,
        sigma_max=0.5,
        rng: torch.Generator = None,
        device: str = "cpu",
    ):
        super().__init__(shape=(1,), device=device, rng=rng)
        self.device = device
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
            torch.rand(batch_size, device=self.device, generator=self.rng)
            * (self.sigma_max - self.sigma_min)
            + self.sigma_min
        )
        return {"sigma": sigma}
