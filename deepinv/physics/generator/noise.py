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

    def __init__(self, sigma_min=0.01, sigma_max=0.5, device: str = "cpu"):
        super().__init__(shape=(1,), device=device)
        self.device = device
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def step(self, batch_size=1, **kwargs):
        r"""
        Generates a batch of noise levels.

        :param int batch_size: batch size

        :return: dictionary with key **'sigma'**: tensor of size (batch_size,).
        :rtype: dict

        """
        sigma = (
            torch.rand(batch_size, device=self.device)
            * (self.sigma_max - self.sigma_min)
            + self.sigma_min
        )
        return {"sigma": sigma}


# if __name__ == "__main__":
#     import deepinv as dinv
#     from deepinv.physics.generator import AccelerationMaskGenerator
#
#     mask_generator = SigmaGenerator() + AccelerationMaskGenerator((32, 32))
#     sigmas = mask_generator.step(4)
