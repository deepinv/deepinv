import torch
from deepinv.physics.generator import PhysicsGenerator


class SigmaGenerator(PhysicsGenerator):
    r"""
    Generator for MRI cartesian acceleration masks.

    It generates a mask of vertical lines for MRI acceleration using fixed sampling in the low frequencies (center of k-space),
    and random uniform sampling in the high frequencies.

    :param tuple img_size: image size.
    :param int acceleration: acceleration factor.
    :param str device: cpu or gpu.

    |sep|

    :Examples:

    >>> generator = PhysicsGenerator((32, 32))
    >>> sigma = generator.step()
    >>> print(sigma)

    """

    def __init__(self, sigma_min=0.01, sigma_max=.5, device: str = "cpu"):
        super().__init__(shape=(1,), device=device)
        self.device = device
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def step(self, batch_size=1):
        r"""

        """
        sigma = torch.rand(batch_size, device=self.device) * (self.sigma_max - self.sigma_min) + self.sigma_min
        return {'sigma': sigma}


if __name__ == "__main__":
    import deepinv as dinv
    from deepinv.physics.generator import AccelerationMaskGenerator
    mask_generator = SigmaGenerator() + AccelerationMaskGenerator((32, 32))
    sigmas = mask_generator.step(4)
    print(sigmas)
