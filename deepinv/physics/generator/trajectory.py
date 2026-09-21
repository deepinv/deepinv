from __future__ import annotations

import torch

from deepinv.physics.generator.base import PhysicsGenerator


class BrownianGenerator(PhysicsGenerator):
    r"""Generates Brownian motion trajectories.

    The trajectory starts at 0 and is reflected at :math:`\pm` ``bound``.

    Generates trajectories of shape ``(batch_size, n_frames)``.

    :param int n_frames: number of time steps.
    :param float dt: time between steps in seconds.
    :param float sigma: diffusion per square-root second.
    :param float bound: maximum absolute value of the trajectory.

    |sep|

    :Example:

    >>> from deepinv.physics.generator import BrownianGenerator
    >>> generator = BrownianGenerator(n_frames=10)
    >>> generator.step(batch_size=2)["pos"].shape
    torch.Size([2, 10])
    """

    def __init__(
        self,
        n_frames: int,
        dt: float = 0.04,
        sigma: float = 0.75,
        bound: float = 3.0,
        rng: torch.Generator = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(rng=rng, device=device, dtype=dtype)
        self.n_frames = n_frames
        self.dt = dt
        self.sigma = sigma
        self.bound = bound

    def step(self, batch_size: int = 1, seed: int = None, **kwargs) -> dict:
        r"""Generate a batch of trajectories.

        :param int batch_size: number of trajectories.
        :param int seed: optional random seed.
        :return: dictionary with key `pos` of shape ``(batch_size, n_frames)``.
        """
        self.rng_manual_seed(seed)
        steps = (
            self.sigma
            * self.dt**0.5
            * torch.randn(
                batch_size, self.n_frames - 1, generator=self.rng, **self.factory_kwargs
            )
        )
        pos = torch.cat(
            [torch.zeros(batch_size, 1, **self.factory_kwargs), steps], dim=1
        ).cumsum(dim=1)
        pos = torch.remainder(pos + self.bound, 4 * self.bound)
        return {"pos": self.bound - (pos - 2 * self.bound).abs()}
