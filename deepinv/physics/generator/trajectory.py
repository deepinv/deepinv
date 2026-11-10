from __future__ import annotations

import torch

from deepinv.physics.generator.base import PhysicsGenerator


class TrajectoryGenerator(PhysicsGenerator):
    r"""Base generator for parameter trajectories over time.

    Generates trajectories of shape ``(batch_size, T, ...)`` of parameters, to be used for
    specifying transforms varying across time.

    .. tip::
        See :ref:`dynamic_imaging` for example usage of trajectory generators with a ``transform`` for modelling transform trajectories.

    :param str param_key: name of the generated parameter.
    :param int T: number of time steps.
    :param torch.Generator rng: torch random generator.
    :param str, torch.device device: device of the generated trajectories.
    :param torch.dtype dtype: data type of the generated trajectories.
    """

    def __init__(
        self,
        param_key: str,
        T: int,
        rng: torch.Generator = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(rng=rng, device=device, dtype=dtype)
        self.param_key = param_key
        self.T = T

    def sample_trajectory(self, batch_size: int) -> torch.Tensor:
        """Sample trajectories. This must be implemented in child classes.

        :param int batch_size: number of trajectories.
        :return torch.Tensor: trajectories of shape ``(batch_size, T, ...)``.
        """
        raise NotImplementedError()

    def step(self, batch_size: int = 1, seed: int = None, **kwargs) -> dict:
        r"""Generate a batch of trajectories.

        :param int batch_size: number of trajectories.
        :param int seed: optional random seed.
        :return: dictionary with key ``param_key`` of shape ``(batch_size, T, ...)``.
        """
        self.rng_manual_seed(seed)
        return {self.param_key: self.sample_trajectory(batch_size)}


class BrownianGenerator(TrajectoryGenerator):
    r"""Generates Brownian motion trajectories.

    The trajectory :math:`p_t` starts at 0. It is generated as

    .. math::
       p_{t+1} = p_{t} + \sigma \sqrt{dt} \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, 1),

    and is reflected at :math:`\pm` ``bound``.

    :param str param_key: name of the generated parameter.
    :param int T: number of time steps.
    :param float dt: time between steps in seconds.
    :param float sigma: diffusion per square-root second.
    :param float bound: maximum absolute value of the trajectory.

    |sep|

    :Example:

    >>> from deepinv.physics.generator import BrownianGenerator
    >>> generator = BrownianGenerator("theta", T=10) + BrownianGenerator("shift", T=10)
    >>> generator.step(batch_size=2)["theta"].shape
    torch.Size([2, 10])
    """

    def __init__(
        self,
        param_key: str,
        T: int,
        dt: float = 0.04,
        sigma: float = 0.75,
        bound: float = 3.0,
        **kwargs,
    ):
        super().__init__(param_key, T, **kwargs)
        self.dt = dt
        self.sigma = sigma
        self.bound = bound

    def sample_trajectory(self, batch_size: int) -> torch.Tensor:
        steps = (
            self.sigma
            * self.dt**0.5
            * torch.randn(
                batch_size, self.T - 1, generator=self.rng, **self.factory_kwargs
            )
        )
        pos = torch.cat(
            [torch.zeros(batch_size, 1, **self.factory_kwargs), steps], dim=1
        ).cumsum(dim=1)
        pos = torch.remainder(pos + self.bound, 4 * self.bound)
        return self.bound - (pos - 2 * self.bound).abs()
