import torch
import torch.nn as nn
from torch import Tensor
import warnings
from typing import Optional, Union
from numpy import ndarray


class BaseSDESolver(nn.Module):
    r"""
    Base class for solving Stochastic Differential Equations (SDEs) from :class:`deepinv.sampling.BaseSDE` of the form:

    ..math:

        d\, x_t = f(x_t, t) d\,t  + g(t) d\,w_t

    where :math:`f` is the drift term, :math:`g` is the diffusion coefficient, and :math:`w_t` is a standard Brownian process.

    :param deepinv.sampling.BaseSDE sde: the SDE to solve.
    :param torch.Generator rng: a random number generator for reproducibility, optional.
    """

    def __init__(
        self,
        sde,
        rng: Optional[torch.Generator] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sde = sde

        self.rng = rng
        if rng is not None:
            self.initial_random_state = rng.get_state()

    def step(self, t0, t1, x0: Tensor, *args, **kwargs) -> Tensor:
        r"""
        Perform a single step with step size from time `t0` to time `t1`, with current state `x0`.

        Args:
            t0: float or Tensor of size (,).
            t1: float or Tensor of size (,).
            y0: Tensor of size (batch_size, d).
            extra0: Any extra state for the solver.

        Returns:
            y1, where y1 is a Tensor of size (batch_size, d).
            extra1: Modified extra state for the solver.
        """
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self, x_init: Tensor, *args, timesteps: Union[Tensor, ndarray] = None, **kwargs
    ) -> Tensor:
        x = x_init
        for t_cur, t_next in zip(timesteps[:-1], timesteps[1:]):
            x = self.step(t_cur, t_next, x, *args, **kwargs)
        return x

    def rng_manual_seed(self, seed: int = None):
        r"""
        Sets the seed for the random number generator.

        :param int seed: the seed to set for the random number generator. If not provided, the current state of the random number generator is used.
            Note: it will be ignored if the random number generator is not initialized.
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

    def randn_like(self, input: torch.Tensor, seed: int = None):
        r"""
        Equivalent to `torch.randn_like` but supports a pseudorandom number generator argument.
        :param int seed: the seed for the random number generator, if `rng` is provided.

        """
        self.rng_manual_seed(seed)
        return torch.empty_like(input).normal_(generator=self.rng)


class EulerSolver(BaseSDESolver):
    def __init__(self, sde, rng: torch.Generator = None, *args, **kwargs):
        super().__init__(sde, rng=rng, *args, **kwargs)

    def step(self, t0, t1, x0: Tensor, *args, **kwargs):
        dt = abs(t1 - t0)
        dW = self.randn_like(x0) * dt**0.5
        drift, diffusion = self.sde.discretize(x0, t0, *args, **kwargs)
        return x0 + drift * dt + diffusion * dW


class HeunSolver(BaseSDESolver):
    def __init__(self, sde, rng: torch.Generator = None, *args, **kwargs):
        super().__init__(sde, rng=rng, *args, **kwargs)

    def step(self, t0, t1, x0: Tensor, *args, **kwargs):
        dt = abs(t1 - t0)
        dW = self.randn_like(x0) * dt**0.5
        drift_0, diffusion_0 = self.sde.discretize(x0, t0, *args, **kwargs)
        x_euler = x0 + drift_0 * dt + diffusion_0 * dW
        drift_1, diffusion_1 = self.sde.discretize(x_euler, t1, *args, **kwargs)

        return (
            x0 + 0.5 * (drift_0 + drift_1) * dt + 0.5 * (diffusion_0 + diffusion_1) * dW
        )


def select_sde_solver(name: str = "euler") -> BaseSDESolver:
    if name.lower() == "euler":
        return EulerSolver
    elif name.lower() == "heun":
        return HeunSolver
    else:
        raise ValueError("Invalid SDE solver name.")
