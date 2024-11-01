import torch
import torch.nn as nn
from torch import Tensor
import warnings
from typing import Optional, Union
from numpy import ndarray


class BaseSDESolver(nn.Module):
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
        Defaults to Euler step
        """
        drift, diffusion = self.sde.discretize(x0, t0, *args, **kwargs)
        dt = t1 - t0
        dW = self.randn_like(x0) * abs(dt) ** 0.5
        return x0 + drift * dt + diffusion * dW

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
