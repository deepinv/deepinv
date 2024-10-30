import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable
import warnings


class SDE_solver(nn.Module):
    def __init__(
        self, drift: Callable, diffusion: Callable, rng: torch.Generator = None
    ):
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion

        self.rng = rng
        if rng is not None:
            self.initial_random_state = rng.get_state()

    def step(self, t0, t1, x0):
        pass

    def sample(self, x_init: Tensor, *args, timesteps: Tensor = None, **kwargs) -> Tensor:
        x = x_init
        for i, t in enumerate(timesteps[:-1]):
            x = self.step(t, timesteps[i + 1], x, *args, **kwargs)
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


class Euler_solver(SDE_solver):
    def __init__(
        self, drift: Callable, diffusion: Callable, rng: torch.Generator = None
    ):
        super().__init__(drift, diffusion, rng=rng)

    def step(self, t0, t1, x0, *args, **kwargs):
        dt = abs(t1 - t0)
        dW = self.randn_like(x0) * dt**0.5
        return x0 + self.drift(x0, t0, *args, **kwargs) * dt + self.diffusion(t0) * dW


class Heun_solver(SDE_solver):
    def __init__(
        self, drift: Callable, diffusion: Callable, rng: torch.Generator = None
    ):
        super().__init__(drift, diffusion, rng=rng)

    def step(self, t0, t1, x0, *args, **kwargs):
        dt = abs(t1 - t0)
        dW = self.randn_like(x0) * dt**0.5
        diff_x0 = self.diffusion(t0)
        drift_x0 = self.drift(x0, t0, *args, **kwargs)
        x_euler = x0 + drift_x0 * dt + diff_x0 * dW
        diff_x1 = self.diffusion(t1)
        drift_x1 = self.drift(x_euler, t1, *args, **kwargs)
        return x0 + 0.5 * (drift_x0 + drift_x1) * dt + 0.5 * (diff_x0 + diff_x1) * dW

