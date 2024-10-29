# %%

import torch
import math
from torch import Tensor
import torch.nn as nn
from typing import Callable
import numpy as np
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

    def sample(
        self,
        x_init: Tensor,
        timesteps: Tensor = None
    ):
        x = x_init
        for i,t in enumerate(timesteps[:-1]) :
            x = self.step(t, timesteps[i+1], x)
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
        super().__init__(drift, diffusion, rng = rng)

    def step(self, t0, t1, x0):
        dt = abs(t1 - t0)
        return x0 + self.drift(x0,t0) * dt + self.diffusion(t0) * self.randn_like(x0) * dt**0.5

class Heun_solver(SDE_solver):
    def __init__(
        self, drift: Callable, diffusion: Callable, rng: torch.Generator = None
    ):
        super().__init__(drift, diffusion, rng = rng)

    def step(self, t0, t1, x0):
        dt = abs(t1 - t0)
        return x0 + self.drift(x0,t0) * dt + self.diffusion(t0) * self.randn_like(x0) * dt**0.5


class DiffusionSDE(nn.Module):
    def __init__(
        self,
        f: Callable = lambda x, t: -x,
        g: Callable = lambda t: math.sqrt(2.0),
        prior: Callable = None,
        rng: torch.Generator = None,
        use_backward_ode = False
    ):
        super().__init__()
        self.use_backward_ode = use_backward_ode
        self.drift_forw = lambda x, t: f(x, t)
        self.diff_forw = lambda t: g(t)
        self.forward_sde = Euler_solver(drift=self.drift_forw, diffusion=self.diff_forw, rng=rng)
        if self.use_backward_ode : 
            self.drift_back = lambda x, t: f(x, t) - 0.5 * (g(t) ** 2) * (-prior.grad(x, t))
        else :
            self.drift_back = lambda x, t: f(x, t) - (g(t) ** 2) * (-prior.grad(x, t))
        self.diff_back = lambda t: g(t)
        self.backward_sde = Euler_solver(drift=self.drift_back, diffusion=self.diff_back, rng=rng)


class EDMSDE(DiffusionSDE):
    def __init__(
        self,
        prior: Callable,
        sigma: Callable =  lambda t: t,
        sigma_prime: Callable =  lambda t: 1.,
        s: Callable =  lambda t: 1.,
        s_prime: Callable = lambda t : 0.,
        beta: Callable = lambda t: 1.0 / t,
        rng: torch.Generator = None
    ): 
        super().__init__(prior=prior, rng=rng)
        self.drift_forw = lambda x, t: (- sigma_prime(t) * sigma(t) + beta(t) * sigma(t) ** 2) * (-prior.grad(x, sigma(t)))
        self.diff_forw = lambda t: sigma(t) * (2 * beta(t)) ** 0.5
        if self.use_backward_ode :
            self.diff_back = lambda t: 0.
            self.drift_back = lambda x, t:  - ((s_prime(t) / s(t)) * x  - (s(t) ** 2) * sigma_prime(t) * sigma(t) * (-prior.grad(x / s(t), sigma(t))))
        else : 
            self.drift_back = lambda x, t: (sigma_prime(t) * sigma(t) + beta(t) * sigma(t) ** 2) * (-prior.grad(x, sigma(t)))
            self.diff_back = self.diff_forw
        self.forward_sde = Euler_solver(drift=self.drift_forw, diffusion=self.diff_forw, rng=rng)
        self.backward_sde = Euler_solver(drift=self.drift_back, diffusion=self.diff_back, rng=rng)